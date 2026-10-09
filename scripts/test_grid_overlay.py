#!/usr/bin/env python3
"""test_grid_overlay.py — coarse battleship-grid path baseline (single VLM call, no CoT).

A baseline to compare against test_pipeline's set-of-marks + chain-of-thought pipeline. The VLM controls
BOTH robots (raph, donnie): it sees the overhead image with a BATTLESHIP GRID overlay and returns, for
each robot, an ordered path of grid cells formed by concatenating adjacent/diagonal cells. The
astar_proj planner then routes each robot through the CENTROIDS of those battleship cells. No
classifier, no chain-of-thought.

Reuses the same segmentation/occupancy, goal separation, planner (node_Path_Translator.astar_proj) and
debug writers as test_pipeline (imported), so the only differences from the main pipeline are the
battleship overlay, the custom single-call prompt, and using battleship cell centroids as the planner's
reference points.

    --mode sim   the Gazebo overhead image  (test_data/overhead.png + poses.json, gazebo calibration)
    --mode real  an AVL_* lab image         (--scene, hardcoded poses, lab_test calibration)

Both modes come from the same scene definitions the other harnesses use: real scenes are read from
test_pipeline_real.SCENES, so an image is never paired with another scene's poses.

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_grid_overlay.py --mode sim \\
        --prompt "Send raph to the chair and donnie to the box"

    python3 src/CoPlanVLM/scripts/test_grid_overlay.py --mode real --scene AVL_3 \\
        --prompt "Send raph to the chair and donnie to the box"

Output (written to --out, default debug/grid_overlay_{sim,real}/):
    marks_overlay.png     — the battleship-grid image sent to the VLM
    robot_paths_waypoints.png — both robots' planned (A*) trajectories + their chosen grid cells
    robot_paths_centroids.png — both robots' reference routes through the chosen cell centroids (pre-A*)
    raw_overhead.png, inflation_overlay.png — identical for every robot, so written once here
  and per robot in <out>/<robot>/: vlm_selections, route_centroids, route_planned.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys

import cv2
from PIL import Image as PILImage

import debug_io
from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world, world_to_ned
from obs_seg.segmenter import segment_frame
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)
from node_Path_Translator import astar_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Executive_API.map_gen import render_battleship_map, _N_COLS, _N_ROWS

# Reuse test_pipeline's helpers (OpenAI client, JSON parse, goal separation, evaluation printer) to
# avoid drift, and test_pipeline_real's SCENES so real-mode images/poses have exactly one definition.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp         # noqa: E402
import test_pipeline_real as tpr   # noqa: E402

# Per-mode defaults, same shape as ablation_test_no_obs_markers._OUT_DIRS. sim and real write to separate
# directories so one mode's results can never clobber the other's.
_OUT_DIRS = {"sim": "debug/grid_overlay_sim", "real": "debug/grid_overlay_real"}
_CAMERAS  = {"sim": "gazebo", "real": "lab_test"}


# ── Prompt (our own; no chain-of-thought) ───────────────────────────────────────
BASELINE_PROMPT = """\
You are a path-planning agent controlling two TurtleBot4 robots named "raph" and "donnie" that share one
workspace. You are given an overhead camera image of the environment with a BATTLESHIP GRID overlay: the
image is divided into labeled cells, 14 columns (A-N, left to right) and 8 rows (1-8, top to bottom),
giving cells such as "A1", "H4", "N8". Columns never go past N and rows never go past 8.
All directions are from the IMAGE's point of view, NOT a robot's or person's perspective: the left/right
of an object means its left/right side as it appears in the image (left = lower column letter, right =
higher column letter), and up/down mean toward the top/bottom of the image (up = lower row number,
down = higher row number).

Identify each robot in the image before planning:
- "raph"   — the round black TurtleBot labeled "raph" (magenta).
- "donnie" — the round black TurtleBot labeled "donnie" (light blue).

YOUR TASK:
For each robot, choose an ordered PATH of grid cells that accomplishes the operator's instruction. Build
each path by concatenating cells that are ADJACENT or DIAGONAL neighbours — each cell must touch the
previous one edge-to-edge or corner-to-corner — starting from the cell the robot is currently in, so the
path is a connected chain the robot can drive along. Use only cells that exist on the grid (never a
column past N or a row past 8). Try to AVOID traveling into obstacles: look at the image and keep each
path on clear, open floor, routing around objects, furniture, walls and other robots rather than driving
a cell onto or through them. You MUST include both robots; if a robot has no task, give it an empty
list.

Respond with EXACTLY one JSON object and nothing else:
{"waypoints": {"raph": ["<cell>", ...], "donnie": ["<cell>", ...]}}

Example — if raph is in A7, donnie is in D2, and the instruction is "Send raph to the chair at C4 and
donnie to the box at H6", a valid response is:
{"waypoints": {"raph": ["A7", "B6", "C5", "C4"], "donnie": ["D2", "E3", "F4", "G5", "H6"]}}
(each path starts at the robot's current cell and every step moves to an adjacent or diagonal cell.)"""

# ── Battleship cell centroids (planner reference points) ─────────────────────────

def _battleship_centroids(w: int, h: int) -> dict[str, tuple[float, float]]:
    """{label: (cx, cy) pixel} for all 14x8 cells — the center of each battleship cell.

    Matches render_battleship_map's geometry (image divided into _N_COLS x _N_ROWS equal cells), so a
    cell's centroid is ((col+0.5)*cell_w, (row+0.5)*cell_h). Fed to astar_proj.build_reference in place
    of the set-of-marks grid_cell_centers.csv.
    """
    cw, ch = w / _N_COLS, h / _N_ROWS
    centers: dict[str, tuple[float, float]] = {}
    for col in range(_N_COLS):
        for row in range(_N_ROWS):
            label = chr(ord("A") + col) + str(row + 1)
            centers[label] = ((col + 0.5) * cw, (row + 0.5) * ch)
    return centers


def _adjacency_warnings(name: str, labels: list) -> None:
    """Warn (non-fatal) when consecutive cells are not 8-connected neighbours, so it's visible when the
    VLM's path is not a connected adjacent/diagonal chain as instructed."""
    prev_rc, prev_lbl = None, None
    for lbl in labels:
        m = re.match(r"^([A-Za-z])(\d+)$", str(lbl).strip())
        if not m:
            print(f"[{name}] malformed cell '{lbl}'")
            prev_rc, prev_lbl = None, None
            continue
        rc = (ord(m.group(1).upper()) - ord("A"), int(m.group(2)) - 1)
        if prev_rc is not None:
            dc, dr = abs(rc[0] - prev_rc[0]), abs(rc[1] - prev_rc[1])
            if max(dc, dr) != 1:   # 0 = repeat, >1 = jump
                print(f"[{name}] non-adjacent step {prev_lbl}->{lbl} (Δcol={dc}, Δrow={dr})")
        prev_rc, prev_lbl = rc, lbl


# ── VLM call (single call; text + battleship image; plain JSON, like the baseline node) ──

def _call_baseline_vlm(prompt: str, map_b64: str, model: str, temperature: float,
                       out_dir: str | None = None) -> tuple[dict, dict]:
    """Mirror the baseline Executive node: system instructions = BASELINE_PROMPT, user content =
    [operator prompt text, battleship overlay image]. Plain JSON reply (no structured schema).

    `map_b64` is the base64 PNG of the overlay, encoded by the caller so the same bytes go to both
    the API and debug_io.save_marks_overlay.

    Returns (routes, usage) where routes = {robot: [cell, ...]} and usage is token counts.
    """
    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(BASELINE_PROMPT)
    print(f'\nOperator instruction: "{prompt}"')
    print("─" * 71)
    # This baseline sends the operator text as a separate user turn, so record it with the system
    # instructions to keep vlm_prompt.txt a complete picture of what the model was given.
    saved_prompt = BASELINE_PROMPT + f'\n\nOperator instruction: "{prompt}"'

    client = tp._openai_client()
    print(f"Planning with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature, instructions=BASELINE_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{map_b64}"},
            ],
        }],
    )
    u = getattr(response, "usage", None)
    usage = {
        "input":  getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "total":  getattr(u, "total_tokens", 0) or 0,
    }

    raw = response.output_text.strip()
    if out_dir is not None:
        debug_io.save_vlm_exchange(out_dir, saved_prompt, raw)

    result = tp._parse_json_reply(raw)
    if result is None:
        sys.exit(f"VLM returned unparseable JSON:\n{raw}")
    # Accept {"waypoints": {...}} or a bare {"raph": [...], "donnie": [...]}.
    routes = result.get("waypoints")
    if not isinstance(routes, dict):
        routes = result if any(k in result for k in ("raph", "donnie")) else {}
    print("─" * 16 + " VLM routes " + "─" * 16)
    for name, labels in routes.items():
        seq = ", ".join(map(str, labels)) if labels else "(empty — holds position)"
        print(f"  {name} ({len(labels) if isinstance(labels, list) else '?'}): {seq}")
    print("─" * 44)
    return routes, usage


def main() -> None:
    # Stage 1: --mode alone, so the real parser below can carry CONCRETE per-mode defaults (which is
    # what --help prints) instead of None sentinels resolved after parsing.
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=("sim", "real"), default="sim",
                             help="sim = Gazebo overhead.png + poses.json; real = an AVL_* lab image "
                                  "chosen with --scene. Default: sim.")
    mode_args, _ = mode_parser.parse_known_args()
    mode = mode_args.mode

    parser = argparse.ArgumentParser(description=__doc__, parents=[mode_parser],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="Operator instruction for the VLM (required)")
    if mode == "real":
        # Registered ONLY in real mode, so `--mode sim --scene AVL_1` fails here with argparse's
        # "unrecognized arguments: --scene". ablation_test_no_obs_markers.py rejects the same
        # combination, but one layer down: it uses parse_known_args and forwards --scene through to
        # test_pipeline.py, whose sim parser has no --scene either.
        parser.add_argument("--scene", choices=list(tpr.SCENES), default=tpr.SCENE,
                            help=f"Image + hardcoded robot poses to run on (default: {tpr.SCENE}, set "
                                 "by the SCENE constant at the top of test_pipeline_real.py)")
    parser.add_argument("--data", default=str(tp.DATA_DIR),
                        help="Dir containing the scene images (+ poses.json in sim mode) "
                             "(default: the test_data/ bundled with the package)")
    parser.add_argument("--camera", default=_CAMERAS[mode], choices=["gazebo", "lab_test"],
                        help=f"Overhead camera calibration for pixel<->world (default: "
                             f"{_CAMERAS[mode]} in {mode} mode)")
    parser.add_argument("--out", default=_OUT_DIRS[mode],
                        help=f"Debug output directory (default: {_OUT_DIRS[mode]})")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the planner call (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    args = parser.parse_args()

    camera = args.camera

    # ── Load the scene's image + robot poses ──────────────────────────────
    # The ONLY mode-dependent stage: both branches leave img_bgr/img_rgb/pil_rgba/w/h plus
    # robot_world_xy and robot_poses_ned defined the same way, so everything below is shared.
    if mode == "sim":
        img_path = os.path.join(args.data, "overhead.png")
        poses_path = os.path.join(args.data, "poses.json")
        if not os.path.exists(img_path):
            sys.exit(f"Image not found: {img_path}\nRun save_overhead.py first.")
        if not os.path.exists(poses_path):
            sys.exit(f"Poses not found: {poses_path}\nRun save_overhead.py first.")
    else:
        scene = tpr.SCENES[args.scene]
        img_path = os.path.join(args.data, scene["image"])
        if not os.path.exists(img_path):
            sys.exit(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_rgba = PILImage.fromarray(img_rgb).convert("RGBA")
    h, w = img_bgr.shape[:2]

    if mode == "sim":
        print(f"Scene: sim ({os.path.basename(img_path)}) — loaded {w}×{h}")
        with open(poses_path) as f:
            poses_raw = json.load(f)
        robot_world_xy = {name: gazebo_to_world(p["x"], p["y"]) for name, p in poses_raw.items()}
        robot_poses_ned = {name: gazebo_to_ned(p["x"], p["y"]) for name, p in poses_raw.items()}
        print("Robot world poses: " +
              ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))
    else:
        print(f"Scene: {args.scene} ({scene['image']}) — loaded {w}×{h}")
        # Planning references use world coords directly; overlay markers use NED (map_gen converts
        # back to world internally), so world -> NED here makes the markers land where intended.
        robot_world_xy = dict(scene["poses"])
        robot_poses_ned = {name: world_to_ned(wx, wy) for name, (wx, wy) in robot_world_xy.items()}
        print("Robot world poses (hardcoded): " +
              ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    os.makedirs(args.out, exist_ok=True)

    # ── Segmentation + occupancy (same as test_pipeline) ──────────────────
    print("Running CLIPSeg segmentation…")
    pix_labels = segment_frame(img_rgb)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    world_poses = [ned_to_world(*p) for p in robot_poses_ned.values() if p]
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    ctx = {"infl": infl}

    # Robot-independent debug artifacts: rendered and written ONCE per run at the top level
    # (they are identical for every robot). `infl_overlay` is reused below as each robot's
    # route_planned.png canvas, so render_inflation_overlay runs only this once.
    os.makedirs(args.out, exist_ok=True)
    infl_overlay = render_inflation_overlay(img_bgr, cleared, infl, meta, camera)
    debug_io.save_raw_overhead(args.out, img_bgr)
    debug_io.save_inflation_overlay(args.out, infl_overlay)

    # ── Battleship overlay image + cell centroids ─────────────────────────
    overlay = render_battleship_map(pil_rgba, robot_poses=robot_poses_ned, camera=camera)
    # Encode once: the same bytes go to the API and to marks_overlay.png.
    _buf = io.BytesIO()
    overlay.save(_buf, format="PNG")
    map_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
    debug_io.save_marks_overlay(args.out, map_b64)
    grid_px = _battleship_centroids(w, h)

    # ── Single VLM call ───────────────────────────────────────────────────
    routes, usage = _call_baseline_vlm(args.prompt, map_b64, args.model, args.temperature,
                                       out_dir=args.out)

    params = dict(ASTAR_PARAMS)

    # ── Goal separation: reserve the poses of robots that will NOT move ───
    # Same block as tp.main(), calling the same grid_planner_utils.separate_goal the live translator
    # node uses, so this baseline is scored on routes the robots could actually execute — without it
    # two robots sent to one target both terminate on the same cell. A robot holding position still
    # occupies its cell, so it is claimed BEFORE the loop: claiming inside would be too late if it
    # happened to be visited second, and reserving up front makes the outcome independent of `routes`
    # key order. "Will not move" covers an empty label list, a malformed route, and a robot with no
    # pose.
    claimed: list = []
    for name, labels in routes.items():
        if isinstance(labels, list) and labels and name in robot_world_xy:
            continue                                  # moving; its goal is claimed after it plans
        pose = robot_world_xy.get(name)
        if pose is not None:
            claimed.append(pose)

    # ── Plan per robot through battleship centroids (astar) ───────────────
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    centroid_routes: dict[str, list] = {}   # pose + chosen battleship-cell centroids (pre-A*)
    for name, labels in routes.items():
        if not isinstance(labels, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose for this scene; skipping.", file=sys.stderr)
            continue

        _adjacency_warnings(name, labels)
        pose_xy = robot_world_xy[name]
        ref, unknown = astar_proj.build_reference(labels, pose_xy, grid_px, camera=camera)
        for lbl in unknown:
            print(f"[{name}] unknown cell '{lbl}'; skipping.")

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

        # Displace the goal off any already-claimed goal BEFORE planning, so A* runs once and the
        # debug figures describe the executed route. Only the final reference point moves —
        # intermediate waypoints and the route itself are untouched. This baseline always plans with
        # A*, so tp.main()'s `planner_name == "astar"` guard collapses away here.
        if tp.MIN_GOAL_SEPARATION > 0 and claimed:
            adjusted = tp.separate_goal(ref[-1], claimed, infl, meta,
                                        min_separation=tp.MIN_GOAL_SEPARATION,
                                        max_shift=float(params["projection_radius"]))
            if adjusted is None:
                print(f"[{name}] goal is within {tp.MIN_GOAL_SEPARATION:.2f} m of another robot and "
                      "no free cell far enough away was found; keeping the original goal.")
            elif math.dist(adjusted, ref[-1]) > 1e-9:
                print(f"[{name}] goal moved {math.dist(adjusted, ref[-1]):.2f} m to clear another "
                      f"robot: ({ref[-1][0]:.2f}, {ref[-1][1]:.2f}) -> "
                      f"({adjusted[0]:.2f}, {adjusted[1]:.2f})")
                ref[-1] = adjusted

        # Recorded AFTER separation so robot_paths_centroids.png shows the reference route that was
        # actually planned, not the pre-adjustment one.
        centroid_routes[name] = list(ref)       # pose -> cell centroids, in order (pre-A*)

        print(f"[{name}] Planning with astar ({len(ref)} reference pts)…")
        world_path, dbg = astar_proj.plan(ref, ctx, meta, params)
        for warn in dbg.get("warnings", []):
            print(f"[{name}] WARNING: {warn}")

        if not world_path:
            print(f"[{name}] planner returned empty path.", file=sys.stderr)
            continue

        length = tp._path_length(world_path)
        path_lengths[name] = length
        world_paths[name] = world_path
        # The EXECUTED endpoint, not the adjusted reference: plan() may project it further, and
        # what the next robot must avoid is where this one actually stops.
        claimed.append(world_path[-1])
        print(f"[{name}] {len(world_path)} waypoints: "
              f"start=({world_path[0][0]:.3f}, {world_path[0][1]:.3f})  "
              f"end=({world_path[-1][0]:.3f}, {world_path[-1][1]:.3f})  "
              f"path length={length:.1f} m")

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        debug_io.save_vlm_selections(out_dir, img_bgr, "battleship", labels, grid_px)
        dbg["start_world"] = pose_xy
        astar_proj.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera,
                              overlay=infl_overlay)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plots ─────────────────────────────────────────
    if world_paths:
        wp_png = os.path.join(args.out, "robot_paths_waypoints.png")
        debug_io.save_paths_with_waypoints(wp_png, img_bgr, world_paths, routes, grid_px, camera)
        print(f"Combined robot paths + VLM cells -> {wp_png}")
    if centroid_routes:
        # Pre-A* reference route through the chosen cell centroids — the one figure that still
        # uses the plain trajectory renderer (there are no VLM picks to overlay on it).
        cpaths_png = os.path.join(args.out, "robot_paths_centroids.png")
        debug_io.save_robot_paths(cpaths_png, img_bgr, centroid_routes, camera)
        print(f"Combined centroid routes -> {cpaths_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
    print(f"Scene: {args.scene if mode == 'real' else 'sim'} ({os.path.basename(img_path)})")
    print("Baseline: battleship grid + astar (no classifier, no CoT)")
    print(f"Tokens: {usage['total']} total "
          f"({usage['input']} input + {usage['output']} output)")
    tp._print_path_lengths(path_lengths)
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
