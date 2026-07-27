#!/usr/bin/env python3
"""test_battleship_baseline.py — coarse battleship-grid path baseline (single VLM call, no CoT).

A baseline to compare against test_pipeline's set-of-marks + chain-of-thought pipeline. The VLM controls
BOTH robots (raph, donnie): it sees the overhead image with a BATTLESHIP GRID overlay and returns, for
each robot, an ordered path of grid cells formed by concatenating adjacent/diagonal cells. The
astar_proj planner then routes each robot through the CENTROIDS of those battleship cells. No
classifier, no chain-of-thought.

Reuses the same segmentation/occupancy, planner (node_Path_Translator.astar_proj) and debug writers as
test_pipeline (imported), so the only differences from the main pipeline are the battleship overlay, the
custom single-call prompt, and using battleship cell centroids as the planner's reference points.

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_battleship_baseline.py \\
        --prompt "Send raph to the chair and donnie to the box"

Output (written to --out, default debug/battleship_baseline/):
    vlm_overlay.png       — the battleship-grid image sent to the VLM
    robot_paths.png       — both robots' planned (A*) trajectories on the clean overhead image
    robot_paths_centroids.png — both robots' reference routes through the chosen cell centroids (pre-A*)
  and per robot in <out>/<robot>/: raw_overhead, segmentation, occ_true, vlm_selections,
    route_centroids, route_planned, occ_inflated, inflation_overlay.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import mask_to_occupancy, create_filtered_occupancy_map, RESOLUTION as _RESOLUTION
from node_Path_Translator import astar_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Executive_API.map_gen import render_battleship_map, _N_COLS, _N_ROWS

# Reuse test_pipeline's helpers (OpenAI client, JSON parse, debug writers) to avoid drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp   # noqa: E402


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

def _call_baseline_vlm(prompt: str, overlay: PILImage.Image, model: str, temperature: float,
                       out_dir: str | None = None) -> tuple[dict, dict]:
    """Mirror the baseline Executive node: system instructions = BASELINE_PROMPT, user content =
    [operator prompt text, battleship overlay image]. Plain JSON reply (no structured schema).

    Returns (routes, usage) where routes = {robot: [cell, ...]} and usage is token counts.
    """
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    map_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(BASELINE_PROMPT)
    print(f'\nOperator instruction: "{prompt}"')
    print("─" * 71)
    if out_dir is not None:
        with open(os.path.join(out_dir, "vlm_prompt.txt"), "w") as f:
            f.write(BASELINE_PROMPT + f'\n\nOperator instruction: "{prompt}"')

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
        with open(os.path.join(out_dir, "vlm_response.txt"), "w") as f:
            f.write(raw)

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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="Operator instruction for the VLM (required)")
    parser.add_argument("--data", default="test_data",
                        help="Dir containing overhead.png + poses.json (default: test_data)")
    parser.add_argument("--camera", default="gazebo", choices=["gazebo", "lab_test"],
                        help="Overhead camera calibration for pixel<->world (default: gazebo)")
    parser.add_argument("--out", default="debug/battleship_baseline",
                        help="Debug output directory (default: debug/battleship_baseline)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the planner call (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    args = parser.parse_args()

    camera = args.camera

    # ── Load saved image + poses ──────────────────────────────────────────
    img_path = os.path.join(args.data, "overhead.png")
    poses_path = os.path.join(args.data, "poses.json")
    if not os.path.exists(img_path):
        sys.exit(f"Image not found: {img_path}\nRun save_overhead.py first.")
    if not os.path.exists(poses_path):
        sys.exit(f"Poses not found: {poses_path}\nRun save_overhead.py first.")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_rgba = PILImage.fromarray(img_rgb).convert("RGBA")
    h, w = img_bgr.shape[:2]
    print(f"Loaded image: {w}×{h}")

    with open(poses_path) as f:
        poses_raw = json.load(f)
    robot_world_xy = {name: gazebo_to_world(p["x"], p["y"]) for name, p in poses_raw.items()}
    robot_poses_ned = {name: gazebo_to_ned(p["x"], p["y"]) for name, p in poses_raw.items()}
    print("Robot world poses: " +
          ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    os.makedirs(args.out, exist_ok=True)

    # ── Segmentation + occupancy (same as test_pipeline) ──────────────────
    print("Running CLIPSeg segmentation…")
    segmenter = TraversabilitySegmenter()
    pix_labels, _ = segmenter.classify(
        img_rgb, traversable_prompts=["the floor"], untraversable_prompts=[""], threshold=0.48)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    world_poses = [ned_to_world(*p) for p in robot_poses_ned.values() if p]
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    ctx = {"infl": infl}

    # ── Battleship overlay image + cell centroids ─────────────────────────
    overlay = render_battleship_map(pil_rgba, robot_poses=robot_poses_ned, camera=camera)
    overlay.convert("RGB").save(os.path.join(args.out, "vlm_overlay.png"))
    grid_px = _battleship_centroids(w, h)

    # ── Single VLM call ───────────────────────────────────────────────────
    routes, usage = _call_baseline_vlm(args.prompt, overlay, args.model, args.temperature,
                                       out_dir=args.out)

    # ── Plan per robot through battleship centroids (astar) ───────────────
    params = dict(ASTAR_PARAMS)
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    centroid_routes: dict[str, list] = {}   # pose + chosen battleship-cell centroids (pre-A*)
    for name, labels in routes.items():
        if not isinstance(labels, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose in poses.json; skipping.", file=sys.stderr)
            continue

        _adjacency_warnings(name, labels)
        pose_xy = robot_world_xy[name]
        ref, unknown = astar_proj.build_reference(labels, pose_xy, grid_px, camera=camera)
        for lbl in unknown:
            print(f"[{name}] unknown cell '{lbl}'; skipping.")
        if len(ref) >= 2:
            centroid_routes[name] = list(ref)   # pose -> cell centroids, in order (pre-A*)

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

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
        print(f"[{name}] {len(world_path)} waypoints: "
              f"start=({world_path[0][0]:.3f}, {world_path[0][1]:.3f})  "
              f"end=({world_path[-1][0]:.3f}, {world_path[-1][1]:.3f})  "
              f"path length={length:.2f} m")

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        tp._save_common_debug(out_dir, img_bgr, pix_labels, grid)
        tp._save_vlm_selections(out_dir, img_bgr, "battleship", labels, grid_px)
        dbg["start_world"] = pose_xy
        astar_proj.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plots ─────────────────────────────────────────
    if world_paths:
        paths_png = os.path.join(args.out, "robot_paths.png")
        tp._save_robot_paths(paths_png, img_bgr, world_paths, camera)
        print(f"Combined robot paths -> {paths_png}")
    if centroid_routes:
        cpaths_png = os.path.join(args.out, "robot_paths_centroids.png")
        tp._save_robot_paths(cpaths_png, img_bgr, centroid_routes, camera)
        print(f"Combined centroid routes -> {cpaths_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
    print("Baseline: battleship grid + astar (no classifier, no CoT)")
    print(f"Tokens: {usage['total']} total "
          f"({usage['input']} input + {usage['output']} output)")
    if path_lengths:
        order = [n for n in ("donnie", "raph") if n in path_lengths]
        order += [n for n in path_lengths if n not in order]   # any others, insertion order
        for name in order:
            print(f"Path length [{name}]: {path_lengths[name]:.2f} m")
        print(f"Path length [total]: {sum(path_lengths.values()):.2f} m")
    else:
        print("Path length: no paths were planned.")
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
