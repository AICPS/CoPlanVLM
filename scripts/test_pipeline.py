#!/usr/bin/env python3
"""test_pipeline.py — offline harness mirroring the Executive's two-call pipeline.

Runs the full VLM planning pipeline on a saved overhead image (no Gazebo, no running ROS nodes):
segmentation → prompt + map overlay → planner → debug images. It reuses the SAME prompt constructor
(prompt_gen.generate_prompt), overlays (map_gen) and planners (node_Path_Translator) as the live
exec/translate nodes, so what you see here is what the robots would get.

By default the classifier LLM chooses the controller from the prompt, and the overlay + CoT follow:
    --planner {nav2point, maneuver, coverage}   optional; if omitted, the classifier chooses it
    --map-overlay {points, marked_obs, battleship}   optional; default marked_obs (all controllers)
    --cot / --no-cot                             chain-of-thought scaffold (default: on)

Usage (from workspace root):
    # Classifier-first (default): only the prompt is required.
    python3 src/CoPlanVLM/scripts/test_pipeline.py \\
        --prompt "Have raph loop around the chair and return to its start"

    # Fully manual: pin the controller and overlay explicitly.
    python3 src/CoPlanVLM/scripts/test_pipeline.py \\
        --planner nav2point --map-overlay points --no-cot \\
        --prompt "Send raph to the chair and donnie to the table"

Output (written to --out, default debug/offline_test_sim/):
    marks_overlay.png     — the exact overlay image sent to the VLM (grid/marks + robot markers)
    vlm_prompt.txt        — the exact system prompt sent to the planner VLM
    vlm_response.txt      — the raw model reply
    robot_paths_waypoints.png — planned trajectories + each robot's labeled VLM waypoint picks
    raw_overhead.png      — the loaded image
    inflation_overlay.png — true obstacles (red) + inflation margin (yellow) on the overhead image
  and per robot in <out>/<robot_name>/:
    vlm_selections.png    — the VLM's raw picks (dots for nav2point/maneuver, shaded cells for coverage)
    route_centroids.png   — planner reference route (robot pose + selected centroids)
    route_planned.png     — planner's obstacle-avoiding path

raw_overhead and inflation_overlay are identical for every robot, so they are written ONCE at the
top level rather than duplicated into each robot's subdirectory.

The sibling live directories are debug/gazebo_sim and debug/deploy_real (written by the running
nodes); debug/offline_test_real is this harness's real-image twin, test_pipeline_real.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

import debug_io
from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world, world_to_pixel
from obs_seg.segmenter import segment_frame
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)

from node_Path_Translator import astar_proj, coverage_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Path_Translator.coverage_proj import PARAMS as COVERAGE_PARAMS
# Same helper and same default the live translator node uses, so the routes drawn here are the
# routes the robots would execute (see grid_planner_utils.separate_goal).
from node_Path_Translator.grid_planner_utils import (
    DEFAULT_MIN_GOAL_SEPARATION as MIN_GOAL_SEPARATION, separate_goal)

from node_Executive_API.prompt_gen import (
    generate_prompt, route_schema, classifier_schema,
    CONTROLLERS, MAP_OVERLAY_TYPES, TASK_ROUTING, classifier_prompt,
    COT_FIELDS, reasoning_text)
from node_Executive_API.map_gen import (
    _N_COLS as _GRID_COLS, _N_ROWS as _GRID_ROWS,
    _DOT_RADIUS, _LABEL_SIZE, _ROBOT_COLORS, _ROBOT_RADIUS)

_PLANNERS = {"astar": astar_proj, "coverage": coverage_proj}

# TASK_ROUTING, route_schema and classifier_schema are imported from prompt_gen (NOT restated here)
# so this harness and the live node share one definition — the whole point of the harness is that
# what it validates is what exec.py executes. prompt_gen is rclpy-free, so importing it keeps this
# script ROS-free too.

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SCRIPT_DIR.parent
_CONFIG_DIR = _PKG_DIR / "config"


def _load_grid_csv(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Returns {label: (u, v)} pixel dict from grid_cell_centers.csv."""
    result = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            result[row["cell"]] = (float(row["center_x"]), float(row["center_y"]))
    return result


def _parse_json_reply(text: str) -> dict | None:
    """Tolerant JSON parse (mirrors exec.py._parse_json_reply)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── VLM calls (classifier + planner), mirroring exec's two-call pipeline ─────────

def _openai_client():
    """OpenAI client keyed like the live pipeline: load config/.env (exactly as the launch files do)
    and read MY_API_KEY, falling back to OPENAI_API_KEY. No manual `source`/`export` needed."""
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_CONFIG_DIR / ".env")
    api_key = os.getenv("MY_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit(f"No API key found — set MY_API_KEY in {_CONFIG_DIR / '.env'} "
                 "(as the launch files use) or export OPENAI_API_KEY.")
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# Default controller selector: runs when --planner is omitted (manual --planner skips it).
def _classify(prompt: str, model: str, temperature: float, robot_names) -> str:
    """First call: text-only classifier -> controller (one of CONTROLLERS).

    Constrained by classifier_schema() (enum over CONTROLLERS), so the model cannot return a task
    type outside the vocabulary — same call exec._classify_task() makes. `robot_names` names the
    robots in the classifier prompt, matching the roster used for the planner call.
    """
    client = _openai_client()
    print(f"Classifying instruction with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature,
        instructions=classifier_prompt(robot_names),
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={"format": {"type": "json_schema", "name": "task_classification",
                         "strict": True, "schema": classifier_schema()}},
    )
    result = _parse_json_reply(response.output_text.strip()) or {}
    task_type = result.get("task_type")
    if task_type not in CONTROLLERS:
        sys.exit(f"Classifier returned unknown task_type {task_type!r}; "
                 f"expected one of {sorted(CONTROLLERS)}.")
    return task_type


def _call_planner_vlm(instructions: str, map_b64: str, result_key: str,
                      model: str, temperature: float, robot_names: list[str],
                      cot: bool = False, allowed_labels: list[str] | None = None,
                      out_dir: str | None = None,
                      controller: str | None = None) -> tuple[dict, dict]:
    """Second call: vision planner. instructions = per-task prompt, image = the overlay (only).

    `map_b64` is the base64 PNG straight from generate_prompt — taken as-is rather than as a PIL
    image so the exact bytes sent are also what debug_io.save_marks_overlay writes to disk, with
    no decode/re-encode round-trip.

    The reply is constrained to a strict route schema (structured outputs) so it can only be the fixed
    {result_key:{robots...}} shape — no invented keys. When allowed_labels is given, every waypoint is
    also restricted to that enum (free cells) so obstacle cells cannot be selected. With cot=True the
    schema also carries a leading "reasoning" string generated before the routes.

    Returns (routes, usage) where usage is {input, output, total} token counts for this call.

    The full system prompt and the full raw model output are printed and, if out_dir is given, saved to
    vlm_prompt.txt / vlm_response.txt.
    """
    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(instructions)
    print("─" * 71)

    schema = route_schema(result_key, robot_names, cot, allowed_labels, controller=controller)
    steps = [f for f, _ in COT_FIELDS[controller]] if cot else []
    keys = "".join(f"{f}, " for f in steps) + f"{result_key}{{{', '.join(robot_names)}}}"
    print(f"Structured output: enforcing schema (keys: {keys})")
    if allowed_labels:
        print(f"  waypoints restricted to {len(allowed_labels)} free cells (obstacle cells excluded)")

    client = _openai_client()
    print(f"Planning with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature, instructions=instructions,
        input=[{
            "role": "user",
            "content": [{"type": "input_image",
                         "image_url": f"data:image/png;base64,{map_b64}"}],
        }],
        text={"format": {"type": "json_schema", "name": "route_plan",
                         "strict": True, "schema": schema}},
    )
    u = getattr(response, "usage", None)
    usage = {
        "input":  getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "total":  getattr(u, "total_tokens", 0) or 0,
    }

    raw = response.output_text.strip()
    # Persist the exchange through the shared writer (same files the live exec node writes).
    if out_dir is not None:
        debug_io.save_vlm_exchange(out_dir, instructions, raw)

    result = _parse_json_reply(raw)
    if result is None:
        sys.exit(f"VLM returned unparseable JSON:\n{raw}")
    routes = result.get(result_key, {})
    if not isinstance(routes, dict):
        sys.exit(f"VLM '{result_key}' is not a dict: {routes!r}")

    # Readable terminal view: reflowed reasoning, then a compact per-robot route listing.
    print("─" * 16 + " VLM response " + "─" * 16)
    trace = reasoning_text(result, controller) if cot else ""
    if trace:
        print("REASONING:")
        print(textwrap.indent(trace, "  "))
        print()
    print(f"{result_key.upper()}:")
    for name, labels in routes.items():
        seq = ", ".join(map(str, labels)) if labels else "(empty — holds position)"
        print(f"  {name} ({len(labels) if isinstance(labels, list) else '?'}): {seq}")
    print("─" * 46)
    return routes, usage


# ── Debug image writers (no rclpy) ──────────────────────────────────────────────

def _path_length(world_path: list) -> float:
    """Total length (metres) of a planned path = sum of Euclidean steps between consecutive waypoints."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(world_path, world_path[1:]))


def _print_path_lengths(path_lengths: dict) -> None:
    """Print each robot's path length: donnie, then raph, then anyone else. One decimal, no total.

    A FIXED order, not dict order: path_lengths is filled in `routes` key order — whatever order the
    VLM happened to emit — so the same two robots would swap places between runs and make two runs
    awkward to compare side by side. Every harness (test_pipeline, _real, _battleship_baseline,
    _regression_only) calls this one function so their evaluation blocks stay directly comparable.
    """
    if not path_lengths:
        print("Path length: no paths were planned.")
        return
    order = [n for n in ("donnie", "raph") if n in path_lengths]
    order += [n for n in path_lengths if n not in order]   # any others, insertion order
    for name in order:
        print(f"Path length [{name}]: {path_lengths[name]:.1f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=list(CONTROLLERS), default=None,
                        help="Controller / low-level planner. If omitted, the classifier LLM chooses it "
                             "from the prompt.")
    parser.add_argument("--map-overlay", dest="map_overlay", choices=list(MAP_OVERLAY_TYPES),
                        default=None,
                        help="Overlay style rendered + described in the prompt. If omitted, defaults to "
                             "marked_obs for every controller.")
    parser.add_argument("--prompt", required=True,
                        help="Operator instruction for the VLM (required in every mode)")
    parser.add_argument("--data", default="test_data",
                        help="Dir containing overhead.png + poses.json (default: test_data)")
    parser.add_argument("--camera", default="gazebo", choices=["gazebo", "lab_test"],
                        help="Overhead camera calibration for pixel<->world (default: gazebo)")
    parser.add_argument("--out", default="debug/offline_test_sim",
                        help="Debug output directory (default: debug/offline_test_sim)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the classifier + planner calls (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    parser.add_argument("--cot", action=argparse.BooleanOptionalAction, default=True,
                        help="Append the controller's chain-of-thought scaffold (--cot / --no-cot). "
                             "Default: on.")
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
    print(f"Loaded image: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    with open(poses_path) as f:
        poses_raw = json.load(f)
    # planning references use world coords; overlay markers use NED (same as translate/exec).
    robot_world_xy = {name: gazebo_to_world(p["x"], p["y"]) for name, p in poses_raw.items()}
    robot_poses_ned = {name: gazebo_to_ned(p["x"], p["y"]) for name, p in poses_raw.items()}
    print("Robot world poses: " +
          ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    os.makedirs(args.out, exist_ok=True)

    # ── Segmentation + occupancy (shared: overlay filtering + planning) ───
    print("Running CLIPSeg segmentation…")
    pix_labels = segment_frame(img_rgb)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    # ── Resolve controller / overlay / CoT (classifier-first, with manual overrides) ──
    if args.planner:                       # manual controller
        task_type = args.planner
    else:                                  # classifier picks the controller (first LLM call)
        task_type = _classify(args.prompt, args.model, args.temperature, list(robot_poses_ned))
        print(f"Classifier chose controller: {task_type}")

    map_overlay = args.map_overlay or "marked_obs"   # default overlay for every controller
    cot = args.cot                         # boolean, default True

    result_key, planner_name = TASK_ROUTING[task_type]
    planner = _PLANNERS[planner_name]
    print(f"Controller: {task_type} | overlay: {map_overlay} | planner: {planner_name} "
          f"(key '{result_key}') | CoT: {cot}")

    # ── Filtered planning context (mirrors exec.run_segmentation) ─────────
    # Edge/footprint clearing + inflation live in obs_seg.occupancy and are applied once here; the
    # overlay renderer and the planner modules both consume this single grid and never redo it.
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

    # ── Build the prompt + overlay image together (single source of truth) ─
    instructions, map_b64 = generate_prompt(
        args.prompt, task_type, map_overlay,
        pil_img=pil_rgba, occ_grid=infl, occ_meta=meta, camera=camera,
        robot_poses=robot_poses_ned, cot=cot, robot_names=list(robot_poses_ned))
    # Byte-exact copy of the image sent to the VLM (no decode/re-encode).
    debug_io.save_marks_overlay(args.out, map_b64)
    # Waypoints are left unconstrained (no enum) — the schema still locks the route shape/keys, but the
    # model may pick any label and we rely on it choosing sensibly (the planner projects picks to free
    # cells). Pass allowed_labels=... to _call_planner_vlm to re-enable the free-cell enum.
    routes, usage = _call_planner_vlm(instructions, map_b64, result_key, args.model, args.temperature,
                                      list(robot_world_xy), cot=cot, out_dir=args.out,
                                      controller=task_type)

    # ── Grid CSV (shared across robots) ───────────────────────────────────
    csv_path = _CONFIG_DIR / "grid_cell_centers.csv"
    if not csv_path.exists():
        sys.exit(f"Grid CSV not found: {csv_path}")
    grid_px = _load_grid_csv(csv_path)

    params = {**ASTAR_PARAMS, **COVERAGE_PARAMS}

    # ── Goal separation: reserve the poses of robots that will NOT move ───
    # Mirrors node_Path_Translator so this harness plans the routes the robots would actually
    # execute — without it, two robots sent to one target both terminate on the same cell and
    # robot_paths_waypoints.png shows a plan that could never run. A robot holding position still
    # occupies its cell, so it is claimed BEFORE the loop: claiming inside would be too late if it
    # happened to be visited second, and reserving up front makes the outcome independent of
    # `routes` key order. "Will not move" covers an empty label list, a malformed route, and a
    # robot with no pose.
    claimed: list = []
    for name, labels in routes.items():
        if isinstance(labels, list) and labels and name in robot_world_xy:
            continue                                  # moving; its goal is claimed after it plans
        pose = robot_world_xy.get(name)
        if pose is not None:
            claimed.append(pose)

    # ── Plan per robot + write debug images ───────────────────────────────
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    for name, labels in routes.items():
        if not isinstance(labels, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose in poses.json; skipping.", file=sys.stderr)
            continue

        # astar trims already-passed centroids; coverage keeps all for the TSP to reorder.
        pose_xy = robot_world_xy[name]
        ref, unknown = planner.build_reference(labels, pose_xy, grid_px, camera=camera)
        for lbl in unknown:
            print(f"[{name}] unknown label '{lbl}'; skipping.")

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

        # Displace the goal off any already-claimed goal BEFORE planning, so A* runs once and the
        # debug figures describe the executed route. Only the final reference point moves —
        # intermediate waypoints and the route itself are untouched.
        if planner_name == "astar" and MIN_GOAL_SEPARATION > 0 and claimed:
            adjusted = separate_goal(ref[-1], claimed, infl, meta,
                                     min_separation=MIN_GOAL_SEPARATION,
                                     max_shift=float(params["projection_radius"]))
            if adjusted is None:
                print(f"[{name}] goal is within {MIN_GOAL_SEPARATION:.2f} m of another robot and no "
                      "free cell far enough away was found; keeping the original goal.")
            elif math.dist(adjusted, ref[-1]) > 1e-9:
                print(f"[{name}] goal moved {math.dist(adjusted, ref[-1]):.2f} m to clear another "
                      f"robot: ({ref[-1][0]:.2f}, {ref[-1][1]:.2f}) -> "
                      f"({adjusted[0]:.2f}, {adjusted[1]:.2f})")
                ref[-1] = adjusted

        print(f"[{name}] Planning with {planner_name} ({len(ref)} reference pts)…")
        world_path, dbg = planner.plan(ref, ctx, meta, params)
        for w in dbg.get("warnings", []):
            print(f"[{name}] WARNING: {w}")

        if not world_path:
            print(f"[{name}] planner returned empty path.", file=sys.stderr)
            continue

        length = _path_length(world_path)
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
        # `map_overlay` (resolved), NOT args.map_overlay — the latter is None unless --map-overlay
        # was passed, which would silently pick the wrong selection style.
        debug_io.save_vlm_selections(out_dir, img_bgr, map_overlay, labels, grid_px)
        dbg["start_world"] = pose_xy
        # save_debug renders inflation_overlay (via occupancy.render_inflation_overlay)
        # and the planned route on top; pass `cleared` (post-override occupancy) as the red layer.
        planner.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera,
                           overlay=infl_overlay)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plot (both robots on the clean overhead image) ─
    if world_paths:
        wp_png = os.path.join(args.out, "robot_paths_waypoints.png")
        debug_io.save_paths_with_waypoints(wp_png, img_bgr, world_paths, routes, grid_px, camera)
        print(f"Combined robot paths + VLM waypoints -> {wp_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
    print(f"Controller: {task_type}"
          + ("  (classifier-chosen)" if args.planner is None else "  (manual)"))
    print(f"Tokens: {usage['total']} total "
          f"({usage['input']} input + {usage['output']} output)")
    _print_path_lengths(path_lengths)
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
