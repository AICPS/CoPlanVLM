#!/usr/bin/env python3
"""test_regression_only.py — raw-pixel-pointing path baseline (single VLM call, no CoT).

The opposite extreme from test_pipeline's set-of-marks + chain-of-thought pipeline: the VLM sees an
UNMARKED overhead image (no grid, no set-of-marks — only a hollow circle + name label per robot so it can
tell raph from donnie) and returns, for each robot, an ordered list of KEY POINTS as raw pixel
coordinates [x, y]. Those pixels are mapped to world coordinates and fed to the astar_proj planner, which
projects each to the nearest free cell and routes A* between them. No classifier, no chain-of-thought.

Reuses the same segmentation/occupancy, planner (node_Path_Translator.astar_proj) and debug writers as
test_pipeline / test_battleship_baseline (imported), so the only differences are the unmarked overlay, the
custom single-call prompt, and using the VLM's raw pixels (not grid-cell centroids) as reference points.

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_regression_only.py \\
        --prompt "Send raph to the chair and donnie to the box"

Output (written to --out, default debug/regression_only/):
    marks_overlay.png     — the unmarked image (robot markers only) sent to the VLM
    robot_paths_waypoints.png — both robots' planned (A*) trajectories + their numbered pixel picks
    raw_overhead.png, inflation_overlay.png — identical for every robot, so written once here
  and per robot in <out>/<robot>/: vlm_selections (numbered points), route_centroids, route_planned.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw

import debug_io
from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world, pixel_to_world
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)
from node_Path_Translator import astar_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Executive_API import map_gen

# Reuse test_pipeline's helpers (OpenAI client, JSON parse, debug writers) to avoid drift.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp   # noqa: E402


# ── Prompt (our own; no chain-of-thought) ───────────────────────────────────────
REGRESSION_PROMPT = """\
You are a path-planning agent controlling two TurtleBot4 robots named "raph" and "donnie" that share one
workspace. You are given an UNMARKED overhead camera image of the environment, {w} pixels wide and {h}
pixels tall. The image has NO grid and NO labels except a small hollow circle drawn on each robot with
its name next to it:
- "raph"   — the round black TurtleBot marked with a magenta circle labeled "raph".
- "donnie" — the round black TurtleBot marked with a light-blue circle labeled "donnie".

All directions are from the IMAGE's point of view, NOT a robot's or person's perspective: the left/right
of an object means its left/right side as it appears in the image, and up/down mean toward the top/bottom
of the image.

YOUR TASK:
For each robot, choose an ordered list of KEY POINTS — from one to many — that the robot must pass through
to accomplish the operator's instruction. Give each point as pixel coordinates [x, y] in the image, where
x runs from 0 at the left edge to {w} at the right edge and y runs from 0 at the top edge to {h} at the
bottom edge. Order the points along the route, starting from a point near the robot's current marker. Try
to AVOID traveling into obstacles: look at the image and keep every point on clear, open floor, routing
around objects, furniture, walls and other robots rather than placing a point on or driving through them.
You MUST include both robots; if a robot has no task, give it an empty list.

Respond with EXACTLY one JSON object and nothing else:
{{"waypoints": {{"raph": [[x, y], ...], "donnie": [[x, y], ...]}}}}

Example — if raph's marker is near [180, 620] and donnie's near [640, 210], and the instruction is "Send
raph to the chair and donnie to the box", a valid response is:
{{"waypoints": {{"raph": [[180, 620], [340, 470], [520, 430]], "donnie": [[640, 210], [820, 360], [1010, 500]]}}}}
(each list starts near the robot's marker and lists the key points to pass through, in travel order.)"""


# ── Unmarked overlay (robot name markers only) ──────────────────────────────────

def _render_robots_only(pil_rgba: PILImage.Image, robot_poses_ned: dict,
                        camera: str) -> PILImage.Image:
    """The 'unmarked' image: a copy of the overhead image with ONLY the robot circles + name labels
    (no grid, no set-of-marks), so the VLM can identify raph vs donnie while pointing on bare floor."""
    result = pil_rgba.copy()
    draw = ImageDraw.Draw(result)
    map_gen._draw_robot_markers(draw, robot_poses_ned, camera, map_gen._LABEL_FONT)
    return result


# ── Pixel points -> world reference route (replaces astar_proj.build_reference) ──

def _points_to_reference(points: list, pose_xy, camera: str) -> tuple[list, list]:
    """Convert the VLM's ordered pixel points to a world-frame reference route.

    ref = [robot pose] + [pixel_to_world(u, v) for each valid [u, v] point]. Any entry that is not a
    2-number pair is collected into `dropped` (printed as a warning) rather than crashing the run.
    """
    ref = [(float(pose_xy[0]), float(pose_xy[1]))]
    dropped = []
    for p in points:
        try:
            u, v = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            dropped.append(p)
            continue
        x, y = pixel_to_world(u, v, camera=camera)
        ref.append((float(x), float(y)))
    return ref, dropped


# ── VLM call (single call; text + unmarked image; plain JSON, like the baseline node) ──

def _call_regression_vlm(prompt: str, map_b64: str, w: int, h: int, model: str,
                         temperature: float, out_dir: str | None = None) -> tuple[dict, dict]:
    """system instructions = REGRESSION_PROMPT (image dims filled in), user content = [operator prompt
    text, unmarked overlay image]. Plain JSON reply (no structured schema).

    `map_b64` is the base64 PNG of the overlay, encoded by the caller so the same bytes go to both
    the API and debug_io.save_marks_overlay.

    Returns (routes, usage) where routes = {robot: [[x, y], ...]} pixel points and usage is token counts.
    """
    instructions = REGRESSION_PROMPT.format(w=w, h=h)

    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(instructions)
    print(f'\nOperator instruction: "{prompt}"')
    print("─" * 71)
    # This baseline sends the operator text as a separate user turn, so record it with the system
    # instructions to keep vlm_prompt.txt a complete picture of what the model was given.
    saved_prompt = instructions + f'\n\nOperator instruction: "{prompt}"'

    client = tp._openai_client()
    print(f"Planning with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature, instructions=instructions,
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
    for name, points in routes.items():
        n = len(points) if isinstance(points, list) else "?"
        seq = ", ".join(f"[{p[0]},{p[1]}]" for p in points if isinstance(p, (list, tuple)) and len(p) >= 2) \
            if points else "(empty — holds position)"
        print(f"  {name} ({n}): {seq}")
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
    parser.add_argument("--out", default="debug/regression_only",
                        help="Debug output directory (default: debug/regression_only)")
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

    # Robot-independent debug artifacts: rendered and written ONCE per run at the top level
    # (they are identical for every robot). `infl_overlay` is reused below as each robot's
    # route_planned.png canvas, so render_inflation_overlay runs only this once.
    os.makedirs(args.out, exist_ok=True)
    infl_overlay = render_inflation_overlay(img_bgr, cleared, infl, meta, camera)
    debug_io.save_raw_overhead(args.out, img_bgr)
    debug_io.save_inflation_overlay(args.out, infl_overlay)

    # ── Unmarked overlay (robot name markers only) ────────────────────────
    overlay = _render_robots_only(pil_rgba, robot_poses_ned, camera)
    # Encode once: the same bytes go to the API and to marks_overlay.png.
    _buf = io.BytesIO()
    overlay.save(_buf, format="PNG")
    map_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
    debug_io.save_marks_overlay(args.out, map_b64)

    # ── Single VLM call ───────────────────────────────────────────────────
    routes, usage = _call_regression_vlm(args.prompt, map_b64, w, h, args.model, args.temperature,
                                         out_dir=args.out)

    # ── Plan per robot through the VLM's raw pixel points (astar) ─────────
    params = dict(ASTAR_PARAMS)
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    # Accumulated selections for the combined figure. This baseline's "labels" are just ordinals,
    # so they are numbered CONTINUOUSLY across robots (raph 1..n, donnie n+1..m) — per-robot
    # restarts would collide in the shared label->pixel dict and mis-place donnie's markers.
    sel_routes: dict[str, list] = {}
    sel_px: dict[str, tuple] = {}
    next_id = 1
    for name, points in routes.items():
        if not isinstance(points, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose in poses.json; skipping.", file=sys.stderr)
            continue

        pose_xy = robot_world_xy[name]
        ref, dropped = _points_to_reference(points, pose_xy, camera)
        for p in dropped:
            print(f"[{name}] malformed point '{p}'; skipping.")

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
        # Reuse save_vlm_selections' point style: ordinal labels + a pixel dict keyed by them.
        valid_pts = [p for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        labels = [str(next_id + i) for i in range(len(valid_pts))]
        grid_px = {lbl: (float(p[0]), float(p[1])) for lbl, p in zip(labels, valid_pts)}
        next_id += len(valid_pts)
        sel_routes[name] = labels
        sel_px.update(grid_px)
        debug_io.save_vlm_selections(out_dir, img_bgr, "points", labels, grid_px)
        dbg["start_world"] = pose_xy
        astar_proj.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera,
                              overlay=infl_overlay)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plot ──────────────────────────────────────────
    if world_paths:
        wp_png = os.path.join(args.out, "robot_paths_waypoints.png")
        debug_io.save_paths_with_waypoints(wp_png, img_bgr, world_paths, sel_routes, sel_px, camera)
        print(f"Combined robot paths + VLM points -> {wp_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
    print("Baseline: unmarked image + raw pixel points + astar (no classifier, no CoT)")
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
