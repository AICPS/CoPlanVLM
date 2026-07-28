#!/usr/bin/env python3
"""test_pipeline_real.py — test_pipeline's two-call pipeline on a REAL lab-deployment image.

Identical to test_pipeline.py (same segmentation, classifier-first controller selection, prompt+overlay,
planner, and debug figures — including robot_paths_waypoints.png) except:
  * it runs on the real overhead image test_data/overhead_real.png (1920x1200), and
  * pixel<->world uses the `lab_test` camera calibration (real intrinsics) instead of `gazebo`, and
  * there is no pose file yet, so robot world poses are HARDCODED (see HARDCODED_WORLD_POSES).

All helpers are reused from test_pipeline (imported as `tp`) so this stays in lockstep with the main
harness; only main() is reimplemented, mirroring tp.main() with the image path, camera default, and pose
source changed.

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    # Classifier-first (default): only the prompt is required.
    python3 src/CoPlanVLM/scripts/test_pipeline_real.py \\
        --prompt "Send raph to the chair and donnie to the table"

    # Fully manual: pin the controller and overlay explicitly.
    python3 src/CoPlanVLM/scripts/test_pipeline_real.py \\
        --planner nav2point --map-overlay points --no-cot \\
        --prompt "Send raph to the chair and donnie to the table"

Output (written to --out, default debug/offline_test_real/): same files as test_pipeline.py.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image as PILImage

import debug_io
from coord_transform import world_to_ned, ned_to_world
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)

# Reuse ALL of test_pipeline's helpers + re-exported symbols so this stays in sync with the main harness.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp   # noqa: E402


# Placeholder robot poses (WORLD frame, metres) until a real pose source exists. raph is inserted first
# so it gets _ROBOT_COLORS[0] (magenta) and donnie gets [1] (cyan), matching the rest of the pipeline.
HARDCODED_WORLD_POSES = {"raph": (3.2, -1.45), "donnie": (-3.55, -1.35)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", choices=list(tp.CONTROLLERS), default=None,
                        help="Controller / low-level planner. If omitted, the classifier LLM chooses it "
                             "from the prompt.")
    parser.add_argument("--map-overlay", dest="map_overlay", choices=list(tp.MAP_OVERLAY_TYPES),
                        default=None,
                        help="Overlay style rendered + described in the prompt. If omitted, defaults to "
                             "marked_obs for every controller.")
    parser.add_argument("--prompt", required=True,
                        help="Operator instruction for the VLM (required in every mode)")
    parser.add_argument("--data", default="test_data",
                        help="Dir containing overhead_real.png (default: test_data)")
    parser.add_argument("--camera", default="lab_test", choices=["gazebo", "lab_test"],
                        help="Overhead camera calibration for pixel<->world (default: lab_test)")
    parser.add_argument("--out", default="debug/offline_test_real",
                        help="Debug output directory (default: debug/offline_test_real)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the classifier + planner calls (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    parser.add_argument("--cot", action=argparse.BooleanOptionalAction, default=True,
                        help="Append the controller's chain-of-thought scaffold (--cot / --no-cot). "
                             "Default: on.")
    args = parser.parse_args()

    camera = args.camera

    # ── Load real image; robot poses are hardcoded (no poses.json yet) ─────
    img_path = os.path.join(args.data, "overhead_real.png")
    if not os.path.exists(img_path):
        sys.exit(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_rgba = PILImage.fromarray(img_rgb).convert("RGBA")
    print(f"Loaded image: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    # Planning references use world coords directly; overlay markers use NED (map_gen converts back to
    # world internally), so world -> NED here makes the markers land at the intended world positions.
    robot_world_xy = dict(HARDCODED_WORLD_POSES)
    robot_poses_ned = {name: world_to_ned(wx, wy) for name, (wx, wy) in HARDCODED_WORLD_POSES.items()}
    print("Robot world poses (hardcoded): " +
          ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    os.makedirs(args.out, exist_ok=True)

    # ── Segmentation + occupancy (shared: overlay filtering + planning) ───
    print("Running CLIPSeg segmentation…")
    segmenter = TraversabilitySegmenter()
    pix_labels, _ = segmenter.classify(
        img_rgb, traversable_prompts=["the floor"], untraversable_prompts=[""], threshold=0.48)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    # ── Resolve controller / overlay / CoT (classifier-first, with manual overrides) ──
    if args.planner:                       # manual controller
        task_type = args.planner
    else:                                  # classifier picks the controller (first LLM call)
        task_type = tp._classify(args.prompt, args.model, args.temperature, list(robot_poses_ned))
        print(f"Classifier chose controller: {task_type}")

    map_overlay = args.map_overlay or "marked_obs"   # default overlay for every controller
    cot = args.cot                         # boolean, default True

    result_key, planner_name = tp.TASK_ROUTING[task_type]
    planner = tp._PLANNERS[planner_name]
    print(f"Controller: {task_type} | overlay: {map_overlay} | planner: {planner_name} "
          f"(key '{result_key}') | CoT: {cot}")

    # ── Filtered planning context (mirrors exec.run_segmentation) ─────────
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
    instructions, map_b64 = tp.generate_prompt(
        args.prompt, task_type, map_overlay,
        pil_img=pil_rgba, occ_grid=infl, occ_meta=meta, camera=camera,
        robot_poses=robot_poses_ned, cot=cot, robot_names=list(robot_poses_ned))
    # Byte-exact copy of the image sent to the VLM (no decode/re-encode).
    debug_io.save_marks_overlay(args.out, map_b64)
    routes, usage = tp._call_planner_vlm(instructions, map_b64, result_key, args.model, args.temperature,
                                         list(robot_world_xy), cot=cot, out_dir=args.out)

    # ── Grid CSV (shared across robots) ───────────────────────────────────
    csv_path = tp._CONFIG_DIR / "grid_cell_centers.csv"
    if not csv_path.exists():
        sys.exit(f"Grid CSV not found: {csv_path}")
    grid_px = tp._load_grid_csv(csv_path)

    params = {**tp.ASTAR_PARAMS, **tp.COVERAGE_PARAMS}

    # ── Plan per robot + write debug images ───────────────────────────────
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    for name, labels in routes.items():
        if not isinstance(labels, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose; skipping.", file=sys.stderr)
            continue

        pose_xy = robot_world_xy[name]
        ref, unknown = planner.build_reference(labels, pose_xy, grid_px, camera=camera)
        for lbl in unknown:
            print(f"[{name}] unknown label '{lbl}'; skipping.")

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

        print(f"[{name}] Planning with {planner_name} ({len(ref)} reference pts)…")
        world_path, dbg = planner.plan(ref, ctx, meta, params)
        for w in dbg.get("warnings", []):
            print(f"[{name}] WARNING: {w}")

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
        debug_io.save_vlm_selections(out_dir, img_bgr, map_overlay, labels, grid_px)
        dbg["start_world"] = pose_xy
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
    if path_lengths:
        for name, length in path_lengths.items():
            print(f"Path length [{name}]: {length:.2f} m")
        print(f"Path length [total]: {sum(path_lengths.values()):.2f} m")
    else:
        print("Path length: no paths were planned.")
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
