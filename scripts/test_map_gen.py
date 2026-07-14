#!/usr/bin/env python3
"""Offline visual comparison of the two map overlay styles.

Generates a battleship-grid composite and a nav2point set-of-marks overlay from the
saved overhead image and writes both PNGs to debug/test_map_gen/ for side-by-side
inspection.  No ROS nodes or simulation required.

Prerequisites:
    colcon build --symlink-install --packages-select talking-turtle
    source install/setup.bash

Usage (from workspace root):
    python3 src/VLM_mission_planning/scripts/test_map_gen.py
    python3 src/VLM_mission_planning/scripts/test_map_gen.py --camera lab_test

Output (debug/test_map_gen/):
    battleship.png  — overhead image composited with the transparent grid overlay
    nav2point.png   — set-of-marks dots at free-space grid centers only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

from coord_transform import gazebo_to_ned, ned_to_world
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)
from node_Executive_API.map_gen import (render_battleship_map, render_grid_points_map,
                                        _SEG_PROMPTS, _SEG_THRESHOLD)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR    = _SCRIPT_DIR.parent
_WS_DIR     = _PKG_DIR.parent.parent   # src/VLM_mission_planning -> src -> workspace root
_IMAGE      = _PKG_DIR / "overhead.png"
_POSES_FILE = _WS_DIR / "test_data" / "poses.json"
_OUT_DIR    = _WS_DIR / "debug" / "test_map_gen"   # workspace-root debug/, not inside the package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="gazebo", choices=["gazebo", "lab_test"])
    args = parser.parse_args()

    if not _IMAGE.exists():
        sys.exit(f"Image not found: {_IMAGE}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = PILImage.open(_IMAGE).convert("RGBA")
    print(f"Loaded: {_IMAGE}  ({img.size[0]}x{img.size[1]})")

    robot_poses: dict[str, tuple[float, float] | None] = {}
    if _POSES_FILE.exists():
        raw = json.loads(_POSES_FILE.read_text())
        for name, coords in raw.items():
            robot_poses[name] = gazebo_to_ned(coords["x"], coords["y"])
        print(f"Loaded poses for: {list(robot_poses)}")

    print("\n[1/2] Generating battleship overlay...")
    render_battleship_map(img, robot_poses=robot_poses, camera=args.camera).save(_OUT_DIR / "battleship.png")
    print(f"  Saved -> {_OUT_DIR / 'battleship.png'}")

    print("\n[2/2] Running CLIPSeg (first run loads the model)...")
    from obs_seg.segmenter import TraversabilitySegmenter
    rgb = np.array(img.convert("RGB"))
    pix_labels, _ = TraversabilitySegmenter().classify(rgb, _SEG_PROMPTS, [], _SEG_THRESHOLD)
    unique, counts = np.unique(pix_labels, return_counts=True)
    print(f"  pix_labels {pix_labels.shape}  [{', '.join(f'{v}:{n}' for v, n in zip(unique, counts))}]")

    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=args.camera)   # raw (no overrides)
    world_poses = [ned_to_world(*p) for p in robot_poses.values() if p]
    # infl = overrides + inflate; cleared = post-override, pre-inflation (for the overlay's red layer).
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    render_grid_points_map(img, infl, meta, args.camera, robot_poses=robot_poses).save(_OUT_DIR / "nav2point.png")
    print(f"  Saved -> {_OUT_DIR / 'nav2point.png'}")

    base_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = render_inflation_overlay(base_bgr, cleared, infl, meta, args.camera)
    cv2.imwrite(str(_OUT_DIR / "inflation_overlay.png"), overlay)
    print(f"  Saved -> {_OUT_DIR / 'inflation_overlay.png'}")

    print(f"\nDone. Open {_OUT_DIR}/ to compare.")


if __name__ == "__main__":
    main()
