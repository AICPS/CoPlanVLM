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
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from node_Executive_API.map_gen import (render_battleship_map, render_nav2point_map,
                                        _SEG_PROMPTS, _SEG_THRESHOLD)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR    = _SCRIPT_DIR.parent
_IMAGE      = _PKG_DIR / "overhead.png"
_OUT_DIR    = _PKG_DIR / "debug" / "test_map_gen"
_RESOLUTION = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="gazebo", choices=["gazebo", "lab_test"])
    args = parser.parse_args()

    if not _IMAGE.exists():
        sys.exit(f"Image not found: {_IMAGE}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = PILImage.open(_IMAGE).convert("RGBA")
    print(f"Loaded: {_IMAGE}  ({img.size[0]}x{img.size[1]})")

    print("\n[1/2] Generating battleship overlay...")
    render_battleship_map(img).save(_OUT_DIR / "battleship.png")
    print(f"  Saved -> {_OUT_DIR / 'battleship.png'}")

    print("\n[2/2] Running CLIPSeg (first run loads the model)...")
    from obs_seg.segmenter import TraversabilitySegmenter
    rgb = np.array(img.convert("RGB"))
    pix_labels, _ = TraversabilitySegmenter().classify(rgb, _SEG_PROMPTS, [], _SEG_THRESHOLD)
    unique, counts = np.unique(pix_labels, return_counts=True)
    print(f"  pix_labels {pix_labels.shape}  [{', '.join(f'{v}:{n}' for v, n in zip(unique, counts))}]")

    render_nav2point_map(img, pix_labels, args.camera, _RESOLUTION).save(_OUT_DIR / "nav2point.png")
    print(f"  Saved -> {_OUT_DIR / 'nav2point.png'}")

    print(f"\nDone. Open {_OUT_DIR}/ to compare.")


if __name__ == "__main__":
    main()
