#!/usr/bin/env python3
"""test_pipeline.py — offline harness for the full VLM→centroids→path pipeline.

Tests CLIPSeg segmentation, occupancy building, and path planning on a saved overhead
image without requiring a running Gazebo simulation or any ROS nodes.

Prerequisites:
    colcon build --symlink-install --packages-select talking-turtle
    source install/setup.bash
    export OPENAI_API_KEY=<key>   (only needed with --prompt)

Usage (from workspace root):
    # Test planner only (no VLM, fastest):
    python3 src/VLM_mission_planning/scripts/test_pipeline.py \\
        --grid-path '{"raph": ["A1","C3","F5"], "donnie": ["N8","K6","H4"]}'

    # Full pipeline including VLM call:
    python3 src/VLM_mission_planning/scripts/test_pipeline.py \\
        --prompt "Send raph to the chair and donnie to the table"

    # Use A* instead of CHOMP:
    python3 src/VLM_mission_planning/scripts/test_pipeline.py \\
        --grid-path '{"raph": ["A1","C3"]}' --planner astar

Output (written to --out, default debug/offline_test/<robot_name>/):
    raw_overhead.png      — the loaded image
    segmentation.png      — CLIPSeg overlay (green=free, red=obstacle)
    grid_overlay.png      — image with battleship grid
    occ_true.png          — raw occupancy map
    route_centroids.png   — VLM-commanded centroids with robot pose prepended
    route_planned.png     — A* obstacle-avoiding path
    occ_inflated.png      — inflated occupancy map
    inflation_overlay.png — inflation margin visualised on overhead image
    (or sdf.png/chomp_route.png for --planner chomp)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from coord_transform import pixel_to_world, world_to_pixel, gazebo_to_world
from obs_seg import FREE, OCCUPIED, UNKNOWN
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import mask_to_occupancy

from node_Path_Translator import chomp_proj, astar_proj
from node_Path_Translator.chomp_proj import PARAMS as CHOMP_PARAMS
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS

_PLANNERS = {"chomp": chomp_proj, "astar": astar_proj}

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


def _composite_grid(img_bgr: np.ndarray, grid_overlay: np.ndarray | None) -> np.ndarray:
    """Alpha-composite the transparent grid onto the image, mirroring mapper.py."""
    if grid_overlay is None:
        return img_bgr
    g = cv2.resize(grid_overlay, (img_bgr.shape[1], img_bgr.shape[0]))
    if g.ndim == 3 and g.shape[2] == 4:
        a = g[..., 3:4].astype(np.float32) / 255.0
        return (img_bgr * (1 - a) + g[..., :3] * a).astype(np.uint8)
    return img_bgr


def _call_vlm(prompt: str, img_bgr: np.ndarray, grid_overlay: np.ndarray | None,
              model: str, temperature: float = 0.0) -> dict:
    """Call GPT-4o with the grid-overlaid overhead image (mirrors exec.py + mapper.py)."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed; run: pip install openai")

    # Import the same system prompt the live exec node uses.
    sys.path.insert(0, str(_PKG_DIR / "nodes"))
    from node_Executive_API.prompt import EXECUTIVE_SYSTEM_PROMPT

    # Composite the grid before encoding — the VLM must see cell labels to name them.
    img_for_vlm = _composite_grid(img_bgr, grid_overlay)
    _, encoded = cv2.imencode(".png", img_for_vlm)
    map_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    client = OpenAI()
    print(f"Calling {model}…")
    response = client.responses.create(
        model=model,
        temperature=temperature,
        instructions=EXECUTIVE_SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt + "\n\njson"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{map_b64}",
                    },
                ],
            }
        ],
    )
    reply = response.output_text.strip()
    result = _parse_json_reply(reply)
    if result is None:
        sys.exit(f"VLM returned unparseable JSON:\n{reply}")
    paths = result.get("paths", {})
    if not isinstance(paths, dict):
        sys.exit(f"VLM 'paths' is not a dict: {paths!r}")
    print(f"VLM analysis: {result.get('analysis', '')}")
    return paths


def _save_common_debug(
    out_dir: str,
    img_bgr: np.ndarray,
    pix_labels: np.ndarray,
    grid: np.ndarray,
    grid_overlay_img: np.ndarray | None,
) -> None:
    """Mirrors translate.py._save_common_debug without rclpy."""
    cv2.imwrite(os.path.join(out_dir, "raw_overhead.png"), img_bgr)

    seg_color = np.zeros((*pix_labels.shape, 3), dtype=np.uint8)
    seg_color[pix_labels == FREE] = (0, 180, 0)
    seg_color[pix_labels == OCCUPIED] = (0, 0, 200)
    seg_color[pix_labels == UNKNOWN] = (128, 128, 128)
    # Resize seg overlay to match image if CLIPSeg output is at a different resolution.
    if seg_color.shape[:2] != img_bgr.shape[:2]:
        seg_color = cv2.resize(seg_color, (img_bgr.shape[1], img_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(
        os.path.join(out_dir, "segmentation.png"),
        (0.5 * img_bgr + 0.5 * seg_color).astype(np.uint8),
    )

    if grid_overlay_img is not None:
        g = cv2.resize(grid_overlay_img, (img_bgr.shape[1], img_bgr.shape[0]))
        if g.ndim == 3 and g.shape[2] == 4:
            a = g[..., 3:4].astype(np.float32) / 255.0
            comp = (img_bgr * (1 - a) + g[..., :3] * a).astype(np.uint8)
        else:
            comp = g[..., :3]
        cv2.imwrite(os.path.join(out_dir, "grid_overlay.png"), comp)

    g2 = np.flipud(grid.T)
    occ = np.full((*g2.shape, 3), 128, np.uint8)
    occ[g2 == FREE] = (255, 255, 255)
    occ[g2 == OCCUPIED] = (0, 0, 0)
    cv2.imwrite(os.path.join(out_dir, "occ_true.png"), occ)


def _save_inflation_overlay(
    out_dir: str,
    img_bgr: np.ndarray,
    grid: np.ndarray,
    meta: dict,
    infl: np.ndarray,
) -> None:
    """Inflated occupancy map + colour overlay (mirrors astar_proj.save_debug inflation section)."""
    # occ_inflated: white=free, black=occupied, gray=unknown
    g2 = np.flipud(infl.T)
    occ_i = np.full((*g2.shape, 3), 128, np.uint8)
    occ_i[g2 == FREE] = (255, 255, 255)
    occ_i[g2 == OCCUPIED] = (0, 0, 0)
    cv2.imwrite(os.path.join(out_dir, "occ_inflated.png"), occ_i)

    # inflation_overlay: red=true obstacle, yellow=inflation-only margin
    h_img, w_img = img_bgr.shape[:2]
    uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
    xs, ys = pixel_to_world(uu, vv)
    gx = np.floor((xs - meta["origin_x"]) / meta["resolution"]).astype(np.int64)
    gy = np.floor((ys - meta["origin_y"]) / meta["resolution"]).astype(np.int64)
    inb = (gx >= 0) & (gx < meta["width"]) & (gy >= 0) & (gy < meta["height"])
    gxc = np.clip(gx, 0, meta["width"] - 1)
    gyc = np.clip(gy, 0, meta["height"] - 1)
    true_occ = inb & (grid[gyc, gxc] != FREE)
    infl_occ = inb & (infl[gyc, gxc] == OCCUPIED)
    margin = infl_occ & ~true_occ
    ov = img_bgr.copy()
    ov[margin] = (0.5 * img_bgr[margin] + 0.5 * np.array([0, 220, 220])).astype(np.uint8)
    ov[true_occ] = (0.5 * img_bgr[true_occ] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "inflation_overlay.png"), ov)


def _save_route_centroids(out_dir: str, img_bgr: np.ndarray, ref: list) -> None:
    """Orange line + blue dots through the reference route (robot pose prepended).

    Mirrors astar_proj.save_debug route_centroids section exactly.
    """
    rc = img_bgr.copy()
    pts = [(int(round(u)), int(round(v))) for (u, v) in (world_to_pixel(x, y) for x, y in ref)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(rc, a, b, (0, 165, 255), 2)
    for p in pts:
        cv2.circle(rc, p, 8, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(out_dir, "route_centroids.png"), rc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="test_data",
                        help="Dir containing overhead.png + poses.json (default: test_data)")
    parser.add_argument("--grid-path",
                        default='{"raph": ["A8","B8","C8","D8","E8","F8","G8","H8","I8","J8"], "donnie": ["A1"]}',
                        help='JSON string {"robot": ["A1","B2",...]} — skips VLM call')
    parser.add_argument("--prompt", default=None,
                        help="User instruction for GPT-4o (used only if --grid-path not set)")
    parser.add_argument("--planner", default="astar", choices=["chomp", "astar"],
                        help="Path planner to use (default: astar)")
    parser.add_argument("--out", default="debug/offline_test",
                        help="Debug output directory (default: debug/offline_test)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for VLM call (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    args = parser.parse_args()

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
    print(f"Loaded image: {img_bgr.shape[1]}×{img_bgr.shape[0]}")

    with open(poses_path) as f:
        poses_raw = json.load(f)
    # Apply the same Gazebo→world conversion as translate.py._make_pose_cb.
    robot_world_xy = {
        name: gazebo_to_world(p["x"], p["y"]) for name, p in poses_raw.items()
    }
    print("Robot world poses: " +
          ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    # ── Grid overlay (needed by VLM and debug images) ────────────────────
    overlay_path = _CONFIG_DIR / "transparent_grid.png"
    grid_overlay_img = (
        cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        if overlay_path.exists() else None
    )
    if grid_overlay_img is None:
        print("Warning: transparent_grid.png not found — VLM will not see the grid.")

    # ── Determine grid paths ──────────────────────────────────────────────
    if args.prompt:
        paths = _call_vlm(args.prompt, img_bgr, grid_overlay_img,
                          model=args.model, temperature=args.temperature)
        print(f"VLM returned paths: {paths}")
    else:
        paths = json.loads(args.grid_path)
        print(f"Using grid-path: {paths}")

    # ── Segmentation + occupancy (shared across robots) ───────────────────
    print("Running CLIPSeg segmentation…")
    segmenter = TraversabilitySegmenter()
    pix_labels, _ = segmenter.classify(
        img_rgb,
        traversable_prompts=["the floor"],
        untraversable_prompts=[""],
        threshold=0.48,
    )
    grid, meta = mask_to_occupancy(pix_labels, resolution=0.05)
    print(
        f"Occupancy grid: {meta['width']}×{meta['height']} cells "
        f"@ {meta['resolution']} m/cell"
    )

    # ── Grid CSV ──────────────────────────────────────────────────────────
    csv_path = _CONFIG_DIR / "grid_cell_centers.csv"
    if not csv_path.exists():
        sys.exit(f"Grid CSV not found: {csv_path}")
    grid_px = _load_grid_csv(csv_path)

    # ── Planner context (built once, shared across robots) ────────────────
    planner = _PLANNERS[args.planner]
    params = {**CHOMP_PARAMS, **ASTAR_PARAMS}
    print(f"Building {args.planner} planner context…")
    ctx = planner.build(grid, meta, params)

    # ── Plan per robot ────────────────────────────────────────────────────
    for name, labels in paths.items():
        if not isinstance(labels, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose in poses.json; skipping.", file=sys.stderr)
            continue

        # Build reference route via the shared astar_proj function.
        pose_xy = robot_world_xy[name]
        ref, unknown = astar_proj.build_reference(labels, pose_xy, grid_px)
        for lbl in unknown:
            print(f"[{name}] unknown label '{lbl}'; skipping.")

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

        print(f"[{name}] Planning with {args.planner} ({len(ref)} reference pts)…")
        world_path, dbg = planner.plan(ref, ctx, meta, params)
        for w in dbg.get("warnings", []):
            print(f"[{name}] WARNING: {w}")

        if not world_path:
            print(f"[{name}] planner returned empty path.", file=sys.stderr)
            continue

        print(
            f"[{name}] {len(world_path)} waypoints: "
            f"start=({world_path[0][0]:.3f}, {world_path[0][1]:.3f})  "
            f"end=({world_path[-1][0]:.3f}, {world_path[-1][1]:.3f})"
        )

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        _save_common_debug(out_dir, img_bgr, pix_labels, grid, grid_overlay_img)
        _save_route_centroids(out_dir, img_bgr, ref)
        if "infl" in ctx:
            _save_inflation_overlay(out_dir, img_bgr, grid, meta, ctx["infl"])
        dbg["start_world"] = pose_xy
        planner.save_debug(out_dir, img_bgr, grid, meta, ctx, dbg, params)
        print(f"[{name}] Debug images -> {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
