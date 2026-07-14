#!/usr/bin/env python3
"""test_pipeline.py — offline harness mirroring the Executive's two-call pipeline.

Runs the full VLM planning pipeline on a saved overhead image (no Gazebo, no running ROS nodes):
segmentation → prompt + map overlay → planner → debug images. It reuses the SAME prompt constructor
(prompt_gen.generate_prompt), overlays (map_gen) and planners (node_Path_Translator) as the live
exec/translate nodes, so what you see here is what the robots would get.

The controller and overlay are chosen explicitly (the classifier is kept in-code but not used here):
    --planner {nav2point, maneuver, coverage}   controller / low-level planner (required)
    --map-overlay {points, marked_obs, battleship}   overlay style rendered + described (required)
    --cot                                        append the generic chain-of-thought scaffold

Usage (from workspace root):
    python3 src/VLM_mission_planning/scripts/test_pipeline.py \\
        --planner nav2point --map-overlay points \\
        --prompt "Send raph to the chair and donnie to the table"

    python3 src/VLM_mission_planning/scripts/test_pipeline.py \\
        --planner coverage --map-overlay battleship \\
        --prompt "Have the robots patrol the left side of the room"

Output (written to --out, default debug/offline_test/):
    vlm_overlay.png       — the exact overlay image sent to the VLM (grid/marks + robot markers)
    robot_paths.png       — both robots' planned trajectories overlaid on the clean overhead image
  and per robot in <out>/<robot_name>/:
    raw_overhead.png      — the loaded image
    segmentation.png      — CLIPSeg overlay (green=free, red=obstacle)
    occ_true.png          — raw occupancy map
    vlm_selections.png    — the VLM's raw picks (dots for nav2point/maneuver, shaded cells for coverage)
    route_centroids.png   — planner reference route (robot pose + selected centroids)
    route_planned.png     — planner's obstacle-avoiding path
    occ_inflated.png      — inflated occupancy map
    inflation_overlay.png — inflation margin visualised on overhead image
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world, world_to_pixel
from obs_seg import FREE, OCCUPIED, UNKNOWN
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import mask_to_occupancy, create_filtered_occupancy_map, RESOLUTION as _RESOLUTION

from node_Path_Translator import astar_proj, coverage_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Path_Translator.coverage_proj import PARAMS as COVERAGE_PARAMS

from node_Executive_API.prompt_gen import (
    generate_prompt, CONTROLLERS, MAP_OVERLAY_TYPES, CLASSIFIER_SYSTEM_PROMPT)
from node_Executive_API.map_gen import (
    _N_COLS as _GRID_COLS, _N_ROWS as _GRID_ROWS,
    _DOT_RADIUS, _LABEL_SIZE, _ROBOT_COLORS, _ROBOT_RADIUS)

_PLANNERS = {"astar": astar_proj, "coverage": coverage_proj}

# Mirror of exec._TASK_ROUTING: task type -> (VLM output key, planner name). Kept in sync by hand
# so this offline harness stays free of rclpy (the live routing lives in node_Executive_API/exec.py).
_TASK_ROUTING = {
    "nav2point": ("waypoints", "astar"),
    "maneuver":  ("waypoints", "astar"),
    "coverage":  ("regions",   "coverage"),
}

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


# NOTE: kept for later use — currently unused because the planner is chosen explicitly via --planner.
def _classify(prompt: str, model: str, temperature: float) -> str:
    """First call: text-only classifier -> controller (one of CONTROLLERS)."""
    client = _openai_client()
    print(f"Classifying instruction with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature,
        instructions=CLASSIFIER_SYSTEM_PROMPT,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    result = _parse_json_reply(response.output_text.strip()) or {}
    task_type = result.get("task_type")
    if task_type not in CONTROLLERS:
        sys.exit(f"Classifier returned unknown task_type {task_type!r}; "
                 f"expected one of {sorted(CONTROLLERS)}.")
    return task_type


def _call_planner_vlm(instructions: str, overlay: PILImage.Image, result_key: str,
                      model: str, temperature: float, out_dir: str | None = None) -> tuple[dict, dict]:
    """Second call: vision planner. instructions = per-task prompt, image = the overlay (only).

    Returns (routes, usage) where usage is {input, output, total} token counts for this call.

    The full system prompt and the full raw model output (the chain-of-thought reasoning trace + final
    JSON, when present) are printed and, if out_dir is given, saved to vlm_prompt.txt / vlm_response.txt.
    """
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    map_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(instructions)
    print("─" * 71)
    if out_dir is not None:
        with open(os.path.join(out_dir, "vlm_prompt.txt"), "w") as f:
            f.write(instructions)

    client = _openai_client()
    print(f"Planning with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature, instructions=instructions,
        input=[{
            "role": "user",
            "content": [{"type": "input_image",
                         "image_url": f"data:image/png;base64,{map_b64}"}],
        }],
    )
    u = getattr(response, "usage", None)
    usage = {
        "input":  getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "total":  getattr(u, "total_tokens", 0) or 0,
    }

    raw = response.output_text.strip()
    print("─" * 12 + " VLM full response (reasoning + output) " + "─" * 12)
    print(raw)
    print("─" * 64)
    if out_dir is not None:
        with open(os.path.join(out_dir, "vlm_response.txt"), "w") as f:
            f.write(raw)

    result = _parse_json_reply(raw)
    if result is None:
        sys.exit(f"VLM returned unparseable JSON:\n{raw}")
    routes = result.get(result_key, {})
    if not isinstance(routes, dict):
        sys.exit(f"VLM '{result_key}' is not a dict: {routes!r}")
    print(f"VLM analysis: {result.get('analysis', '')}")
    return routes, usage


# ── Debug image writers (no rclpy) ──────────────────────────────────────────────

def _save_common_debug(
    out_dir: str,
    img_bgr: np.ndarray,
    pix_labels: np.ndarray,
    grid: np.ndarray,
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

    g2 = np.flipud(grid.T)
    occ = np.full((*g2.shape, 3), 128, np.uint8)
    occ[g2 == FREE] = (255, 255, 255)
    occ[g2 == OCCUPIED] = (0, 0, 0)
    cv2.imwrite(os.path.join(out_dir, "occ_true.png"), occ)


def _cell_rect(label: str, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Battleship cell label (e.g. 'C4') -> pixel rect (x0,y0,x1,y1); None if malformed/out of range."""
    m = re.match(r"^([A-Za-z])(\d+)$", label.strip())
    if not m:
        return None
    col = ord(m.group(1).upper()) - ord("A")
    row = int(m.group(2)) - 1
    if not (0 <= col < _GRID_COLS and 0 <= row < _GRID_ROWS):
        return None
    cw, ch = w / _GRID_COLS, h / _GRID_ROWS
    return (int(col * cw), int(row * ch), int((col + 1) * cw), int((row + 1) * ch))


def _save_vlm_selections(out_dir: str, img_bgr: np.ndarray, map_overlay: str,
                         labels: list, grid_px: dict) -> None:
    """Plot the VLM's raw selections for one robot (before planning).

    The style mirrors the OVERLAY the VLM actually saw (not the controller):
    points / marked_obs: labeled dots drawn in the SAME set-of-marks style as vlm_overlay.png
                         (blue dot + gray-backed blue label), with a slightly larger label font.
    battleship:          the chosen grid cells, shaded and outlined.
    """
    if map_overlay == "battleship":
        vis = img_bgr.copy()
        h, w = img_bgr.shape[:2]
        fill = vis.copy()
        rects = [r for r in (_cell_rect(lbl, w, h) for lbl in labels) if r is not None]
        for x0, y0, x1, y1 in rects:
            cv2.rectangle(fill, (x0, y0), (x1, y1), (0, 200, 0), -1)
        vis = cv2.addWeighted(fill, 0.4, vis, 0.6, 0)
        for x0, y0, x1, y1 in rects:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 150, 0), 2)
        cv2.imwrite(os.path.join(out_dir, "vlm_selections.png"), vis)
        return

    # nav2point / maneuver — mirror render_grid_points_map's mark style (PIL), slightly larger font.
    label_size = _LABEL_SIZE + 6
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=label_size)
    except OSError:
        font = ImageFont.load_default()
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    r, pad = _DOT_RADIUS, 2
    for lbl in labels:
        uv = grid_px.get(lbl)
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(0, 0, 255))
        tx, ty = u + r + 3, v - label_size // 2
        bbox = draw.textbbox((tx, ty), lbl, font=font)
        box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(box, fill=(220, 220, 220))
        draw.text((tx, ty), lbl, fill=(0, 0, 255), font=font)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, "vlm_selections.png"), out)


def _path_length(world_path: list) -> float:
    """Total length (metres) of a planned path = sum of Euclidean steps between consecutive waypoints."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(world_path, world_path[1:]))


def _save_robot_paths(out_path: str, img_bgr: np.ndarray,
                      world_paths: dict[str, list], camera: str) -> None:
    """Draw every robot's planned trajectory on the clean overhead image in one figure.

    Each path uses the SAME per-robot color as its overlay marker (map_gen._ROBOT_COLORS, assigned by
    insertion order): a hollow circle at the start pose, a polyline through the waypoints, and a filled
    dot at the end, plus a small legend.
    """
    vis = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (name, world_path) in enumerate(world_paths.items()):
        rgb = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        pts = np.array([[int(round(u)), int(round(v))]
                        for u, v in (world_to_pixel(x, y, camera) for x, y in world_path)],
                       dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False, color=bgr, thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(vis, tuple(pts[0]), _ROBOT_RADIUS, bgr, 3, lineType=cv2.LINE_AA)   # start: hollow
        cv2.circle(vis, tuple(pts[-1]), 8, bgr, -1, lineType=cv2.LINE_AA)             # end: filled
        cv2.putText(vis, name, (12, 34 + i * 34), font, 1.0, bgr, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--planner", required=True, choices=list(CONTROLLERS),
                        help="Controller / low-level planner (required). Chosen explicitly; the "
                             "classifier is kept in-code but not used here for now.")
    parser.add_argument("--map-overlay", required=True, dest="map_overlay",
                        choices=list(MAP_OVERLAY_TYPES),
                        help="Overlay style rendered + described in the prompt (required).")
    parser.add_argument("--prompt", required=True,
                        help="Operator instruction for the VLM (required in every mode)")
    parser.add_argument("--data", default="test_data",
                        help="Dir containing overhead.png + poses.json (default: test_data)")
    parser.add_argument("--camera", default="gazebo", choices=["gazebo", "lab_test"],
                        help="Overhead camera calibration for pixel<->world (default: gazebo)")
    parser.add_argument("--out", default="debug/offline_test",
                        help="Debug output directory (default: debug/offline_test)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the classifier + planner calls (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    parser.add_argument("--cot", action="store_true",
                        help="Append the generic chain-of-thought reasoning scaffold to the prompt.")
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
    segmenter = TraversabilitySegmenter()
    pix_labels, _ = segmenter.classify(
        img_rgb, traversable_prompts=["the floor"], untraversable_prompts=[""], threshold=0.48)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    # ── Controller (planner) chosen explicitly via --planner ──────────────
    task_type = args.planner
    print(f"Planner (controller): {task_type}")
    result_key, planner_name = _TASK_ROUTING[task_type]
    planner = _PLANNERS[planner_name]
    print(f"Planner: {planner_name}  (VLM output key: '{result_key}')")

    # ── Filtered planning context (mirrors exec.run_segmentation) ─────────
    # Edge/footprint clearing + inflation live in obs_seg.occupancy and are applied once here; the
    # overlay renderer and the planner modules both consume this single grid and never redo it.
    world_poses = [ned_to_world(*p) for p in robot_poses_ned.values() if p]
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    ctx = {"infl": infl}

    # ── Build the prompt + overlay image together (single source of truth) ─
    print(f"Overlay: {args.map_overlay}   CoT: {args.cot}")
    instructions, map_b64 = generate_prompt(
        args.prompt, task_type, args.map_overlay,
        pil_img=pil_rgba, occ_grid=infl, occ_meta=meta, camera=camera,
        robot_poses=robot_poses_ned, cot=args.cot)
    overlay = PILImage.open(io.BytesIO(base64.b64decode(map_b64))).convert("RGBA")
    overlay.convert("RGB").save(os.path.join(args.out, "vlm_overlay.png"))
    routes, usage = _call_planner_vlm(instructions, overlay, result_key, args.model, args.temperature,
                                      out_dir=args.out)
    print(f"VLM routes: {routes}")

    # ── Grid CSV (shared across robots) ───────────────────────────────────
    csv_path = _CONFIG_DIR / "grid_cell_centers.csv"
    if not csv_path.exists():
        sys.exit(f"Grid CSV not found: {csv_path}")
    grid_px = _load_grid_csv(csv_path)

    params = {**ASTAR_PARAMS, **COVERAGE_PARAMS}

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
        print(f"[{name}] {len(world_path)} waypoints: "
              f"start=({world_path[0][0]:.3f}, {world_path[0][1]:.3f})  "
              f"end=({world_path[-1][0]:.3f}, {world_path[-1][1]:.3f})  "
              f"path length={length:.2f} m")

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        _save_common_debug(out_dir, img_bgr, pix_labels, grid)
        _save_vlm_selections(out_dir, img_bgr, args.map_overlay, labels, grid_px)
        dbg["start_world"] = pose_xy
        # save_debug renders occ_inflated + inflation_overlay (via occupancy.render_inflation_overlay)
        # and the planned route on top; pass `cleared` (post-override occupancy) as the red layer.
        planner.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plot (both robots on the clean overhead image) ─
    if world_paths:
        paths_png = os.path.join(args.out, "robot_paths.png")
        _save_robot_paths(paths_png, img_bgr, world_paths, camera)
        print(f"Combined robot paths -> {paths_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
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
