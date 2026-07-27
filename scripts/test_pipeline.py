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

Output (written to --out, default debug/offline_test/):
    vlm_overlay.png       — the exact overlay image sent to the VLM (grid/marks + robot markers)
    robot_paths.png       — both robots' planned trajectories overlaid on the clean overhead image
    robot_paths_waypoints.png — planned trajectories + each robot's labeled VLM waypoint picks
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
import textwrap
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


# Default controller selector: runs when --planner is omitted (manual --planner skips it).
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


def _format_reasoning(text: str, width: int = 100) -> str:
    """Reflow a single-line reasoning string into readable lines: each numbered step "N)" starts a new
    line, lettered sub-steps "a)"/"b)" are indented, and long lines are wrapped to `width`."""
    text = re.sub(r"\s*(?<![\w])(\d+\))\s*", r"\n\1 ", text)         # "1)" .. -> own line
    text = re.sub(r"\s*(?<![\w])([a-z]\))\s+", r"\n   \1 ", text)    # "a)"/"b)" -> indented
    out = []
    for ln in text.splitlines():
        ln = ln.rstrip()
        if not ln:
            continue
        indent = len(ln) - len(ln.lstrip())
        out.append(textwrap.fill(ln, width=width, subsequent_indent=" " * (indent + 4)))
    return "\n".join(out)


def _route_schema(result_key: str, robot_names: list[str], cot: bool,
                  allowed_labels: list[str] | None = None) -> dict:
    """Strict JSON Schema for the planner reply — enforced via Responses API structured outputs.

    Always: {result_key: {<robot>: [labels] for each robot}}. additionalProperties:false + a fixed
    robot roster means the model CANNOT emit extra keys (e.g. a "raph_donnie" shared route) — the
    structure is guaranteed by constrained decoding, not by asking in the prompt.

    When allowed_labels is given, every waypoint is constrained to that enum (the free/blue-dot cells),
    so the model physically CANNOT select an obstacle cell — obstacle avoidance enforced at decode time,
    not merely requested in the prompt.

    When cot=True a leading free-text "reasoning" string field is added. Structured outputs fill
    properties in declaration order, so the model writes its full reasoning trace FIRST and the routes
    are conditioned on it — real chain-of-thought, in one call, with the answer keys still locked.
    """
    item_schema = {"type": "string"}
    if allowed_labels:
        item_schema = {"type": "string", "enum": list(allowed_labels)}
    route_obj = {
        "type": "object", "additionalProperties": False,
        "required": list(robot_names),
        "properties": {name: {"type": "array", "items": item_schema} for name in robot_names},
    }
    properties: dict = {}
    required: list[str] = []
    if cot:                                    # declared first -> generated (reasoned) first
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    properties[result_key] = route_obj
    required.append(result_key)
    return {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}


def _call_planner_vlm(instructions: str, overlay: PILImage.Image, result_key: str,
                      model: str, temperature: float, robot_names: list[str],
                      cot: bool = False, allowed_labels: list[str] | None = None,
                      out_dir: str | None = None) -> tuple[dict, dict]:
    """Second call: vision planner. instructions = per-task prompt, image = the overlay (only).

    The reply is constrained to a strict route schema (structured outputs) so it can only be the fixed
    {result_key:{robots...}} shape — no invented keys. When allowed_labels is given, every waypoint is
    also restricted to that enum (free cells) so obstacle cells cannot be selected. With cot=True the
    schema also carries a leading "reasoning" string generated before the routes.

    Returns (routes, usage) where usage is {input, output, total} token counts for this call.

    The full system prompt and the full raw model output are printed and, if out_dir is given, saved to
    vlm_prompt.txt / vlm_response.txt.
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

    schema = _route_schema(result_key, robot_names, cot, allowed_labels)
    keys = ("reasoning, " if cot else "") + f"{result_key}{{{', '.join(robot_names)}}}"
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
    if out_dir is not None:                       # keep the exact JSON (single line) on disk
        with open(os.path.join(out_dir, "vlm_response.txt"), "w") as f:
            f.write(raw)

    result = _parse_json_reply(raw)
    if result is None:
        sys.exit(f"VLM returned unparseable JSON:\n{raw}")
    routes = result.get(result_key, {})
    if not isinstance(routes, dict):
        sys.exit(f"VLM '{result_key}' is not a dict: {routes!r}")

    # Readable terminal view: reflowed reasoning, then a compact per-robot route listing.
    print("─" * 16 + " VLM response " + "─" * 16)
    if result.get("reasoning"):
        print("REASONING:")
        print(_format_reasoning(str(result["reasoning"])))
        print()
    print(f"{result_key.upper()}:")
    for name, labels in routes.items():
        seq = ", ".join(map(str, labels)) if labels else "(empty — holds position)"
        print(f"  {name} ({len(labels) if isinstance(labels, list) else '?'}): {seq}")
    print("─" * 46)
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


def _draw_dashed_polyline(img: np.ndarray, pts: np.ndarray, color, thickness: int,
                          dash: float = 18.0, phase: float = 0.0) -> None:
    """Draw a dashed polyline through `pts` (Nx2 pixel points).

    Marches along the path in small steps, drawing where (arc_length + phase) % (2*dash) < dash — equal
    dash/gap, period 2*dash. `phase` shifts the pattern so a second robot's dashes fall in the first's
    gaps (pass phase=i*dash), making overlapping paths show as alternating colored dashes.
    """
    period = 2.0 * dash
    dist = float(phase)
    step = 2.0
    for a, b in zip(pts[:-1].astype(float), pts[1:].astype(float)):
        seg = b - a
        seglen = float(np.hypot(seg[0], seg[1]))
        if seglen == 0.0:
            continue
        direction = seg / seglen
        t = 0.0
        while t < seglen:
            s = min(step, seglen - t)
            if (dist + t) % period < dash:           # inside a dash -> draw this sub-segment
                p1 = a + direction * t
                p2 = a + direction * (t + s)
                cv2.line(img, (int(round(p1[0])), int(round(p1[1]))),
                         (int(round(p2[0])), int(round(p2[1]))), color, thickness, cv2.LINE_AA)
            t += s
        dist += seglen


def _draw_robot_legend(vis: np.ndarray, names: list[str], colors: list) -> None:
    """Boxed top-left legend: each robot name in its path color on a light-gray background."""
    if not names:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, lthick, pad, ygap = 1.3, 2, 12, 10
    sizes = [cv2.getTextSize(n, font, scale, lthick)[0] for n in names]
    row_h = max(h for _, h in sizes) + ygap
    box_w = max(w for w, _ in sizes) + 2 * pad
    box_h = row_h * len(names) + ygap
    x0, y0 = 8, 8
    cv2.rectangle(vis, (x0, y0), (x0 + box_w, y0 + box_h), (225, 225, 225), -1)   # light-gray fill
    cv2.rectangle(vis, (x0, y0), (x0 + box_w, y0 + box_h), (170, 170, 170), 1)    # subtle border
    for i, name in enumerate(names):
        baseline_y = y0 + ygap + row_h * i + sizes[i][1]
        cv2.putText(vis, name, (x0 + pad, baseline_y), font, scale, colors[i], lthick, cv2.LINE_AA)


def _save_robot_paths(out_path: str, img_bgr: np.ndarray,
                      world_paths: dict[str, list], camera: str) -> None:
    """Draw every robot's planned trajectory on the clean overhead image in one figure.

    Each path uses the SAME per-robot color as its overlay marker (map_gen._ROBOT_COLORS, assigned by
    insertion order): a hollow circle at the start pose, a DASHED polyline through the waypoints (the
    per-robot dash phase is staggered so overlapping paths stay both-visible as alternating colored
    dashes), and a filled dot at the end, plus a boxed legend.
    """
    vis = img_bgr.copy()
    dash = 18.0
    colors = []
    for i, (name, world_path) in enumerate(world_paths.items()):
        rgb = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        colors.append(bgr)
        pts = np.array([[int(round(u)), int(round(v))]
                        for u, v in (world_to_pixel(x, y, camera) for x, y in world_path)],
                       dtype=np.int32)
        _draw_dashed_polyline(vis, pts, bgr, thickness=6, dash=dash, phase=i * dash)
        cv2.circle(vis, tuple(pts[0]), _ROBOT_RADIUS, bgr, 4, lineType=cv2.LINE_AA)   # start: hollow
        cv2.circle(vis, tuple(pts[-1]), 9, bgr, -1, lineType=cv2.LINE_AA)             # end: filled

    _draw_robot_legend(vis, list(world_paths), colors)
    cv2.imwrite(out_path, vis)


def _save_paths_with_waypoints(out_path: str, img_bgr: np.ndarray,
                               world_paths: dict[str, list], routes: dict[str, list],
                               grid_px: dict, camera: str) -> None:
    """Combined figure: each robot's planned A* trajectory AND its raw VLM-selected waypoints on the
    clean overhead image, in the robot's color (sibling to robot_paths.png).

    Per robot (same color assignment as _save_robot_paths, so colors match robot_paths.png): a staggered
    DASHED polyline for the planned path with a hollow start circle, plus each VLM-chosen grid label
    drawn as a filled dot with its label text (light-gray backed) at that label's pixel center. Only
    robots that produced a planned path are shown; their waypoints come from `routes`.
    """
    vis = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    dash = 18.0
    names = list(world_paths)
    colors = []
    for i, name in enumerate(names):
        rgb = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        colors.append(bgr)

        pts = np.array([[int(round(u)), int(round(v))]
                        for u, v in (world_to_pixel(x, y, camera) for x, y in world_paths[name])],
                       dtype=np.int32)
        _draw_dashed_polyline(vis, pts, bgr, thickness=6, dash=dash, phase=i * dash)
        cv2.circle(vis, tuple(pts[0]), _ROBOT_RADIUS, bgr, 4, lineType=cv2.LINE_AA)   # start: hollow

        # VLM-selected waypoints: a blue filled dot + large label text at each chosen label's center.
        blue = (255, 0, 0)   # BGR
        lbl_scale, lbl_thick = 1.3, 3
        for lbl in routes.get(name, []):
            uv = grid_px.get(lbl)
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            cv2.circle(vis, (u, v), 8, blue, -1, lineType=cv2.LINE_AA)
            tx, ty = u + 13, v - 8
            (tw, th), base = cv2.getTextSize(str(lbl), font, lbl_scale, lbl_thick)
            cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + base), (225, 225, 225), -1)
            cv2.putText(vis, str(lbl), (tx, ty), font, lbl_scale, bgr, lbl_thick, cv2.LINE_AA)

    _draw_robot_legend(vis, names, colors)
    cv2.imwrite(out_path, vis)


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
    parser.add_argument("--out", default="debug/offline_test",
                        help="Debug output directory (default: debug/offline_test)")
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
    segmenter = TraversabilitySegmenter()
    pix_labels, _ = segmenter.classify(
        img_rgb, traversable_prompts=["the floor"], untraversable_prompts=[""], threshold=0.48)
    grid, meta = mask_to_occupancy(pix_labels, _RESOLUTION, camera=camera)
    print(f"Occupancy grid: {meta['width']}×{meta['height']} cells @ {meta['resolution']} m/cell")

    # ── Resolve controller / overlay / CoT (classifier-first, with manual overrides) ──
    if args.planner:                       # manual controller
        task_type = args.planner
    else:                                  # classifier picks the controller (first LLM call)
        task_type = _classify(args.prompt, args.model, args.temperature)
        print(f"Classifier chose controller: {task_type}")

    map_overlay = args.map_overlay or "marked_obs"   # default overlay for every controller
    cot = args.cot                         # boolean, default True

    result_key, planner_name = _TASK_ROUTING[task_type]
    planner = _PLANNERS[planner_name]
    print(f"Controller: {task_type} | overlay: {map_overlay} | planner: {planner_name} "
          f"(key '{result_key}') | CoT: {cot}")

    # ── Filtered planning context (mirrors exec.run_segmentation) ─────────
    # Edge/footprint clearing + inflation live in obs_seg.occupancy and are applied once here; the
    # overlay renderer and the planner modules both consume this single grid and never redo it.
    world_poses = [ned_to_world(*p) for p in robot_poses_ned.values() if p]
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    ctx = {"infl": infl}

    # ── Build the prompt + overlay image together (single source of truth) ─
    instructions, map_b64 = generate_prompt(
        args.prompt, task_type, map_overlay,
        pil_img=pil_rgba, occ_grid=infl, occ_meta=meta, camera=camera,
        robot_poses=robot_poses_ned, cot=cot)
    overlay = PILImage.open(io.BytesIO(base64.b64decode(map_b64))).convert("RGBA")
    overlay.convert("RGB").save(os.path.join(args.out, "vlm_overlay.png"))
    # Waypoints are left unconstrained (no enum) — the schema still locks the route shape/keys, but the
    # model may pick any label and we rely on it choosing sensibly (the planner projects picks to free
    # cells). Pass allowed_labels=... to _call_planner_vlm to re-enable the free-cell enum.
    routes, usage = _call_planner_vlm(instructions, overlay, result_key, args.model, args.temperature,
                                      list(robot_world_xy), cot=cot, out_dir=args.out)

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

        wp_png = os.path.join(args.out, "robot_paths_waypoints.png")
        _save_paths_with_waypoints(wp_png, img_bgr, world_paths, routes, grid_px, camera)
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
