"""Shared debug-artifact writers (plain library, NOT a ROS node).

Single source of truth for persisting run artifacts, so the live nodes and the offline
harnesses write the same files the same way. No rclpy, no node, no entry point — this is
imported by both `node_Executive_API` / `node_Path_Translator` (live) and the
`scripts/test_*.py` harnesses (ROS-free), which is only possible from a plain library under
`nodes/`; `scripts/` is not importable (no `__init__.py`, installed to `share/`).

Error handling lives HERE, once. Every public function creates its output directory if
needed and NEVER raises: a debug-write problem must not abort a plan. Failures are reported
through the caller-supplied ``on_error`` callable, which defaults to ``print``. ROS nodes pass
``on_error=self.get_logger().warn`` so messages reach the ROS log without this module ever
importing rclpy.

Artifacts written here:
    raw_overhead.png            the overhead frame the run was planned from
    marks_overlay.png           the exact overlay image sent to the VLM (byte-for-byte)
    inflation_overlay.png       obstacles (red) + inflation margin (yellow) on that frame
    vlm_prompt.txt              the exact system prompt sent to the planner VLM
    vlm_response.txt            the raw model reply
    vlm_selections.png          one robot's raw VLM picks, before planning
    robot_paths_waypoints.png   every robot's planned trajectory + its labeled VLM picks

Everything except vlm_selections.png is robot-independent, so callers write those ONCE per plan
at the debug-dir top level; only the per-robot artifacts live in <debug_dir>/<robot>/.

Route/occupancy visualisations (route_planned, inflation_overlay, …) are written by the
planner modules' own ``save_debug`` — see node_Path_Translator/astar_proj.py
"""
from __future__ import annotations

import base64
import os
import re

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

from coord_transform import world_to_pixel
# Style constants come from map_gen so a robot's PATH colour is the same colour as its MARKER on
# the overlay the VLM saw, and selection dots match the set-of-marks style. Importing map_gen (not
# copying the values) is what keeps those in lock-step; map_gen never imports this module, so there
# is no cycle, and it pulls in no ROS dependency.
from node_Executive_API.map_gen import (
    _N_COLS, _N_ROWS, _DOT_RADIUS, _LABEL_SIZE, _ROBOT_COLORS, _ROBOT_RADIUS)


def _safe(what: str, fn, on_error) -> bool:
    """Run `fn`, reporting any failure via `on_error` instead of raising. Returns success."""
    try:
        fn()
        return True
    except Exception as exc:  # noqa: BLE001 — debug saving must never break the caller
        (on_error or print)(f"Debug save failed ({what}): {exc}")
        return False


def save_raw_overhead(out_dir: str, base_bgr: np.ndarray, *, on_error=None) -> bool:
    """Write `base_bgr` (BGR) as <out_dir>/raw_overhead.png."""
    def _write():
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "raw_overhead.png"), base_bgr)
    return _safe("raw_overhead.png", _write, on_error)


def save_marks_overlay(out_dir: str, png_b64: str, *, on_error=None) -> bool:
    """Write the overlay image actually sent to the VLM as <out_dir>/marks_overlay.png.

    `png_b64` is the base64 PNG string every caller already holds — the exact value interpolated
    into the request's ``data:image/png;base64,...`` URL (produced by map_gen._to_b64, or by the
    baselines encoding their own overlay). The bytes are decoded and written verbatim, so the
    file is a byte-exact copy of the image the model received: no re-encode, and no chance of
    the saved artifact drifting from what was sent.

    Whatever overlay a caller sends is what lands here — the set-of-marks image (blue dots / red
    X) in production, the battleship grid for that baseline, the unmarked frame for the
    regression baseline. Robot-independent: write it once per plan, not per robot.
    """
    def _write():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "marks_overlay.png"), "wb") as f:
            f.write(base64.b64decode(png_b64))
    return _safe("marks_overlay.png", _write, on_error)


def save_inflation_overlay(out_dir: str, overlay_bgr: np.ndarray, *, on_error=None) -> bool:
    """Write an already-rendered inflation overlay as <out_dir>/inflation_overlay.png.

    Takes the rendered image rather than the grids, because the caller renders it once (via
    obs_seg.occupancy.render_inflation_overlay, its single source of truth) and reuses the same
    array as the canvas for every robot's route_planned.png — re-rendering or re-reading it per
    robot is pure waste. Robot-independent: write it once per plan, not per robot.
    """
    def _write():
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "inflation_overlay.png"), overlay_bgr)
    return _safe("inflation_overlay.png", _write, on_error)


def save_vlm_exchange(out_dir: str, prompt: str, response: str, *, on_error=None) -> bool:
    """Write the planner exchange: <out_dir>/vlm_prompt.txt and <out_dir>/vlm_response.txt.

    `prompt` is the assembled system prompt, `response` the raw model reply (kept verbatim so
    the exact JSON can be re-parsed). Both overwrite each run — only the latest is kept.
    """
    def _write():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "vlm_prompt.txt"), "w") as f:
            f.write(prompt)
        with open(os.path.join(out_dir, "vlm_response.txt"), "w") as f:
            f.write(response)
    return _safe("vlm_prompt/response.txt", _write, on_error)


# ── VLM selection / trajectory figures ────────────────────────────────────────
# Moved here from scripts/test_pipeline.py so the live nodes can produce them too — scripts/ is
# not importable from a node. Rendering is unchanged.

def _cell_rect(label: str, w: int, h: int):
    """Battleship cell label (e.g. 'C4') -> pixel rect (x0,y0,x1,y1); None if malformed/out of range."""
    m = re.match(r"^([A-Za-z])(\d+)$", label.strip())
    if not m:
        return None
    col = ord(m.group(1).upper()) - ord("A")
    row = int(m.group(2)) - 1
    if not (0 <= col < _N_COLS and 0 <= row < _N_ROWS):
        return None
    cw, ch = w / _N_COLS, h / _N_ROWS
    return (int(col * cw), int(row * ch), int((col + 1) * cw), int((row + 1) * ch))


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


def _draw_robot_legend(vis: np.ndarray, names, colors) -> None:
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


def save_vlm_selections(out_dir: str, img_bgr: np.ndarray, map_overlay: str,
                        labels, grid_px: dict, *, on_error=None) -> bool:
    """Plot one robot's raw VLM selections (before planning) as <out_dir>/vlm_selections.png.

    The style mirrors the OVERLAY the VLM actually saw (not the controller):
    points / marked_obs: labeled dots drawn in the SAME set-of-marks style as marks_overlay.png
                         (blue dot + gray-backed blue label), with a slightly larger label font.
    battleship:          the chosen grid cells, shaded and outlined.
    """
    def _write():
        os.makedirs(out_dir, exist_ok=True)
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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                      size=label_size)
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
    return _safe("vlm_selections.png", _write, on_error)


def _draw_paths(vis: np.ndarray, world_paths: dict, camera: str):
    """Draw each robot's dashed trajectory + named hollow start marker, in place.

    Returns ``(colors, ends)``: the per-robot BGR color and final pixel point, both in
    `world_paths` insertion order. Shared by save_robot_paths and save_paths_with_waypoints so
    the two assign colors identically (map_gen._ROBOT_COLORS by insertion order) and stagger the
    dash phase the same way, keeping overlapping routes both-visible as alternating dashes.

    Each start circle is labeled with its robot's name in that robot's own color, so a figure can be
    read without cross-referencing the legend — which matters most when the two dashed routes cross.
    """
    dash = 18.0
    colors, ends = [], []
    for i, name in enumerate(world_paths):
        rgb = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        colors.append(bgr)
        pts = np.array([[int(round(u)), int(round(v))]
                        for u, v in (world_to_pixel(x, y, camera) for x, y in world_paths[name])],
                       dtype=np.int32)
        _draw_dashed_polyline(vis, pts, bgr, thickness=6, dash=dash, phase=i * dash)
        cv2.circle(vis, tuple(pts[0]), _ROBOT_RADIUS, bgr, 4, lineType=cv2.LINE_AA)   # start: hollow
        _draw_robot_name(vis, name, tuple(pts[0]), bgr)
        ends.append(tuple(pts[-1]))
    return colors, ends


def _draw_robot_name(vis: np.ndarray, name: str, center, bgr) -> None:
    """Write `name` beside a start circle at `center`, in that robot's color, in place.

    Same gray-backed style as the waypoint labels so the two read as one annotation layer. Placed
    to the upper-right of the marker, flipping to the left / below when that would run off the
    frame — robots often start near an edge, where an unclamped label would be cut in half.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, pad = 1.1, 3, 3
    (tw, th), base = cv2.getTextSize(name, font, scale, thick)
    h, w = vis.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    tx = cx + _ROBOT_RADIUS + 6                       # default: right of the circle
    if tx + tw + pad > w:
        tx = cx - _ROBOT_RADIUS - 6 - tw              # too close to the right edge -> go left
    ty = cy - _ROBOT_RADIUS - 6                       # default: above the circle
    if ty - th - pad < 0:
        ty = cy + _ROBOT_RADIUS + 6 + th              # too close to the top -> go below
    tx = max(pad, min(tx, w - tw - pad))
    ty = max(th + pad, min(ty, h - base - pad))

    cv2.rectangle(vis, (tx - pad, ty - th - pad), (tx + tw + pad, ty + base), (225, 225, 225), -1)
    cv2.putText(vis, name, (tx, ty), font, scale, bgr, thick, cv2.LINE_AA)


def save_robot_paths(out_path: str, img_bgr: np.ndarray, world_paths: dict, camera: str, *,
                     on_error=None) -> bool:
    """Draw every robot's trajectory on the clean overhead image, written to `out_path`.

    Each path uses the SAME per-robot color as its overlay marker: a hollow circle at the start,
    a staggered DASHED polyline through the waypoints, a filled dot at the end, plus a legend.

    Retained for test_battleship_baseline's robot_paths_centroids.png, which renders the
    reference route through the chosen cell centroids (pre-A*). The general combined figure is
    save_paths_with_waypoints.
    """
    def _write():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        vis = img_bgr.copy()
        colors, ends = _draw_paths(vis, world_paths, camera)
        for color, end in zip(colors, ends):
            cv2.circle(vis, end, 9, color, -1, lineType=cv2.LINE_AA)   # end: filled
        _draw_robot_legend(vis, list(world_paths), colors)
        cv2.imwrite(out_path, vis)
    return _safe(os.path.basename(out_path), _write, on_error)


def save_paths_with_waypoints(out_path: str, img_bgr: np.ndarray, world_paths: dict,
                              routes: dict, grid_px: dict, camera: str, *,
                              on_error=None) -> bool:
    """Combined figure: every robot's planned trajectory AND its raw VLM-selected waypoints.

    Per robot: a staggered DASHED polyline for the planned path with a hollow start circle, plus
    each VLM-chosen grid label drawn as a filled dot with its label text (light-gray backed) at
    that label's pixel center. Only robots that produced a planned path are shown; their
    waypoints come from `routes`, and labels absent from `grid_px` are skipped.
    """
    def _write():
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        vis = img_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        colors, _ends = _draw_paths(vis, world_paths, camera)
        blue = (255, 0, 0)   # BGR
        lbl_scale, lbl_thick = 1.3, 3
        for i, name in enumerate(world_paths):
            for lbl in routes.get(name, []):
                uv = grid_px.get(lbl)
                if uv is None:
                    continue
                u, v = int(round(uv[0])), int(round(uv[1]))
                cv2.circle(vis, (u, v), 8, blue, -1, lineType=cv2.LINE_AA)
                tx, ty = u + 13, v - 8
                (tw, th), base = cv2.getTextSize(str(lbl), font, lbl_scale, lbl_thick)
                cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + base),
                              (225, 225, 225), -1)
                cv2.putText(vis, str(lbl), (tx, ty), font, lbl_scale, colors[i], lbl_thick,
                            cv2.LINE_AA)
        _draw_robot_legend(vis, list(world_paths), colors)
        cv2.imwrite(out_path, vis)
    return _safe(os.path.basename(out_path), _write, on_error)
