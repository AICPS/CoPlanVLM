#!/usr/bin/env python3
"""test_regression_only.py — coordinate-pointing path baseline (single VLM call, no CoT).

The opposite extreme from test_pipeline's set-of-marks + chain-of-thought pipeline: the VLM sees an
UNMARKED overhead image (no grid, no set-of-marks — only a hollow circle + name label per robot so it can
tell raph from donnie) and returns, for each robot, an ordered list of KEY POINTS as coordinates [x, y].
Those points are mapped to world coordinates and fed to the astar_proj planner, which projects each to
the nearest free cell and routes A* between them. No classifier, no chain-of-thought.

COORDINATES ARE NORMALIZED, not raw pixels. The VLM answers on a 0-GRID_MAX grid spanning the image
(x = 0 at the left edge, GRID_MAX at the right; y = 0 at the top, GRID_MAX at the bottom) and this script
converts back with _from_grid. Two reasons, both about measuring pointing rather than arithmetic:
  * for vision the image is rescaled so its short side is 768 px, so a 1936x1216 frame reaches the model
    as roughly 1223x768 — asking for coordinates in the original resolution makes it silently rescale;
  * sim (1936x1216) and real (1920x1200) frames would otherwise need different coordinate spaces for the
    same physical scene. Normalized, one prompt is correct for both.
The robots' TRUE normalized positions are given in the prompt as calibration anchors (_anchor_lines) —
they are the only points in the image whose location is known exactly.

Reuses the same segmentation/occupancy, goal separation, planner (node_Path_Translator.astar_proj) and
debug writers as test_pipeline / test_battleship_baseline (imported), so the only differences are the
unmarked overlay, the custom single-call prompt, and using the VLM's own coordinates (not grid-cell
centroids) as reference points.

    --mode sim   the Gazebo overhead image  (test_data/overhead.png + poses.json, gazebo calibration)
    --mode real  an AVL_* lab image         (--scene, hardcoded poses, lab_test calibration)

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_regression_only.py --mode sim \\
        --prompt "Send raph to the chair and donnie to the box"

    python3 src/CoPlanVLM/scripts/test_regression_only.py --mode real --scene AVL_3 \\
        --prompt "Send raph to the chair and donnie to the box"

Output (written to --out, default debug/regression_only_{sim,real}/):
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
import math
import os
import sys

import cv2
from PIL import Image as PILImage, ImageDraw

import debug_io
from coord_transform import (gazebo_to_world, gazebo_to_ned, ned_to_world, world_to_ned,
                             pixel_to_world, world_to_pixel)
from obs_seg.segmenter import segment_frame
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)
from node_Path_Translator import astar_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Executive_API import map_gen

# Reuse test_pipeline's helpers (OpenAI client, JSON parse, goal separation, evaluation printer) to
# avoid drift, and test_pipeline_real's SCENES so real-mode images/poses have exactly one definition.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp         # noqa: E402
import test_pipeline_real as tpr   # noqa: E402

# Per-mode defaults, same shape as test_pipeline_no_markers._OUT_DIRS. sim and real write to separate
# directories so one mode's results can never clobber the other's.
_OUT_DIRS = {"sim": "debug/regression_only_sim", "real": "debug/regression_only_real"}
_CAMERAS  = {"sim": "gazebo", "real": "lab_test"}

# Normalized coordinate range the VLM answers in: [0, GRID_MAX] on BOTH axes, spanning the image.
GRID_MAX = 1000


# ── Prompt (our own; no chain-of-thought) ───────────────────────────────────────
REGRESSION_PROMPT = """\
You are a path-planning agent controlling two TurtleBot4 robots named "raph" and "donnie" that share one
workspace. You are given an overhead camera image of the environment. There is a small hollow circle drawn on each robot with its name next to it:
- "raph"   — the round black TurtleBot marked with a magenta circle labeled "raph".
- "donnie" — the round black TurtleBot marked with a light-blue circle labeled "donnie".

All directions are from the IMAGE's point of view, NOT a robot's or person's perspective: the left/right
of an object means its left/right side as it appears in the image, and up/down mean toward the top/bottom
of the image. Cardinal directions are fixed to the image: NORTH/UP is toward the top, SOUTH/DOWN toward
the bottom, WEST/LEFT toward the left edge, EAST/RIGHT toward the right edge.
The robots must avoid all obstacles while moving.

COORDINATE SYSTEM:
Give every point as [x, y] on a NORMALIZED grid that does not depend on the image's pixel size:
  x = 0 at the LEFT edge, x = {gmax} at the RIGHT edge
  y = 0 at the TOP edge,  y = {gmax} at the BOTTOM edge
So [{ghalf}, {ghalf}] is the exact centre of the image, [0, 0] the top-left corner, [{gmax}, {gmax}] the
bottom-right corner, and [{ghalf}, 0] the middle of the top edge. Both numbers must be INTEGERS between
0 and {gmax}. Never report pixel coordinates and never give a value outside 0-{gmax}.

KNOWN POSITIONS — the robots' true coordinates on that grid. Use them to calibrate every estimate:
{anchors}
Judge every other point by comparing it to these: something halfway between the robots is halfway between
their coordinates, something further right than "raph" has a larger x, something above "donnie" has a
smaller y.

YOUR TASK:
For each robot, choose an ordered list of KEY POINTS the robot must travel to in order to accomplish the
operator's instruction, in travel order. Do NOT repeat the robot's own current position — the route
already starts there; list only where it must GO. Give as FEW points as the task needs: if the robot can
reach its destination on a clear straight run, ONE point — the destination — is the correct answer. Add
intermediate points only where they are needed to steer around something. Try to AVOID traveling into
obstacles: look at the image and keep every point on clear, open floor, and remember the robot drives in
a straight line between consecutive points, so no straight segment may cut through an object, furniture,
a wall or the other robot. You MUST include both robots; if a robot has no task, give it an empty list.

Respond with EXACTLY one JSON object and nothing else:
{{"waypoints": {{"raph": [[x, y], ...], "donnie": [[x, y], ...]}}}}

Example 1 — a simple "go there" task. For a DIFFERENT image than this one, where raph is at [180, 620]
and donnie at [640, 210], the chair is at [520, 430], the box is at [950, 500], and both robots have open
floor in front of them, the instruction "Send raph to the chair and donnie to the box" gives:
{{"waypoints": {{"raph": [[520, 430]], "donnie": [[950, 500]]}}}}
(one point each — just the destination. No detour is needed, so no extra points are invented.)

Example 2 — For another scene donnie is at [640, 210], the chair is at [640, 820],
and a large table covers the middle of the image around [640, 500], directly between them. Driving
straight down would cut through the table, so the route steps around its WEST side before turning back to
the chair. The instruction "Send donnie to the chair" gives:
{{"waypoints": {{"raph": [], "donnie": [[380, 320], [380, 720], [640, 820]]}}}}
(three points: two to clear the table on the west, then the destination. Every straight segment between
consecutive points — [640,210]->[380,320], [380,320]->[380,720], [380,720]->[640,820] — stays on open
floor. raph has no task, so it gets an empty list.)"""


# ── Unmarked overlay (robot name markers only) ──────────────────────────────────

def _render_robots_only(pil_rgba: PILImage.Image, robot_poses_ned: dict,
                        camera: str) -> PILImage.Image:
    """The 'unmarked' image: a copy of the overhead image with ONLY the robot circles + name labels
    (no grid, no set-of-marks), so the VLM can identify raph vs donnie while pointing on bare floor."""
    result = pil_rgba.copy()
    draw = ImageDraw.Draw(result)
    map_gen._draw_robot_markers(draw, robot_poses_ned, camera, map_gen._LABEL_FONT)
    return result


# ── Normalized grid <-> pixels ───────────────────────────────────────────────────

def _to_grid(u: float, v: float, w: int, h: int) -> tuple[int, int]:
    """Pixel (u, v) -> normalized integer grid coords, the space the VLM answers in.

    Each axis is normalized by its OWN extent, so one grid unit is w/GRID_MAX px horizontally and
    h/GRID_MAX px vertically — the grid is not square in metres on a non-square image. That is
    deliberate: it is what makes one prompt correct for both the 1936x1216 sim frame and the 1920x1200
    lab frames, and the model is only ever told about edges, never about aspect ratio.
    """
    return int(round(u / w * GRID_MAX)), int(round(v / h * GRID_MAX))


def _from_grid(gx: float, gy: float, w: int, h: int) -> tuple[float, float]:
    """Normalized grid coords -> pixel (u, v). Inverse of _to_grid, up to integer rounding."""
    return gx / GRID_MAX * w, gy / GRID_MAX * h


def _anchor_lines(robot_world_xy: dict, w: int, h: int, camera: str) -> str:
    """The robots' TRUE positions, expressed on the same normalized grid the VLM must answer in.

    These are the only points in the image whose location is known exactly, so they are the only honest
    calibration references available — every other point the model must judge relative to them. Robots
    are listed in a fixed order (raph, donnie, then any others) so the prompt is reproducible, and a
    robot with no pose simply contributes no line.
    """
    colors = {"raph": "magenta circle", "donnie": "light-blue circle"}
    order = [n for n in ("raph", "donnie") if n in robot_world_xy]
    order += [n for n in robot_world_xy if n not in order]
    lines = []
    for name in order:
        gx, gy = _to_grid(*world_to_pixel(*robot_world_xy[name], camera=camera), w, h)
        marker = f" ({colors[name]})" if name in colors else ""
        lines.append(f'- "{name}"{marker} is at [{gx}, {gy}].')
    return "\n".join(lines)


# ── Normalized points -> world reference route (replaces astar_proj.build_reference) ──

def _points_to_reference(points: list, pose_xy, w: int, h: int,
                         camera: str) -> tuple[list, list, list, int]:
    """Convert the VLM's ordered normalized points to a world-frame reference route.

    ref = [robot pose] + [pixel_to_world(_from_grid(gx, gy)) for each valid point]. Returns
    (ref, pixels, dropped, n_clamped) where:
        pixels    the same points in PIXEL coordinates, so the debug figures plot exactly what was
                  planned instead of re-deriving the conversion (and silently plotting normalized
                  numbers as pixels, which would pile every marker into the top-left corner);
        dropped   entries that are not 2-number pairs — warned about, not fatal;
        n_clamped how many coordinates fell outside [0, GRID_MAX] and were pulled back to the edge.
                  Clamping rather than dropping keeps one bad number from silently shortening a route
                  (which would flatter the path-length comparison); the count is reported as a metric
                  of whether the model understood the coordinate space at all.
    """
    ref = [(float(pose_xy[0]), float(pose_xy[1]))]
    pixels: list[tuple[float, float]] = []
    dropped: list = []
    n_clamped = 0
    for p in points:
        try:
            gx, gy = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            dropped.append(p)
            continue
        cgx, cgy = min(max(gx, 0.0), GRID_MAX), min(max(gy, 0.0), GRID_MAX)
        if (cgx, cgy) != (gx, gy):
            n_clamped += 1
        u, v = _from_grid(cgx, cgy, w, h)
        x, y = pixel_to_world(u, v, camera=camera)
        pixels.append((u, v))
        ref.append((float(x), float(y)))
    return ref, pixels, dropped, n_clamped


# ── Reply schema (structured outputs) ────────────────────────────────────────────

def _waypoints_schema(robot_names) -> dict:
    """Strict JSON Schema for the reply, enforced by the Responses API via constrained decoding.

    Shape: {"waypoints": {<robot>: [[x, y], ...] for every robot}}.

    Without this the model is free to return something that LOOKS like JSON but is not — the observed
    failure was a ```json fence wrapping entries annotated with trailing comments:

        "raph": [
            [150, 950],  // Green box
        ]

    which json.loads rejects, aborting the run after CLIPSeg has already segmented and the image has
    already been paid for. additionalProperties:false plus a fixed robot roster also means the model
    cannot invent a third key or drop a robot. This is the same mechanism test_pipeline uses
    (prompt_gen.route_schema); the baseline simply never adopted it.

    Strict mode does not support minItems/maxItems, so a point is typed "array of integer" rather than
    "exactly two integers" — _points_to_reference still drops any entry that is not a 2-number pair.
    """
    point = {"type": "array", "items": {"type": "integer"}}
    return {
        "type": "object", "additionalProperties": False, "required": ["waypoints"],
        "properties": {
            "waypoints": {
                "type": "object", "additionalProperties": False,
                "required": list(robot_names),
                "properties": {name: {"type": "array", "items": point} for name in robot_names},
            },
        },
    }


# ── VLM call (single call; text + unmarked image; schema-enforced JSON) ──

def _call_regression_vlm(prompt: str, map_b64: str, anchors: str, model: str,
                         temperature: float, robot_names, out_dir: str | None = None
                         ) -> tuple[dict, dict]:
    """system instructions = REGRESSION_PROMPT (anchors filled in), user content = [operator prompt
    text, unmarked overlay image]. Reply shape is enforced by _waypoints_schema (structured outputs),
    so a fenced or comment-annotated reply cannot be produced in the first place.

    `map_b64` is the base64 PNG of the overlay, encoded by the caller so the same bytes go to both
    the API and debug_io.save_marks_overlay. `anchors` is the _anchor_lines block. The image is sent
    with image_url alone — no `detail` — matching exec.py and every other harness, so the condition
    differs from them only in the prompt and the overlay.

    Returns (routes, usage) where routes = {robot: [[x, y], ...]} NORMALIZED points and usage is
    token counts.
    """
    instructions = REGRESSION_PROMPT.format(gmax=GRID_MAX, ghalf=GRID_MAX // 2, anchors=anchors)

    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(instructions)
    print(f'\nOperator instruction: "{prompt}"')
    print("─" * 71)
    # This baseline sends the operator text as a separate user turn, so record it with the system
    # instructions to keep vlm_prompt.txt a complete picture of what the model was given.
    saved_prompt = instructions + f'\n\nOperator instruction: "{prompt}"'

    client = tp._openai_client()
    print(f"Structured output: enforcing schema (robots: {list(robot_names)})")
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
        text={"format": {"type": "json_schema", "name": "waypoint_plan",
                         "strict": True, "schema": _waypoints_schema(robot_names)}},
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
    # Stage 1: --mode alone, so the real parser below can carry CONCRETE per-mode defaults (which is
    # what --help prints) instead of None sentinels resolved after parsing.
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=("sim", "real"), default="sim",
                             help="sim = Gazebo overhead.png + poses.json; real = an AVL_* lab image "
                                  "chosen with --scene. Default: sim.")
    mode_args, _ = mode_parser.parse_known_args()
    mode = mode_args.mode

    parser = argparse.ArgumentParser(description=__doc__, parents=[mode_parser],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="Operator instruction for the VLM (required)")
    if mode == "real":
        # Registered ONLY in real mode, so `--mode sim --scene AVL_1` fails with argparse's
        # "unrecognized arguments: --scene" exactly the way the other harnesses do in sim mode.
        parser.add_argument("--scene", choices=list(tpr.SCENES), default=tpr.SCENE,
                            help=f"Image + hardcoded robot poses to run on (default: {tpr.SCENE}, set "
                                 "by the SCENE constant at the top of test_pipeline_real.py)")
    parser.add_argument("--data", default="test_data",
                        help="Dir containing the scene images (+ poses.json in sim mode) "
                             "(default: test_data)")
    parser.add_argument("--camera", default=_CAMERAS[mode], choices=["gazebo", "lab_test"],
                        help=f"Overhead camera calibration for pixel<->world (default: "
                             f"{_CAMERAS[mode]} in {mode} mode)")
    parser.add_argument("--out", default=_OUT_DIRS[mode],
                        help=f"Debug output directory (default: {_OUT_DIRS[mode]})")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for the planner call (default: gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="VLM sampling temperature; 0=deterministic (default: 0.0)")
    args = parser.parse_args()

    camera = args.camera

    # ── Load the scene's image + robot poses ──────────────────────────────
    # The ONLY mode-dependent stage: both branches leave img_bgr/img_rgb/pil_rgba/w/h plus
    # robot_world_xy and robot_poses_ned defined the same way, so everything below is shared.
    if mode == "sim":
        img_path = os.path.join(args.data, "overhead.png")
        poses_path = os.path.join(args.data, "poses.json")
        if not os.path.exists(img_path):
            sys.exit(f"Image not found: {img_path}\nRun save_overhead.py first.")
        if not os.path.exists(poses_path):
            sys.exit(f"Poses not found: {poses_path}\nRun save_overhead.py first.")
    else:
        scene = tpr.SCENES[args.scene]
        img_path = os.path.join(args.data, scene["image"])
        if not os.path.exists(img_path):
            sys.exit(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        sys.exit(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_rgba = PILImage.fromarray(img_rgb).convert("RGBA")
    # w and h come from the LOADED image and are threaded into every _to_grid/_from_grid call, so the
    # same prompt is correct for the 1936x1216 sim frame and the 1920x1200 lab frames alike. Never
    # replace these with a constant.
    h, w = img_bgr.shape[:2]

    if mode == "sim":
        print(f"Scene: sim ({os.path.basename(img_path)}) — loaded {w}×{h}")
        with open(poses_path) as f:
            poses_raw = json.load(f)
        robot_world_xy = {name: gazebo_to_world(p["x"], p["y"]) for name, p in poses_raw.items()}
        robot_poses_ned = {name: gazebo_to_ned(p["x"], p["y"]) for name, p in poses_raw.items()}
        print("Robot world poses: " +
              ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))
    else:
        print(f"Scene: {args.scene} ({scene['image']}) — loaded {w}×{h}")
        # Planning references use world coords directly; overlay markers use NED (map_gen converts
        # back to world internally), so world -> NED here makes the markers land where intended.
        robot_world_xy = dict(scene["poses"])
        robot_poses_ned = {name: world_to_ned(wx, wy) for name, (wx, wy) in robot_world_xy.items()}
        print("Robot world poses (hardcoded): " +
              ", ".join(f"{n}=({xy[0]:.3f}, {xy[1]:.3f})" for n, xy in robot_world_xy.items()))

    os.makedirs(args.out, exist_ok=True)

    # ── Segmentation + occupancy (same as test_pipeline) ──────────────────
    print("Running CLIPSeg segmentation…")
    pix_labels = segment_frame(img_rgb)
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
    # The anchors are computed from the SAME robot_world_xy the planner uses, so what the model is
    # told is exactly where the robots are — a wrong anchor would be worse than no anchor.
    anchors = _anchor_lines(robot_world_xy, w, h, camera)
    routes, usage = _call_regression_vlm(args.prompt, map_b64, anchors, args.model, args.temperature,
                                         list(robot_world_xy), out_dir=args.out)

    params = dict(ASTAR_PARAMS)

    # ── Goal separation: reserve the poses of robots that will NOT move ───
    # Same block as tp.main(), calling the same grid_planner_utils.separate_goal the live translator
    # node uses, so every condition in the study is scored under identical post-processing — without
    # it two robots sent to one target both terminate on the same cell. A robot holding position still
    # occupies its cell, so it is claimed BEFORE the loop: claiming inside would be too late if it
    # happened to be visited second, and reserving up front makes the outcome independent of `routes`
    # key order. "Will not move" covers an empty point list, a malformed route, and a robot with no
    # pose.
    claimed: list = []
    for name, points in routes.items():
        if isinstance(points, list) and points and name in robot_world_xy:
            continue                                  # moving; its goal is claimed after it plans
        pose = robot_world_xy.get(name)
        if pose is not None:
            claimed.append(pose)

    # ── Plan per robot through the VLM's coordinate picks (astar) ─────────
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    total_clamped = 0
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
            print(f"[{name}] no pose for this scene; skipping.", file=sys.stderr)
            continue

        pose_xy = robot_world_xy[name]
        ref, pixels, dropped, n_clamped = _points_to_reference(points, pose_xy, w, h, camera)
        for p in dropped:
            print(f"[{name}] malformed point '{p}'; skipping.")
        if n_clamped:
            total_clamped += n_clamped
            print(f"[{name}] {n_clamped} coordinate(s) outside 0-{GRID_MAX}; clamped to the image edge.")

        if len(ref) < 2:
            print(f"[{name}] fewer than 2 reference points after filtering; skipping.")
            continue

        # Displace the goal off any already-claimed goal BEFORE planning, so A* runs once and the
        # debug figures describe the executed route. Only the final reference point moves —
        # intermediate waypoints and the route itself are untouched. This baseline always plans with
        # A*, so tp.main()'s `planner_name == "astar"` guard collapses away here.
        if tp.MIN_GOAL_SEPARATION > 0 and claimed:
            adjusted = tp.separate_goal(ref[-1], claimed, infl, meta,
                                        min_separation=tp.MIN_GOAL_SEPARATION,
                                        max_shift=float(params["projection_radius"]))
            if adjusted is None:
                print(f"[{name}] goal is within {tp.MIN_GOAL_SEPARATION:.2f} m of another robot and "
                      "no free cell far enough away was found; keeping the original goal.")
            elif math.dist(adjusted, ref[-1]) > 1e-9:
                print(f"[{name}] goal moved {math.dist(adjusted, ref[-1]):.2f} m to clear another "
                      f"robot: ({ref[-1][0]:.2f}, {ref[-1][1]:.2f}) -> "
                      f"({adjusted[0]:.2f}, {adjusted[1]:.2f})")
                ref[-1] = adjusted

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
        # The EXECUTED endpoint, not the adjusted reference: plan() may project it further, and
        # what the next robot must avoid is where this one actually stops.
        claimed.append(world_path[-1])
        print(f"[{name}] {len(world_path)} waypoints: "
              f"start=({world_path[0][0]:.3f}, {world_path[0][1]:.3f})  "
              f"end=({world_path[-1][0]:.3f}, {world_path[-1][1]:.3f})  "
              f"path length={length:.1f} m")

        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        # Reuse save_vlm_selections' point style: ordinal labels + a pixel dict keyed by them. The
        # pixels come from _points_to_reference, NOT from the raw reply — the reply is normalized, so
        # plotting it directly would pile every marker into the top-left corner of the image.
        labels = [str(next_id + i) for i in range(len(pixels))]
        grid_px = {lbl: px for lbl, px in zip(labels, pixels)}
        next_id += len(pixels)
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
    print(f"Scene: {args.scene if mode == 'real' else 'sim'} ({os.path.basename(img_path)})")
    print(f"Baseline: unmarked image + normalized 0-{GRID_MAX} coordinates + astar "
          "(no classifier, no CoT)")
    print(f"Tokens: {usage['total']} total "
          f"({usage['input']} input + {usage['output']} output)")
    # How often the model answered outside the stated coordinate space — the direct measure of
    # whether it understood the grid, and the metric to compare against the old pixel prompt.
    print(f"Out-of-range picks (clamped): {total_clamped}")
    tp._print_path_lengths(path_lengths)
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
