#!/usr/bin/env python3
"""test_convoi_prompting.py — CoNVOI-style numbered free-space marking (single VLM call, no CoT).

The fourth marking condition in the study. Where test_pipeline marks EVERY grid cell (blue dot if
free, red X if blocked, alphanumeric label on both) and test_battleship_baseline lays a labeled grid
over the whole image, this condition marks the image ONLY where the robot can actually drive, with
plain sequential numbers:

  * candidate points are the same 14x8 cell centres every other condition uses;
  * a candidate that is not FREE in the INFLATED planning grid gets nothing drawn — a hole in the
    pattern, no red marker, no label;
  * a candidate that IS free gets the next integer, walking the grid in raster order (top-left,
    across to the right, then the left end of the next row down).

So the numbering runs 1..N with no gaps while the spatial layout has gaps. A number's VALUE therefore
carries no information about where it is — that is the point of the condition, and it is why the
prompt describes the marks without ever describing how the numbers are laid out. The occupancy map
does the obstacle reasoning; the VLM only does the semantic selection ("which of these drivable spots
gets me to the chair").

The VLM returns, for each robot, an ordered list of those numbers. They are looked up to pixels and
fed through astar_proj.build_reference — the SAME call the set-of-marks harnesses make for their
maneuver / nav2point label sequences — then planned with astar_proj. No classifier, no
chain-of-thought, and no adjacency constraint on the sequence (unlike the battleship baseline).

Reuses the same segmentation/occupancy, goal separation, planner (node_Path_Translator.astar_proj) and
debug writers as test_pipeline / test_regression_only (imported), so the only differences are the
overlay, the custom single-call prompt, and the label scheme.

    --mode sim   the Gazebo overhead image  (test_data/overhead.png + poses.json, gazebo calibration)
    --mode real  an AVL_* lab image         (--scene, hardcoded poses, lab_test calibration)

Prerequisites:
    colcon build --symlink-install --packages-select coplan_vlm
    source install/setup.bash

Usage (from workspace root):
    python3 src/CoPlanVLM/scripts/test_convoi_prompting.py --mode sim \\
        --prompt "Send raph to the chair and donnie to the box"

    python3 src/CoPlanVLM/scripts/test_convoi_prompting.py --mode real --scene AVL_3 \\
        --prompt "Send raph to the chair and donnie to the box"

Output (written to --out, default debug/convoi_{sim,real}/):
    marks_overlay.png     — the numbered free-space image sent to the VLM
    robot_paths_waypoints.png — both robots' planned (A*) trajectories + their chosen numbers
    raw_overhead.png, inflation_overlay.png — identical for every robot, so written once here
  and per robot in <out>/<robot>/: vlm_selections, route_centroids, route_planned.
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
from PIL import Image as PILImage, ImageDraw, ImageFont

import debug_io
from coord_transform import gazebo_to_world, gazebo_to_ned, ned_to_world, world_to_ned
from obs_seg.segmenter import segment_frame
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               render_inflation_overlay, RESOLUTION as _RESOLUTION)
from node_Path_Translator import astar_proj
from node_Path_Translator.astar_proj import PARAMS as ASTAR_PARAMS
from node_Executive_API import map_gen
from node_Executive_API.map_gen import _N_COLS, _N_ROWS

# Reuse test_pipeline's helpers (OpenAI client, JSON parse, goal separation, evaluation printer) to
# avoid drift, and test_pipeline_real's SCENES so real-mode images/poses have exactly one definition.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pipeline as tp         # noqa: E402
import test_pipeline_real as tpr   # noqa: E402

# Per-mode defaults, same shape as test_pipeline_no_markers._OUT_DIRS. sim and real write to separate
# directories so one mode's results can never clobber the other's.
_OUT_DIRS = {"sim": "debug/convoi_sim", "real": "debug/convoi_real"}
_CAMERAS  = {"sim": "gazebo", "real": "lab_test"}

# ── Number mark appearance ─────────────────────────────────────────────────────
# Yellow, RGB. Drawn as bare text centred on the point — no dot, no X, no backing box, unlike every
# other renderer in map_gen (which gray-backs its labels). The absence of a marker glyph is part of
# the condition: the number IS the mark.
# A pale yellow rather than a saturated one: raising the blue channel off 0 lifts the glyphs away from
# the warm yellow-brown of the lab carpet and the yellow floor tape (which a saturated yellow blends
# into), while staying unmistakably yellow — the prompt tells the model to look for yellow numbers.
_NUM_COLOR = (255, 243, 150)

# Point size of the numbers. THIS is the knob to turn to make the marks bigger or smaller.
# Deliberately a local font rather than map_gen._LABEL_FONT (28 pt): that one is shared by the
# battleship grid labels, the set-of-marks labels and the robot name labels, so raising it there
# would silently change what the OTHER conditions in the study look like. Larger than 28 because
# these numbers carry no dot to draw the eye — and because the model sees the image rescaled to a
# 768 px short side, which shrinks every glyph by roughly 0.63x on the way in.
_NUM_SIZE = 38
# BOLD (DejaVuSans-Bold, not the regular face map_gen loads): with no dot or backing box behind them,
# the numbers rely entirely on stroke weight to stay readable over carpet speckle and shadow, and the
# heavier stroke survives the downscale to the model's 768 px short side far better than a thin one.
# Falls back to the regular face, then to PIL's bitmap default, so a machine without DejaVu still runs.
try:
    _NUM_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   size=_NUM_SIZE)
except OSError:
    try:
        _NUM_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                       size=_NUM_SIZE)
    except OSError:
        _NUM_FONT = ImageFont.load_default()


# ── Prompt (our own; no chain-of-thought) ───────────────────────────────────────
# Deliberately silent on how the numbers are laid out. They are assigned in raster order over the
# free cells only, so consecutive numbers are usually — but NOT always — neighbours, and the gaps
# left by obstacles mean a number's value cannot be turned into a position. Telling the model any
# rule here would be telling it a rule that is wrong wherever an obstacle intervenes.
CONVOI_PROMPT = """\
You are a path-planning agent controlling two TurtleBot4 robots named "raph" and "donnie" that share one
workspace. You are given an overhead camera image of the environment. There is a small hollow circle drawn on each robot with its name next to it:
- "raph"   — the round black TurtleBot marked with a magenta circle labeled "raph".
- "donnie" — the round black TurtleBot marked with a light-blue circle labeled "donnie".

All directions are from the IMAGE's point of view, NOT a robot's or person's perspective: the left/right
of an object means its left/right side as it appears in the image, and up/down mean toward the top/bottom
of the image. Cardinal directions are fixed to the image: NORTH/UP is toward the top, SOUTH/DOWN toward
the bottom, WEST/LEFT toward the left edge, EAST/RIGHT toward the right edge.

THE NUMBERS ON THE IMAGE:
The image is marked with yellow numbers. Each number sits at a location a robot can safely drive to —
every numbered spot is on clear, open floor. Anywhere with NO number is either an obstacle or too close
to one to be safe, so the empty regions of the image show you where the robots cannot go.
The numbers are NAMES, NOT COORDINATES: a number's value tells you nothing about where it is, and two
numbers that are close in value are not necessarily close together in the image. Judge every number
only by WHERE YOU SEE IT in the image.

YOUR TASK:
For each robot, choose an ordered list of NUMBERS the robot must drive to in order to accomplish the
operator's instruction, in travel order. Use ONLY numbers that actually appear in the image — never
invent a number you cannot see. Do NOT include a number for the robot's own current position — the
route already starts there; list only where it must GO.
Give as FEW numbers as the task needs: if the robot can reach its destination on a clear straight run,
ONE number — the one nearest the destination — is the correct answer. Add intermediate numbers only
where they are needed to steer around something, remembering that the robot drives in a straight line
between consecutive numbers, so no straight segment may cut through an object, furniture, a wall or the
other robot. You MUST include both robots; if a robot has no task, give it an empty list.
For tasks requiring the robots to survey or patrol a region, please generate routes that completely cover
the designated area. If no robot name or number of robots is mentioned, please try to divide the work 
evenly and efficiently between robots.


Respond with EXACTLY one JSON object and nothing else:
{"waypoints": {"raph": [<number>, ...], "donnie": [<number>, ...]}}

Example — for a DIFFERENT image than this one, where the chair is next to the number 44 and there is a
clear run to it from raph, while donnie must get to a box next to the number 61 on the far side of a
table, the instruction "Send raph to the chair and donnie to the box" gives:
{"waypoints": {"raph": [44], "donnie": [23, 39, 61]}}
(raph gets one number, the destination, because no detour is needed. donnie gets three: 23 and 39 are
numbers on open floor that carry it around the WEST end of the table, then 61 is the destination. Note
that donnie's numbers are not consecutive and not in any particular numeric order — they were chosen by
where they appear in the image, which is the only thing that matters.)"""


# ── Numbered free points (the marking scheme) ────────────────────────────────────

def _numbered_free_points(w: int, h: int, occ_grid, occ_meta,
                          camera: str) -> tuple[dict[str, tuple[float, float]], int]:
    """Assign 1..N to the FREE 14x8 cell centres in raster order. Returns ({label: (u, v)}, n_blocked).

    Candidate geometry matches render_battleship_map / grid_cell_centers.csv — the image divided into
    _N_COLS x _N_ROWS equal cells, centre at ((col+0.5)*cell_w, (row+0.5)*cell_h) — but derived from
    the LOADED image's w/h rather than read from the CSV, which is hardcoded for the 1936x1216 sim
    frame and would be a few pixels off on the 1920x1200 lab frames.

    Free/blocked comes from map_gen._cell_is_free against the INFLATED planning grid: the identical
    classification behind the red X marks and blocked_cell_labels, so a spot that carries a number can
    never be a spot the planner then refuses to route through.

    Raster order is row-major: the top row left-to-right, then the next row down, and so on. Blocked
    candidates are SKIPPED without consuming a number, so the returned labels are exactly "1".."N"
    with no gaps while the drawn pattern has holes wherever an obstacle is. That is what makes the
    number a pure name: its value cannot be inverted back to a position.

    Keys are strings because astar_proj.build_reference does a `label in grid_px` lookup and the
    debug_io writers key their pixel dicts the same way.
    """
    cw, ch = w / _N_COLS, h / _N_ROWS
    points: dict[str, tuple[float, float]] = {}
    n_blocked = 0
    n = 0
    for row in range(_N_ROWS):
        for col in range(_N_COLS):
            u, v = (col + 0.5) * cw, (row + 0.5) * ch
            if not map_gen._cell_is_free(u, v, occ_grid, occ_meta, camera):
                n_blocked += 1
                continue
            n += 1
            points[str(n)] = (u, v)
    return points, n_blocked


def _render_numbered_map(pil_rgba: PILImage.Image, grid_px: dict, robot_poses_ned: dict,
                         camera: str) -> PILImage.Image:
    """Draw each number centred on its point in yellow, plus the robot circles + name labels.

    No dot, no X and no backing box: the number itself is the only thing drawn at the point (PIL's
    anchor="mm" centres the glyphs on the coordinate rather than hanging them off its top-left). The
    robot markers stay because the VLM still has to tell raph from donnie; their name labels are given
    the numbers' bounding boxes as `occupied` so _draw_robot_markers' collision search moves a name
    off any number it would otherwise cover.
    """
    result = pil_rgba.copy()
    draw = ImageDraw.Draw(result)
    occupied: list[tuple] = []
    for label, (u, v) in grid_px.items():
        draw.text((u, v), label, fill=_NUM_COLOR, font=_NUM_FONT, anchor="mm")
        occupied.append(draw.textbbox((u, v), label, font=_NUM_FONT, anchor="mm"))
    # Robot names keep map_gen's shared _LABEL_FONT so the identification channel looks identical to
    # every other condition; only the numbers scale with _NUM_SIZE.
    map_gen._draw_robot_markers(draw, robot_poses_ned, camera, map_gen._LABEL_FONT,
                                occupied=occupied)
    return result


# ── Reply schema (structured outputs) ────────────────────────────────────────────

def _waypoints_schema(robot_names) -> dict:
    """Strict JSON Schema for the reply, enforced by the Responses API via constrained decoding.

    Shape: {"waypoints": {<robot>: [<int>, ...] for every robot}} — a flat list of integers, since
    this condition's labels are single numbers rather than the regression baseline's [x, y] pairs.

    Without this the model is free to return something that LOOKS like JSON but is not (the observed
    failure elsewhere in the study was a ```json fence around entries carrying // comments), which
    aborts the run after CLIPSeg has already segmented and the image has already been paid for.
    additionalProperties:false plus a fixed robot roster also means the model cannot invent a third
    key or drop a robot.

    Deliberately NOT an enum of the valid numbers. prompt_gen.route_schema supports constraining
    waypoints to free-cell labels, but the study leaves that off, and turning it on here would make
    "Invalid picks (no such number)" identically zero — destroying the one metric that measures
    whether the model actually read the marks.
    """
    return {
        "type": "object", "additionalProperties": False, "required": ["waypoints"],
        "properties": {
            "waypoints": {
                "type": "object", "additionalProperties": False,
                "required": list(robot_names),
                "properties": {name: {"type": "array", "items": {"type": "integer"}}
                               for name in robot_names},
            },
        },
    }


# ── VLM call (single call; text + numbered image; schema-enforced JSON) ──

def _call_convoi_vlm(prompt: str, map_b64: str, model: str, temperature: float,
                     robot_names, out_dir: str | None = None) -> tuple[dict, dict]:
    """system instructions = CONVOI_PROMPT, user content = [operator prompt text, numbered overlay
    image]. Reply shape is enforced by _waypoints_schema (structured outputs), so a fenced or
    comment-annotated reply cannot be produced in the first place.

    `map_b64` is the base64 PNG of the overlay, encoded by the caller so the same bytes go to both
    the API and debug_io.save_marks_overlay. The image is sent with image_url alone — no `detail` —
    matching exec.py and every other harness, so the condition differs from them only in the prompt
    and the overlay.

    Returns (routes, usage) where routes = {robot: [<number>, ...]} and usage is token counts.
    """
    print("─" * 16 + " VLM full prompt (system instructions) " + "─" * 16)
    print(CONVOI_PROMPT)
    print(f'\nOperator instruction: "{prompt}"')
    print("─" * 71)
    # This baseline sends the operator text as a separate user turn, so record it with the system
    # instructions to keep vlm_prompt.txt a complete picture of what the model was given.
    saved_prompt = CONVOI_PROMPT + f'\n\nOperator instruction: "{prompt}"'

    client = tp._openai_client()
    print(f"Structured output: enforcing schema (robots: {list(robot_names)})")
    print(f"Planning with {model}…")
    response = client.responses.create(
        model=model, temperature=temperature, instructions=CONVOI_PROMPT,
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
    for name, nums in routes.items():
        n = len(nums) if isinstance(nums, list) else "?"
        seq = ", ".join(map(str, nums)) if nums else "(empty — holds position)"
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
    # w and h come from the LOADED image and are threaded into _numbered_free_points, so the marks
    # land on cell centres for the 1936x1216 sim frame and the 1920x1200 lab frames alike. Never
    # replace these with a constant or with grid_cell_centers.csv.
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

    # ── Numbered free-space overlay ───────────────────────────────────────
    # Built AFTER segmentation (unlike the other baselines' overlays), because which points get a
    # number is decided by the inflated grid this stage produces.
    grid_px, n_blocked = _numbered_free_points(w, h, infl, meta, camera)
    n_total = _N_COLS * _N_ROWS
    print(f"Numbered {len(grid_px)} free points (1-{len(grid_px)}); "
          f"{n_blocked} of {n_total} candidates blocked and left unmarked.")
    if not grid_px:
        sys.exit("No free grid points to mark — nothing the VLM could select. Check segmentation.")
    overlay = _render_numbered_map(pil_rgba, grid_px, robot_poses_ned, camera)
    # Encode once: the same bytes go to the API and to marks_overlay.png.
    _buf = io.BytesIO()
    overlay.save(_buf, format="PNG")
    map_b64 = base64.b64encode(_buf.getvalue()).decode("utf-8")
    debug_io.save_marks_overlay(args.out, map_b64)

    # ── Single VLM call ───────────────────────────────────────────────────
    routes, usage = _call_convoi_vlm(args.prompt, map_b64, args.model, args.temperature,
                                     list(robot_world_xy), out_dir=args.out)

    params = dict(ASTAR_PARAMS)

    # ── Goal separation: reserve the poses of robots that will NOT move ───
    # Same block as tp.main(), calling the same grid_planner_utils.separate_goal the live translator
    # node uses, so every condition in the study is scored under identical post-processing — without
    # it two robots sent to one target both terminate on the same cell. A robot holding position still
    # occupies its cell, so it is claimed BEFORE the loop: claiming inside would be too late if it
    # happened to be visited second, and reserving up front makes the outcome independent of `routes`
    # key order. "Will not move" covers an empty number list, a malformed route, and a robot with no
    # pose.
    claimed: list = []
    for name, nums in routes.items():
        if isinstance(nums, list) and nums and name in robot_world_xy:
            continue                                  # moving; its goal is claimed after it plans
        pose = robot_world_xy.get(name)
        if pose is not None:
            claimed.append(pose)

    # ── Plan per robot through the VLM's numbered picks (astar) ───────────
    path_lengths: dict[str, float] = {}
    world_paths: dict[str, list] = {}
    sel_routes: dict[str, list] = {}   # per robot, the labels that actually resolved to a point
    total_invalid = 0
    for name, nums in routes.items():
        if not isinstance(nums, list):
            print(f"[{name}] route is not a list; skipping.", file=sys.stderr)
            continue
        if name not in robot_world_xy:
            print(f"[{name}] no pose for this scene; skipping.", file=sys.stderr)
            continue

        pose_xy = robot_world_xy[name]
        # The schema types picks as integers while grid_px is keyed by string, so normalise here.
        # str(int(n)) also collapses a stray 12.0 onto "12" rather than losing the pick.
        labels = []
        for n in nums:
            try:
                labels.append(str(int(n)))
            except (TypeError, ValueError):
                labels.append(str(n))
        ref, unknown = astar_proj.build_reference(labels, pose_xy, grid_px, camera=camera)
        for lbl in unknown:
            print(f"[{name}] no such number '{lbl}' on the image; skipping.")
        total_invalid += len(unknown)
        labels = [lbl for lbl in labels if lbl not in unknown]

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

        sel_routes[name] = labels

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
        # "points" style: the chosen numbers plotted at their own pixel positions. grid_px is the
        # WHOLE numbering (labels are globally unique), so one dict serves every robot.
        debug_io.save_vlm_selections(out_dir, img_bgr, "points", labels, grid_px)
        dbg["start_world"] = pose_xy
        astar_proj.save_debug(out_dir, img_bgr, cleared, meta, ctx, dbg, params, camera=camera,
                              overlay=infl_overlay)
        print(f"[{name}] Debug images -> {out_dir}/")

    # ── Combined trajectory plot ──────────────────────────────────────────
    if world_paths:
        wp_png = os.path.join(args.out, "robot_paths_waypoints.png")
        debug_io.save_paths_with_waypoints(wp_png, img_bgr, world_paths, sel_routes, grid_px, camera)
        print(f"Combined robot paths + VLM numbers -> {wp_png}")

    # ── Evaluation summary ────────────────────────────────────────────────
    print("─" * 27 + " Evaluation " + "─" * 27)
    print(f"Scene: {args.scene if mode == 'real' else 'sim'} ({os.path.basename(img_path)})")
    print("Baseline: convoi-style numbered free space + astar (no classifier, no CoT)")
    print(f"Tokens: {usage['total']} total "
          f"({usage['input']} input + {usage['output']} output)")
    print(f"Numbered free points: {len(grid_px)} ({n_blocked} of {n_total} blocked, not marked)")
    # How often the model named a number that is not on the image — the direct measure of whether it
    # read the marks at all, and the counterpart to the regression baseline's clamped-pick count.
    print(f"Invalid picks (no such number): {total_invalid}")
    tp._print_path_lengths(path_lengths)
    print("─" * 66)

    print("Done.")


if __name__ == "__main__":
    main()
