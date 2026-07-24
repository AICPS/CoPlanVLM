"""In-repo prompt construction for the Executive (path-planner) node.

Each operator instruction goes through two LLM calls:

1. CLASSIFIER (CLASSIFIER_SYSTEM_PROMPT) — text-only; tags the instruction with a controller
   ("nav2point", "maneuver", or "coverage").

2. PLANNER (generate_prompt(...)) — vision call; returns the assembled system prompt AND the overlay
   image (rendered here) so the two can never disagree. The model returns a per-robot route as JSON.

generate_prompt composes the planner prompt from independent, swappable pieces, always in a fixed order:
  1. COMMON_PREAMBLE                       — robot descriptions, image context, global rules (shared)
  2. MAP_OVERLAY_DESCRIPTION[overlay]      — how THIS overlay marks the map
  (image)                                  — the matching overlay image, rendered + returned as base64
  4. CONTROLLER_EXPLAIN[controller]        — what to select, output schema, example + the operator line
  5. COT_BLOCKS[controller]                — optional per-controller chain-of-thought scaffold

A single ``map_overlay_type`` selects BOTH the description text (piece 2) and the map_gen renderer that
draws the image, so the prompt and the image stay in lock-step.
"""

from __future__ import annotations

from node_Executive_API.map_gen import (
    render_grid_points_map, render_obstacle_marked_map, render_battleship_map,
    blocked_cell_labels, _to_b64)


# ── Axis vocabularies (the only valid controller / overlay names) ──────────────
CONTROLLERS       = ("nav2point", "maneuver", "coverage")
MAP_OVERLAY_TYPES = ("points", "marked_obs", "battleship")


# ── 1. Shared preamble (every prompt) ──────────────────────────────────────────

COMMON_PREAMBLE = """\
You are a path-planning agent for two TurtleBot4 robots sharing one workspace.
You will be shown an overhead camera image of the environment with a map overlay.

The two robots — identify each by its appearance in the image:
- "raph"   — the round black TurtleBot labeled "raph" with magenta in the image.
- "donnie" — the round black TurtleBot labeled "donnie" in light blue in the image.

Locate each robot in the image before planning. General rules that apply to all tasks:
- You MUST include BOTH robots in every response, even if the instruction only mentions one.
  A robot with no task holds its position: return an empty list for each robot that should not move.
"""


# ── 2. Map-overlay descriptions (how each overlay marks the map) ───────────────
# Each entry describes the marks in the rendered image and the overlay-specific rules for choosing
# labels, so the controller pieces below can stay overlay-neutral.

_POINTS_OVERLAY = """\
MAP OVERLAY — Set of Marks:
The image shows blue dots at a regular grid of candidate locations, each labeled with an alphanumeric
identifier. The grid has exactly 14 columns (A–N, left to right) and 8 rows (1–8, top to bottom),
giving points like "A1", "H4", "N8". Columns never go past N and rows never go past 8.
A dot is placed at every grid point regardless of what lies beneath it, so a dot may fall on an
obstacle — use the image to judge which points are on clear, reachable floor and choose destinations
only from those.
Choose ONLY from labels that exist on this grid; NEVER invent or select a point outside it (no column
beyond N, no row beyond 8, e.g. "G9" or "P4" do not exist)."""

_MARKED_OBS_OVERLAY = """\
MAP OVERLAY — Set of Marks with Impassable Points:
The image shows a labeled mark at every point of a regular grid with exactly 14 columns (A–N, left to
right) and 8 rows (1–8, top to bottom), giving points like "A1", "H4", "N8". Columns never go past N
and rows never go past 8. Each mark has one of two forms:
- a BLUE DOT marks a FREE point the robot CAN navigate to;
- a RED X marks a point that is NOT passable — the robot CANNOT stand on or travel through a red X
  point (it is blocked/occupied).
A red X ONLY means that point is impassable because an object has been percieved there; it does NOT tell you what is there.
It is likely although not guaranteed that objects refered to in the user instruction will be marked by a red X.
Identify the objects and targets named in the instruction from the image itself, not from the red X marks.
Every mark, blue or red, is labeled with its alphanumeric identifier.
Choose robot destinations and routes ONLY from blue-dot points.
NEVER invent or select a point outside the grid (no column beyond N, no row beyond 8, e.g. "G9" or
"P4" do not exist)."""

_BATTLESHIP_OVERLAY = """\
MAP OVERLAY — Battleship Grid:
The image shows a grid overlaid on the environment with exactly 14 columns (A–N, left to right) and
8 rows (1–8, top to bottom), giving cells such as "A1", "H4", "N8". Columns never go past N and rows
never go past 8.
NEVER invent or select a cell outside the grid (no column beyond N, no row beyond 8, e.g. "G9" or
"P4" do not exist)."""

MAP_OVERLAY_DESCRIPTION = {
    "points":     _POINTS_OVERLAY,
    "marked_obs": _MARKED_OBS_OVERLAY,
    "battleship": _BATTLESHIP_OVERLAY,
}

# map_overlay_type -> the map_gen renderer that draws the matching overlay image. Paired with the
# descriptions above so a single map_overlay_type drives both text and image.
_OVERLAY_RENDERERS = {
    "points":     render_grid_points_map,
    "marked_obs": render_obstacle_marked_map,
    "battleship": render_battleship_map,
}


# ── 4. Controller explanations (what to select + output schema + example) ──────
# Overlay-neutral: they reference "the labeled locations shown in the overlay"; the overlay piece above
# supplies the mark-specific rules. NOT f-strings, so the JSON braces are literal.

_NAV2POINT_EXPLAIN = """\
YOUR TASK:
Select a single goal location or a sequence of goal locations for each robot to accomplish the task specified in the instruction, 
choosing from the labeled locations shown in the overlay. The destination(s) matter; the exact path taken does
not, as an A* path planner will route each robot to the goal locations. Output only the goal point(s)
for each robot in order — do not list intermediate steps. A robot with no task should have an empty list.

OUTPUT FORMAT (exactly one JSON object):
{"waypoints": {"raph": ["<goal_point>", ...], "donnie": ["<goal_point>", ...]}}

Example — if the instruction is "Send raph to the chair and then the person. Have donnie stay in place.", a valid response is:
{"waypoints": {"raph": ["C4", "H8"], "donnie": []}}"""

_MANEUVER_EXPLAIN = """\
YOUR TASK:
Plan a specific route for each robot as an ordered list of waypoints, chosen from the labeled locations
shown in the overlay. The route taken matters — follow the constraints in the instruction (loops,
avoidance corridors, formations, etc.). Step between locations that are adjacent/close together to draw
a coherent path.

OUTPUT FORMAT (exactly one JSON object):
{"waypoints": {"raph": ["<point>", ...], "donnie": ["<point>", ...]}}

Example — if the instruction is "Have the robots do a loop around the chair", a valid response is:
{"waypoints": {"raph": ["A7", "B6", "C5", "D4", "D3", "C3", "B3", "B4", "B5", "B6", "A7"],
              "donnie": ["D3", "D4", "D5", "C5", "B5", "B4", "B3", "C3", "D2"]}}"""

_COVERAGE_EXPLAIN = """\
YOUR TASK:
Select the locations each robot should visit to cover or patrol the area described in the instruction,
choosing from the labeled locations shown in the overlay. You choose which to cover. If no robot is
specified, split coverage roughly equally between raph and donnie. If only one robot is specified,
return an empty list for the other. The order does not matter — a lower-level controller will find the
most efficient patrol route.

OUTPUT FORMAT (exactly one JSON object):
{"regions": {"raph": ["<cell>", ...], "donnie": ["<cell>", ...]}}

Example — if the instruction is "Have the robots patrol the left side of the map", a valid response is:
{"regions": {"raph": ["A5", "A6", "A7", "A8", "B5", "B6", "B7", "B8", "C5", "C6", "C7", "C8", "D5", "D6", "D7", "D8", "E5", "E6", "E7", "E8"],
             "donnie": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4", "E1", "E2", "E3", "E4"]}}"""

CONTROLLER_EXPLAIN = {
    "nav2point": _NAV2POINT_EXPLAIN,
    "maneuver":  _MANEUVER_EXPLAIN,
    "coverage":  _COVERAGE_EXPLAIN,
}


# ── 5. Per-controller chain-of-thought scaffolds ───────────────────────────────
# Appended when cot=True (selected by controller via COT_BLOCKS). The response schema carries a leading
# "reasoning" string field generated BEFORE the route keys, so these only guide WHAT to reason about —
# the structure (reasoning first, then the locked route object) is enforced by the structured-output
# schema, not by this text.
#
# coverage is a placeholder copy of the original generic block for now, kept as a separate constant so
# it can be reworked into its own custom reasoning later without affecting the others.

_NAV2POINT_COT = """\
CoT REASONING STEPS — Before choosing routes, work through these in the "reasoning" field, in order:
1) Identify the objects/locations named in the operator's instruction and localize each with its grid
   label(s) (e.g. "chair: C4", "yellow box: H8").
2) Based off the instruction, list the alpha numeric points (e.g. B4) that the robots should travel to.
3) For each desired point, check whether its cell is blocked (Check the the list of blocked points in the MAP_OVERLAY)): 
   if it is blocked, list the direction the robot should be relative to the obstacle, 
   then the nearest FREE (blue-dot) point the robot can stand on in that direction;
   if the desired point is already free, use it as-is. Also note which robot (raph or donnie) is currently nearest to
   that cell (e.g. "green box E8 is blocked -> travel to D6, nearest robot donnie").
"""

_COVERAGE_COT = """\
REASONING — Before deciding the routes, work through the task in the "reasoning" field, in this order:
1. RELEVANT OBJECTS: Identify the objects/regions named in the operator's instruction. 
   List where they are located on the map (e.g. "chair: C4", "yellow box: H8"). 
   List where each robot is on the map initially (e.g. "raph: A7", "donnie: D2").
2. SELECT_REGIONS: List all regions that the robots should cover/visit.
3. ROBOT ASSIGNMENT: Divide the regions evenly between the two robots.
Only after this reasoning, fill in the route object with your final choice."""

# Inserted into _MANEUVER_COT (step 1b) only for the marked_obs overlay, which is the only overlay that
# draws red X marks. Omitted otherwise so the model is not asked to inventory marks that do not exist.
_RED_MARKER_SUBSTEP = """\
   c) Label each red marker: for every red X in the image, note its label and the object at or near it
      (it may sit beside, not directly under, the mark); write "unknown" if unclear.
"""

# ── Previous maneuver CoT (dense adjacent-cell path). Commented out, kept for reference. ──
# _MANEUVER_COT = """\
# CoT REASONING STEPS — Before choosing routes, work through these in the "reasoning" field, in order:
# 1) Identify and locate the robots and key objects in the scene.
#    a) Robots: give raph's and donnie's current position as the labeled grid point nearest each robot
#       in the image (e.g. "raph: A7", "donnie: D2").
#    b) Task features: list everything the instruction requires you to perceive to carry out the task —
#       not only goal objects, but also boundaries or lines to avoid/not cross (e.g. caution tape),
#       regions to stay within or out of, and landmarks to go around. Give the grid label(s) each one
#       occupies or spans (e.g. "yellow box: H8", "caution tape: E4-E7", "chair to loop: C4").
# %RED_MARKER_STEP%
# 2) Split the instruction into one subtask per robot, naming which robot (raph or donnie) performs each.
#    Keep every spatial constraint and landmark named in the instruction — words like behind / around /
#    left of / between / via and the object they refer to. Do not shorten or drop these. If only one
#    robot is needed, write "stay" for the other.
# 3) Describe in text the specific route each robot must follow to get to the goal.
#    List any key intermediate locations (e.g. [by chair H4]) each robot should pass through.
# 4) For each robot please list the final location (e.g. F3) each robot should finish at.
# 5) List any locations (e.g. D5) each robot should avoid.
# 6) Build each robot's final path: an ordered list of grid labels from its current position (1a),
#    through the intermediate locations (4), to its final location (3), stepping between adjacent/nearby
#    labels and keeping off the avoid locations (5). Put this list into the route object.

_MANEUVER_COT = """\
CoT REASONING STEPS — Before choosing routes, work through these in the "reasoning" field, in order.
A path planner will connect your consecutive waypoints with a collision-free path, so give only the KEY
waypoints that define the maneuver's shape — do NOT list every adjacent cell or hand-trace around
obstacles (skipping cells is expected).

1) Identify and locate the robots and key objects in the scene.
   a) Robots: give raph's and donnie's current position as the labeled grid point nearest each robot
      in the image (e.g. "raph: A7", "donnie: D2").
   b) Task features: list everything the instruction requires you to perceive to carry out the task —
      not only goal objects, but also boundaries or lines to avoid/not cross (e.g. caution tape),
      regions to stay within or out of, and landmarks to go around. Give the grid label(s) each one
      occupies or spans (e.g. "yellow box: H8", "caution tape: E4-E7", "chair to loop: C4").
%RED_MARKER_STEP%
2) Split the instruction into one subtask per robot, naming which robot (raph or donnie) performs each.
   Keep every spatial constraint and landmark named in the instruction — words like behind / around /
   left of / between / via and the object they refer to. Do not shorten or drop these. If only one
   robot is needed, write "stay" for the other.
3) Ground the spatial words in the grid (columns A->N run left-to-right, rows 1->8 top-to-bottom).
   Turn each constraint or goal location into concrete cells: the "far side" of an object from a robot is the side
   across the object from that robot; "between X and Y" is the cells whose column/row fall between
   theirs; to go "around" or "behind" an object, name the free corridor cells along the required side
   (e.g. an object spanning F3-H3 -> pass above it via F2, G2, H2)
   (e.g. an object at B5 -> approach from right via C5, B5).
5) For each robot, break its subtask into ordered legs, each with a short purpose, giving only the KEY
   waypoint(s) that realize it (include its start from 1a and its final goal cell). Do NOT choose a leg
   waypoint that is on an avoid location from step 4. For a loop or circuit around an object, give
   waypoints on several DIFFERENT sides of it (not just the far side), so the route encircles the object
   instead of going out and doubling back the same way:
      leg 1 <purpose>: <label(s)>   ...   leg N <purpose>: <label(s)>
   Example — "raph: loop around the chair at C4 and return to its start at A7":
      leg 1 approach the chair: C5 ; leg 2 circle it via each side: D4, C3, B4, C5 ;
      leg 3 return to start: A7
6) Assemble each robot's route as a SHORT ordered list of the key waypoints from step 5 (start -> legs
   -> final goal). Keep it sparse: consecutive waypoints may be far apart and the planner fills the gaps
   collision-free. Never place a waypoint on an avoid location from step 4.
7) Verify and fix before writing the JSON: the route starts at the robot, follows the legs in order,
   ends at the final goal, avoids every step-4 location, and truly satisfies the spatial constraint
   (correct side of the landmark). Then put the route into the route object using the prescribed
   output structure."""

# controller -> its chain-of-thought block. Total over CONTROLLERS (validated in generate_prompt).
COT_BLOCKS = {
    "nav2point": _NAV2POINT_COT,   # TODO: custom block (copy of generic for now)
    "maneuver":  _MANEUVER_COT,
    "coverage":  _COVERAGE_COT,    # TODO: custom block (copy of generic for now)
}


def _operator_line(instruction: str) -> str:
    return f'The operator\'s instruction is: "{instruction}"'


# ── Constructor ────────────────────────────────────────────────────────────────

def generate_prompt(instruction, controller, map_overlay_type, *,
                    pil_img, occ_grid, occ_meta, camera, robot_poses=None, cot=False):
    """Assemble the planner system prompt AND render the matching overlay image.

    Returns ``(prompt_text, image_b64)``. ``controller`` selects the task semantics + output schema
    (piece 4); ``map_overlay_type`` selects BOTH the overlay description (piece 2) and the map_gen
    renderer that draws the image, so text and image always agree. ``cot=True`` appends the
    controller's chain-of-thought scaffold (piece 5); for maneuver the red-marker sub-step is included
    only when ``map_overlay_type == "marked_obs"``.

    Render inputs: ``pil_img`` (RGBA overhead frame), the pre-inflated planning grid ``occ_grid`` +
    ``occ_meta`` (used by the points/marked_obs renderers; ignored by battleship), ``camera`` calibration
    key, and ``robot_poses`` (NED) for the robot markers.
    """
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller {controller!r}; expected one of {CONTROLLERS}")
    if map_overlay_type not in MAP_OVERLAY_TYPES:
        raise ValueError(
            f"unknown map_overlay_type {map_overlay_type!r}; expected one of {MAP_OVERLAY_TYPES}")

    # Render the overlay image with the renderer paired to this overlay type (battleship takes no grid).
    if map_overlay_type == "battleship":
        overlay_img = render_battleship_map(pil_img, robot_poses=robot_poses, camera=camera)
    else:
        overlay_img = _OVERLAY_RENDERERS[map_overlay_type](
            pil_img, occ_grid, occ_meta, camera, robot_poses=robot_poses)
    image_b64 = _to_b64(overlay_img)

    # For the marked_obs overlay, append the exact red-X (impassable) labels so the text lists the same
    # blocked points the image marks (same _cell_is_free classification via blocked_cell_labels).
    overlay_desc = MAP_OVERLAY_DESCRIPTION[map_overlay_type]
    if map_overlay_type == "marked_obs":
        blocked = blocked_cell_labels(occ_grid, occ_meta, camera)
        listing = ", ".join(blocked) if blocked else "(none)"
        overlay_desc = f"{overlay_desc}\nThe points marked impassable (red X) in this image are: {listing}. Do not choose one of these points as a waypoint for the robots."

    # Assemble the text prompt in the fixed piece order.
    parts = [
        COMMON_PREAMBLE,
        overlay_desc,
        CONTROLLER_EXPLAIN[controller],
        _operator_line(instruction),
    ]
    if cot:
        cot_block = COT_BLOCKS[controller]
        if controller == "maneuver":
            red = _RED_MARKER_SUBSTEP if map_overlay_type == "marked_obs" else ""
            # Strip the whole sentinel line (incl. its newline) so omitting the sub-step leaves no
            # blank line; _RED_MARKER_SUBSTEP carries its own trailing newline when present.
            cot_block = cot_block.replace("%RED_MARKER_STEP%\n", red)
        parts.append(cot_block)
    return "\n\n".join(parts), image_b64


# ── Classifier prompt ──────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """\
You are a robot navigation classifier for two TurtleBot robots named raph and donnie.
You receive an operator instruction and route it to the motion planner best suited to carry it out.
Choose exactly one task type:

- "nav2point": Best for tasks where each robot drives to one or more goal locations in order.
  The DESTINATION(S) matter; the exact path taken does not.
  Ex 1. "Send each robot to the nearest box."
  Ex 2. "Send raph to the chair and then the person. Have donnie stay in place."

- "coverage": Best for tasks that involve patrolling or sweeping an area.
  The planner selects high-level regions to cover, not a specific path. A low level controller 
  will ensure all seleected regions are visited efficiently.
  Ex 1. "Have a robot patrol the perimeter of the boxes."
  Ex 2. "Have one robot patrol the left side of the room and the other patrol the right side."

- "maneuver": Best for tasks where the specific route matters, not just the destination.
  Use this when the instruction constrains HOW the robot travels (loops, specific routes, avoidance etc.).
  Use this task type only when "nav2point" and "coverage" are insufficient.
  Do not use it for static formations (e.g. surround, block...etc.), relative final positions, or coordinated destination assignments.
  Ex 1. "Have raph do a loop around the chair and return to its start location."
  Ex 2. "Have the robots navigate to the chair, staying as far away from the people as possible."

Respond with EXACTLY one JSON object and nothing else:
{"task_type": "nav2point" | "maneuver" | "coverage"}
"""
