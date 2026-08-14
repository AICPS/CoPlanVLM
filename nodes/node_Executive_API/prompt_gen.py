"""In-repo prompt construction for the Executive (path-planner) node.

Each operator instruction goes through two LLM calls:

1. CLASSIFIER (classifier_prompt(robot_names)) — text-only; tags the instruction with a controller
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

Both calls are constrained by strict JSON schemas built here (``classifier_schema`` /
``route_schema``), and ``TASK_ROUTING`` maps the classifier's answer onto the model output key and the
downstream planner. Prompt text, overlay image, reply schema and planner routing therefore all live in
this one module — the live node (exec.py) and the offline harness (scripts/test_pipeline.py) import
them rather than restating them, so the two can never drift.

ROBOT NAMES: every prompt piece is a template whose robot names are %ROBOT_A% / %ROBOT_B% sentinels,
filled per call from the ``robot_names`` roster the caller passes (exec's ROS parameter, i.e. the
launch file's bot_name / bot2_name). The same roster drives ``route_schema``, so the prompt always
describes exactly the robots the reply schema accepts, and renaming a robot is a launch-file edit.
Exactly two robots are supported — anything else raises rather than silently mis-prompting.
"""

from __future__ import annotations

from node_Executive_API.map_gen import (
    render_grid_points_map, render_obstacle_marked_map, render_battleship_map,
    blocked_cell_labels, _to_b64, ROBOT_COLOR_NAMES)


# ── Axis vocabularies (the only valid controller / overlay names) ──────────────
CONTROLLERS       = ("nav2point", "maneuver", "coverage")
MAP_OVERLAY_TYPES = ("points", "marked_obs", "battleship")

# The overlay production uses for every controller. Lives here (not in exec.py) because the path
# translator also needs it: it renders vlm_selections.png in the style matching the overlay the VLM
# actually saw, and /vlm_plan does not carry the overlay type.
PRODUCTION_MAP_OVERLAY = "marked_obs"

# Task type -> (model output key, downstream planner name published on /vlm_plan).
# nav2point / maneuver return ordered "waypoints" routed through A*; coverage returns an unordered
# "regions" cell set routed through the coverage (TSP) planner. node_Path_Translator resolves the
# planner name via its own _PLANNERS table, so this is the single hand-off contract between the
# Executive and the translator.
TASK_ROUTING = {
    "nav2point": ("waypoints", "astar"),
    "maneuver":  ("waypoints", "astar"),
    "coverage":  ("regions",   "coverage"),
}


# ── 1. Shared preamble (every prompt) ──────────────────────────────────────────

COMMON_PREAMBLE = """\
You are a path-planning agent for two TurtleBot4 robots sharing one workspace.
You will be shown an overhead camera image of the environment with a map overlay.

The two robots — identify each by its appearance in the image:
%ROBOT_ROSTER%

General rules that apply to all tasks:
- You MUST include BOTH robots in every response, even if the instruction only mentions one.
  A robot with no task holds its position: return an empty list for each robot that should not move.
"""


# ── 2. Map-overlay descriptions (how each overlay marks the map) ───────────────
# Each entry describes the marks in the rendered image and the overlay-specific rules for choosing
# labels, so the controller pieces below can stay overlay-neutral.

_POINTS_OVERLAY = """\
MAP OVERLAY — Set of Marks:
The image shows blue dots at a regular grid of candidate locations, each labeled with an alphanumeric
identifier. The grid has exactly 14 columns (A-N, left to right) and 8 rows (1-8, top to bottom),
giving points like "A1", "H4", "N8". Columns never go past N and rows never go past 8.
Directions are cardinal and fixed to the image: WEST/LEFT is a lower column letter (A1 is west of
B1), EAST/RIGHT a higher one (H2 is east of G2), NORTH/UP a lower row number (C3 is north of C5),
and SOUTH/DOWN a higher one.
A dot is placed at every grid point regardless of what lies beneath it, so a dot may fall on an
obstacle — use the image to judge which points are on clear, reachable floor and choose destinations
only from those.
Choose ONLY from labels that exist on this grid; NEVER invent or select a point outside it (no column
beyond N, no row beyond 8, e.g. "G9" or "P4" do not exist)."""

_MARKED_OBS_OVERLAY = """\
MAP OVERLAY — Set of Marks with Impassable Points:
The image shows a labeled mark at every point of a regular grid with exactly 14 columns (A-N, left to
right) and 8 rows (1-8, top to bottom), giving points like "A1", "H4", "N8". Columns never go past N
and rows never go past 8.
Directions are cardinal and fixed to the image: WEST/LEFT is a lower column letter (A1 is west of
B1), EAST/RIGHT a higher one (H2 is east of G2), NORTH/UP a lower row number (C3 is north of an object covering points C4, C5, D5),
and SOUTH/DOWN a higher one (G5 is south of G4).
Each mark has one of two forms:
- a BLUE DOT marks a FREE point the robot CAN navigate to;
- a RED X marks a point that is NOT passable — the robot CANNOT stand on or travel through a red X
  point (it is blocked/occupied). Red X marks will often mark objects of interest.
Choose robot destinations and routes ONLY from blue-dot points.
NEVER invent or select a point outside the grid (no column beyond N, no row beyond 8, e.g. "G9" or
"P4" do not exist)."""

_BATTLESHIP_OVERLAY = """\
MAP OVERLAY — Battleship Grid:
The image shows a grid overlaid on the environment with exactly 14 columns (A-N, left to right) and
8 rows (1-8, top to bottom), giving cells such as "A1", "H4", "N8". Columns never go past N and rows
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
{"waypoints": {"%ROBOT_A%": ["<goal_point>", ...], "%ROBOT_B%": ["<goal_point>", ...]}}

Example — if the instruction is "Send %ROBOT_A% to the chair and then the person. Have %ROBOT_B% stay in place.", a valid response is:
{"waypoints": {"%ROBOT_A%": ["C4", "H8"], "%ROBOT_B%": []}}"""

_MANEUVER_EXPLAIN = """\
YOUR TASK:
Plan a specific route for each robot as an ordered list of waypoints, chosen from the labeled locations
shown in the overlay. The route taken matters so follow the instruction carefully. Step between locations that have clear paths between them to draw a coherent path.

OUTPUT FORMAT (exactly one JSON object):
{"waypoints": {"%ROBOT_A%": ["<point>", ...], "%ROBOT_B%": ["<point>", ...]}}

Example — if the instruction is "Have the robots do a loop around the chair", a valid response is:
{"waypoints": {"%ROBOT_A%": ["A7", "B6", "C5", "D4", "D3", "C3", "B3", "B4", "B5", "B6", "A7"],
              "%ROBOT_B%": ["D3", "D4", "D5", "C5", "B5", "B4", "B3", "C3", "D2"]}}"""

_COVERAGE_EXPLAIN = """\
YOUR TASK:
Select the locations each robot should visit to cover or patrol the area described in the instruction,
choosing from the labeled locations shown in the overlay. You choose which to cover. If no robot is
specified, split coverage roughly equally between %ROBOT_A% and %ROBOT_B%. If only one robot is specified,
return an empty list for the other. The order does not matter — a lower-level controller will find the
most efficient patrol route.

OUTPUT FORMAT (exactly one JSON object):
{"regions": {"%ROBOT_A%": ["<cell>", ...], "%ROBOT_B%": ["<cell>", ...]}}

Example — if the instruction is "Have the robots patrol the left side of the map", a valid response is:
{"regions": {"%ROBOT_A%": ["A5", "A6", "A7", "A8", "B5", "B6", "B7", "B8", "C5", "C6", "C7", "C8", "D5", "D6", "D7", "D8", "E5", "E6", "E7", "E8"],
             "%ROBOT_B%": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4", "E1", "E2", "E3", "E4"]}}"""

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
1) GROUNDING — locate everything the task depends on.
   a) Give each robot's current position as the labeled grid point nearest it in the image
      (e.g. "%ROBOT_A%: A7", "%ROBOT_B%: D2").
   b) Identify every object/goal location named in the operator's instruction and give its grid
      label(s) (e.g. "chair: C4", "yellow box: H8").
   c) %RED_MARKER_STEP%
2) TASK ALLOCATION.
   a) Classify the task as ONE robot (say which), BOTH, or UNSPECIFIED. "the robots"/"both"/"each"
      means BOTH; "a robot" or a single robot name means ONE; otherwise UNSPECIFIED — divide the work
      appropriately between them.
   b) List every point the robots should visit, and for each one name which robot is currently closest.
3) ROUTE CONSTRUCTION — assign the points to each robot, in visit order. Every robot chosen in 2a MUST
   get at least one point, and if both are sent to the same target give them two DIFFERENT nearby
   points — two robots cannot occupy one cell.
4) VERIFICATION — for each assigned point, check whether its cell is blocked (see the blocked points
   listed in the MAP OVERLAY). If it is blocked, you MUST give all four parts in order: name the
   obstacle, state which SIDE of it the robot should be on (north/south/east/west), then the nearest
   FREE (blue-dot) point on that side. Never go straight from "blocked" to a replacement point.
      <robot> -> <label> is blocked by <obstacle> -> approach from the <direction> -> travel to <label>
   If the point is already free, use it as-is: "<robot> -> <label> is free -> travel to <label>".
   Example — "%ROBOT_B% -> E8 is blocked by the green box -> approach from the WEST -> travel to D8".
"""

_COVERAGE_COT = """\
REASONING — Before deciding the routes, work through these in the "reasoning" field, in order:
1) GROUNDING — locate everything the task depends on.
   a) Give each robot's current position as the labeled grid point nearest it in the image
      (e.g. "%ROBOT_A%: A7", "%ROBOT_B%: D2").
   b) Identify every object/region named in the operator's instruction and give the grid label(s)
      each one occupies or spans (e.g. "chair: C4", "yellow box: G8,G7,H8,H7").
   c) %RED_MARKER_STEP%
   d) Describe which regions the instruction requires the robots to cover or visit, then list every
      grid region that matches that description. To survey completely AROUND an object, list the full
      ring of free regions on every side of it, not just one side
      (e.g. survey around yellow box G4,G5,H4,H5 -> G3,H3,I4,I5,H6,G6,F5,F4).
      To cover an AREA or region, list every region INSIDE it, not the ring around it
      (e.g. survey the area spanning A1-B3 -> A1,A2,A3,B1,B2,B3).
2) TASK ALLOCATION — divide the regions from 1d evenly between %ROBOT_A% and %ROBOT_B%.
   Try to assign points to the closest robot while maintaining a roughly even split.
3) VERIFICATION — for each region selected in step 2, state whether it is in the blocked list given in
   the MAP OVERLAY. For each blocked one, state WHY it was chosen, then REPLACE it with the nearby
   free region (or regions) that best fulfills that same purpose. Every blocked region must be
   replaced — never simply drop one. Only after this reasoning, fill in the route object with your
   final choice."""

# Body of the red-marker sub-step, substituted into the %RED_MARKER_STEP% sentinel only for the
# marked_obs overlay — the only overlay that draws red X marks. Omitted otherwise so the model is not
# asked to inventory marks that do not exist.
#
# Deliberately carries NO step label: each CoT block writes its own ("   c) %RED_MARKER_STEP%"), so a
# block is free to number it differently without touching this text. The continuation line keeps its
# own indent, so it lines up under whatever label the caller supplies. _cot_block drops the ENTIRE
# sentinel line when the sub-step is omitted, which is what stops a bare "c)" being left behind.
_RED_MARKER_SUBSTEP = """Label each red marker: for every red X in the image, give its grid label AND the
      object under it or the closest object near it (it may sit beside, not directly under, the mark) —
      one entry per marker, e.g. "H8: person", "C4: chair", "G3: box"; write "unknown" if unclear."""

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
waypoints that define the maneuver's shape.

1) Identify and locate the robots and key objects in the scene.
   a) Robots: give %ROBOT_A%'s and %ROBOT_B%'s current position as the labeled grid point nearest each robot
      in the image (e.g. "%ROBOT_A%: A7", "%ROBOT_B%: D2").
   b) Task features: list everything the instruction requires you to perceive to carry out the task.
      Give the grid label(s) each one occupies or spans (e.g. "yellow box: H8", "caution tape: E4-E7", "box: C4, D4, C3, D3").
   c) %RED_MARKER_STEP%
2) Split the instruction into two subtasks, one per each robot. Do not summarize or drop any spatial constraints or landmarks — keep every word like behind / around / left of...
   Use the robot the instruction names; if it names none, list which robot would be best. If the
   instruction gives a robot no work, its subtask is "stay".
   For each subtask, list the final location each robot should end at.
      subtask 1 <subtask kept in full>: <robot> (<why that robot>), ends at <label>
      subtask 2 <subtask kept in full>: <robot> (<why that robot>), ends at <label>
   Example — "have one robot wait behind the chair at C4 while the other approaches the person at H8
   from the west":
      subtask 1 wait behind the chair at C4: %ROBOT_B% (already at C2, nearest), ends at C3
      subtask 2 approach the person at H8 from the west: %ROBOT_A% (already at F7, nearest), ends at G8
   Example — "have %ROBOT_A% loop around the chair at C4 and then travel to %ROBOT_B%":
      subtask 1 loop around the chair at C4 and then travel to %ROBOT_B%: %ROBOT_A% (named in the
      instruction), ends at B7 (beside %ROBOT_B%, which is at A7)
      subtask 2 stay: %ROBOT_B% (the instruction gives it no work), ends at A7 where it already is,
      so its route is empty
   Example — "have %ROBOT_A% go around the west side of the pallet stack at E3-F5 to reach E6, while
   %ROBOT_B% approaches the cart at L7 from the north":
      subtask 1 go around the west side of the pallet stack at E3-F5 to reach E6: %ROBOT_A% (already
      at B2, nearest), ends at E6 (the west side is column D, so the route must run down column D and
      round the south end of the stack — NOT straight across open floor)
      subtask 2 approach the cart at L7 from the north: %ROBOT_B% (already at M4, nearest), ends at L6
3) For each robot, break its subtask into ordered legs, each with a clear and detailed description, giving only the KEY
   waypoint(s) that realize it. Turn each spatial constraint into concrete cells on the REQUIRED SIDE:
      (e.g. an object spanning F3-H3 -> pass north/above it via F2, G2, H2)
   For a loop or circuit around an object, give
   waypoints on several DIFFERENT sides of it (not just the far side), so the route encircles the object
   instead of going out and doubling back the same way:
      (e.g. an object at G5, G6, F6 -> loop around it via F5,G4,H5,H6,G7,E6,F5)
      leg 1 <purpose>: <label(s)>   ...   leg N <purpose>: <label(s)>
   Example — "%ROBOT_A%: loop around the chair at C4 and return to its start at A7":
      leg 1 approach the chair: C5 ; leg 2 circle it via each side: D4, C3, B4, C5 ;
      leg 3 return to start: A7
   Example — "%ROBOT_A%: travel around the left side of the chair at D3,D4 before continuing to the
   person at F7". One leg per clause, and the side constraint becomes cells on that side:
      leg 1 pass the chair on its LEFT/WEST side: C3, C4 (left of column D is column C, so the route
      runs down column C — the leg must sit beside the chair, not on open floor away from it) ;
      leg 2 continue to the person: F6 (just north of the person at F7)
4) Assemble each robot's route as a SHORT ordered list of the key waypoints from step 3 (start -> legs
   -> final goal). Keep it sparse: consecutive waypoints may be far apart and the planner fills the gaps
   collision-free. If both robots are sent to the same target, end their routes on two DIFFERENT
   nearby points — two robots cannot occupy one cell.
"""

# controller -> ordered (field_name, description) for the reply's per-step reasoning fields, mirroring
# the numbered steps of that controller's COT_BLOCKS entry above. route_schema declares them in this
# order, so the model fills each step in sequence BEFORE emitting routes, and strict mode makes every
# one mandatory — a step cannot be silently skipped the way it could with one big "reasoning" string.
#
# Swap a task category's reasoning format by editing its entry here (and its COT_BLOCKS prose): the
# per-category selection is the same mechanism COT_BLOCKS uses, so the three categories stay
# independent. Collapsing an entry to a single ("reasoning", ...) pair restores the old behaviour for
# that category alone.
#
# The field count MUST equal the number of numbered steps in the matching block; the descriptions are
# short restatements, since the block itself carries the detailed guidance.
COT_FIELDS = {
    "nav2point": [
        ("step1_ground",     "a) each robot's current grid point; b) the objects/goal locations named "
                             "in the instruction with their label(s); c) each red marker and the "
                             "object at or near it."),
        ("step2_allocate",   "a) ONE robot (say which), BOTH, or UNSPECIFIED, and how the work "
                             "divides; b) every point to visit, each with the robot closest to it."),
        ("step3_route",      "The points assigned to each robot, in visit order."),
        ("step4_verify",     "For each assigned point: whether it is blocked, and the free point used "
                             "instead plus the direction."),
    ],
    "maneuver": [
        ("step1_locate",     "a) each robot's current grid point; b) task features (goals, boundaries "
                             "to avoid, landmarks) with the label(s) each occupies or spans."),
        ("step2_subtasks",   "One subtask per robot with every spatial constraint and landmark kept, "
                             "which robot performs each, and the cell each ends at. A robot the "
                             "instruction gives no work to gets the subtask \"stay\"."),
        ("step3_legs",       "Per robot: ordered legs with a short purpose and only the KEY waypoints, "
                             "each spatial constraint turned into cells on the required side."),
        ("step4_assemble",   "Per robot: the short ordered route from start through the legs to the goal."),
    ],
    "coverage": [
        ("step1_ground",     "a) each robot's current grid point; b) the objects/regions named in the "
                             "instruction with the label(s) each occupies or spans; c) each red marker "
                             "and the object at or near it; d) a description of which regions must be "
                             "covered, then every region matching it."),
        ("step2_allocate",   "How those regions divide evenly between the robots."),
        ("step3_verify",     "For each selected region: whether it is blocked, why it was chosen, and "
                             "the nearby free region(s) replacing it."),
    ],
}

# controller -> its chain-of-thought block. Total over CONTROLLERS (validated in generate_prompt).
COT_BLOCKS = {
    "nav2point": _NAV2POINT_COT,   # TODO: custom block (copy of generic for now)
    "maneuver":  _MANEUVER_COT,
    "coverage":  _COVERAGE_COT,    # TODO: custom block (copy of generic for now)
}

# Every controller must have both a prose block and a matching set of reply fields.
assert set(COT_FIELDS) == set(COT_BLOCKS) == set(CONTROLLERS), (
    "COT_FIELDS / COT_BLOCKS / CONTROLLERS disagree: "
    f"{sorted(COT_FIELDS)} / {sorted(COT_BLOCKS)} / {sorted(CONTROLLERS)}")


def reasoning_text(result: dict, controller: str) -> str:
    """Join a planner reply's per-step reasoning fields into one readable block, in step order.

    `result` is the decoded reply; `controller` selects which COT_FIELDS list to read. Fields absent
    from the reply are skipped, so a cot=False reply (no reasoning fields at all) yields "" rather
    than raising — callers can treat the empty string as "nothing to report".

    Shared by exec.py's log line and the offline harness's terminal output so the two render the
    trace identically.
    """
    fields = COT_FIELDS.get(controller, ())
    return "\n".join(f"{name}: {result[name]}" for name, _ in fields if result.get(name))


def _operator_line(instruction: str) -> str:
    return f'The operator\'s instruction is: "{instruction}"'


# ── Robot-name substitution ────────────────────────────────────────────────────
# The prompt pieces above are TEMPLATES: every robot name is a %ROBOT_A% / %ROBOT_B% sentinel, and
# the roster block is %ROBOT_ROSTER%. They are filled in per call from the roster the launch file
# configured (exec's `robot_names` parameter), so renaming a robot there is enough — the prompt, the
# reply schema (route_schema) and the marker colours all follow from the same list.
#
# Sentinels rather than f-strings because these strings are full of literal JSON braces.

DEFAULT_ROBOT_NAMES = ("raph", "donnie")


def _check_roster(robot_names) -> list:
    """Return the roster as a list, or raise if it is not exactly two robots.

    The prompts (roster block, worked examples, CoT scaffolds) are written for two robots, so any
    other count would silently produce text that disagrees with route_schema's robot list. Failing
    here turns that into an immediate, readable error instead of a confused model.
    """
    names = list(robot_names)
    if len(names) != 2:
        raise ValueError(
            f"prompts support exactly 2 robots, got {names}. Set the `robot_names` parameter to two "
            "names (see the launch files' bot_name / bot2_name).")
    return names


def _fill(text: str, robot_names) -> str:
    """Substitute the robot sentinels in a prompt template.

    Single place for the substitution rules so the pieces cannot drift apart. Colours come from
    map_gen.ROBOT_COLOR_NAMES, index-aligned with the _ROBOT_COLORS the overlay actually draws, so
    the description can never contradict the image.
    """
    names = _check_roster(robot_names)
    # Pad the quoted names to a common width so the em-dashes line up whatever the names are.
    width = max(len(n) for n in names) + 2          # +2 for the surrounding quotes
    roster = "\n".join(
        f'- {chr(34) + name + chr(34):<{width}} — the round black TurtleBot labeled "{name}" with '
        f'{ROBOT_COLOR_NAMES[i % len(ROBOT_COLOR_NAMES)]} in the image.'
        for i, name in enumerate(names))
    return (text.replace("%ROBOT_ROSTER%", roster)
                .replace("%ROBOT_A%", names[0])
                .replace("%ROBOT_B%", names[1]))


def _common_preamble(robot_names) -> str:
    """Piece 1: robot roster + global rules."""
    return _fill(COMMON_PREAMBLE, robot_names)


def _controller_explain(controller: str, robot_names) -> str:
    """Piece 4: what to select, output schema and worked example for this controller."""
    return _fill(CONTROLLER_EXPLAIN[controller], robot_names)


def _cot_block(controller: str, robot_names, map_overlay_type: str) -> str:
    """Piece 5: the controller's chain-of-thought scaffold.

    A scaffold may carry a %RED_MARKER_STEP% sentinel, filled only for the marked_obs overlay — the
    only overlay that draws red X marks, so the model is never asked to inventory marks that do not
    exist. Keyed on the sentinel being PRESENT rather than on a controller name, so adding it to
    another block needs no change here.

    The block owns the step LABEL ("   c) %RED_MARKER_STEP%") and _RED_MARKER_SUBSTEP is only the
    body, so each block can number the sub-step to suit its own scheme. That means omission has to
    drop the ENTIRE line, label included — substituting an empty body would leave a bare "c)" behind.
    """
    block = COT_BLOCKS[controller]
    if "%RED_MARKER_STEP%" in block:
        keep = map_overlay_type == "marked_obs"
        kept_lines = []
        for line in block.splitlines(keepends=True):
            if "%RED_MARKER_STEP%" not in line:
                kept_lines.append(line)
            elif keep:
                kept_lines.append(line.replace("%RED_MARKER_STEP%", _RED_MARKER_SUBSTEP))
        block = "".join(kept_lines)
    return _fill(block, robot_names)


def classifier_prompt(robot_names=DEFAULT_ROBOT_NAMES) -> str:
    """System prompt for the CLASSIFIER call, with the configured robot names filled in.

    Public because both exec.py and the offline harnesses make this call directly.
    """
    return _fill(_CLASSIFIER_SYSTEM_PROMPT, robot_names)


# ── Constructor ────────────────────────────────────────────────────────────────

def generate_prompt(instruction, controller, map_overlay_type, *,
                    pil_img, occ_grid, occ_meta, camera, robot_poses=None, cot=False,
                    robot_names=DEFAULT_ROBOT_NAMES):
    """Assemble the planner system prompt AND render the matching overlay image.

    Returns ``(prompt_text, image_b64)``. ``controller`` selects the task semantics + output schema
    (piece 4); ``map_overlay_type`` selects BOTH the overlay description (piece 2) and the map_gen
    renderer that draws the image, so text and image always agree. ``cot=True`` appends the
    controller's chain-of-thought scaffold (piece 5); for maneuver the red-marker sub-step is included
    only when ``map_overlay_type == "marked_obs"``.

    ``robot_names`` (exactly two) names the robots throughout the prompt. Pass the SAME roster used
    for ``route_schema`` — the prompt describes the robots the schema will accept, and its order sets
    which marker colour belongs to which robot. Callers normally pass exec's ``robot_names``
    parameter, i.e. whatever the launch file configured.

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

    # Assemble the text prompt in the fixed piece order. The overlay description carries no robot
    # names, so it is used verbatim; the other pieces are filled from the roster.
    parts = [
        _common_preamble(robot_names),
        overlay_desc,
        _controller_explain(controller, robot_names),
        _operator_line(instruction),
    ]
    if cot:
        parts.append(_cot_block(controller, robot_names, map_overlay_type))
    return "\n\n".join(parts), image_b64


# ── Reply schemas (Responses API structured outputs) ───────────────────────────

def route_schema(result_key, robot_names, cot, allowed_labels=None, controller=None):
    """Strict JSON Schema for the PLANNER reply — enforced via Responses API structured outputs.

    Always: {result_key: {<robot>: [labels] for each robot}}. additionalProperties:false + a fixed
    robot roster means the model CANNOT emit extra keys (e.g. a "raph_donnie" shared route) — the
    structure is guaranteed by constrained decoding, not by asking in the prompt.

    When allowed_labels is given, every waypoint is constrained to that enum (the free/blue-dot cells),
    so the model physically CANNOT select an obstacle cell — obstacle avoidance enforced at decode time,
    not merely requested in the prompt. Callers currently leave it unset and let the planner project
    picks onto free cells instead (see map_gen.free_cell_labels to re-enable).

    When cot=True the controller's COT_FIELDS are declared as one required string field PER REASONING
    STEP, ahead of result_key, and `controller` is therefore required. One field per step rather than a
    single "reasoning" blob because strict mode requires every declared property: the model cannot
    address three of six steps and leave the rest unwritten. Structured outputs also fill properties in
    declaration order, so the steps are produced in sequence — each conditioned on the previous — and
    the routes are conditioned on all of them. Read the trace back with reasoning_text().
    """
    if cot and controller not in COT_FIELDS:
        raise ValueError(
            f"cot=True needs a controller in {tuple(COT_FIELDS)}, got {controller!r} — the reasoning "
            "fields are per task category (see COT_FIELDS).")
    item_schema = {"type": "string"}
    if allowed_labels:
        item_schema = {"type": "string", "enum": list(allowed_labels)}
    route_obj = {
        "type": "object", "additionalProperties": False,
        "required": list(robot_names),
        "properties": {name: {"type": "array", "items": item_schema} for name in robot_names},
    }
    properties: dict = {}
    required: list = []
    if cot:                # declared first, in step order -> generated (reasoned) first, in order
        for fname, desc in COT_FIELDS[controller]:
            properties[fname] = {"type": "string", "description": desc}
            required.append(fname)
    properties[result_key] = route_obj
    required.append(result_key)
    return {"type": "object", "additionalProperties": False,
            "required": required, "properties": properties}


def classifier_schema():
    """Strict JSON Schema for the CLASSIFIER reply: exactly one of CONTROLLERS.

    The enum makes an out-of-vocabulary task type impossible at decode time, rather than something
    callers detect and reject afterwards. CONTROLLERS is the single source of the vocabulary, so
    adding a controller extends the classifier automatically.
    """
    return {
        "type": "object", "additionalProperties": False,
        "required": ["task_type"],
        "properties": {"task_type": {"type": "string", "enum": list(CONTROLLERS)}},
    }


# ── Classifier prompt ──────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a robot navigation classifier for two TurtleBot robots named %ROBOT_A% and %ROBOT_B%.
You receive an operator instruction and route it to the motion planner best suited to carry it out.
Choose exactly one task type:

- "nav2point": Best for tasks where each robot drives to one or more goal locations in order.
  The DESTINATION(S) matter; the exact path taken does not.
  Ex 1. "Send each robot to the nearest box."
  Ex 2. "Send %ROBOT_A% to the chair and then the person. Have %ROBOT_B% stay in place."

- "coverage": Best for tasks that involve patrolling or sweeping an area.
  The planner selects high-level regions to cover, not a specific path. A low level controller 
  will ensure all seleected regions are visited efficiently.
  Ex 1. "Have a robot patrol the perimeter of the boxes."
  Ex 2. "Have one robot patrol the left side of the room and the other patrol the right side."

- "maneuver": Best for tasks where the specific route matters, not just the destination.
  Use this when the instruction constrains HOW the robot travels (loops, specific routes, avoidance etc.).
  Use this task type only when "nav2point" and "coverage" are insufficient.
  Do not use it for static formations (e.g. surround, block...etc.), relative final positions, or coordinated destination assignments.
  Ex 1. "Have %ROBOT_A% do a loop around the chair and return to its start location."
  Ex 2. "Have the robots navigate to the chair, staying as far away from the people as possible."

Respond with EXACTLY one JSON object and nothing else:
{"task_type": "nav2point" | "maneuver" | "coverage"}
"""
