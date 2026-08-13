"""Grid path-planning algorithms (plain library, NOT a ROS node).

Operates purely on an int8 occupancy grid as produced by obs_seg.occupancy
(FREE=0 traversable, anything else = blocked). Intended for the *inflated* grid,
where the robot can be treated as a point.

Cells are (gx, gy) = (column, row); grid is indexed grid[gy, gx].

Lives inside node_Path_Translator, not at the top of nodes/, because its only consumers are
this package's planners (astar_proj, coverage_proj); the top level is reserved for libraries
genuinely shared across nodes (coord_transform, obs_seg, debug_io). Public API:
``from .grid_planner_utils import astar, line_of_sight, project_to_free, separate_goal,
simplify_path_los``.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from obs_seg import FREE, OCCUPIED
from obs_seg.occupancy import cell_to_world, world_to_cell

Cell = Tuple[int, int]

# m — minimum spacing between two robots' final goals. Defined here rather than at either call site
# so node_Path_Translator (its ROS parameter default) and scripts/test_pipeline.py cannot drift
# apart: a mismatch would make the offline debug figures describe a route the robots never execute.
DEFAULT_MIN_GOAL_SEPARATION = 0.5

# 8-connected neighborhood (dx, dy)
_NEIGHBORS8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _free(grid: np.ndarray, gx: int, gy: int) -> bool:
    h, w = grid.shape
    return 0 <= gx < w and 0 <= gy < h and grid[gy, gx] == FREE


def separate_goal(goal_xy, claimed, infl: np.ndarray, meta: dict, *,
                  min_separation: float, max_shift: float) -> Optional[Tuple[float, float]]:
    """Nearest free cell to `goal_xy` that is >= `min_separation` from every claimed goal.

    Returns a world (x, y) — `goal_xy`'s own cell centre when it is already clear — or None if no
    acceptable cell lies within `max_shift` metres, in which case the caller keeps the original goal
    and warns.

    A scratch COPY of the inflated grid gets a disc of radius `min_separation` stamped OCCUPIED
    around each claimed goal, then project_to_free's BFS ring search finds the nearest cell that is
    neither an obstacle nor inside a claimed disc — one search handles both. `infl` itself is never
    modified: the grid A* searches afterwards is untouched, so a robot may still travel THROUGH
    another's claimed disc and a parked robot never blocks a corridor.

    `max_shift` bounds the displacement. It is needed because adjusting the reference before
    planning bypasses astar_proj's own projection_radius check (by then the cell is already free,
    so its measured displacement is zero); without it a goal could slide far off target to escape
    a disc.

    Kept ROS-free here, rather than as a node method, so node_Path_Translator and the offline
    harness (scripts/test_pipeline.py) run the SAME separation — otherwise the debug figures would
    describe a route the robots never execute, which is exactly when the figure matters most.
    """
    h, w = infl.shape
    res = meta["resolution"]
    scratch = infl.copy()
    yy, xx = np.ogrid[0:h, 0:w]
    r2 = (min_separation / res) ** 2
    for (cx_w, cy_w) in claimed:
        cx, cy = world_to_cell(cx_w, cy_w, meta)
        scratch[((xx - cx) ** 2 + (yy - cy) ** 2) <= r2] = OCCUPIED

    goal_cell = world_to_cell(goal_xy[0], goal_xy[1], meta)
    cell = project_to_free(scratch, goal_cell)
    if cell is None:
        return None
    shift = math.hypot((cell[0] - goal_cell[0]) * res, (cell[1] - goal_cell[1]) * res)
    if shift > max_shift:
        return None
    return cell_to_world(cell[0], cell[1], meta)


def project_to_free(grid: np.ndarray, cell: Cell, max_radius_cells: int = 200) -> Optional[Cell]:
    """Return `cell` if FREE, else the nearest FREE cell via BFS ring search.

    Returns None if no FREE cell is found within `max_radius_cells`, or the start
    cell is outside the grid.
    """
    gx, gy = cell
    h, w = grid.shape
    if not (0 <= gx < w and 0 <= gy < h):
        return None
    if _free(grid, gx, gy):
        return (gx, gy)

    seen = np.zeros((h, w), dtype=bool)
    seen[gy, gx] = True
    q: deque = deque([(gx, gy, 0)])
    while q:
        cx, cy, d = q.popleft()
        if d > max_radius_cells:
            return None
        for dx, dy in _NEIGHBORS8:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx]:
                if grid[ny, nx] == FREE:
                    return (nx, ny)
                seen[ny, nx] = True
                q.append((nx, ny, d + 1))
    return None


def astar(grid: np.ndarray, start: Cell, goal: Cell) -> Optional[List[Cell]]:
    """8-connected A* between two FREE cells (Euclidean cost + heuristic).

    Returns the inclusive cell path [start, ..., goal], or None if unreachable
    (or if start/goal is not FREE).
    """
    if not _free(grid, *start) or not _free(grid, *goal):
        return None
    if start == goal:
        return [start]

    def h(c: Cell) -> float:
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    open_heap: List[Tuple[float, Cell]] = [(h(start), start)]
    g_cost = {start: 0.0}
    came_from: dict = {}
    closed = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        cx, cy = cur
        for dx, dy in _NEIGHBORS8:
            nx, ny = cx + dx, cy + dy
            if not _free(grid, nx, ny):
                continue
            step = 1.41421356 if (dx and dy) else 1.0
            ng = g_cost[cur] + step
            nb = (nx, ny)
            if ng < g_cost.get(nb, float("inf")):
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), nb))
    return None


def line_of_sight(grid: np.ndarray, a: Cell, b: Cell) -> bool:
    """True if every cell on the Bresenham segment a->b is FREE."""
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if not _free(grid, x, y):
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def simplify_path_los(grid: np.ndarray, path: List[Cell]) -> List[Cell]:
    """Greedy string-pull: from the current cell, jump to the farthest cell still in
    line-of-sight, and repeat. First and last cells are always preserved.
    """
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
