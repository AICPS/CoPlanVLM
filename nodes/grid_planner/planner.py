"""Grid path-planning algorithms (plain library, NOT a ROS node).

Operates purely on an int8 occupancy grid as produced by obs_seg.occupancy
(FREE=0 traversable, anything else = blocked). Intended for the *inflated* grid,
where the robot can be treated as a point.

Cells are (gx, gy) = (column, row); grid is indexed grid[gy, gx].
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from obs_seg import FREE

Cell = Tuple[int, int]

# 8-connected neighborhood (dx, dy)
_NEIGHBORS8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _free(grid: np.ndarray, gx: int, gy: int) -> bool:
    h, w = grid.shape
    return 0 <= gx < w and 0 <= gy < h and grid[gy, gx] == FREE


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
