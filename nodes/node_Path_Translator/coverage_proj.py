"""Coverage (region-sweep) planner — a pluggable translate.py planner module.

Method: project each region-cell centroid (plus the robot pose) onto the nearest FREE cell of the
INFLATED occupancy grid, order the projected points with an open Travelling-Salesman tour that
starts at the robot's pose (no return), then stitch consecutive anchors with A* + line-of-sight
thinning. Unlike astar_proj the input cells are an *unordered* set to cover — the order is chosen
here, not by the VLM.

Ordering cost is the true geodesic (obstacle-aware) distance: for each anchor we run one
single-source Dijkstra expansion over the free grid (n runs, not C(n,2) pairwise A*), so the cost
matrix scales to the full grid (14x8 = 112 cells). The tour is solved with nearest-neighbour +
2-opt — no external TSP solver is needed at this size.

Pluggable planner interface (shared with astar_proj):
    build_reference(labels, pose_xy, grid_px, camera) -> (ref, unknown)
    plan(reference_xy, ctx, meta, params) -> (world_path, debug)
    save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params, camera) -> None

The planning grid is inflated upstream (exec.run_segmentation / test_pipeline), so ctx is simply
{"infl": <inflated grid>} built by the caller — this module never inflates.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import numpy as np

from coord_transform import pixel_to_world
from obs_seg import FREE
from obs_seg.occupancy import world_to_cell, cell_to_world
from grid_planner import astar, project_to_free, simplify_path_los

from . import astar_proj


PARAMS: dict = {
    "projection_radius": 1.0,   # m; region cells with no free cell within this radius are dropped
}

# 8-connected neighbourhood (dx, dy) with Euclidean step costs — matches grid_planner.astar.
_NEIGHBORS8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
_DIAG = 1.41421356


def _p(params, key):
    v = params.get(key, PARAMS[key])
    return PARAMS[key] if v is None else v


def build_reference(labels, pose_xy, grid_px, camera=None):
    """Convert an unordered label set to a world-frame reference: [pose, centroids...].

    Coverage order is irrelevant and chosen later by the TSP, so — unlike astar_proj.build_reference
    — nothing is trimmed here; every known label becomes a centroid and the robot pose is prepended
    as the fixed tour start.

    Returns (ref, unknown) where ref is [(x, y), ...] and unknown lists labels absent from grid_px.
    """
    centroids = []
    unknown = []
    for label in labels:
        if label not in grid_px:
            unknown.append(label)
            continue
        u, v = float(grid_px[label][0]), float(grid_px[label][1])
        x, y = pixel_to_world(u, v, camera=camera)
        centroids.append((float(x), float(y)))

    ref = []
    if pose_xy is not None:
        ref.append((float(pose_xy[0]), float(pose_xy[1])))
    ref.extend(centroids)
    return ref, unknown


def _dijkstra(grid: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """Single-source shortest-path costs from `start` to every FREE cell (8-connected, Euclidean).

    Returns an (H, W) float array of costs; unreachable/blocked cells are +inf.
    """
    h, w = grid.shape
    dist = np.full((h, w), math.inf, dtype=np.float64)
    sx, sy = start
    if not (0 <= sx < w and 0 <= sy < h and grid[sy, sx] == FREE):
        return dist
    dist[sy, sx] = 0.0
    heap: List[Tuple[float, int, int]] = [(0.0, sx, sy)]
    while heap:
        d, cx, cy = heapq.heappop(heap)
        if d > dist[cy, cx]:
            continue
        for dx, dy in _NEIGHBORS8:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == FREE:
                nd = d + (_DIAG if (dx and dy) else 1.0)
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, nx, ny))
    return dist


def _open_tsp(cost: List[List[float]]) -> List[int]:
    """Open TSP (fixed start at index 0, no return edge): nearest-neighbour + 2-opt.

    `cost` is a full n×n symmetric matrix (math.inf for unreachable pairs). Returns a visiting
    order (a permutation of range(n) beginning with 0). Handles ~100+ nodes comfortably.
    """
    n = len(cost)
    if n <= 2:
        return list(range(n))

    # Nearest-neighbour construction from the fixed start.
    unvisited = set(range(1, n))
    order = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j: cost[cur][j])
        order.append(nxt)
        unvisited.discard(nxt)
        cur = nxt

    # 2-opt improvement. Index 0 stays fixed (segment reversals start at i >= 1). The last node has
    # no trailing edge (open path), which the k+1 == n branch accounts for.
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = order[i - 1]
            b = order[i]
            for k in range(i + 1, n):
                c = order[k]
                if k + 1 < n:
                    d = order[k + 1]
                    delta = (cost[a][c] + cost[b][d]) - (cost[a][b] + cost[c][d])
                else:
                    delta = cost[a][c] - cost[a][b]   # reversing the open tail
                if delta < -1e-9:
                    order[i:k + 1] = order[i:k + 1][::-1]
                    b = order[i]
                    improved = True
    return order


def plan(reference_xy, ctx, meta, params):
    """reference_xy: world (x,y) points [robot pose, region centroids...]. Returns (world_path, debug)."""
    infl = ctx["infl"]
    proj_radius = float(_p(params, "projection_radius"))
    res = meta["resolution"]
    warnings: List[str] = []

    # Project every reference point onto the nearest free cell, dropping any whose nearest free cell
    # is farther than projection_radius. The robot pose (index 0) is kept as the fixed tour start.
    anchors: List[Tuple[int, int]] = []
    for idx, (x, y) in enumerate(reference_xy):
        orig_cell = world_to_cell(x, y, meta)
        free_cell = project_to_free(infl, orig_cell)
        if free_cell is None:
            warnings.append(f"Cell ({x:.2f},{y:.2f}) has no free cell nearby; skipping.")
            continue
        dist_m = math.hypot((free_cell[0] - orig_cell[0]) * res,
                            (free_cell[1] - orig_cell[1]) * res)
        if idx != 0 and dist_m > proj_radius:
            warnings.append(
                f"Cell ({x:.2f},{y:.2f}) nearest free cell is {dist_m:.2f} m away "
                f"(limit {proj_radius:.2f} m); dropping.")
            continue
        anchors.append(free_cell)

    # Deduplicate while preserving order (start stays first); TSP over the unique cell set.
    seen = set()
    uniq: List[Tuple[int, int]] = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    anchors = uniq

    if len(anchors) <= 1:
        # Only the start survived (or nothing) — no region to sweep.
        world_path = [cell_to_world(gx, gy, meta) for (gx, gy) in anchors]
        return world_path, {"warnings": warnings, "reference_xy": list(reference_xy),
                            "anchors": anchors, "full_cells": list(anchors)}

    # Geodesic cost matrix: one Dijkstra expansion per anchor (n runs), read off at the anchors.
    n = len(anchors)
    dmaps = [_dijkstra(infl, a) for a in anchors]
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                gx, gy = anchors[j]
                cost[i][j] = float(dmaps[i][gy, gx])   # +inf if unreachable

    order = _open_tsp(cost)
    ordered = [anchors[k] for k in order]

    # Stitch consecutive anchors in tour order with A* + LOS thinning (only n-1 A* runs).
    full_cells = [ordered[0]]
    current = ordered[0]
    for nxt in ordered[1:]:
        seg = astar(infl, current, nxt)
        if seg is None:
            warnings.append(f"No A* path from {current} to {nxt}; skipping cell.")
            continue
        seg = simplify_path_los(infl, seg)
        full_cells.extend(seg[1:])      # drop duplicate shared endpoint
        current = nxt

    world_path = [cell_to_world(gx, gy, meta) for (gx, gy) in full_cells]
    debug = {"reference_xy": list(reference_xy), "anchors": ordered,
             "full_cells": full_cells, "warnings": warnings}
    return world_path, debug


def save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params, camera=None):
    """Route-specific debug — identical visualisation to astar_proj (anchors + planned route)."""
    return astar_proj.save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params, camera=camera)
