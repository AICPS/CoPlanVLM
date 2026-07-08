"""A* projection planner — the original translate.py path method, extracted as a module.

Method: project each reference point (robot pose + VLM centroids) onto the nearest FREE cell of
the INFLATED occupancy grid, then pairwise A* between consecutive anchors with line-of-sight
thinning. Reuses grid_planner + obs_seg.occupancy verbatim.

Pluggable planner interface (shared with chomp_proj):
    build(grid, meta, params) -> ctx
    plan(reference_xy, ctx, meta, params) -> (world_path, debug)
    save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params) -> None
"""
from __future__ import annotations

import math
import os
from typing import List

import cv2
import numpy as np

from coord_transform import pixel_to_world, world_to_pixel
from obs_seg import FREE, OCCUPIED
from obs_seg.occupancy import inflate_occupancy, world_to_cell, cell_to_world
from grid_planner import project_to_free, astar, simplify_path_los


PARAMS: dict = {
    "projection_radius": 1.0,   # m; waypoints with no free cell within this radius are dropped
}


def _p(params, key):
    v = params.get(key, PARAMS[key])
    return PARAMS[key] if v is None else v


def build_reference(labels, pose_xy, grid_px, camera=None):
    """Convert a label list to a world-frame reference route.

    Shared between translate.py and test_pipeline.py — the single implementation of
    centroid-label -> world-coordinate conversion.

    Args:
        labels:   ordered list of grid-cell label strings from the VLM
        pose_xy:  robot world position (x, y) prepended as the first point, or None
        grid_px:  {label: indexable[0]=u, [1]=v} — accepts (u,v) tuples or np.array([u,v,1])
        camera:   camera calibration key (e.g. "gazebo", "lab_test") — required

    On a replan the robot has usually driven past its first few centroids, so any centroid before
    the one nearest the current pose is dropped — this prevents the new route from backtracking.
    On the initial plan the robot sits at the start, so the nearest centroid is effectively the
    first and nothing is trimmed.

    Returns:
        (ref, unknown) where ref is [(x,y),...] and unknown is [label,...] for any
        labels not found in grid_px.
    """
    # Convert known labels -> world centroids, tracking unknown labels.
    centroids = []
    unknown = []
    for label in labels:
        if label not in grid_px:
            unknown.append(label)
            continue
        u, v = float(grid_px[label][0]), float(grid_px[label][1])
        x, y = pixel_to_world(u, v, camera=camera)
        centroids.append((float(x), float(y)))

    # Drop centroids before the one nearest the current pose (skip already-passed waypoints).
    if pose_xy is not None and len(centroids) >= 2:
        px, py = float(pose_xy[0]), float(pose_xy[1])
        nearest = min(range(len(centroids)),
                      key=lambda i: math.hypot(centroids[i][0] - px, centroids[i][1] - py))
        centroids = centroids[nearest:]

    ref = []
    if pose_xy is not None:
        ref.append((float(pose_xy[0]), float(pose_xy[1])))
    ref.extend(centroids)
    return ref, unknown


def build(grid, meta, params):
    """Inflate the occupancy grid once per /grid_path message; reused for every robot."""
    infl = inflate_occupancy(grid, meta["resolution"], params.get("inflation_radius", 0.5))
    return {"infl": infl}


def plan(reference_xy, ctx, meta, params):
    """reference_xy: world (x,y) points [robot pose, centroids...]. Returns (world_path, debug)."""
    infl = ctx["infl"]
    proj_radius = float(_p(params, "projection_radius"))
    res = meta["resolution"]
    warnings: List[str] = []

    # Project each reference point onto the nearest free cell, dropping any whose nearest
    # free cell is farther than projection_radius metres away.
    anchors = []
    for (x, y) in reference_xy:
        orig_cell = world_to_cell(x, y, meta)
        free_cell = project_to_free(infl, orig_cell)
        if free_cell is None:
            warnings.append(f"Waypoint ({x:.2f},{y:.2f}) has no free cell nearby; skipping.")
            continue
        dist_m = math.hypot(
            (free_cell[0] - orig_cell[0]) * res,
            (free_cell[1] - orig_cell[1]) * res,
        )
        if dist_m > proj_radius:
            warnings.append(
                f"Waypoint ({x:.2f},{y:.2f}) nearest free cell is {dist_m:.2f} m away "
                f"(limit {proj_radius:.2f} m); dropping."
            )
            continue
        anchors.append(free_cell)
    if not anchors:
        return [], {"warnings": warnings, "reference_xy": list(reference_xy)}

    # Pairwise A* + per-segment line-of-sight thinning (anchors preserved).
    full_cells = [anchors[0]]
    current = anchors[0]
    for nxt in anchors[1:]:
        seg = astar(infl, current, nxt)
        if seg is None:
            warnings.append(f"No A* path from {current} to {nxt}; skipping waypoint.")
            continue
        seg = simplify_path_los(infl, seg)
        full_cells.extend(seg[1:])      # drop duplicate shared endpoint
        current = nxt

    world_path = [cell_to_world(gx, gy, meta) for (gx, gy) in full_cells]
    debug = {"reference_xy": list(reference_xy), "anchors": anchors,
             "full_cells": full_cells, "warnings": warnings}
    return world_path, debug


def _i(uv):
    return (int(round(uv[0])), int(round(uv[1])))


def save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params, camera=None):
    """Route-specific debug: route_centroids, route_planned, occ_inflated, inflation_overlay."""
    infl = ctx["infl"]
    base = base_bgr
    ref = debug.get("reference_xy", [])
    full_cells = debug.get("full_cells", [])
    anchors = debug.get("anchors", [])
    start_world = debug.get("start_world")

    # Naive route through the reference centroids (orange line, blue dots).
    rc = base.copy()
    ref_px = [_i(world_to_pixel(x, y, camera=camera)) for (x, y) in ref]
    for a, b in zip(ref_px, ref_px[1:]):
        cv2.line(rc, a, b, (0, 165, 255), 2)
    for p in ref_px:
        cv2.circle(rc, p, 8, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(out_dir, "route_centroids.png"), rc)

    # Obstacle-avoiding planned route (green line; red=anchor, yellow=intermediate; magenta=start).
    rp = base.copy()
    ppx = [_i(world_to_pixel(*cell_to_world(gx, gy, meta), camera=camera)) for (gx, gy) in full_cells]
    for a, b in zip(ppx, ppx[1:]):
        cv2.line(rp, a, b, (0, 200, 0), 2)
    anchor_set = {(int(a[0]), int(a[1])) for a in anchors}
    for (cell, p) in zip(full_cells, ppx):
        color = (0, 0, 255) if (int(cell[0]), int(cell[1])) in anchor_set else (0, 255, 255)
        cv2.circle(rp, p, 5, color, -1)
    if start_world is not None:
        cv2.circle(rp, _i(world_to_pixel(start_world[0], start_world[1], camera=camera)), 6, (255, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, "route_planned.png"), rp)

    # Inflated occupancy map (white=free, black=occupied, gray=unknown), reoriented to match image.
    def viz(gmap):
        g2 = np.flipud(gmap.T)
        out = np.full((*g2.shape, 3), 128, np.uint8)
        out[g2 == FREE] = (255, 255, 255)
        out[g2 == OCCUPIED] = (0, 0, 0)
        return out
    cv2.imwrite(os.path.join(out_dir, "occ_inflated.png"), viz(infl))

    # Inflation margin overlay (pixel space): cyan=inflation margin, blue=true obstacle.
    h_img, w_img = base.shape[:2]
    uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
    xs, ys = pixel_to_world(uu, vv, camera=camera)
    gx = np.floor((xs - meta["origin_x"]) / meta["resolution"]).astype(np.int64)
    gy = np.floor((ys - meta["origin_y"]) / meta["resolution"]).astype(np.int64)
    inb = (gx >= 0) & (gx < meta["width"]) & (gy >= 0) & (gy < meta["height"])
    gxc = np.clip(gx, 0, meta["width"] - 1)
    gyc = np.clip(gy, 0, meta["height"] - 1)
    true_occ = inb & (grid[gyc, gxc] != FREE)
    infl_occ = inb & (infl[gyc, gxc] == OCCUPIED)
    margin = infl_occ & ~true_occ
    ov = base.copy()
    ov[margin] = (0.5 * base[margin] + 0.5 * np.array([0, 220, 220])).astype(np.uint8)
    ov[true_occ] = (0.5 * base[true_occ] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "inflation_overlay.png"), ov)
