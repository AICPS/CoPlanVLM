#!/usr/bin/env python3
"""
Occupancy-map construction from a pixel-space traversability label map (ROS-free).
==================================================================================

Two steps:

1. `mask_to_occupancy` — resample a per-pixel label map (FREE/OCCUPIED/UNKNOWN,
   from the segmenter) into a metric occupancy grid in the robot's world frame,
   using the SHARED pixel->world transform (coord_transform.pixel_to_world) — the
   exact same function the path translator uses, so the two can never disagree.

   Aggregation is CONSERVATIVE: each cell is blocked if ANY of its pixels is an
   obstacle, else unknown if any pixel is unknown, else free. Cells with no pixel
   coverage are unknown.

2. `inflate_occupancy` — grow obstacles (and unknown, treated as blocked) by
   `INFLATION_RADIUS` metres so a planner can treat the robot as a point.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from coord_transform import pixel_to_world

from . import FREE, OCCUPIED, UNKNOWN

# Single source of truth for the pipeline's occupancy-grid parameters. These are fixed properties
# of the pipeline (not runtime-tunable): exec uses them directly and the offline test harnesses
# import them, so each value lives in exactly one place.
#   RESOLUTION       — metric size of one occupancy cell (m/cell).
#   INFLATION_RADIUS — obstacle dilation margin (m). TurtleBot4 radius ~0.17 m + margin.
# inflate_occupancy always uses INFLATION_RADIUS (no per-call override), so the inflated grid can
# never disagree between producers/consumers.
RESOLUTION       = 0.05
INFLATION_RADIUS = 0.45

# Pre-inflation traversability overrides (see create_filtered_occupancy_map). Both promote UNKNOWN to
# FREE — they never override a genuine OCCUPIED cell — and are applied before inflation.
#   EDGE_CLEAR_MARGIN  — outer border ring cleared; CLIPSeg confidence is poor at the image margins,
#                        so the frame's outer ring is spuriously UNKNOWN (untraversable).
#   ROBOT_CLEAR_RADIUS — disk cleared around each robot's own pose; a robot occludes the floor it
#                        stands on in the overhead frame, so its footprint is UNKNOWN (self-blocking).
EDGE_CLEAR_MARGIN  = 0.5
ROBOT_CLEAR_RADIUS = 0.5

# internal priority codes for conservative aggregation (higher wins)
_C_NONE, _C_FREE, _C_UNKNOWN, _C_OBSTACLE = -1, 0, 1, 2


def world_to_cell(x, y, meta) -> Tuple[int, int]:
    """World (x, y) metres -> occupancy-grid cell index (gx=col, gy=row).

    `meta` is the dict returned by mask_to_occupancy (resolution, origin_x, origin_y,
    width, height). Inverse of the cell layout used to build the grid.
    """
    gx = int(np.floor((x - meta["origin_x"]) / meta["resolution"]))
    gy = int(np.floor((y - meta["origin_y"]) / meta["resolution"]))
    return gx, gy


def cell_to_world(gx, gy, meta) -> Tuple[float, float]:
    """Occupancy-grid cell (gx=col, gy=row) -> world (x, y) metres at the cell center."""
    x = meta["origin_x"] + (gx + 0.5) * meta["resolution"]
    y = meta["origin_y"] + (gy + 0.5) * meta["resolution"]
    return x, y


def mask_to_occupancy(pix_labels: np.ndarray,
                      resolution: float,
                      camera: str = None) -> Tuple[np.ndarray, dict]:
    """Resample a pixel label map into a metric OccupancyGrid array.

    pix_labels : int8 (H_img, W_img) with values FREE / OCCUPIED / UNKNOWN. `camera` selects
    the pixel->world calibration (e.g. "gazebo", "lab_test") — must be provided explicitly.
    Returns (grid int8 (H, W), meta) where grid[gy, gx] uses FREE/OCCUPIED/UNKNOWN and
    meta = {resolution, origin_x, origin_y, width, height}. grid is row-major in (gy, gx);
    flatten C-order to fill nav_msgs/OccupancyGrid.data.
    """
    h_img, w_img = pix_labels.shape

    # world extent from the four image corners (transform is axis-aligned)
    cu = np.array([0, w_img - 1, 0, w_img - 1], dtype=float)
    cv = np.array([0, 0, h_img - 1, h_img - 1], dtype=float)
    cx, cy = pixel_to_world(cu, cv, camera=camera)
    x_min, x_max = float(cx.min()), float(cx.max())
    y_min, y_max = float(cy.min()), float(cy.max())

    width = max(1, int(np.ceil((x_max - x_min) / resolution)))
    height = max(1, int(np.ceil((y_max - y_min) / resolution)))

    # world coord of every pixel, then its target cell index
    uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
    xs, ys = pixel_to_world(uu, vv, camera=camera)
    gx = np.floor((xs - x_min) / resolution).astype(np.int64)
    gy = np.floor((ys - y_min) / resolution).astype(np.int64)
    in_bounds = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)

    # map pixel labels -> priority codes (obstacle beats unknown beats free)
    pix_code = np.full(pix_labels.shape, _C_FREE, dtype=np.int16)
    pix_code[pix_labels == UNKNOWN] = _C_UNKNOWN
    pix_code[pix_labels == OCCUPIED] = _C_OBSTACLE

    # conservative scatter-max into cells
    cell_code = np.full(height * width, _C_NONE, dtype=np.int16)
    flat = gy[in_bounds] * width + gx[in_bounds]
    np.maximum.at(cell_code, flat, pix_code[in_bounds])
    cell_code = cell_code.reshape(height, width)

    grid = np.full((height, width), UNKNOWN, dtype=np.int8)
    grid[cell_code == _C_FREE] = FREE
    grid[cell_code == _C_OBSTACLE] = OCCUPIED
    # _C_UNKNOWN and _C_NONE remain UNKNOWN

    meta = dict(resolution=float(resolution), origin_x=x_min, origin_y=y_min,
                width=width, height=height)
    return grid, meta


def _apply_free_overrides(grid: np.ndarray, meta: dict, world_poses=None) -> np.ndarray:
    """Promote UNKNOWN cells to FREE near the map edges and around robot footprints.

    Internal helper for create_filtered_occupancy_map (the public entry point) — applied to the raw
    grid BEFORE inflation. Two overrides, both of which ONLY affect cells that are
    currently UNKNOWN (a genuine OCCUPIED cell is never cleared):

      * Edge ring: every cell within EDGE_CLEAR_MARGIN metres of the map border. CLIPSeg confidence
        is poor at the image margins, so the outer ring is spuriously UNKNOWN.
      * Robot disks: every cell within ROBOT_CLEAR_RADIUS metres of a robot's world pose. A robot
        occludes the floor it stands on, so its own footprint reads as UNKNOWN.

    Args:
        grid:        int8 (H, W) occupancy grid (FREE/OCCUPIED/UNKNOWN); NOT modified.
        meta:        grid meta from mask_to_occupancy (resolution, origin_x, origin_y, width, height).
        world_poses: iterable of (x, y) robot positions in the WORLD frame, or None. Callers convert
                     from NED (ned_to_world) before passing. None/empty -> edge clearing only.

    Returns a new grid (copy) with the overrides applied.
    """
    out = grid.copy()
    height, width = out.shape
    res = meta["resolution"]

    # Edge ring: outer k-cell border, where k = EDGE_CLEAR_MARGIN in cells.
    k = int(round(EDGE_CLEAR_MARGIN / res))
    if k > 0:
        edge = np.zeros((height, width), dtype=bool)
        edge[:k, :] = edge[-k:, :] = edge[:, :k] = edge[:, -k:] = True
        out[edge & (out == UNKNOWN)] = FREE

    # Robot disks: ROBOT_CLEAR_RADIUS around each pose's cell.
    if world_poses:
        yy, xx = np.ogrid[0:height, 0:width]
        rc2 = (ROBOT_CLEAR_RADIUS / res) ** 2
        for (x, y) in world_poses:
            gx, gy = world_to_cell(x, y, meta)
            if not (0 <= gx < width and 0 <= gy < height):
                continue
            disk = ((xx - gx) ** 2 + (yy - gy) ** 2) <= rc2
            out[disk & (out == UNKNOWN)] = FREE

    return out


def inflate_occupancy(grid: np.ndarray, resolution: float) -> np.ndarray:
    """Grow obstacles (and unknown, treated as blocked) by `INFLATION_RADIUS` metres.

    Exact metric distance transform: any cell within `INFLATION_RADIUS` of a blocked
    cell becomes OCCUPIED. The input grid is not modified. The radius is fixed (the module
    constant, not a parameter) so every caller inflates to the same margin.
    """
    blocked = (grid == OCCUPIED) | (grid == UNKNOWN)
    # distanceTransform measures distance (px) from each nonzero pixel to the nearest
    # zero pixel; set blocked=0 so free cells get their distance to the nearest obstacle.
    free_img = np.where(blocked, 0, 1).astype(np.uint8)
    dist_m = cv2.distanceTransform(free_img, cv2.DIST_L2, 5) * resolution

    inflated = grid.copy()
    inflated[dist_m < INFLATION_RADIUS] = OCCUPIED
    return inflated


def create_filtered_occupancy_map(grid: np.ndarray, meta: dict, world_poses=None,
                                  return_cleared: bool = False):
    """Turn a raw occupancy grid into the final planning grid: traversability overrides, then inflate.

    Single entry point (used by exec and the offline harnesses) so the "clear edges/footprints, then
    inflate obstacles" sequence lives in exactly one place. `grid`/`meta` come from mask_to_occupancy;
    `world_poses` is an iterable of world-frame (x, y) robot positions, or None (callers convert from
    NED). The input grid is not modified — pass it separately if you want to keep the raw map.

    Returns the inflated planning grid. If `return_cleared` is True, returns
    ``(inflated, cleared)`` where `cleared` is the post-override, pre-inflation grid (useful for
    debug visualizations that want to show what the overrides changed before inflation).
    """
    cleared = _apply_free_overrides(grid, meta, world_poses)
    inflated = inflate_occupancy(cleared, meta["resolution"])
    return (inflated, cleared) if return_cleared else inflated


def render_inflation_overlay(base_bgr: np.ndarray, occ: np.ndarray, infl: np.ndarray,
                             meta: dict, camera: str) -> np.ndarray:
    """Return a BGR debug overlay of the occupancy drawn on the overhead photo.

    Single source of truth for the inflation overlay, shared by astar_proj.save_debug (hence the live
    Path Translator node and test_pipeline) and test_map_gen. Two translucent layers:
        red    — cells blocked in `occ`. Pass the post-override, pre-inflation grid (the `cleared`
                 grid from create_filtered_occupancy_map) so this layer shows the true obstacles fed
                 to inflation and NOT regions that were cleared to free before inflating.
        yellow — inflation-only margin: OCCUPIED in `infl` but not blocked in `occ`.

    `base_bgr` is an (H, W, 3) BGR image; the returned image is a copy (the input is not modified).
    Every image pixel is mapped to its grid cell via the shared pixel->world transform, so the overlay
    lines up with the occupancy grid regardless of resolution.
    """
    h_img, w_img = base_bgr.shape[:2]
    uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
    xs, ys = pixel_to_world(uu, vv, camera=camera)
    gx = np.floor((xs - meta["origin_x"]) / meta["resolution"]).astype(np.int64)
    gy = np.floor((ys - meta["origin_y"]) / meta["resolution"]).astype(np.int64)
    inb = (gx >= 0) & (gx < meta["width"]) & (gy >= 0) & (gy < meta["height"])
    gxc = np.clip(gx, 0, meta["width"] - 1)
    gyc = np.clip(gy, 0, meta["height"] - 1)
    blocked = inb & (occ[gyc, gxc] != FREE)
    infl_occ = inb & (infl[gyc, gxc] == OCCUPIED)
    margin = infl_occ & ~blocked
    ov = base_bgr.copy()
    ov[margin]  = (0.5 * base_bgr[margin]  + 0.5 * np.array([0, 220, 220])).astype(np.uint8)
    ov[blocked] = (0.5 * base_bgr[blocked] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
    return ov
