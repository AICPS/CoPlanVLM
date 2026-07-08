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

2. `inflate_occupancy` — grow obstacles (and unknown, treated as blocked) by a
   metric radius (default 0.25 m; TurtleBot4 radius ~0.17 m + margin) so a planner
   can treat the robot as a point.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from coord_transform import pixel_to_world

from . import FREE, OCCUPIED, UNKNOWN

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


def inflate_occupancy(grid: np.ndarray, resolution: float,
                      inflation_radius: float = 0.25) -> np.ndarray:
    """Grow obstacles (and unknown, treated as blocked) by `inflation_radius` metres.

    Exact metric distance transform: any cell within `inflation_radius` of a blocked
    cell becomes OCCUPIED. The input grid is not modified. Default 0.25 m (TurtleBot4
    radius ~0.17 m + margin).
    """
    blocked = (grid == OCCUPIED) | (grid == UNKNOWN)
    # distanceTransform measures distance (px) from each nonzero pixel to the nearest
    # zero pixel; set blocked=0 so free cells get their distance to the nearest obstacle.
    free_img = np.where(blocked, 0, 1).astype(np.uint8)
    dist_m = cv2.distanceTransform(free_img, cv2.DIST_L2, 5) * resolution

    inflated = grid.copy()
    inflated[dist_m < inflation_radius] = OCCUPIED
    return inflated
