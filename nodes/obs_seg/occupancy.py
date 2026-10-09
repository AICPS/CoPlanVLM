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
#   EDGE_CLEAR_MARGIN_PX — outer border ring cleared, defined in SOURCE-IMAGE PIXELS. CLIPSeg confidence
#                          is poor at the image margins, and that unreliable band is ~constant in image
#                          pixels regardless of camera mounting height — whereas a fixed METRIC width
#                          maps to very different image regions per camera (0.5 m = ~61 px on the high
#                          gazebo cam vs ~112 px on the lower lab cam). Converted to occupancy cells
#                          per-axis via the image->grid resample ratio (see _px_margin_to_cells).
#   ROBOT_CLEAR_RADIUS   — disk cleared around each robot's own pose (METRES; a physical footprint). A
#                          robot occludes the floor it stands on, so its footprint reads as UNKNOWN.
EDGE_CLEAR_MARGIN_PX = 65
ROBOT_CLEAR_RADIUS   = 0.65

# Post-inflation hard perimeter wall (see _stamp_edge_wall / create_filtered_occupancy_map). The outer
# EDGE_WALL_MARGIN-metre ring is forced OCCUPIED AFTER inflation, so the frame border is a crisp
# fixed-width wall with no inward inflation halo. This is a physical standoff distance, so it stays in
# METRES (unlike the pixel-based clear); kept separate so the two tune independently.
EDGE_WALL_MARGIN     = 0.25

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

    # img_width/img_height (the source pixel dims) let the edge overrides define their margin in image
    # pixels instead of metres — the grid is an axis-aligned resample of the image, so a pixel border
    # converts to a per-axis cell band via grid_cells/img_pixels (see _px_margin_to_cells).
    meta = dict(resolution=float(resolution), origin_x=x_min, origin_y=y_min,
                width=width, height=height, img_width=int(w_img), img_height=int(h_img))
    return grid, meta


def _px_margin_to_cells(margin_px: float, meta: dict) -> Tuple[int, int]:
    """A source-image pixel margin -> (k_col, k_row) occupancy-cell border thickness per axis.

    The occupancy grid is an axis-aligned resample of the source image (mask_to_occupancy scatters every
    image pixel into a world cell), so the outer ring of the grid corresponds to the image border and
    cells-per-pixel is grid_cells/img_pixels along each axis. Defining a margin in image pixels (not
    metres) keeps the cleared band consistent across cameras with different mounting heights: CLIPSeg's
    unreliable border is ~constant in pixels, while its metric width is not. Requires img_width/img_height
    in meta (added by mask_to_occupancy); if absent, returns (0, 0) so no band is applied.
    """
    iw, ih = meta.get("img_width"), meta.get("img_height")
    if not iw or not ih:
        return 0, 0
    k_col = int(round(margin_px * meta["width"] / iw))
    k_row = int(round(margin_px * meta["height"] / ih))
    return k_col, k_row


def _apply_free_overrides(grid: np.ndarray, meta: dict, world_poses=None) -> np.ndarray:
    """Promote UNKNOWN cells to FREE near the map edges and around robot footprints.

    Internal helper for create_filtered_occupancy_map (the public entry point) — applied to the raw
    grid BEFORE inflation. Two overrides, both of which ONLY affect cells that are
    currently UNKNOWN (a genuine OCCUPIED cell is never cleared):

      * Edge ring: every cell within EDGE_CLEAR_MARGIN_PX SOURCE-IMAGE PIXELS of the map border (converted
        to a per-axis cell band). CLIPSeg confidence is poor at the image margins, so the outer ring is
        spuriously UNKNOWN; a pixel width keeps this consistent across camera mounting heights.
      * Robot disks: every cell within ROBOT_CLEAR_RADIUS metres of a robot's world pose. A robot
        occludes the floor it stands on, so its own footprint reads as UNKNOWN.

    Args:
        grid:        int8 (H, W) occupancy grid (FREE/OCCUPIED/UNKNOWN); NOT modified.
        meta:        grid meta from mask_to_occupancy (resolution, origin_*, width/height, img_*).
        world_poses: iterable of (x, y) robot positions in the WORLD frame, or None. Callers convert
                     from NED (ned_to_world) before passing. None/empty -> edge clearing only.

    Returns a new grid (copy) with the overrides applied.
    """
    out = grid.copy()
    height, width = out.shape
    res = meta["resolution"]

    # Edge ring: outer border of EDGE_CLEAR_MARGIN_PX image pixels, as per-axis cell bands.
    k_col, k_row = _px_margin_to_cells(EDGE_CLEAR_MARGIN_PX, meta)
    if k_col > 0 or k_row > 0:
        edge = np.zeros((height, width), dtype=bool)
        if k_row > 0:
            edge[:k_row, :] = edge[-k_row:, :] = True
        if k_col > 0:
            edge[:, :k_col] = edge[:, -k_col:] = True
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


def _stamp_edge_wall(grid: np.ndarray, meta: dict) -> None:
    """Force the outer EDGE_WALL_MARGIN-metre ring OCCUPIED, in place.

    Applied AFTER inflation so the perimeter is a crisp fixed-width wall with no inward inflation halo:
    the edge UNKNOWN is first cleared to FREE before inflation (see _apply_free_overrides), then this
    reasserts a hard boundary of exactly EDGE_WALL_MARGIN metres. Unlike the UNKNOWN-only edge clear, this
    overrides whatever is in the ring (FREE/UNKNOWN/OCCUPIED alike), so the planner keeps the robot off
    the frame border.
    """
    k = int(round(EDGE_WALL_MARGIN / meta["resolution"]))
    if k > 0:
        grid[:k, :] = grid[-k:, :] = grid[:, :k] = grid[:, -k:] = OCCUPIED


def create_filtered_occupancy_map(grid: np.ndarray, meta: dict, world_poses=None,
                                  return_cleared: bool = False):
    """Turn a raw occupancy grid into the final planning grid: traversability overrides, inflate, wall.

    Single entry point (used by exec and the offline harnesses) so the "clear edges/footprints, inflate
    obstacles, then stamp the hard perimeter wall" sequence lives in exactly one place. `grid`/`meta` come
    from mask_to_occupancy; `world_poses` is an iterable of world-frame (x, y) robot positions, or None
    (callers convert from NED). The input grid is not modified — pass it separately if you want to keep
    the raw map.

    The outer EDGE_WALL_MARGIN-metre ring is forced OCCUPIED AFTER inflation, giving a crisp fixed-width
    border with no inward inflation halo (the edge UNKNOWN was cleared to FREE before inflation). It is
    stamped into `inflated` ONLY, deliberately not into `cleared`: the wall is a synthetic standoff this
    code invents, not something the segmenter detected, so it belongs in the debug overlays' margin
    layer (yellow) alongside the inflation halo rather than in the detected-obstacle layer (red).
    Because it is absent from `cleared` but present in `inflated`, render_inflation_overlay classifies
    it as margin with no special-casing. Where the segmenter DID find a real obstacle overlapping the
    border, that cell stays OCCUPIED in `cleared` (_apply_free_overrides never clears a true OCCUPIED)
    and correctly renders red.

    Returns the inflated planning grid. If `return_cleared` is True, returns
    ``(inflated, cleared)`` where `cleared` is the post-override, pre-inflation grid used as the red
    layer in debug visualizations. `cleared` is a RENDERING input only — never plan against it; the
    planning grid, perimeter wall included, is `inflated`.
    """
    cleared = _apply_free_overrides(grid, meta, world_poses)
    inflated = inflate_occupancy(cleared, meta["resolution"])
    # Hard perimeter wall, stamped post-inflation so it stays a crisp EDGE_WALL_MARGIN-metre border.
    # Planning grid only — see the docstring for why `cleared` deliberately does not get it.
    _stamp_edge_wall(inflated, meta)
    return (inflated, cleared) if return_cleared else inflated


def render_inflation_overlay(base_bgr: np.ndarray, occ: np.ndarray, infl: np.ndarray,
                             meta: dict, camera: str) -> np.ndarray:
    """Return a BGR debug overlay of the occupancy drawn on the overhead photo.

    Single source of truth for the inflation overlay, shared by astar_proj.save_debug (hence the live
    Path Translator node and the offline harnesses). Two translucent layers:
        red    — cells blocked in `occ`. Pass the post-override, pre-inflation grid (the `cleared`
                 grid from create_filtered_occupancy_map) so this layer shows the true obstacles fed
                 to inflation and NOT regions that were cleared to free before inflating.
        yellow — margin: OCCUPIED in `infl` but not blocked in `occ`. That is the inflation halo AND
                 the EDGE_WALL_MARGIN perimeter wall, both of which are buffers this code adds rather
                 than things the segmenter saw (see create_filtered_occupancy_map).

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
