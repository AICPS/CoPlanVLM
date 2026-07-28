"""Per-task-type overhead-map builders for the Executive node.

The classifier tags each instruction with a controller; ``prompt_gen.generate_prompt`` then renders
the overhead overlay best suited to that behavior via the ``render_*`` functions here (paired to a
``map_overlay_type``). Each renderer is a pure PIL-in/PIL-out function; ``_node_to_pil`` and ``_to_b64``
extract the live camera frame off the ``ExecutiveApiNode`` (``node.camera_image``, ``node.bridge``,
``node.camera_matrix``, ``node.dist_coeffs``). Static map assets are loaded and cached here.

CLIPSeg lives here (not in translator_node.py). ``run_segmentation(node)`` is called by exec once per
replan tick; it runs CLIPSeg on the current camera frame, stores ``node.pix_labels``, and writes
the result to ``_OCC_FILE`` so that translator_node.py can read it instead of running its own instance.
This keeps the model in one process and avoids a ROS topic for the occupancy data.
"""

from __future__ import annotations

import base64
import csv
import io
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from ament_index_python.packages import get_package_share_directory

from coord_transform import pixel_to_world, world_to_pixel, ned_to_world
from obs_seg import FREE
from obs_seg.occupancy import (mask_to_occupancy, create_filtered_occupancy_map,
                               world_to_cell, RESOLUTION)

# ── Shared occupancy file ──────────────────────────────────────────────────────
# exec writes the occupancy snapshot here; translate reads it instead of running CLIPSeg or
# rebuilding/inflating the grid itself. The .npz bundles the raw pixel labels (for debug), the raw
# metric occupancy grid (for debug), the INFLATED planning grid, and the grid meta.
# get_package_share_directory returns <ws>/install/coplan_vlm/share/coplan_vlm/;
# 4 levels up is the colcon workspace root (standard isolated-install layout).
_OCC_FILE = os.path.normpath(os.path.join(
    get_package_share_directory('coplan_vlm'), '..', '..', '..', '..',
    'debug', 'coplan_vlm_occupancy.npz'))

# ── Grid / set-of-marks tuning ────────────────────────────────────────────────
_N_COLS        = 14
_N_ROWS        = 8
_GRID_COLOR    = (0, 0, 255, 255)   # blue, ~70 % opacity
_GRID_WIDTH    = 3                  # line thickness in pixels
_DOT_RADIUS    = 10                 # nav2point mark dot radius
_LABEL_SIZE    = 28                 # pt; grid cell labels and nav2point marks
_ROBOT_RADIUS  = 22                 # hollow circle radius for robot markers
_ROBOT_COLORS  = [                  # per-robot colors (RGB), cycled by index
    (220,  20, 220),  # magenta  – robot 0
    (  0, 200, 200),  # cyan     – robot 1
    (255, 140,   0),  # orange   – robot 2
    (  0, 200,  80),  # green    – robot 3
]
# Human-readable names for _ROBOT_COLORS, SAME ORDER. prompt_gen tells the VLM which colour marker
# belongs to which robot, so these must stay index-aligned with the tuples above — otherwise the
# prompt would describe a colour the image does not draw.
ROBOT_COLOR_NAMES = ("magenta", "light blue", "orange", "green")
_SEG_PROMPTS   = ["the floor"]
_SEG_THRESHOLD = 0.48

# Label font (loaded once): DejaVuSans at _LABEL_SIZE, falling back to PIL's default if unavailable.
try:
    _LABEL_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                     size=_LABEL_SIZE)
except OSError:
    _LABEL_FONT = ImageFont.load_default()

# Occupancy-grid parameters (resolution, inflation radius) are the single source of truth in
# obs_seg.occupancy (RESOLUTION, INFLATION_RADIUS) — not duplicated here.


@lru_cache(maxsize=1)
def _get_segmenter():
    """Load TraversabilitySegmenter (CLIPSeg) once on first call; deferred import."""
    from obs_seg.segmenter import TraversabilitySegmenter
    return TraversabilitySegmenter()


@lru_cache(maxsize=1)
def _load_grid_centers() -> list[tuple[str, float, float]]:
    """Return [(label, u, v), ...] pixel coords from grid_cell_centers.csv (cached, read once)."""
    pkg_dir = get_package_share_directory('coplan_vlm')
    csv_path = os.path.join(pkg_dir, 'config', 'grid_cell_centers.csv')
    with open(csv_path, newline='') as f:
        return [(r['cell'], float(r['center_x']), float(r['center_y']))
                for r in csv.DictReader(f)]


# ── Segmentation (runs in exec, result shared with translate via file) ─────────

def run_segmentation(node) -> None:
    """Run CLIPSeg on the current camera frame and write the occupancy snapshot to _OCC_FILE.

    Called by exec._run_plan before map builder dispatch so every task type produces a fresh
    occupancy snapshot. Inflation happens HERE (once, upstream) via obs_seg.occupancy.inflate_occupancy
    so the planner modules never inflate; translator_node.py reads the pre-inflated grid directly.
    Stores node.pix_labels (still used by the nav2point map builder) and saves an .npz bundling the
    camera frame, the raw pixel labels, the raw metric grid, the inflated planning grid, and the meta.
    No-op if no camera image is available yet.

    The snapshot carries `frame_bgr` — the undistorted frame everything here was computed from — so
    translator_node.py renders its debug figures on the SAME frame the VLM saw. Without it translate would
    fall back to its own latest camera image, which is a good ten to forty seconds newer (classifier
    call + CLIPSeg + planner call all happen in between) and shows the robots already moved.
    """
    if node.camera_image is None:
        return
    cv_img = node.bridge.imgmsg_to_cv2(node.camera_image, desired_encoding='rgba8')
    if node.camera_matrix is not None and node.dist_coeffs is not None:
        cv_img = cv2.undistort(cv_img, node.camera_matrix, node.dist_coeffs)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    pix_labels, _ = _get_segmenter().classify(rgb, _SEG_PROMPTS, [], _SEG_THRESHOLD)
    node.pix_labels = pix_labels

    grid, meta = mask_to_occupancy(pix_labels, RESOLUTION, camera=node.camera_name)
    # Clear the edge ring + each robot's own footprint (at the frame's captured poses), then inflate.
    world_poses = [ned_to_world(x, y)
                   for (x, y) in (node.camera_image_poses or {}).values() if (x, y) is not None]
    infl, cleared = create_filtered_occupancy_map(grid, meta, world_poses, return_cleared=True)
    # Cache the inflated grid + meta on the node so the nav2point overlay filters against the same
    # single inflated grid (rather than re-inflating from pix_labels).
    node.occ_grid = infl
    node.occ_meta = meta

    # npz: frame_bgr = the undistorted frame all of the above was computed from (translate draws its
    # debug figures on it); grid = raw CLIPSeg map; cleared = post-override, pre-inflation (red layer
    # for overlays); infl = final planning grid (overrides + inflation), what the planner consumes.
    os.makedirs(os.path.dirname(_OCC_FILE), exist_ok=True)
    np.savez(_OCC_FILE, frame_bgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
             pix_labels=pix_labels, grid=grid, cleared=cleared, infl=infl,
             resolution=meta["resolution"], origin_x=meta["origin_x"],
             origin_y=meta["origin_y"], width=meta["width"], height=meta["height"])


# ── Pure render functions (PIL in → PIL out; no ROS types) ────────────────────

def _overlaps(box: tuple, occupied: list[tuple]) -> bool:
    """Return True if box (x0,y0,x1,y1) intersects any rect in occupied."""
    x0, y0, x1, y1 = box
    return any(x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0
               for ox0, oy0, ox1, oy1 in occupied)


def _draw_robot_markers(draw: ImageDraw.ImageDraw,
                        robot_poses: dict | None,
                        camera: str,
                        font,
                        occupied: list[tuple] | None = None) -> None:
    """Draw a hollow circle + gray-box name label for each robot with a known pose.

    `occupied` is a list of (x0,y0,x1,y1) bounding boxes already drawn on the image.
    The label is placed in the first of 8 candidate positions (right, left, above, below,
    four diagonals) that does not collide with any occupied box.
    """
    if not robot_poses:
        return
    occ: list[tuple] = list(occupied) if occupied else []
    r = _ROBOT_RADIUS
    pad = 2
    for i, (name, ned_xy) in enumerate(robot_poses.items()):
        if ned_xy is None:
            continue
        wx, wy = ned_to_world(ned_xy[0], ned_xy[1])
        u, v = world_to_pixel(wx, wy, camera=camera)
        u, v = int(round(u)), int(round(v))
        color = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
        draw.ellipse((u - r, v - r, u + r, v + r), outline=color, width=3)

        # Measure text so candidate offsets are exact.
        tb = draw.textbbox((0, 0), name, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # 8 candidate positions: right, left, above, below, then four diagonals.
        candidates = [
            (u + r + 4,      v - th // 2),
            (u - tw - r - 4, v - th // 2),
            (u - tw // 2,    v - r - th - 4),
            (u - tw // 2,    v + r + 4),
            (u + r + 4,      v - r - th),
            (u + r + 4,      v + r),
            (u - tw - r - 4, v - r - th),
            (u - tw - r - 4, v + r),
        ]
        tx, ty = candidates[0]
        for cx, cy in candidates:
            cb = draw.textbbox((cx, cy), name, font=font)
            cbox = (cb[0] - pad, cb[1] - pad, cb[2] + pad, cb[3] + pad)
            if not _overlaps(cbox, occ):
                tx, ty = cx, cy
                break

        bbox = draw.textbbox((tx, ty), name, font=font)
        box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(box, fill=(220, 220, 220))
        draw.text((tx, ty), name, fill=color, font=font)
        # Reserve both the circle and label so subsequent robots avoid them.
        occ.append((u - r, v - r, u + r, v + r))
        occ.append(box)


def render_battleship_map(img_rgba: PILImage.Image,
                          robot_poses: dict | None = None,
                          camera: str | None = None) -> PILImage.Image:
    """Draw a labeled battleship grid over an RGBA image.

    Divides the image into _N_COLS × _N_ROWS cells with blue grid lines, then places
    a gray-backgrounded alphanumeric label (e.g. 'A1') in the top-left of each cell.
    """
    w, h = img_rgba.size
    cell_w = w / _N_COLS
    cell_h = h / _N_ROWS

    # Draw grid lines on a transparent layer so they blend with the image.
    grid_layer = PILImage.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    for i in range(1, _N_COLS):
        x = int(round(i * cell_w))
        gd.line([(x, 0), (x, h - 1)], fill=_GRID_COLOR, width=_GRID_WIDTH)
    for j in range(1, _N_ROWS):
        y = int(round(j * cell_h))
        gd.line([(0, y), (w - 1, y)], fill=_GRID_COLOR, width=_GRID_WIDTH)
    gd.rectangle([(0, 0), (w - 1, h - 1)], outline=_GRID_COLOR, width=_GRID_WIDTH)
    result = PILImage.alpha_composite(img_rgba, grid_layer)

    font = _LABEL_FONT
    draw = ImageDraw.Draw(result)
    pad = 2
    occupied: list[tuple] = []
    for col_idx in range(_N_COLS):
        for row_idx in range(_N_ROWS):
            label = chr(ord('A') + col_idx) + str(row_idx + 1)
            tx = int(col_idx * cell_w) + 4
            ty = int(row_idx * cell_h) + 4
            bbox = draw.textbbox((tx, ty), label, font=font)
            box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
            draw.rectangle(box, fill=(220, 220, 220))
            draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
            occupied.append(box)
    if robot_poses and camera is not None:
        _draw_robot_markers(draw, robot_poses, camera, font, occupied=occupied)
    return result


def _render_marks(img_rgba: PILImage.Image, centers: list, occ_grid: np.ndarray | None,
                  occ_meta: dict | None, camera: str,
                  robot_poses: dict | None = None) -> PILImage.Image:
    """Draw labeled set-of-marks dots for `centers` [(label, u, v), …] on an RGBA image.

    `occ_grid`/`occ_meta` are the pre-inflated planning grid + meta (built once upstream in
    run_segmentation); marks are filtered directly against that single inflated grid — this never
    inflates. When occ_grid is None, all provided marks are shown (no obstacle filtering).
    """
    if occ_grid is not None:
        free_cells = []
        for label, u, v in centers:
            wx, wy = pixel_to_world(u, v, camera=camera)
            gx, gy = world_to_cell(wx, wy, occ_meta)
            if (0 <= gx < occ_meta['width'] and 0 <= gy < occ_meta['height']
                    and occ_grid[gy, gx] == FREE):
                free_cells.append((label, u, v))
    else:
        free_cells = list(centers)

    font = _LABEL_FONT
    result = img_rgba.copy()
    draw = ImageDraw.Draw(result)
    r = _DOT_RADIUS
    pad = 2
    occupied: list[tuple] = []
    for label, u, v in free_cells:
        dot_box = (u - r, v - r, u + r, v + r)
        draw.ellipse(dot_box, fill=(0, 0, 255))
        occupied.append(dot_box)
        tx, ty = u + r + 3, v - _LABEL_SIZE // 2
        bbox = draw.textbbox((tx, ty), label, font=font)
        box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(box, fill=(220, 220, 220))
        draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
        occupied.append(box)
    _draw_robot_markers(draw, robot_poses, camera, font, occupied=occupied)
    return result


def render_grid_points_map(img_rgba: PILImage.Image, occ_grid: np.ndarray | None,
                           occ_meta: dict | None, camera: str,
                           robot_poses: dict | None = None) -> PILImage.Image:
    """Draw a blue set-of-marks dot at EVERY grid center (all 112 cells); no obstacle filtering.

    ``occ_grid``/``occ_meta`` are accepted for a uniform renderer interface but ignored here — every
    grid point is marked regardless of occupancy (passing occ_grid=None skips the free-cell filter).
    """
    return _render_marks(img_rgba, _load_grid_centers(), None, None, camera, robot_poses)


def _cell_is_free(u: float, v: float, occ_grid: np.ndarray | None,
                  occ_meta: dict | None, camera: str) -> bool:
    """Is the grid center at pixel (u, v) FREE in the inflated grid? None grid -> treated as free.

    Non-free = occupied / unknown / out-of-bounds of the inflated planning grid. Shared by the marked
    overlay and the textual grid so their blocked/open classification can never disagree.
    """
    if occ_grid is None:
        return True
    wx, wy = pixel_to_world(u, v, camera=camera)
    gx, gy = world_to_cell(wx, wy, occ_meta)
    return (0 <= gx < occ_meta['width'] and 0 <= gy < occ_meta['height']
            and occ_grid[gy, gx] == FREE)


def render_obstacle_marked_map(img_rgba: PILImage.Image, occ_grid: np.ndarray | None,
                               occ_meta: dict | None, camera: str,
                               robot_poses: dict | None = None) -> PILImage.Image:
    """Draw a labeled mark at EVERY grid center, distinguishing free vs obstacle cells.

    Unlike render_grid_points_map (which hides obstacle cells), this marks them: FREE cells get a blue
    dot + blue label, non-free cells (occupied / unknown / out-of-bounds of the inflated grid) get a
    red X + red label. Both keep their alphanumeric label + light-gray text box, so the VLM sees where
    obstacles / objects of interest are. When occ_grid is None, every cell is treated as free.
    """
    font = _LABEL_FONT
    result = img_rgba.copy()
    draw = ImageDraw.Draw(result)
    r = _DOT_RADIUS
    pad = 2
    occupied: list[tuple] = []
    for label, u, v in _load_grid_centers():
        is_free = _cell_is_free(u, v, occ_grid, occ_meta, camera)
        color = (0, 0, 255) if is_free else (255, 0, 0)   # blue free, red obstacle (RGB)
        if is_free:
            draw.ellipse((u - r, v - r, u + r, v + r), fill=color)
        else:
            draw.line([(u - r, v - r), (u + r, v + r)], fill=color, width=3)
            draw.line([(u - r, v + r), (u + r, v - r)], fill=color, width=3)
        occupied.append((u - r, v - r, u + r, v + r))
        tx, ty = u + r + 3, v - _LABEL_SIZE // 2
        bbox = draw.textbbox((tx, ty), label, font=font)
        box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(box, fill=(220, 220, 220))
        draw.text((tx, ty), label, fill=color, font=font)
        occupied.append(box)
    _draw_robot_markers(draw, robot_poses, camera, font, occupied=occupied)
    return result


def grid_status_text(occ_grid: np.ndarray | None, occ_meta: dict | None, camera: str) -> str:
    """Return a 2D ASCII grid of cell status: '.' = open (FREE), 'X' = blocked (non-free).

    Rows 1-8 top->bottom, columns A-N left->right — matching the image and the occupancy classification
    used by render_obstacle_marked_map (via _cell_is_free), so the visual marks and this text agree.
    Intended to be embedded in the marked-text / CoT prompts.
    """
    centers = {label: (u, v) for label, u, v in _load_grid_centers()}
    header = "     " + " ".join(chr(ord('A') + c) for c in range(_N_COLS))
    lines = [header]
    for row in range(1, _N_ROWS + 1):
        cells = []
        for c in range(_N_COLS):
            label = f"{chr(ord('A') + c)}{row}"
            uv = centers.get(label)
            free = _cell_is_free(uv[0], uv[1], occ_grid, occ_meta, camera) if uv else False
            cells.append("." if free else "X")
        lines.append(f"  {row}  " + " ".join(cells))
    return "\n".join(lines)


def blocked_cell_labels(occ_grid: np.ndarray | None, occ_meta: dict | None, camera: str) -> list[str]:
    """Return the alphanumeric labels of grid cells marked NOT passable (red X).

    Uses the same _cell_is_free classification as render_obstacle_marked_map and grid_status_text, so
    this list, the image's red X marks, and the ascii grid can never disagree. Order follows
    _load_grid_centers() (CSV order). occ_grid=None -> every cell free -> empty list.
    """
    return [label for label, u, v in _load_grid_centers()
            if not _cell_is_free(u, v, occ_grid, occ_meta, camera)]


def free_cell_labels(occ_grid: np.ndarray | None, occ_meta: dict | None, camera: str) -> list[str]:
    """Return the labels of grid cells that ARE passable (blue dots) — the complement of
    blocked_cell_labels, using the same _cell_is_free classification. Order follows _load_grid_centers()
    (CSV order). occ_grid=None -> every cell free -> all labels.

    Intended as the allowed-value set for structured-output enums, so the VLM cannot select an
    obstacle cell as a waypoint.
    """
    return [label for label, u, v in _load_grid_centers()
            if _cell_is_free(u, v, occ_grid, occ_meta, camera)]


# ── Node wrappers (ROS image extraction + base64 encoding) ────────────────────

def _node_to_pil(node) -> PILImage.Image:
    """Extract and undistort the node's current camera frame as an RGBA PIL Image."""
    cv_img = node.bridge.imgmsg_to_cv2(node.camera_image, desired_encoding='rgba8')
    if node.camera_matrix is not None and node.dist_coeffs is not None:
        cv_img = cv2.undistort(cv_img, node.camera_matrix, node.dist_coeffs)
    return PILImage.fromarray(cv_img).convert("RGBA")


def _to_b64(pil_img: PILImage.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')
