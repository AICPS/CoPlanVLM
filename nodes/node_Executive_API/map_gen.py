"""Per-task-type overhead-map builders for the Executive node.

The classifier tags each instruction with a task type; the Executive then renders the overhead map
best suited to that behavior via ``MAP_BUILDERS[task_type](node)``. Each builder receives the
``ExecutiveApiNode`` so it can read the live camera frame and intrinsics off it
(``node.camera_image``, ``node.bridge``, ``node.camera_matrix``, ``node.dist_coeffs``). Static map
assets (e.g. the grid overlay PNG) are loaded and cached here, not on the node.

CLIPSeg lives here (not in translate.py). ``run_segmentation(node)`` is called by exec once per
replan tick; it runs CLIPSeg on the current camera frame, stores ``node.pix_labels``, and writes
the result to ``_OCC_FILE`` so that translate.py can read it instead of running its own instance.
This keeps the model in one process and avoids a ROS topic for the occupancy data.

Mirrors ``prompt_gen.PROMPT_BUILDERS``.
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

from coord_transform import pixel_to_world
from obs_seg import FREE
from obs_seg.occupancy import mask_to_occupancy, inflate_occupancy, world_to_cell

# ── Shared occupancy file ──────────────────────────────────────────────────────
# exec writes pix_labels here; translate reads it instead of running CLIPSeg itself.
# get_package_share_directory returns <ws>/install/talking-turtle/share/talking-turtle/;
# 4 levels up is the colcon workspace root (standard isolated-install layout).
_OCC_FILE = os.path.normpath(os.path.join(
    get_package_share_directory('talking-turtle'), '..', '..', '..', '..',
    'debug', 'talking_turtle_pix_labels.npy'))

# ── Grid / set-of-marks tuning ────────────────────────────────────────────────
_N_COLS        = 14
_N_ROWS        = 8
_GRID_COLOR    = (0, 0, 255, 255)   # blue, ~70 % opacity
_GRID_WIDTH    = 3                  # line thickness in pixels
_DOT_RADIUS    = 10                 # nav2point mark dot radius
_LABEL_SIZE    = 28                 # pt; grid cell labels and nav2point marks
_SEG_PROMPTS   = ["the floor"]
_SEG_THRESHOLD = 0.48


@lru_cache(maxsize=1)
def _get_segmenter():
    """Load TraversabilitySegmenter (CLIPSeg) once on first call; deferred import."""
    from obs_seg.segmenter import TraversabilitySegmenter
    return TraversabilitySegmenter()


@lru_cache(maxsize=1)
def _load_grid_centers() -> list[tuple[str, float, float]]:
    """Return [(label, u, v), ...] pixel coords from grid_cell_centers.csv (cached, read once)."""
    pkg_dir = get_package_share_directory('talking-turtle')
    csv_path = os.path.join(pkg_dir, 'config', 'grid_cell_centers.csv')
    with open(csv_path, newline='') as f:
        return [(r['cell'], float(r['center_x']), float(r['center_y']))
                for r in csv.DictReader(f)]


# ── Segmentation (runs in exec, result shared with translate via file) ─────────

def run_segmentation(node) -> None:
    """Run CLIPSeg on the current camera frame; save pix_labels to node and to _OCC_FILE.

    Called by exec._run_plan before map builder dispatch so every task type produces a fresh
    occupancy snapshot. translate.py reads _OCC_FILE instead of running its own CLIPSeg instance.
    No-op if no camera image is available yet.
    """
    if node.camera_image is None:
        return
    cv_img = node.bridge.imgmsg_to_cv2(node.camera_image, desired_encoding='rgba8')
    if node.camera_matrix is not None and node.dist_coeffs is not None:
        cv_img = cv2.undistort(cv_img, node.camera_matrix, node.dist_coeffs)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
    pix_labels, _ = _get_segmenter().classify(rgb, _SEG_PROMPTS, [], _SEG_THRESHOLD)
    node.pix_labels = pix_labels
    os.makedirs(os.path.dirname(_OCC_FILE), exist_ok=True)
    np.save(_OCC_FILE, pix_labels)


# ── Pure render functions (PIL in → PIL out; no ROS types) ────────────────────

def render_battleship_map(img_rgba: PILImage.Image) -> PILImage.Image:
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

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  size=_LABEL_SIZE)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(result)
    pad = 2
    for col_idx in range(_N_COLS):
        for row_idx in range(_N_ROWS):
            label = chr(ord('A') + col_idx) + str(row_idx + 1)
            tx = int(col_idx * cell_w) + 4
            ty = int(row_idx * cell_h) + 4
            bbox = draw.textbbox((tx, ty), label, font=font)
            draw.rectangle((bbox[0] - pad, bbox[1] - pad,
                            bbox[2] + pad, bbox[3] + pad), fill=(220, 220, 220))
            draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
    return result


def render_nav2point_map(img_rgba: PILImage.Image, pix_labels: np.ndarray | None,
                         camera: str, resolution: float) -> PILImage.Image:
    """Draw set-of-marks dots at free-space grid centers on an RGBA image.

    When pix_labels is None, all grid marks are shown (no obstacle filtering).
    """
    if pix_labels is not None:
        grid, meta = mask_to_occupancy(pix_labels, resolution, camera=camera)
        inflated = inflate_occupancy(grid, resolution)
        free_cells = []
        for label, u, v in _load_grid_centers():
            wx, wy = pixel_to_world(u, v, camera=camera)
            gx, gy = world_to_cell(wx, wy, meta)
            if (0 <= gx < meta['width'] and 0 <= gy < meta['height']
                    and inflated[gy, gx] == FREE):
                free_cells.append((label, u, v))
    else:
        free_cells = list(_load_grid_centers())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  size=_LABEL_SIZE)
    except OSError:
        font = ImageFont.load_default()

    result = img_rgba.copy()
    draw = ImageDraw.Draw(result)
    r = _DOT_RADIUS
    for label, u, v in free_cells:
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(0, 0, 255))
        tx, ty = u + r + 3, v - _LABEL_SIZE // 2
        bbox = draw.textbbox((tx, ty), label, font=font)
        pad = 2
        draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
                        fill=(220, 220, 220))
        draw.text((tx, ty), label, fill=(0, 0, 255), font=font)
    return result


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


def generate_battleship_map(node) -> str:
    """Composite the live overhead frame with the Battleship grid overlay; return base64 PNG."""
    if node.camera_image is None:
        raise RuntimeError("No camera image received yet — is the camera publishing?")
    return _to_b64(render_battleship_map(_node_to_pil(node)))


# ── Per-task map builders ──────────────────────────────────────────────────────

def gen_nav2point_map(node) -> str:
    """Set-of-marks overlay: draw labeled dots only at grid cell centers in free space."""
    if node.camera_image is None:
        raise RuntimeError("No camera image received yet — is the camera publishing?")
    return _to_b64(render_nav2point_map(
        _node_to_pil(node), node.pix_labels, node.camera_name, node.resolution))


def gen_manouver_map(node) -> str:
    """Map overlay for multi-waypoint maneuvers — the Battleship grid."""
    return generate_battleship_map(node)


def gen_coverage_map(node) -> str:
    """Map overlay for area coverage. TODO: coverage-specific overlay; grid for now."""
    return generate_battleship_map(node)


# Dispatch registry: task type -> map builder. Keys match prompt_gen.PROMPT_BUILDERS.
MAP_BUILDERS = {
    "nav2point": gen_nav2point_map,
    "manouver": gen_manouver_map,
    "coverage": gen_coverage_map,
}
