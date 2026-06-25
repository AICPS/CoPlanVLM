"""Shared overhead-image <-> world-frame coordinate transform (plain library, NOT a ROS node).

Single source of truth so the path translator (node_Path_Translator), the occupancy-map
generator (obs_seg) and the visualizer can never disagree on the conversion. No rclpy, no node,
no entry point -- just importable functions plus the calibration they use.

Convention (straight-down overhead camera, ground plane):
    x = WORLD_ORIGIN_X + (v - ORIGIN_PIXEL_V) * METRES_PER_PIXEL_Y   # world x from image row v
    y = WORLD_ORIGIN_Y + (u - ORIGIN_PIXEL_U) * METRES_PER_PIXEL_X   # world y from image col u
where (u, v) = pixel (column, row).
"""

# ── Linear calibration for the overhead camera (THE SINGLE SOURCE OF TRUTH) ──────────────
# Derived from the ideal nadir pinhole sim (ids_overhead): z=9.5 m, 80 deg HFOV, 1936x1216 ->
# f=1153.62 px, scale = 9.5/1153.62 = 0.008235 m/px (isotropic); nadir (5.5,0) -> image centre.
# NOTE: these values are SPECIFIC TO THE sim_world world (camera height 9.5 m). A different
# world/camera height (e.g. house at z=10 m) would need a recomputed scale.

# --- old calibration from previous (hand-tuned; wrong scale + H4-cell-pixel anchor) ---
# WORLD_ORIGIN_X = 0.0
# WORLD_ORIGIN_Y = 5.5
# METRES_PER_PIXEL_X = 1.0 / 138.0
# METRES_PER_PIXEL_Y = -1.0 / 152.0
# ORIGIN_PIXEL_U = 1035.0
# ORIGIN_PIXEL_V = 532.0

WORLD_ORIGIN_X = 0.0          # camera nadir world Y (the anchor maps here)
WORLD_ORIGIN_Y = 5.5         # camera nadir world X
METRES_PER_PIXEL_X = 0.008235     # world y per image column   (= 9.5 / 1153.62)
METRES_PER_PIXEL_Y = -0.008235    # world x per image row      (axis inverted)
ORIGIN_LABEL = "H4"          # legacy reference only; no longer used by the transform
# Anchor pixel = camera nadir = principal point = image centre (1936/2, 1216/2).
ORIGIN_PIXEL_U = 968.0
ORIGIN_PIXEL_V = 608.0


def pixel_to_world(u, v):
    """Image pixel (u=col, v=row) -> world (x, y) in metres.

    Fully self-contained: the whole calibration (anchor pixel, anchor world coord, scale) lives
    in this module. Scalar or numpy-array inputs for u, v both work (vectorized). Returns (x, y).
    """
    x = WORLD_ORIGIN_X + (v - ORIGIN_PIXEL_V) * METRES_PER_PIXEL_Y
    y = WORLD_ORIGIN_Y + (u - ORIGIN_PIXEL_U) * METRES_PER_PIXEL_X
    return x, y


def world_to_pixel(x, y):
    """World (x, y) metres -> image pixel (u=col, v=row). Exact inverse of pixel_to_world."""
    u = ORIGIN_PIXEL_U + (y - WORLD_ORIGIN_Y) / METRES_PER_PIXEL_X
    v = ORIGIN_PIXEL_V + (x - WORLD_ORIGIN_X) / METRES_PER_PIXEL_Y
    return u, v


def gazebo_to_world(x, y):
    """Gazebo physical frame -> camera/world (image) frame.

    The overhead camera image axes are rotated 90 deg relative to Gazebo, so a point at Gazebo
    (x, y) lies at camera/world (y, x) — a pure axis swap. world_to_gazebo is the inverse.
    """
    return y, x


def world_to_gazebo(x, y):
    """Camera/world (image) frame -> Gazebo physical frame. Inverse of gazebo_to_world.

    Currently also a pure x<->y swap, but defined explicitly (rather than reusing
    gazebo_to_world) so the inverse stays correct if the gazebo<->world relationship ever
    becomes more than a swap.
    """
    return y, x