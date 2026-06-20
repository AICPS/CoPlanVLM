"""Shared overhead-image <-> world-frame coordinate transform (plain library, NOT a ROS node).

Single source of truth so the path translator (node_Path_Translator) and the
occupancy-map generator (obs_seg) can never disagree on the conversion. No rclpy,
no node, no entry point -- just an importable function.

Convention (straight-down overhead camera, ground plane):
    x = x0 + (v - v0) * sy        # world x from image row v   (sy typically -1/152)
    y = y0 + (u - u0) * sx        # world y from image col u   (sx typically  1/138)
where (u, v) = pixel (column, row) and (u0, v0) = pixel of the world origin label.
"""


def pixel_to_world(u, v, x0, y0, sx, sy, u0, v0):
    """Image pixel (u=col, v=row) -> world (x, y) in metres.

    Scalar or numpy-array inputs for u, v both work (vectorized). Returns (x, y).
    """
    x = x0 + (v - v0) * sy
    y = y0 + (u - u0) * sx
    return x, y


def world_to_pixel(x, y, x0, y0, sx, sy, u0, v0):
    """World (x, y) metres -> image pixel (u=col, v=row). Exact inverse of pixel_to_world.

    Used to draw world-frame paths back onto the overhead image for debug overlays.
    Returns (u, v).
    """
    u = u0 + (y - y0) / sx
    v = v0 + (x - x0) / sy
    return u, v
