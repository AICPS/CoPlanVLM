"""Shared coordinate transforms (plain library, NOT a ROS node).

Single source of truth so the path translator, the occupancy-map generator (obs_seg), the
controller and the visualizer can never disagree on frames. No rclpy, no node, no entry point.

Four frames, all defined relative to the canonical WORLD frame
=============================================================
1. WORLD (canonical): origin directly under the camera nadir at (0, 0); +x to the RIGHT of
   the image, +y to the TOP of the image; z up; +yaw counter-clockwise (on the image).
   ALL planning and control happen in this frame.
2. PIXEL: overhead image, per-camera pinhole. Image column u increases right, row v increases
   DOWN. Selectable camera in {"gazebo", "lab_test"}.
3. NED: North-East-Down (MoCap room; and sim odom_to_pose emits it too). z is DOWN, so it is a
   reflection of WORLD in the ground plane — the yaw is NOT a plain negation (see below).
4. GAZEBO: sim ground-truth frame. Only used by odom_to_pose to produce NED in sim.

Yaw is always derived from the SAME position transform (project a point one step ahead and take
the resulting heading), so axis swaps / z-flips / origin offsets are handled automatically. Use
the *_pose (x, y, yaw) helpers rather than hand-coding a per-frame yaw formula.
"""
import math

# ── Camera intrinsics (per camera) ───────────────────────────────────────────────────────
# Perfect-pinhole model: metres_per_pixel = height / f, principal point at (cx, cy).
# gazebo: ideal nadir sim (ids_overhead) z=9.5 m, 80 deg HFOV, 1936x1216 -> f=1153.6 px,
#   scale 9.5/1153.6 = 0.008235 m/px; principal point = image centre.
# lab_test: real intrinsic calibration (camera_matrix K, 1920x1200 raw images) z=4.27 m.
#   fx != fy here, so the pinhole scale differs per axis; the yaw/pixel maths below handle it.
# fx=fy=1153.6187 == height / 0.008235, so gazebo reproduces the old 0.008235 m/px exactly.
CAMERAS = {
    "gazebo":   {"height": 9.5, "fx": 1153.6187, "fy": 1153.6187, "cx": 968.0, "cy": 608.0},
    "lab_test": {"height": 4.27, "fx": 959.5390439489479, "fy": 948.7315120801915, "cx": 988.0379239269574, "cy": 621.2961259804192},
}


def _cam(camera):
    if camera is None:
        raise ValueError(
            "camera must be specified explicitly — pass camera='gazebo' or 'lab_test'. "
            "The global set_active_camera mechanism has been removed.")
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera {camera!r}; expected one of {sorted(CAMERAS)}")
    return CAMERAS[camera]


# ── Frame offsets (placeholders; tune once real geometry is known) ───────────────────────
# NED origin vs world origin (applied in NED metres). 0 assumes MoCap origin under the nadir.
NED_OFFSET_X = 0.0
NED_OFFSET_Y = 0.0
# GAZEBO origin vs world origin (applied in Gazebo metres). Backed out from the previous
# calibration: the nadir sits at Gazebo (5.5, 0). Confirm in sim.
GZ_OFFSET_X = 5.5
GZ_OFFSET_Y = 0.0


# ── Angle helpers ────────────────────────────────────────────────────────────────────────
def wrap_to_pi(angle):
    """Wrap an angle (radians) to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x, y, z, w):
    """Planar yaw (rotation about the frame's z axis) from a quaternion, in [-pi, pi].

    Matches tf_transformations.euler_from_quaternion()[2].
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ── WORLD <-> PIXEL (per camera) ─────────────────────────────────────────────────────────
def world_to_pixel(x, y, camera=None):
    """World (x, y) metres -> image pixel (u=col, v=row) for `camera` (default: active)."""
    c = _cam(camera)
    u = c["cx"] + x * c["fx"] / c["height"]
    v = c["cy"] - y * c["fy"] / c["height"]      # image row grows downward -> minus
    return u, v


def pixel_to_world(u, v, camera=None):
    """Image pixel (u=col, v=row) -> world (x, y) metres for `camera` (default: active).

    Scalar or numpy-array inputs both work (vectorized).
    """
    c = _cam(camera)
    x = (u - c["cx"]) * c["height"] / c["fx"]
    y = -(v - c["cy"]) * c["height"] / c["fy"]
    return x, y


def world_yaw_to_pixel(yaw, camera=None):
    """World yaw -> pixel yaw. A world heading direction (cos yaw, sin yaw) maps to the pixel
    direction (fx*cos yaw, -fy*sin yaw) -- the image row axis grows downward (v = -y), and
    anisotropic focal lengths (fx != fy) scale the two axes differently. When fx == fy this
    reduces to pixel_yaw = -world_yaw."""
    c = _cam(camera)
    return math.atan2(-c["fy"] * math.sin(yaw), c["fx"] * math.cos(yaw))


def pixel_yaw_to_world(yaw, camera=None):
    """Pixel yaw -> world yaw. Inverse of world_yaw_to_pixel: a pixel direction (cos, sin)
    unscales to the world direction (cos/fx, -sin/fy). When fx == fy this reduces to
    world_yaw = -pixel_yaw."""
    c = _cam(camera)
    return math.atan2(-math.sin(yaw) / c["fy"], math.cos(yaw) / c["fx"])


def world_to_pixel_pose(x, y, yaw, camera=None):
    """World pose (x, y, yaw) -> pixel pose (u, v, yaw_pixel)."""
    u, v = world_to_pixel(x, y, camera)
    return u, v, world_yaw_to_pixel(yaw, camera)


def pixel_to_world_pose(u, v, yaw, camera=None):
    """Pixel pose (u, v, yaw_pixel) -> world pose (x, y, yaw)."""
    x, y = pixel_to_world(u, v, camera)
    return x, y, pixel_yaw_to_world(yaw, camera)


# ── WORLD <-> NED ────────────────────────────────────────────────────────────────────────
def world_to_ned(x, y):
    """World (x, y) -> NED (x=North, y=East). ned_x = world_y, ned_y = world_x (+ offset).

    A pure axis swap: NED North (+x) = world +y (image up), NED East (+y) = world +x
    (image right).
    """
    return y + NED_OFFSET_X, x + NED_OFFSET_Y


def ned_to_world(nx, ny):
    """NED (x, y) -> world (x, y). Inverse of world_to_ned."""
    return ny - NED_OFFSET_Y, nx - NED_OFFSET_X


def ned_yaw_to_world(yaw):
    """NED yaw -> world yaw. NED is z-down and swapped, so world_yaw = 90deg - ned_yaw."""
    return wrap_to_pi(math.pi / 2 - yaw)


def world_yaw_to_ned(yaw):
    """World yaw -> NED yaw. Inverse (an involution): ned_yaw = 90deg - world_yaw."""
    return wrap_to_pi(math.pi / 2 - yaw)


def ned_to_world_pose(nx, ny, yaw):
    """NED pose (x, y, yaw) -> world pose (x, y, yaw)."""
    wx, wy = ned_to_world(nx, ny)
    return wx, wy, ned_yaw_to_world(yaw)


def world_to_ned_pose(x, y, yaw):
    """World pose (x, y, yaw) -> NED pose (x, y, yaw)."""
    nx, ny = world_to_ned(x, y)
    return nx, ny, world_yaw_to_ned(yaw)


# ── WORLD <-> GAZEBO ─────────────────────────────────────────────────────────────────────
def gazebo_to_world(gx, gy):
    """Gazebo (x, y) -> world (x, y). Identity direction with an origin offset."""
    return gx - GZ_OFFSET_X, gy - GZ_OFFSET_Y


def world_to_gazebo(x, y):
    """World (x, y) -> Gazebo (x, y). Inverse of gazebo_to_world."""
    return x + GZ_OFFSET_X, y + GZ_OFFSET_Y


def gazebo_yaw_to_world(yaw):
    """Gazebo yaw -> world yaw. Gazebo axes align with world (identity direction)."""
    return wrap_to_pi(yaw)


def world_yaw_to_gazebo(yaw):
    """World yaw -> Gazebo yaw. Inverse (identity)."""
    return wrap_to_pi(yaw)


def gazebo_to_world_pose(gx, gy, yaw):
    """Gazebo pose (x, y, yaw) -> world pose (x, y, yaw)."""
    wx, wy = gazebo_to_world(gx, gy)
    return wx, wy, gazebo_yaw_to_world(yaw)


def world_to_gazebo_pose(x, y, yaw):
    """World pose (x, y, yaw) -> Gazebo pose (x, y, yaw)."""
    gx, gy = world_to_gazebo(x, y)
    return gx, gy, world_yaw_to_gazebo(yaw)


# ── GAZEBO -> NED (sim odom_to_pose: gazebo ground truth published as NED) ────────────────
def gazebo_to_ned(gx, gy):
    """Gazebo (x, y) -> NED (x, y), composing gazebo->world->ned."""
    return world_to_ned(*gazebo_to_world(gx, gy))


def gazebo_yaw_to_ned(yaw):
    """Gazebo yaw -> NED yaw (gazebo->world is identity, world->ned is 90deg - yaw)."""
    return world_yaw_to_ned(gazebo_yaw_to_world(yaw))


def gazebo_to_ned_pose(gx, gy, yaw):
    """Gazebo pose (x, y, yaw) -> NED pose (x, y, yaw)."""
    nx, ny = gazebo_to_ned(gx, gy)
    return nx, ny, gazebo_yaw_to_ned(yaw)
