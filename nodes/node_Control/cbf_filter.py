"""Control-Barrier-Function QP safety filter (plain library, NOT a ROS node).

Sits between the nominal P controller and the ``/cmd_vel`` publisher: it takes the nominal
``[v, omega]`` and returns the CLOSEST command that keeps the robot clear of (a) the other robots
and (b) nearby static obstacles from the occupancy map. The nominal controller is untouched — when
no constraint is active the nominal command is returned byte-identical.

No rclpy import: everything here is pure numpy so it can be unit-tested without a ROS graph. It
lives inside ``node_Control`` rather than at the top of ``nodes/`` for the same reason
``grid_planner_utils.py`` lives inside ``node_Path_Translator``: the top level is reserved for
libraries genuinely shared across nodes, and this one has a single consumer.

MODEL
-----
Differential drive ``q = [x, y, theta]``, ``u = [v, omega]``, all in the repo's WORLD frame::

    x' = v cos(theta)     y' = v sin(theta)     theta' = omega

Because ``u`` cannot move the robot sideways, the barrier is enforced on a LOOK-AHEAD POINT a
distance ``ell`` in front of the robot, whose velocity is a full-rank function of ``u``::

    p_l   = [x + ell cos(th), y + ell sin(th)]
    G(th) = [[cos(th), -ell sin(th)],          p_l' = G(th) u
             [sin(th),  ell cos(th)]]

``G``'s omega column is scaled by ``ell``, so omega moves the look-ahead point ``1/ell`` times less
than v does. At ell = 0.10 that is 10x, which is why this filter predominantly BRAKES rather than
steering around obstacles. Raising ``ell`` buys steering authority at the cost of pushing the
protected point further from the robot centre (see the radius convention below).

BARRIER AND CONSTRAINT
----------------------
For an obstacle/peer centre ``p_k`` with required separation ``R_k``, and ``dp = p_l - p_k``::

    h_k  = |dp|^2 - R_k^2                       (>= 0 means safe)
    h_k' = 2 dp^T G(th) u
    ZCBF: h_k' >= -gamma_k h_k
    row : -2 dp^T G(th) u <= max(gamma_k h_k, 0)          (A u <= b)

The ``max(..., 0)`` clamp matters. Without it, a robot that starts already inside the unsafe set
(h < 0) faces a constraint demanding h' > 0 — "actively move away" — which is INFEASIBLE when the
obstacle is dead ahead and reverse is disallowed, since then the v coefficient is negative and the
omega coefficient is exactly zero. The QP would fail, the fallback would stop the robot, and
stopping never restores h: a permanent freeze. Clamping degrades a violated row to "do not get
worse" (h' >= 0), which u = 0 always satisfies. Consequences:

  * ``u = 0`` satisfies every row whenever every ``b_k >= 0``, which the clamp guarantees, so the
    QP is feasible UNCONDITIONALLY. Solver failure can then only be numerical, and the safe-stop
    fallback is always itself constraint-satisfying. No slack variables are needed.
  * Recovery still works: v is pinned to 0 while pointing into the obstacle, but the robot may
    still rotate, and once the heading turns far enough that the v coefficient is non-negative,
    forward motion is permitted again. It escapes by turning.

RADIUS / INFLATION CONVENTION
-----------------------------
Map obstacles come from the ``infl`` grid, which ``obs_seg.occupancy`` has ALREADY dilated by
``INFLATION_RADIUS`` (0.45 m ~ robot radius + margin) — the very grid A* plans on. So the filter
adds only ``obstacle_margin`` and does NOT re-add the robot radius; doing so would double-inflate.
Using the planner's own grid also means the filter's safe set matches the planner's, so it never
fights a correctly-tracked path::

    R_obstacle = obstacle_margin

The peer arrives as a raw pose, so it needs the full two-robot footprint — plus ``ell``, because
the barrier protects the look-ahead point while the body extends BEHIND it. Seen from that point
the robot's own footprint has an effective radius of ``ell + radius_self`` on a rear approach, so
omitting the ``ell`` term would silently erode clearance as ``ell`` is tuned up::

    R_robot = ell + radius_self + radius_other + robot_margin

LIMITATIONS
-----------
The peer is treated as STATIC within each solve. The resulting error in h' is bounded by
``2|dp| v_peer``, which in distance terms is exactly the peer's own top speed; ``robot_margin``
absorbs it (>= (v_self + v_peer) * reaction_latency + tracking error). This protection assumes both
robots run this filter — a peer with it disabled, driving at full speed, cannot be avoided from one
side alone. Poses are assumed fresh; there is no staleness detection. Safety further depends on
pose/map/model accuracy, and the guarantee is an invariance one: starting OUTSIDE the safe set
gives only non-worsening, not recovery. Finally, the filter enforces safety but not progress — two
robots meeting head-on will both brake and can deadlock.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import osqp
from scipy import sparse

from obs_seg import FREE

# QP objective weights, W = diag(w_v, w_omega). Hardcoded rather than exposed as parameters to keep
# the surface small. Only the RATIO matters: scaling W leaves the argmin unchanged.
#
# Equal weights would be dimensionally meaningless (adding m^2/s^2 to rad^2/s^2), making the
# trade-off an artifact of the chosen units. Normalising each deviation by the actuation available
# to it gives w_v : w_omega = 1/v_max^2 : 1/omega_max^2 = (0.3/0.15)^2 = 4 — v deviations cost more
# because v_max is the tighter limit. Both terms then read as "fraction of available actuation,
# squared".
#
# The ratio r = w_v/w_omega sets the braking-vs-steering split: the two channels contribute to h'
# in proportion ell^2 * r, so they are equal only at r = 1/ell^2 = 100. At r = 4 the filter is
# ~96% braking, which is the intended behaviour here.
W_V = 100.0
W_OMEGA = 1.0

# OSQP settings, chosen by measurement over 400 poses sampled from a real occupancy snapshot.
# OSQP is an ADMM solver, so these are a reliability/accuracy trade-off in BOTH directions:
#
#   defaults (eps 1e-3)     0 failures, but residual constraint violation up to 2.8e-4
#   eps 1e-9                violation ~1e-16, but 13% of solves hit the iteration limit
#   eps 1e-4, 10k iters     3 failures / 1500
#   eps 1e-4, 50k iters     0 failures / 1500, violation <= 3.7e-6       <- chosen
#
# Tight tolerances are actively harmful here: an unconverged solve becomes a safe stop, and at 10 Hz
# a 13% stop rate is a robot that stutters constantly. `polishing` (the OSQP >= 1.0 spelling of the
# old `polish`; requirements.txt pins that major) refines the solution on the active set, which is
# what makes a loose eps safe — with it, an inactive constraint set reproduces the nominal command
# to ~1e-16. The high iteration cap costs nothing measurable (p99 1.2 ms, worst 2.1 ms observed
# against a 100 ms control period) because it is only reached on rare hard geometries.
_OSQP_SETTINGS = dict(eps_abs=1e-4, eps_rel=1e-4, polishing=True, max_iter=50000, verbose=False)


@dataclass
class CBFConfig:
    """Filter tuning — the SINGLE SOURCE OF TRUTH for these values.

    ``control.py`` declares one ROS parameter per field and seeds each one from the default here,
    so editing a number in this class changes both the tests and the running robot. Overriding a
    parameter (launch file, ``ros2 param set``) still wins at runtime; the point is that there is
    no second hard-coded copy to drift out of sync.

    Exceptions: ``v_max`` / ``omega_max`` are the actuator clamps, which the nominal controller
    needs too, so ``control.py`` owns them (``max_linear_vel`` / ``max_angular_vel``) and always
    passes them in. The values below are only what a bare ``CBFConfig()`` gets in tests.
    """

    lookahead: float = 0.10         # m     — protected point ahead of the robot centre
    gamma_robot: float = 1.5        #       — ZCBF gain, peer rows (gamma*dt = 0.15 << 1)
    gamma_obstacle: float = 1.5     #       — ZCBF gain, map rows
    radius_self: float = 0.17       # m     — TurtleBot 4 body radius (see obs_seg INFLATION_RADIUS)
    radius_other: float = 0.17      # m
    robot_margin: float = 0.4      # m     — >= (v_self + v_peer) * latency + tracking error
    # m — added to the ALREADY-inflated grid; do NOT re-add the robot radius. MUST stay below half
    # the occupancy resolution (0.05/2 = 0.025): a point on the shared edge between a free and a
    # blocked cell is only res/2 from the blocked cell's CENTRE, so a larger margin puts the whole
    # ring of free cells adjacent to the inflated region at h <= 0 — and that ring is exactly where
    # A* routes, since shortest paths hug obstacles. At 0.05 (the grid pitch itself) 3.6% of
    # legitimately free cells start out unsafe, and the robot freezes there. Real clearance comes
    # from the grid's own 0.45 m inflation, not from this margin.
    obstacle_margin: float = 0.1
    query_radius: float = 1.0       # m     — map obstacles beyond this are ignored
    max_obstacles: int = 1          #       — cap on map rows, nearest-first
    downsample: float = 0.25        # m     — bucket size when thinning obstacle cells
    v_min: float = -0.05            # m/s 
    v_max: float = 0.15             # m/s   — actuator clamp; control.py overrides (see docstring)
    omega_max: float = 0.4          # rad/s — actuator clamp; control.py overrides
    # Post-solve feasibility tolerance. Sized against the measured worst-case residual (3.7e-6, so
    # ~27x headroom), NOT set as tight as possible: a violation of even 1e-3 corresponds to under
    # 0.1 mm of extra approach per 0.1 s tick — a thousandth of the safety margin — whereas an
    # over-tight tolerance turns harmless numerical residue into a safe stop, which is a far worse
    # failure. This check exists to catch a GROSS solver failure, not to police micrometres.
    tol: float = 1e-4

    @property
    def r_robot(self) -> float:
        """Required look-ahead-point separation from a peer's centre (see module docstring)."""
        return self.lookahead + self.radius_self + self.radius_other + self.robot_margin

    @property
    def r_obstacle(self) -> float:
        """Required separation from an `infl` cell centre — margin only; the grid is pre-inflated."""
        return self.obstacle_margin


@dataclass
class FilterResult:
    """Filtered command plus everything the node needs for throttled diagnostics."""

    v: float
    omega: float
    modified: bool          # did the filter change the nominal command?
    n_obstacles: int        # map-obstacle rows in the QP
    peer_active: bool       # was a peer row included?
    min_h: float            # smallest barrier value this tick (inf when no rows)
    status: str             # "no-constraints" | osqp status string | failure reason
    solve_ms: float
    fallback: str = ""      # non-empty => safe stop was substituted, and why

    @property
    def command(self) -> tuple[float, float]:
        return self.v, self.omega


# ── Peer roster ───────────────────────────────────────────────────────────────────────────

def resolve_peers(robot_name: str, robot_names: Sequence[str]) -> list[str]:
    """Return the roster minus this robot. NEVER returns `robot_name` itself.

    An empty or misspelled `robot_name` matches no roster entry, so returning "everyone else" would
    leave this robot's OWN name in the list: it would subscribe to its own pose, measure a distance
    of ~0 to itself, and the peer row would stop it dead — a launch-file typo presenting as a
    mysteriously stationary robot. Returning an empty list instead means "no peer constraint", which
    the caller reports loudly.
    """
    names = [str(n) for n in robot_names]
    if not robot_name or robot_name not in names:
        return []
    return [n for n in names if n != robot_name]


# ── Geometry ──────────────────────────────────────────────────────────────────────────────

def lookahead_point(x: float, y: float, theta: float, ell: float) -> np.ndarray:
    """The point `ell` metres ahead of the robot, whose velocity `u` can fully control."""
    return np.array([x + ell * math.cos(theta), y + ell * math.sin(theta)], dtype=float)


def lookahead_jacobian(theta: float, ell: float) -> np.ndarray:
    """G(theta), mapping u = [v, omega] to the look-ahead point's velocity."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -ell * s],
                     [s, ell * c]], dtype=float)


def barrier(p_lookahead: np.ndarray, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """h = |p_l - p_k|^2 - R_k^2 for each centre. Positive = safe, negative = inside the radius."""
    dp = np.atleast_2d(p_lookahead) - np.atleast_2d(centres)
    return (dp * dp).sum(axis=1) - np.asarray(radii, dtype=float) ** 2


def blocked_cell_centres(grid: np.ndarray, meta: dict) -> np.ndarray:
    """World-frame centres of every non-FREE cell, as an (M, 2) array.

    Vectorised equivalent of calling ``obs_seg.occupancy.cell_to_world`` on each blocked cell (a
    unit test asserts the two agree). Called once per occupancy snapshot, not per control tick.
    Non-FREE rather than == OCCUPIED to match the repo's ``_cell_is_free`` convention, though the
    inflated grid is binary in practice.
    """
    gy, gx = np.nonzero(np.asarray(grid) != FREE)
    res = float(meta["resolution"])
    xs = float(meta["origin_x"]) + (gx + 0.5) * res
    ys = float(meta["origin_y"]) + (gy + 0.5) * res
    return np.column_stack([xs, ys])


def select_obstacles(points: np.ndarray, robot_xy, lookahead_xy, *, query_radius: float,
                     downsample: float, max_count: int) -> np.ndarray:
    """Pick a small, well-spread set of obstacle points near the robot.

    A 1.5 m radius at 0.05 m/cell covers ~2800 cells, so feeding every one to the QP is both slow
    and pointless (they are mostly duplicates of the same wall). Three stages:

    1. keep points within `query_radius` of the robot CENTRE;
    2. bucket onto a `downsample`-metre lattice, keeping the point NEAREST THE LOOK-AHEAD POINT in
       each bucket — the look-ahead point is what the barrier measures from, so this keeps the
       binding representative of each obstacle rather than an arbitrary one;
    3. sort by distance to the look-ahead point and keep the closest `max_count`.

    Sorting before the cap is what makes the cap safe: the constraints that are dropped are always
    the least binding ones.
    """
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 2))
    pts = np.atleast_2d(pts)

    robot_xy = np.asarray(robot_xy, dtype=float)
    look = np.asarray(lookahead_xy, dtype=float)

    d_robot = pts - robot_xy
    pts = pts[(d_robot * d_robot).sum(axis=1) <= query_radius * query_radius]
    if pts.shape[0] == 0:
        return np.empty((0, 2))

    d_look = pts - look
    dist2 = (d_look * d_look).sum(axis=1)

    if downsample > 0.0:
        order = np.argsort(dist2, kind="stable")            # nearest-to-look-ahead first
        keys = np.floor(pts[order] / downsample).astype(np.int64)
        # Fuse the 2-D lattice index into one integer so np.unique can work on a 1-D array.
        flat = keys[:, 0] * np.int64(1 << 32) + keys[:, 1]
        # np.unique returns the FIRST occurrence of each key; because `order` is sorted by distance,
        # that first occurrence is the closest point in its bucket.
        _, first = np.unique(flat, return_index=True)
        keep = order[np.sort(first)]
        pts, dist2 = pts[keep], dist2[keep]

    if max_count is not None and pts.shape[0] > max_count:
        nearest = np.argsort(dist2, kind="stable")[:max_count]
        pts = pts[nearest]
    return pts


# ── Constraint assembly ───────────────────────────────────────────────────────────────────

def build_rows(p_lookahead: np.ndarray, G: np.ndarray, centres: np.ndarray,
               radii: np.ndarray, gammas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack the CBF rows: returns (A, b, h) with ``A u <= b``.

    ``b = max(gamma * h, 0)`` — see the module docstring for why the clamp is what keeps the QP
    feasible for a robot that starts inside the unsafe set.
    """
    centres = np.atleast_2d(np.asarray(centres, dtype=float))
    if centres.size == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,))
    radii = np.asarray(radii, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    dp = p_lookahead - centres                  # (M, 2)
    h = (dp * dp).sum(axis=1) - radii ** 2      # (M,)
    A = -2.0 * (dp @ G)                         # (M, 2): row i is -2 dp_i^T G
    b = np.maximum(gammas * h, 0.0)
    return A, b, h


def solve_qp(u_nom: np.ndarray, A: np.ndarray, b: np.ndarray, cfg: CBFConfig):
    """min 1/2 (u-u_nom)^T W (u-u_nom) s.t. A u <= b and the actuator box. Returns (u, status).

    Expanded into OSQP's ``1/2 u^T P u + q^T u`` with ``P = W`` and ``q = -W u_nom`` (the constant
    term is dropped, which does not move the argmin). The box bounds are appended as identity rows
    so OSQP's two-sided ``l <= A u <= u`` form carries both.
    """
    W = np.diag([W_V, W_OMEGA])
    P = sparse.triu(sparse.csc_matrix(W), format="csc")
    q = -(W @ np.asarray(u_nom, dtype=float))

    A_all = sparse.csc_matrix(np.vstack([np.atleast_2d(A).reshape(-1, 2), np.eye(2)]))
    lo = np.concatenate([np.full(len(b), -np.inf), [cfg.v_min, -cfg.omega_max]])
    up = np.concatenate([np.asarray(b, dtype=float), [cfg.v_max, cfg.omega_max]])

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A_all, l=lo, u=up, **_OSQP_SETTINGS)
    # raise_error=False: a failed solve is reported through info.status and handled by the caller's
    # fallback, which is the behaviour we want (and OSQP's default is due to flip to raising).
    res = prob.solve(raise_error=False)
    return np.asarray(res.x, dtype=float), str(res.info.status)


class CBFSafetyFilter:
    """Wraps the pieces above into one call. Holds config and the solver-failure count."""

    def __init__(self, config: CBFConfig | None = None):
        self.cfg = config or CBFConfig()
        self.failure_count = 0

    # -- fallback ---------------------------------------------------------------------------
    def _stop(self, reason: str, *, n_obs: int = 0, peer: bool = False,
              min_h: float = math.inf, solve_ms: float = 0.0) -> FilterResult:
        """Safe stop. Zero linear AND angular: there is no verified emergency-turn behaviour here.

        Never returns the unfiltered nominal command — a filter that falls back to "whatever the
        controller wanted" provides no guarantee at exactly the moment it is most needed.
        """
        self.failure_count += 1
        return FilterResult(0.0, 0.0, True, n_obs, peer, min_h, reason, solve_ms, fallback=reason)

    def _clip(self, u: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        return np.array([min(max(u[0], cfg.v_min), cfg.v_max),
                         min(max(u[1], -cfg.omega_max), cfg.omega_max)])

    # -- main entry point -------------------------------------------------------------------
    def filter(self, *, nominal, robot_pose, peer_points=(), obstacle_points=None) -> FilterResult:
        """Return the closest safe command to `nominal`.

        Args:
            nominal:         (v, omega) from the nominal controller, ALREADY clamped to the
                             actuator limits — anchoring the objective at an unexecutable command
                             would make "minimum deviation" meaningless.
            robot_pose:      (x, y, theta) in the world frame.
            peer_points:     iterable of peer (x, y) world positions; None entries are skipped.
            obstacle_points: (M, 2) world-frame obstacle centres, or None.
        """
        cfg = self.cfg
        u_nom = np.asarray(nominal, dtype=float)
        if u_nom.shape != (2,) or not np.all(np.isfinite(u_nom)):
            return self._stop("non-finite nominal command")
        if robot_pose is None or not np.all(np.isfinite(np.asarray(robot_pose, dtype=float))):
            return self._stop("non-finite robot pose")

        x, y, theta = (float(v) for v in robot_pose)
        p_l = lookahead_point(x, y, theta, cfg.lookahead)
        G = lookahead_jacobian(theta, cfg.lookahead)

        centres: list[np.ndarray] = []
        radii: list[float] = []
        gammas: list[float] = []

        peers = [np.asarray(p, dtype=float) for p in peer_points
                 if p is not None and np.all(np.isfinite(np.asarray(p, dtype=float)))]
        for p in peers:
            centres.append(p)
            radii.append(cfg.r_robot)
            gammas.append(cfg.gamma_robot)

        selected = np.empty((0, 2))
        if obstacle_points is not None and len(obstacle_points):
            selected = select_obstacles(
                obstacle_points, (x, y), p_l, query_radius=cfg.query_radius,
                downsample=cfg.downsample, max_count=cfg.max_obstacles)
            for p in selected:
                centres.append(p)
                radii.append(cfg.r_obstacle)
                gammas.append(cfg.gamma_obstacle)

        n_obs = int(selected.shape[0])
        peer_active = bool(peers)

        # No rows at all: return the nominal EXACTLY. Short-circuiting rather than solving a
        # trivially-unconstrained QP keeps "filter inactive => nominal preserved" exact rather than
        # true-to-solver-tolerance, which is what the regression check relies on.
        if not centres:
            u = self._clip(u_nom)
            return FilterResult(float(u[0]), float(u[1]),
                                modified=not np.allclose(u, u_nom, atol=1e-12),
                                n_obstacles=0, peer_active=False, min_h=math.inf,
                                status="no-constraints", solve_ms=0.0)

        A, b, h = build_rows(p_l, G, np.vstack(centres), np.array(radii), np.array(gammas))
        min_h = float(h.min())

        # If the nominal command already satisfies every row, it IS the optimum: the objective's
        # unconstrained minimiser is u_nom, so a feasible u_nom is globally optimal and the QP would
        # simply return it. Short-circuiting is exact, not an approximation — and it means the
        # solver is only invoked on the ticks where the filter actually has work to do, which is a
        # minority of them. (It also avoids OSQP's "no active set detected" chatter on stdout, which
        # bypasses verbose=False because it is emitted from C.)
        u_nom_clipped = self._clip(u_nom)
        if np.all(A @ u_nom_clipped <= b) and np.allclose(u_nom_clipped, u_nom, atol=1e-12):
            return FilterResult(float(u_nom[0]), float(u_nom[1]), modified=False,
                                n_obstacles=n_obs, peer_active=peer_active, min_h=min_h,
                                status="nominal-feasible", solve_ms=0.0)

        t0 = time.perf_counter()
        try:
            u, status = solve_qp(u_nom, A, b, cfg)
        except Exception as exc:                                   # noqa: BLE001
            return self._stop(f"solver raised: {exc}", n_obs=n_obs, peer=peer_active, min_h=min_h,
                              solve_ms=(time.perf_counter() - t0) * 1e3)
        solve_ms = (time.perf_counter() - t0) * 1e3

        if u is None or u.shape != (2,) or not np.all(np.isfinite(u)):
            return self._stop(f"non-finite solution ({status})", n_obs=n_obs, peer=peer_active,
                              min_h=min_h, solve_ms=solve_ms)
        if status != "solved":
            return self._stop(f"solver status '{status}'", n_obs=n_obs, peer=peer_active,
                              min_h=min_h, solve_ms=solve_ms)

        # Clip to the actuator box: bounds are enforced exactly here rather than trusted to the
        # solver. With polish enabled the correction is ~1e-9, far too small to disturb a CBF row,
        # and the validation below would catch it if it were not.
        u = self._clip(u)
        if np.any(A @ u > b + cfg.tol):
            worst = float(np.max(A @ u - b))
            return self._stop(f"constraint violated by {worst:.2e} ({status})", n_obs=n_obs,
                              peer=peer_active, min_h=min_h, solve_ms=solve_ms)

        return FilterResult(float(u[0]), float(u[1]),
                            modified=not np.allclose(u, u_nom, atol=1e-9),
                            n_obstacles=n_obs, peer_active=peer_active, min_h=min_h,
                            status=status, solve_ms=solve_ms)
