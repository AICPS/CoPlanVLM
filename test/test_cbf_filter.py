"""Unit + closed-loop tests for the CBF-QP safety filter (no ROS graph, no sim, no API calls).

Run from the workspace root with the workspace sourced (the same requirement the scripts/ harnesses
have, since `nodes/` reaches sys.path via the colcon install):

    source install/setup.bash
    python3 -m pytest src/CoPlanVLM/test/ -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from obs_seg import FREE, OCCUPIED
from obs_seg.occupancy import cell_to_world
from node_Control.cbf_filter import (
    CBFConfig, CBFSafetyFilter, barrier, blocked_cell_centres, build_rows, lookahead_jacobian,
    lookahead_point, resolve_peers, select_obstacles,
)

DT = 0.1          # the control period node_Control runs at


def cfg(**kw) -> CBFConfig:
    return CBFConfig(**kw)


# ── 1. look-ahead geometry ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theta, expected", [
    (0.0,          (1.1, 2.0)),
    (math.pi / 2,  (1.0, 2.1)),
    (math.pi,      (0.9, 2.0)),
])
def test_lookahead_point_at_cardinal_headings(theta, expected):
    assert lookahead_point(1.0, 2.0, theta, 0.1) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, math.pi / 2, math.pi])
def test_lookahead_jacobian_at_cardinal_headings(theta):
    ell = 0.1
    G = lookahead_jacobian(theta, ell)
    c, s = math.cos(theta), math.sin(theta)
    assert np.allclose(G, [[c, -ell * s], [s, ell * c]], atol=1e-12)
    # det G = ell: invertible for ell > 0, and the omega column is ell times weaker than the v
    # column — the reason this filter brakes rather than steers.
    assert np.linalg.det(G) == pytest.approx(ell, abs=1e-12)
    assert np.linalg.norm(G[:, 1]) == pytest.approx(ell * np.linalg.norm(G[:, 0]), abs=1e-12)


# ── 2. barrier values ─────────────────────────────────────────────────────────────────────

def test_barrier_inside_on_and_outside_radius():
    p = np.array([0.0, 0.0])
    centres = np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]])   # inside / on / outside
    h = barrier(p, centres, np.full(3, 1.0))
    assert h[0] < 0
    assert h[1] == pytest.approx(0.0, abs=1e-12)
    assert h[2] > 0
    assert h == pytest.approx([0.25 - 1.0, 0.0, 4.0 - 1.0], abs=1e-12)


# ── 3-4. constraint signs and the clamp ───────────────────────────────────────────────────

def test_cbf_row_sign_for_obstacle_ahead():
    """Obstacle dead ahead: driving forward must be penalised, turning must have no effect."""
    theta = 0.0
    p_l = lookahead_point(0.0, 0.0, theta, 0.1)
    G = lookahead_jacobian(theta, 0.1)
    obstacle = np.array([[1.0, 0.0]])
    A, b, h = build_rows(p_l, G, obstacle, np.array([0.3]), np.array([1.5]))

    assert h[0] > 0                       # still safe
    assert A[0, 0] > 0                    # +v pushes A u up toward the bound => forward penalised
    assert A[0, 1] == pytest.approx(0.0, abs=1e-12)   # omega cannot change h at all here
    assert b[0] == pytest.approx(1.5 * h[0])


def test_violated_row_is_clamped_to_non_worsening():
    """h < 0 must give b = 0 ("do not get worse"), never a positive demand to move away."""
    p_l = lookahead_point(0.0, 0.0, 0.0, 0.1)
    G = lookahead_jacobian(0.0, 0.1)
    obstacle = np.array([[0.2, 0.0]])                 # well inside R = 0.5
    A, b, h = build_rows(p_l, G, obstacle, np.array([0.5]), np.array([1.5]))
    assert h[0] < 0
    assert b[0] == 0.0
    # u = 0 satisfies it, which is what keeps the QP feasible from inside the unsafe set.
    assert (A @ np.zeros(2))[0] <= b[0] + 1e-12


def test_multiple_obstacles_stack_into_correct_shapes():
    p_l = lookahead_point(0.0, 0.0, 0.0, 0.1)
    G = lookahead_jacobian(0.0, 0.1)
    centres = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    A, b, h = build_rows(p_l, G, centres, np.full(3, 0.3), np.full(3, 1.5))
    assert A.shape == (3, 2) and b.shape == (3,) and h.shape == (3,)
    for i, centre in enumerate(centres):
        expect = -2.0 * (p_l - centre) @ G
        assert A[i] == pytest.approx(expect, abs=1e-12)


# ── 5. no constraints => nominal preserved exactly ────────────────────────────────────────

def test_no_constraints_returns_nominal_exactly():
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.2), robot_pose=(0.0, 0.0, 0.0))
    assert (r.v, r.omega) == (0.15, 0.2)          # exact, not approx
    assert r.status == "no-constraints"
    assert not r.modified and not r.peer_active and r.n_obstacles == 0


def test_distant_obstacle_does_not_significantly_modify_command():
    f = CBFSafetyFilter(cfg())
    nominal = (0.15, 0.05)
    r = f.filter(nominal=nominal, robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[1.4, 0.0]]))
    assert r.n_obstacles == 1                      # inside query radius, so it IS constrained
    assert (r.v, r.omega) == pytest.approx(nominal, abs=1e-6)   # but far from binding


# ── 6-7. obstacle ahead brakes; obstacle behind does not ──────────────────────────────────

def test_obstacle_ahead_reduces_unsafe_forward_command():
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[0.30, 0.0]]))
    assert r.modified
    assert r.v < 0.15
    assert r.fallback == ""


def test_obstacle_behind_does_not_block_forward_motion():
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[-0.30, 0.0]]))
    assert r.v == pytest.approx(0.15, abs=1e-6)    # driving away is always allowed


# ── 8. peer handling ──────────────────────────────────────────────────────────────────────

def test_peer_uses_its_configured_combined_radius():
    c = cfg()
    assert c.r_robot == pytest.approx(0.10 + 0.17 + 0.17 + 0.10)
    # The lookahead term is what preserves body-to-body clearance on a rear approach.
    assert c.r_robot - (c.lookahead + c.radius_self) - c.radius_other == pytest.approx(c.robot_margin)

    f = CBFSafetyFilter(c)
    peer = (0.6, 0.0)
    r_peer = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0), peer_points=[peer])
    assert r_peer.peer_active and r_peer.n_obstacles == 0
    assert r_peer.v < 0.15                          # peer at 0.6 m is inside R_robot = 0.54 + margin

    # Same geometry as a map obstacle uses the SMALLER radius, so it is not yet binding.
    r_map = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0),
                     obstacle_points=np.array([peer]))
    assert r_map.v > r_peer.v


def test_missing_peer_pose_drops_that_constraint():
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0), peer_points=[None])
    assert not r.peer_active
    assert (r.v, r.omega) == (0.15, 0.0)
    assert r.fallback == ""


# ── 9. peer roster / self-avoidance guard ─────────────────────────────────────────────────

@pytest.mark.parametrize("name, roster, expected", [
    ("raph",   ["raph", "donnie"], ["donnie"]),
    ("donnie", ["raph", "donnie"], ["raph"]),
    ("raph",   ["raph", "donnie", "leo"], ["donnie", "leo"]),
    ("",       ["raph", "donnie"], []),          # unset -> no peer, never self
    ("Raph",   ["raph", "donnie"], []),          # typo   -> no peer, never self
    ("mikey",  ["raph", "donnie"], []),
])
def test_resolve_peers_never_returns_self(name, roster, expected):
    peers = resolve_peers(name, roster)
    assert peers == expected
    assert name not in peers        # the brick-wall failure mode: self-avoidance stops the robot


# ── 10. bounds, validation, fallback ──────────────────────────────────────────────────────

def test_velocity_bounds_are_respected():
    c = cfg(v_max=0.15, omega_max=0.3)
    f = CBFSafetyFilter(c)
    r = f.filter(nominal=(0.15, 0.3), robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[0.35, 0.35]]))
    assert c.v_min - 1e-9 <= r.v <= c.v_max + 1e-9
    assert -c.omega_max - 1e-9 <= r.omega <= c.omega_max + 1e-9


def test_non_finite_nominal_returns_safe_stop():
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(float("nan"), 0.0), robot_pose=(0.0, 0.0, 0.0))
    assert (r.v, r.omega) == (0.0, 0.0)
    assert r.fallback and f.failure_count == 1


def test_solver_failure_returns_zero_safe_stop(monkeypatch):
    import node_Control.cbf_filter as mod
    monkeypatch.setattr(mod, "solve_qp",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[0.3, 0.0]]))
    assert (r.v, r.omega) == (0.0, 0.0)
    assert "boom" in r.fallback and f.failure_count == 1


def test_post_validation_rejects_a_constraint_violating_solution(monkeypatch):
    """If the solver ever returns something infeasible, the filter must safe-stop, not publish it."""
    import node_Control.cbf_filter as mod
    monkeypatch.setattr(mod, "solve_qp", lambda *a, **k: (np.array([0.15, 0.0]), "solved"))
    f = CBFSafetyFilter(cfg())
    r = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, 0.0),
                 obstacle_points=np.array([[0.22, 0.0]]))
    assert (r.v, r.omega) == (0.0, 0.0)
    assert "constraint violated" in r.fallback


def test_returned_command_satisfies_all_constraints():
    """Sweep geometries and assert A u <= b + tol on every solve that was not a fallback."""
    rng = np.random.default_rng(0)
    c = cfg()
    f = CBFSafetyFilter(c)
    for _ in range(200):
        theta = rng.uniform(-math.pi, math.pi)
        obstacles = rng.uniform(-1.2, 1.2, size=(rng.integers(1, 6), 2))
        nominal = (rng.uniform(0.0, c.v_max), rng.uniform(-c.omega_max, c.omega_max))
        r = f.filter(nominal=nominal, robot_pose=(0.0, 0.0, theta), obstacle_points=obstacles)
        if r.fallback:
            continue
        p_l = lookahead_point(0.0, 0.0, theta, c.lookahead)
        G = lookahead_jacobian(theta, c.lookahead)
        sel = select_obstacles(obstacles, (0.0, 0.0), p_l, query_radius=c.query_radius,
                               downsample=c.downsample, max_count=c.max_obstacles)
        A, b, _ = build_rows(p_l, G, sel, np.full(len(sel), c.r_obstacle),
                             np.full(len(sel), c.gamma_obstacle))
        assert np.all(A @ np.array([r.v, r.omega]) <= b + c.tol)


# ── 11. the freeze case: starting inside the unsafe set ───────────────────────────────────

def test_starting_inside_unsafe_set_stays_feasible_and_can_escape():
    """The issue this filter's clamp exists for: h < 0, obstacle dead ahead, no reverse."""
    c = cfg()
    f = CBFSafetyFilter(c)
    obstacle = np.array([[0.12, 0.0]])              # inside R_obstacle of the look-ahead point

    r = f.filter(nominal=(0.15, 0.2), robot_pose=(0.0, 0.0, 0.0), obstacle_points=obstacle)
    assert r.fallback == "", "QP must stay feasible from inside the unsafe set"
    assert r.min_h < 0
    assert r.v == pytest.approx(0.0, abs=1e-6), "must not drive deeper in"

    # Turned away, forward motion is permitted again -> it escapes by rotating.
    r_away = f.filter(nominal=(0.15, 0.0), robot_pose=(0.0, 0.0, math.pi),
                      obstacle_points=obstacle)
    assert r_away.fallback == ""
    assert r_away.v > 0.0


# ── 12. obstacle selection ────────────────────────────────────────────────────────────────

def test_selection_filters_by_radius_buckets_and_caps():
    grid_pts = np.array([[x, 0.0] for x in np.arange(0.0, 3.0, 0.05)])
    sel = select_obstacles(grid_pts, (0.0, 0.0), np.array([0.1, 0.0]),
                           query_radius=1.0, downsample=0.25, max_count=3)
    assert len(sel) == 3
    assert np.all(np.abs(sel[:, 0]) <= 1.0 + 1e-9)          # radius honoured
    # capped set is the CLOSEST ones, so dropping the rest can never drop a binding constraint
    d = np.abs(sel[:, 0] - 0.1)
    assert np.all(np.diff(np.sort(d)) >= 0)
    assert d.max() < 0.9


def test_selection_keeps_bucket_point_nearest_the_lookahead():
    pts = np.array([[0.30, 0.0], [0.31, 0.0], [0.32, 0.0]])   # all one 0.25 m bucket
    sel = select_obstacles(pts, (0.0, 0.0), np.array([0.1, 0.0]),
                           query_radius=2.0, downsample=0.25, max_count=8)
    assert len(sel) == 1
    assert sel[0] == pytest.approx([0.30, 0.0])


def test_blocked_cell_centres_matches_cell_to_world():
    """The vectorised transform must agree with obs_seg's scalar one, cell for cell."""
    meta = {"resolution": 0.05, "origin_x": -1.0, "origin_y": -2.0, "width": 6, "height": 4}
    grid = np.full((4, 6), FREE, dtype=np.int8)
    blocked = [(1, 2), (3, 0), (5, 3)]           # (gx, gy)
    for gx, gy in blocked:
        grid[gy, gx] = OCCUPIED
    got = blocked_cell_centres(grid, meta)
    want = np.array([cell_to_world(gx, gy, meta) for gx, gy in blocked])
    assert np.allclose(np.sort(got, axis=0), np.sort(want, axis=0))


# ── 13-14. closed-loop integration ────────────────────────────────────────────────────────

def _run_closed_loop(f: CBFSafetyFilter, *, goal, obstacle_points=None, peer=None,
                     start=(0.0, 0.0, 0.0), steps=400):
    """Step the unicycle under nominal-controller + filter. Returns (metrics, trace).

    The nominal controller mirrors node_Control: bearing-to-goal P heading term with kp_yaw, a
    45 deg heading gate on forward motion, kP_pos on distance, both clamped — so what is exercised
    here is the same command the filter sees on the robot. Crucially it also mirrors the node's
    REGIME GATING: while turning in place the filter is not applied at all, because a circular
    robot pivoting about its centre cannot change any obstacle distance.

    Returns `(metrics, trace)` where metrics carries:
        min_h_driving  smallest barrier value over the steps where the robot was TRANSLATING.
                       h is not checked while pivoting: the look-ahead point swings during a turn,
                       so h may legitimately dip there without the body moving at all.
        min_body_gap   smallest centre-to-centre distance to any obstacle/peer over ALL steps —
                       the physical quantity that actually has to hold.
    """
    c = f.cfg
    x, y, th = start
    kp_yaw, kp_pos, gate = 1.5, 0.75, math.radians(45.0)
    centres, radii = [], []
    if peer is not None:
        centres.append(np.asarray(peer, dtype=float))
        radii.append(c.r_robot)
    if obstacle_points is not None:
        for p in np.atleast_2d(obstacle_points):
            centres.append(np.asarray(p, dtype=float))
            radii.append(c.r_obstacle)
    centres_arr = np.array(centres)
    radii_arr = np.array(radii)

    min_h_driving = math.inf
    min_body_gap = math.inf
    trace = []
    for _ in range(steps):
        dx, dy = goal[0] - x, goal[1] - y
        dist = math.hypot(dx, dy)
        yaw_err = math.atan2(math.sin(math.atan2(dy, dx) - th), math.cos(math.atan2(dy, dx) - th))
        omega_nom = max(-c.omega_max, min(c.omega_max, kp_yaw * yaw_err))

        # Same two regimes as node_Control.publish_velocity.
        turning_in_place = abs(yaw_err) > gate
        v_nom = 0.0 if turning_in_place else max(0.0, min(c.v_max, kp_pos * dist))

        if turning_in_place:
            v_cmd, w_cmd = v_nom, omega_nom          # pivot: unfiltered, by design
        else:
            r = f.filter(nominal=(v_nom, omega_nom), robot_pose=(x, y, th),
                         peer_points=[peer] if peer is not None else (),
                         obstacle_points=obstacle_points)
            assert r.fallback == "", f"unexpected fallback: {r.fallback}"
            v_cmd, w_cmd = r.v, r.omega

        x += v_cmd * math.cos(th) * DT
        y += v_cmd * math.sin(th) * DT
        th += w_cmd * DT

        if not turning_in_place:
            p_l = lookahead_point(x, y, th, c.lookahead)
            min_h_driving = min(min_h_driving, float(barrier(p_l, centres_arr, radii_arr).min()))
        body = np.hypot(centres_arr[:, 0] - x, centres_arr[:, 1] - y).min()
        min_body_gap = min(min_body_gap, float(body))
        trace.append((x, y, th))
    return {"min_h_driving": min_h_driving, "min_body_gap": min_body_gap}, trace


def test_static_map_obstacle_integration_stays_outside_boundary():
    """Nominal controller drives straight at an obstacle; the filter must stop it short."""
    c = cfg()
    f = CBFSafetyFilter(c)
    obstacle = np.array([[1.0, 0.0]])
    m, trace = _run_closed_loop(f, goal=(2.0, 0.0), obstacle_points=obstacle)
    # Bounded by the solver's own feasibility tolerance, not by an arbitrary epsilon: the ZCBF is
    # enforced per-sample while the trajectory is integrated discretely, so h can undershoot by
    # about that much. At these radii a full `tol` dip is well under a millimetre of distance.
    assert m["min_h_driving"] >= -c.tol, f"barrier went negative while driving ({m})"
    # The physical claim: the body never reaches the obstacle. `infl` is already dilated by
    # INFLATION_RADIUS, so a blocked-cell centre is not the true obstacle — the body merely has to
    # stay off the cell itself.
    assert m["min_body_gap"] > c.r_obstacle, f"body entered the obstacle ({m})"
    assert trace[-1][0] < 1.0, "robot should stop short of the obstacle, not pass through it"


def test_stationary_peer_integration_stays_outside_boundary():
    """Same, with the other robot parked on the straight-line path."""
    c = cfg()
    f = CBFSafetyFilter(c)
    peer = (1.0, 0.0)
    m, trace = _run_closed_loop(f, goal=(2.0, 0.0), peer=peer)
    assert m["min_h_driving"] >= -c.tol, f"barrier went negative while driving ({m})"
    # The claim that actually matters: the two BODIES never touch, at any point in the run —
    # including during unfiltered pivots, which is exactly what skipping the filter there assumes.
    assert m["min_body_gap"] > c.radius_self + c.radius_other, f"robots touched ({m})"
