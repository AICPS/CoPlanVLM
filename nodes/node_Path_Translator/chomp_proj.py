"""CHOMP trajectory optimizer — deform the VLM reference route the minimum amount to be safe.

Minimizes  U = w_dev * deviation-from-reference
             + w_obs * obstacle-cost (from a signed distance field)
             + w_smooth * smoothness
by covariant gradient descent. The start is held fixed at the robot pose; every other point
(goal included, unless chomp_fix_goal) is free and anchored to its resampled reference point, so
a point only moves when an obstacle pushes it — minimal deviation from the VLM route.

Uses only numpy + cv2 (SDF via cv2.distanceTransform; no scipy, no grid_planner / projection).

Pluggable planner interface (shared with astar_proj):
    build(grid, meta, params) -> ctx
    plan(reference_xy, ctx, meta, params) -> (world_path, debug)
    save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params) -> None
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from coord_transform import pixel_to_world, world_to_pixel
from obs_seg import FREE, OCCUPIED, UNKNOWN
from obs_seg.occupancy import inflate_occupancy


PARAMS: dict = {
    "inflation_radius":  0.5,
    "chomp_num_points":  25,
    "chomp_w_dev":       0.01,
    "chomp_w_obs":       1.0,
    "chomp_w_smooth":    0.0,
    "chomp_w_space":     0.0,
    "chomp_clearance":   0.05,
    "chomp_step":        0.01,
    "chomp_max_step":    0.05,
    "chomp_max_iters":   500,
    "chomp_tol":         1e-4,
    "chomp_covariant":   True,
    "chomp_fix_goal":    False,
    "chomp_noise":       0.03,
}


def _p(params, key):
    v = params.get(key, PARAMS[key])
    return PARAMS[key] if v is None else v


def build(grid, meta, params):
    """Inflate the occupancy grid, then build a signed distance field (metres) from it.

    Inflating first gives CHOMP the same robot-footprint clearance as A* — the SDF then
    provides a smooth gradient for the optimizer, and chomp_clearance adds a soft margin on
    top of the inflated boundary. ctx["infl"] is exposed so callers can save the inflation
    overlay without rebuilding it separately.
    """
    res = meta["resolution"]
    infl = inflate_occupancy(grid, res, _p(params, "inflation_radius"))
    blocked = (infl == OCCUPIED) | (infl == UNKNOWN)
    free_u8 = np.where(blocked, 0, 1).astype(np.uint8)
    blk_u8 = np.where(blocked, 1, 0).astype(np.uint8)
    d_out = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5) * res   # +: free -> nearest obstacle
    d_in = cv2.distanceTransform(blk_u8, cv2.DIST_L2, 5) * res     # inside obstacle -> nearest free
    sdf = d_out - d_in
    g_row, g_col = np.gradient(sdf)        # d/d(row=gy), d/d(col=gx); array is [gy, gx]
    grad_x = g_col / res                    # world x <-> column (gx)
    grad_y = g_row / res                    # world y <-> row (gy)
    return {"sdf": sdf, "grad_x": grad_x, "grad_y": grad_y, "infl": infl}


def _bilinear(arr, cx, cy):
    """Bilinear sample of arr (indexed [row=cy, col=cx]) at continuous (cx, cy). Vectorized."""
    h, w = arr.shape
    cx = np.clip(cx, 0.0, w - 1.0)
    cy = np.clip(cy, 0.0, h - 1.0)
    x0 = np.floor(cx).astype(int)
    y0 = np.floor(cy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = cx - x0
    fy = cy - y0
    v00 = arr[y0, x0]; v01 = arr[y0, x1]
    v10 = arr[y1, x0]; v11 = arr[y1, x1]
    return (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy)
            + v10 * (1 - fx) * fy + v11 * fx * fy)


def _resample(ref, n):
    """Resample a polyline to n points evenly spaced by arc length."""
    pts = np.asarray(ref, dtype=float)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) else np.zeros((n, 2))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0.0, total, n)
    x = np.interp(s, cum, pts[:, 0])
    y = np.interp(s, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def plan(reference_xy, ctx, meta, params):
    """reference_xy: world (x,y) points [robot pose, centroids...]. Returns (world_path, debug)."""
    if len(reference_xy) < 1:
        return [], {}

    N = max(int(_p(params, "chomp_num_points")), 2)
    w_dev = float(_p(params, "chomp_w_dev"))
    w_obs = float(_p(params, "chomp_w_obs"))
    w_smooth = float(_p(params, "chomp_w_smooth"))
    w_space = float(_p(params, "chomp_w_space"))
    eps = float(_p(params, "chomp_clearance"))
    eta = float(_p(params, "chomp_step"))
    max_step = float(_p(params, "chomp_max_step"))
    max_iters = int(_p(params, "chomp_max_iters"))
    tol = float(_p(params, "chomp_tol"))
    covariant = bool(_p(params, "chomp_covariant"))
    fix_goal = bool(_p(params, "chomp_fix_goal"))
    noise = float(_p(params, "chomp_noise"))
    rng = np.random.default_rng()

    sdf = ctx["sdf"]; grad_x = ctx["grad_x"]; grad_y = ctx["grad_y"]
    res = meta["resolution"]; ox = meta["origin_x"]; oy = meta["origin_y"]

    r = _resample(reference_xy, N)     # fixed reference anchors r_i
    xi = r.copy()                       # decision variables, initialized at the reference
    # Target inter-point spacing: mean segment length of the evenly-resampled reference.
    # Fixed for the whole run so the term always pulls toward the original uniform grid.
    d_target = float(np.mean(np.linalg.norm(np.diff(r, axis=0), axis=1)))

    # Smoothness Hessian: tridiagonal Laplacian of the first-difference term, Neumann ends.
    L = (2.0 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
    L[0, 0] = 1.0
    L[N - 1, N - 1] = 1.0

    # Free indices: start (0) always fixed; goal (N-1) fixed only if requested.
    free = list(range(1, N))
    if fix_goal and free:
        free = free[:-1]
    free = np.array(free, dtype=int)
    if free.size == 0:
        world_path = [(float(x), float(y)) for (x, y) in xi]
        return world_path, {"reference_resampled": r, "optimized": xi}

    # Covariant metric = Hessian of the convex (deviation + smoothness) terms over the free
    # block, regularized by w_dev*I. The bare Laplacian is near-singular with a free goal, so its
    # inverse blows up — adding w_dev*I keeps M well-conditioned and the step stable.
    m_inv = np.linalg.inv(w_dev * np.eye(free.size) + w_smooth * L[np.ix_(free, free)]) \
        if covariant else None

    for it in range(max_iters):
        cx = (xi[free, 0] - ox) / res
        cy = (xi[free, 1] - oy) / res
        d = _bilinear(sdf, cx, cy)
        gxv = _bilinear(grad_x, cx, cy)
        gyv = _bilinear(grad_y, cx, cy)
        # CHOMP obstacle "bowl" derivative c'(d): -1 inside, ramp in [0, eps], 0 beyond eps.
        cprime = np.where(d < 0.0, -1.0, np.where(d <= eps, (d - eps) / eps, 0.0))

        g = w_dev * (xi - r) + w_smooth * (L @ xi)      # deviation + smoothness (all points)
        g[free, 0] += w_obs * cprime * gxv               # obstacle push (free points only)
        g[free, 1] += w_obs * cprime * gyv
        # Spacing term: pull each point so all inter-point distances equal d_target.
        # ∂U/∂ξ_i = 2*w_space*[(d_{i-1}-d*)*û_{i-1} - (d_i-d*)*û_i]
        diffs = np.diff(xi, axis=0)                                          # (N-1, 2)
        lens = np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-9)  # (N-1, 1)
        units = diffs / lens                                                 # (N-1, 2)
        errs = lens.squeeze() - d_target                                     # (N-1,)
        g_space = np.zeros_like(xi)
        g_space[1:]  += 2.0 * w_space * errs[:, None] * units  # ξ_i is right end of seg i-1
        g_space[:-1] -= 2.0 * w_space * errs[:, None] * units  # ξ_i is left end of seg i
        g += g_space
        gf = g[free]
        noise_t = noise * (1.0 - it / max_iters)         # anneal noise to 0 over the run
        if noise_t <= 0.0 and np.linalg.norm(gf) < tol:
            break
        delta = eta * (m_inv @ gf if covariant else gf)
        # Per-point step clamp: keeps descent stable regardless of weight/step tuning.
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(1.0, max_step / np.maximum(norms, 1e-9))
        xi[free] -= delta
        # Annealed Gaussian noise breaks saddles (e.g. a reference straight through an obstacle
        # centre) so the path commits to one side instead of sliding along; -> 0 so it settles.
        if noise_t > 0.0:
            xi[free] += noise_t * rng.standard_normal(xi[free].shape)

    world_path = [(float(x), float(y)) for (x, y) in xi]
    debug = {"reference_resampled": r, "optimized": xi}
    return world_path, debug


def _i(uv):
    return (int(round(uv[0])), int(round(uv[1])))


def save_debug(out_dir, base_bgr, grid, meta, ctx, debug, params):
    """CHOMP debug: SDF heatmap + reference-vs-optimized route overlay."""
    base = base_bgr

    # SDF heatmap (reoriented to match the overhead image, like the occupancy maps).
    sdf = ctx["sdf"]
    g2 = np.flipud(sdf.T)
    norm = cv2.normalize(g2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "sdf.png"), cv2.applyColorMap(norm, cv2.COLORMAP_JET))

    # Reference (orange) vs optimized (green) path; yellow dots on optimized; magenta start.
    rp = base.copy()
    ref = debug.get("reference_resampled", [])
    opt = debug.get("optimized", [])
    ref_px = [_i(world_to_pixel(x, y)) for (x, y) in ref]
    opt_px = [_i(world_to_pixel(x, y)) for (x, y) in opt]
    for a, b in zip(ref_px, ref_px[1:]):
        cv2.line(rp, a, b, (0, 165, 255), 2)
    for a, b in zip(opt_px, opt_px[1:]):
        cv2.line(rp, a, b, (0, 200, 0), 2)
    for p in opt_px:
        cv2.circle(rp, p, 3, (0, 255, 255), -1)
    sw = debug.get("start_world")
    if sw is not None:
        cv2.circle(rp, _i(world_to_pixel(sw[0], sw[1])), 6, (255, 0, 255), -1)
    cv2.imwrite(os.path.join(out_dir, "chomp_route.png"), rp)

    # Inflated occupancy + colour overlay (mirrors astar_proj; red=true obstacle, yellow=margin).
    infl = ctx.get("infl")
    if infl is not None:
        g2 = np.flipud(infl.T)
        occ_i = np.full((*g2.shape, 3), 128, np.uint8)
        occ_i[g2 == FREE] = (255, 255, 255)
        occ_i[g2 == OCCUPIED] = (0, 0, 0)
        cv2.imwrite(os.path.join(out_dir, "occ_inflated.png"), occ_i)

        h_img, w_img = base.shape[:2]
        uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
        xs, ys = pixel_to_world(uu, vv)
        gx = np.floor((xs - meta["origin_x"]) / meta["resolution"]).astype(np.int64)
        gy = np.floor((ys - meta["origin_y"]) / meta["resolution"]).astype(np.int64)
        inb = (gx >= 0) & (gx < meta["width"]) & (gy >= 0) & (gy < meta["height"])
        gxc = np.clip(gx, 0, meta["width"] - 1)
        gyc = np.clip(gy, 0, meta["height"] - 1)
        true_occ = inb & (grid[gyc, gxc] != FREE)
        infl_occ = inb & (infl[gyc, gxc] == OCCUPIED)
        margin = infl_occ & ~true_occ
        ov = base.copy()
        ov[margin] = (0.5 * base[margin] + 0.5 * np.array([0, 220, 220])).astype(np.uint8)
        ov[true_occ] = (0.5 * base[true_occ] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "inflation_overlay.png"), ov)
