#!/usr/bin/env python3
"""
translator_node.py — ROS 2 entry point for label -> metric path planning.

Subscribes to `/vlm_plan`, a JSON wrapper naming the planner and the per-robot label lists:

    {"planner": "astar" | "coverage", "routes": {robot_name: [labels]}}

(A bare {robot_name: [labels]} object is also accepted and defaults to the astar planner.) The node
reads the occupancy snapshot exec wrote to _OCC_FILE — the pre-inflated grid plus the camera frame it
was computed from — turns each robot's labels into a world-frame reference route (robot pose
prepended), and DELEGATES the path computation to the named planner module:

    astar    -> astar_proj    : project-to-free + pairwise A* + LOS thinning (ordered waypoints)
    coverage -> coverage_proj : project-to-free + open-TSP ordering + A* stitching (region sweep)

Each planner exposes build_reference(...) / plan(reference_xy, ctx, meta, params) / save_debug(...).
Inflation is done once upstream (exec.run_segmentation), so this node never inflates — it just wraps
the inflated grid as ctx = {"infl": infl}. This node keeps everything ROS-specific (I/O, pose
caching; debug artifacts go through the shared debug_io writer) and stays thin; the algorithms live in the
planner modules. Each robot's result is published as a Float32MultiArray [x1,y1,x2,y2,...] on
/<robot>/waypoint_path.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

import debug_io
from coord_transform import ned_to_world
from obs_seg.occupancy import render_inflation_overlay
from .grid_planner_utils import DEFAULT_MIN_GOAL_SEPARATION, separate_goal
# Which overlay the VLM saw — needed so vlm_selections.png is drawn in the matching style.
# /vlm_plan does not carry it, so both nodes read it from the shared prompt_gen constant.
from node_Executive_API.prompt_gen import PRODUCTION_MAP_OVERLAY

# Occupancy snapshot written by exec (map_gen.run_segmentation): pre-inflated planning grid plus the
# raw grid / pixel labels / meta for debug. Read here instead of rebuilding or re-inflating.
_OCC_FILE = os.path.normpath(os.path.join(
    get_package_share_directory('coplan_vlm'), '..', '..', '..', '..',
    'debug', 'coplan_vlm_occupancy.npz'))

from . import astar_proj, coverage_proj
from .astar_proj import PARAMS as ASTAR_PARAMS
from .coverage_proj import PARAMS as COVERAGE_PARAMS

_PLANNERS = {"astar": astar_proj, "coverage": coverage_proj}


# ──────────────────────────────────────────────────────────────────────────────
class SimplePathTranslator(Node):
    """Pixel-label -> metre waypoint translator; delegates planning to a planner module."""

    def __init__(self):
        super().__init__("simple_path_translator")

        # ─── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("grid_csv", "")
        self.declare_parameter("label_column", "cell")
        self.declare_parameter("u_column", "center_x")
        self.declare_parameter("v_column", "center_y")
        self.declare_parameter("vlm_plan_topic", "/vlm_plan")
        # One robot per entry; each robot's plan is published to /<name>/waypoint_path.
        self.declare_parameter("robot_names", ["raph", "donnie"])

        # Which overhead camera calibration to use for pixel<->world (grid labels, occupancy):
        # "gazebo" (sim) or "lab_test" (hardware). Set by the launch file. Poses arrive in NED.
        self.declare_parameter("camera", "gazebo")

        # ─── Obstacle-aware planning params ───────────────────────────────────
        # Resolution is NOT a param here: it travels in the occupancy snapshot's meta (written by
        # exec) and is read from there per message, so translate never needs its own copy.
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "")                 # set by launch; empty = off

        # Minimum distance (metres) between two robots' FINAL goals. The VLM often sends both robots
        # to the same location; when a later-planned robot's goal falls within this radius of one
        # already claimed, its goal is displaced to the nearest free cell outside. 0 disables it.
        # Default from grid_planner_utils so the offline harness uses the same number.
        self.declare_parameter("min_goal_separation", DEFAULT_MIN_GOAL_SEPARATION)

        # ─── Planner knobs (astar + coverage) ────────────────────────────────
        # The planner is chosen per message (from the /vlm_plan "planner" field), not by a param.
        for k, v in {**ASTAR_PARAMS, **COVERAGE_PARAMS}.items():
            self.declare_parameter(k, v)

        # ─── Read parameters once ─────────────────────────────────────────────
        grid_csv       = self.get_parameter("grid_csv").get_parameter_value().string_value
        self.lbl_col   = self.get_parameter("label_column").get_parameter_value().string_value
        self.u_col     = self.get_parameter("u_column").get_parameter_value().string_value
        self.v_col     = self.get_parameter("v_column").get_parameter_value().string_value
        vlm_plan_topic = self.get_parameter("vlm_plan_topic").get_parameter_value().string_value
        self.robot_names = list(self.get_parameter("robot_names").value)

        self.camera_name: str = self.get_parameter("camera").get_parameter_value().string_value

        self.save_debug = self.get_parameter("save_debug").value
        self.debug_dir = self.get_parameter("debug_dir").value
        self.min_goal_separation = float(self.get_parameter("min_goal_separation").value)

        # All planner tunables gathered into one dict, passed to the per-message planner's
        # plan()/save_debug(). The planner module itself is selected per message in _on_path_msg.
        self.plan_params = {k: self.get_parameter(k).value for k in {**ASTAR_PARAMS, **COVERAGE_PARAMS}}

        # ─── Debug output dir ─────────────────────────────────────────────────
        # The overlay the VLM actually saw is written by exec (marks_overlay.png) — it is the node
        # that renders it. This node writes the frame, the occupancy view and the route figures, all
        # on the frame exec bundled into the occupancy snapshot, so every figure of one plan shows
        # the same instant as marks_overlay.png.
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            self.get_logger().info(f"Debug artifacts -> {self.debug_dir}")
        elif self.save_debug:
            self.get_logger().warn("save_debug=true but debug_dir empty; debug saving disabled.")

        # ─── Load grid pixel data ─────────────────────────────────────────────
        # Per-cell pixel centres for label -> world lookups (the pixel<->world calibration itself
        # lives in coord_transform).
        self.grid_px: Dict[str, np.ndarray] = self._load_grid_csv(grid_csv)

        # ─── ROS 2 I/O ────────────────────────────────────────────────────────
        # No overhead-image subscription: debug figures are drawn on the frame exec bundles into the
        # occupancy snapshot, so they are guaranteed to show the instant the VLM saw. Subscribing
        # here would only re-stream full-resolution frames over the network to duplicate a picture
        # this node already has on disk — and an unsynchronised, un-undistorted one at that.
        self.sub = self.create_subscription(String, vlm_plan_topic, self._on_path_msg, 10)
        self.world_pubs: Dict[str, object] = {
            name: self.create_publisher(Float32MultiArray, f"/{name}/waypoint_path", 10)
            for name in self.robot_names
        }

        # Cache each robot's latest world (x, y) so every plan starts at the robot's current
        # position. /<name>/ned/pose_stamped is published BEST_EFFORT, so match that QoS.
        pose_qos = QoSProfile(depth=1)
        pose_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.robot_xy: Dict[str, object] = {name: None for name in self.robot_names}
        for name in self.robot_names:
            self.create_subscription(
                PoseStamped, f"/{name}/ned/pose_stamped", self._make_pose_cb(name), pose_qos)

        self.get_logger().info("✓")

    # ------------------------------------------------------------------
    def _make_pose_cb(self, name: str):
        """Cache the robot's position in the world frame (NED pose -> world)."""
        def _cb(msg: PoseStamped) -> None:
            self.robot_xy[name] = ned_to_world(msg.pose.position.x, msg.pose.position.y)
        return _cb

    # ------------------------------------------------------------------
    def _separate_goal(self, goal_xy, claimed, infl, meta, max_shift):
        """This node's `min_goal_separation` bound to grid_planner_utils.separate_goal.

        The algorithm lives in the ROS-free library so scripts/test_pipeline.py runs exactly the
        same separation; see that function for the scratch-grid and max_shift rationale.
        """
        return separate_goal(goal_xy, claimed, infl, meta,
                             min_separation=self.min_goal_separation, max_shift=max_shift)

    # ------------------------------------------------------------------
    def _snapshot_frame(self, snap) -> np.ndarray | None:
        """Return the BGR frame every debug figure of this plan is drawn on, or None if absent.

        The frame comes from exec's occupancy snapshot and nowhere else: it is the undistorted image
        the segmentation, the inflated grid and exec's marks_overlay.png all describe, so every
        artifact of one plan shows a single instant. exec writes it in run_segmentation before
        publishing /vlm_plan, so a snapshot reached from _on_path_msg always carries it; None means
        an exec too old to write it, and the plan is executed with no debug figures rather than with
        misleading ones.
        """
        if "frame_bgr" not in snap.files:
            self.get_logger().warn(
                "Occupancy snapshot has no frame_bgr (exec too old?) — skipping debug figures for "
                "this plan; planning and publishing are unaffected.")
            return None
        return np.ascontiguousarray(snap["frame_bgr"])

    # ------------------------------------------------------------------
    def _load_grid_csv(self, csv_path: str) -> Dict[str, np.ndarray]:
        """Read CSV and build {label: [u, v, 1]} dict."""
        centres: Dict[str, np.ndarray] = {}
        p = Path(csv_path)
        if not p.exists():
            self.get_logger().error(f"Grid CSV not found: {csv_path}")
            return centres
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    label = row[self.lbl_col]
                    u = float(row[self.u_col])
                    v = float(row[self.v_col])
                    centres[label] = np.array([u, v, 1.0])
                except KeyError:
                    self.get_logger().error(
                        "CSV missing one of the required columns "
                        f"({self.lbl_col}, {self.u_col}, {self.v_col}).")
                    break
                except ValueError:
                    self.get_logger().warn(f"Skipping malformed row: {row}")
        self.get_logger().debug(f"Loaded {len(centres)} grid centres.")
        return centres

    # ------------------------------------------------------------------
    def _build_reference(self, name, labels, planner):
        """Robot pose (if known) + label centroids -> world reference route. Returns (ref, cur_xy).

        Delegates label->world conversion to the chosen planner's build_reference (astar trims
        already-passed centroids; coverage keeps all, since the TSP reorders them).
        """
        cur_xy = self.robot_xy.get(name)
        if cur_xy is None:
            self.get_logger().warn(
                f"[{name}] No current pose yet; route will start at the first label.")
        ref, unknown = planner.build_reference(labels, cur_xy, self.grid_px, camera=self.camera_name)
        for lbl in unknown:
            self.get_logger().warn(f"[{name}] Unknown label '{lbl}' – skipping.")
        return ref, cur_xy

    # ------------------------------------------------------------------
    def _publish(self, name, world_path, cur_xy) -> None:
        """Publish a world path as Float32MultiArray [x1,y1,...], starting at the actual pose."""
        flat: List[float] = []
        for (x, y) in world_path:
            flat += [float(x), float(y)]
        # The controller's route should begin exactly where the robot is.
        if cur_xy is not None and len(flat) >= 2:
            flat[0], flat[1] = float(cur_xy[0]), float(cur_xy[1])
        arr = Float32MultiArray()
        arr.data = flat
        self.world_pubs[name].publish(arr)

    # ──────────────────────────────────────────────────────────────────
    #  Subscription callback
    # ──────────────────────────────────────────────────────────────────
    def _on_path_msg(self, msg: String) -> None:
        """Plan a path for each robot in the /vlm_plan message via the message-selected planner."""
        try:
            data = json.loads(msg.data)
            assert isinstance(data, dict)
        except Exception as e:
            self.get_logger().error(
                f"Bad /vlm_plan message (expect {{planner, routes}} or {{robot: [labels]}}): {e}")
            return

        # Unwrap the {planner, routes} wrapper; a bare {robot: [labels]} object defaults to astar.
        if "routes" in data:
            planner_name = data.get("planner", "astar")
            plans = data.get("routes", {})
        else:
            planner_name, plans = "astar", data
        if not isinstance(plans, dict):
            self.get_logger().error(f"/vlm_plan 'routes' is not an object; got {plans!r}.")
            return
        planner = _PLANNERS.get(planner_name)
        if planner is None:
            self.get_logger().warn(
                f"Unknown planner '{planner_name}' in /vlm_plan; skipping.")
            return

        # Read the pre-inflated occupancy snapshot written by exec (map_gen.run_segmentation). exec
        # runs CLIPSeg and inflates once; translate consumes the shared file so neither step repeats.
        if not os.path.exists(_OCC_FILE):
            self.get_logger().warn(
                "Occupancy file not found — has exec published a plan yet? Skipping.")
            return
        snap = np.load(_OCC_FILE)
        # snap also carries "pix_labels" (raw CLIPSeg output); nothing here consumes it since the
        # segmentation PNG was dropped — map_gen still bundles it for offline inspection.
        grid = snap["grid"]
        # Post-override, pre-inflation grid — the red layer for the inflation overlay. Fall back to the
        # raw grid for older snapshots that predate the `cleared` key.
        cleared = snap["cleared"] if "cleared" in snap.files else grid
        meta = {"resolution": float(snap["resolution"]),
                "origin_x": float(snap["origin_x"]), "origin_y": float(snap["origin_y"]),
                "width": int(snap["width"]), "height": int(snap["height"])}
        ctx = {"infl": snap["infl"]}   # inflated upstream; planners never inflate

        # Debug context shared by the per-robot artifacts and the combined figure written after the
        # loop. `base` is built once here (not per robot) because the combined figure needs it too,
        # and it comes from the snapshot — exec captured that frame before the classifier call, and
        # the occupancy above plus exec's marks_overlay.png both describe it, so all of this run's
        # figures show one instant.
        base = self._snapshot_frame(snap) if (self.save_debug and self.debug_dir) else None
        debug_on = base is not None
        world_paths: Dict[str, list] = {}
        overlay = None

        # ─── Goal separation: reserve the poses of robots that will NOT move ──────────────────
        # A robot holding position still occupies its cell, so a mover must not be routed onto it.
        # This runs BEFORE the loop because claiming inside it would be too late: if the stationary
        # robot happened to be planned second, the mover's colliding path would already have been
        # published. Reserving up front also makes the outcome independent of `routes` key order.
        # "Will not move" covers an empty label list ("stay in place"), a robot the model omitted,
        # and a malformed route — the robot is physically there in every case.
        claimed: List[tuple] = []
        if self.min_goal_separation > 0:
            for rname in self.robot_names:
                route = plans.get(rname)
                if isinstance(route, list) and route:
                    continue                      # moving; its goal is claimed after it plans
                pose = self.robot_xy.get(rname)
                if pose is None:
                    self.get_logger().warn(
                        f"[{rname}] holding position but no pose received yet; cannot reserve it.")
                    continue
                claimed.append(pose)

        # Robot-independent artifacts: written ONCE per plan at the debug_dir top level, not once
        # per robot (they are byte-identical for every robot — ~15 MB of redundant PNG encoding).
        # Done BEFORE the loop so the frame and the occupancy view exist even when no robot manages
        # to plan, which is exactly when they are most useful. `overlay` is also reused as each
        # robot's route_planned.png canvas below, so it is rendered only this once.
        if debug_on:
            warn = self.get_logger().warn
            overlay = render_inflation_overlay(base, cleared, ctx["infl"], meta, self.camera_name)
            debug_io.save_raw_overhead(self.debug_dir, base, on_error=warn)
            debug_io.save_inflation_overlay(self.debug_dir, overlay, on_error=warn)

        for name, labels in plans.items():
            if name not in self.world_pubs:
                self.get_logger().warn(
                    f"No publisher for robot '{name}' (not in robot_names); skipping.")
                continue
            if not isinstance(labels, list):
                self.get_logger().warn(f"[{name}] Route is not a list; skipping.")
                continue

            ref, cur_xy = self._build_reference(name, labels, planner)
            if not ref:
                self.get_logger().warn(f"[{name}] No valid waypoints in route; nothing to plan.")
                continue

            # build_reference prepends this robot's own pose, so len(ref) >= 2 means it has an
            # actual destination; a length-1 ref is "stay in place" (already reserved above).
            moving = len(ref) >= 2

            # Displace the goal off any already-claimed goal BEFORE planning, so A* runs once and
            # the debug figures describe the executed route with no extra plumbing. Only the final
            # reference point moves — intermediate waypoints and the route itself are untouched.
            if moving and planner_name == "astar" and self.min_goal_separation > 0 and claimed:
                adjusted = self._separate_goal(
                    ref[-1], claimed, ctx["infl"], meta,
                    max_shift=float(self.plan_params["projection_radius"]))
                if adjusted is None:
                    self.get_logger().warn(
                        f"[{name}] goal is within {self.min_goal_separation:.2f} m of another robot "
                        "and no free cell far enough away was found; keeping the original goal.")
                elif math.dist(adjusted, ref[-1]) > 1e-9:
                    self.get_logger().info(
                        f"[{name}] goal moved {math.dist(adjusted, ref[-1]):.2f} m to clear another "
                        f"robot: ({ref[-1][0]:.2f}, {ref[-1][1]:.2f}) -> "
                        f"({adjusted[0]:.2f}, {adjusted[1]:.2f})")
                    ref[-1] = adjusted

            world_path, dbg = planner.plan(ref, ctx, meta, self.plan_params)
            for w in dbg.get("warnings", []):
                self.get_logger().warn(f"[{name}] {w}")
            if not world_path:
                self.get_logger().warn(f"[{name}] planner produced no path; skipping.")
                continue

            self._publish(name, world_path, cur_xy)
            world_paths[name] = world_path
            if moving:
                # The EXECUTED endpoint, not the adjusted reference: plan() may project it further,
                # and what the next robot must avoid is where this one actually stops.
                claimed.append(world_path[-1])
            self.get_logger().info(
                f"[{name}] Planned path: {len(world_path)} waypoints "
                f"(grid {meta['width']}x{meta['height']} @ {meta['resolution']} m, "
                f"planner={planner_name}).")

            if debug_on:
                try:
                    out_dir = os.path.join(self.debug_dir, name)
                    os.makedirs(out_dir, exist_ok=True)
                    # The VLM's raw picks, drawn in the style of the overlay it actually saw.
                    debug_io.save_vlm_selections(out_dir, base, PRODUCTION_MAP_OVERLAY,
                                                 labels, self.grid_px, on_error=warn)
                    dbg["start_world"] = cur_xy
                    # `cleared` (post-override occupancy) is the overlay's red layer; `overlay` is
                    # the one rendered above, reused as this robot's route_planned.png canvas.
                    planner.save_debug(out_dir, base, cleared, meta, ctx, dbg, self.plan_params,
                                       camera=self.camera_name, overlay=overlay)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"[{name}] Debug save failed: {exc}")

        # Combined figure for the whole plan (all robots on one image), at the debug_dir top level
        # alongside the per-robot subdirectories — the same layout the offline harness produces.
        if debug_on and world_paths:
            debug_io.save_paths_with_waypoints(
                os.path.join(self.debug_dir, "robot_paths_waypoints.png"),
                base, world_paths, plans, self.grid_px, self.camera_name,
                on_error=self.get_logger().warn)


# ──────────────────────────────────────────────────────────────────────────────
#  Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = SimplePathTranslator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
