#!/usr/bin/env python3
"""
simple_path_translator.py — ROS 2 entry point for label -> metric path planning.

Subscribes to `/vlm_plan`, a JSON wrapper naming the planner and the per-robot label lists:

    {"planner": "astar" | "coverage", "routes": {robot_name: [labels]}}

(A bare {robot_name: [labels]} object is also accepted and defaults to the astar planner.) The node
reads the pre-inflated occupancy snapshot exec wrote to _OCC_FILE, turns each robot's labels into a
world-frame reference route (robot pose prepended), and DELEGATES the path computation to the named
planner module:

    astar    -> astar_proj    : project-to-free + pairwise A* + LOS thinning (ordered waypoints)
    coverage -> coverage_proj : project-to-free + open-TSP ordering + A* stitching (region sweep)

Each planner exposes build_reference(...) / plan(reference_xy, ctx, meta, params) / save_debug(...).
Inflation is done once upstream (exec.run_segmentation), so this node never inflates — it just wraps
the inflated grid as ctx = {"infl": infl}. This node keeps everything ROS-specific (I/O, pose
caching, the shared occupancy/segmentation debug images) and stays thin; the algorithms live in the
planner modules. Each robot's result is published as a Float32MultiArray [x1,y1,x2,y2,...] on
/<robot>/waypoint_path.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from coord_transform import ned_to_world
from obs_seg import FREE, OCCUPIED, UNKNOWN

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
        self.declare_parameter("image_topic", "/ids_overhead/image")
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "")                 # set by launch; empty = off

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
        self.image_topic = self.get_parameter("image_topic").value

        self.save_debug = self.get_parameter("save_debug").value
        self.debug_dir = self.get_parameter("debug_dir").value

        # All planner tunables gathered into one dict, passed to the per-message planner's
        # plan()/save_debug(). The planner module itself is selected per message in _on_path_msg.
        self.plan_params = {k: self.get_parameter(k).value for k in {**ASTAR_PARAMS, **COVERAGE_PARAMS}}

        # ─── Debug grid overlay (shared) ──────────────────────────────────────
        self._grid_overlay = None
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            try:
                gpath = os.path.join(get_package_share_directory("coplan_vlm"),
                                     "config", "transparent_grid.png")
                self._grid_overlay = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)  # BGRA
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Could not load grid overlay for debug: {exc}")
            self.get_logger().info(f"Debug artifacts -> {self.debug_dir}")
        elif self.save_debug:
            self.get_logger().warn("save_debug=true but debug_dir empty; debug saving disabled.")

        # ─── Load grid pixel data ─────────────────────────────────────────────
        # Per-cell pixel centres for label -> world lookups (the pixel<->world calibration itself
        # lives in coord_transform).
        self.grid_px: Dict[str, np.ndarray] = self._load_grid_csv(grid_csv)

        # ─── Overhead image (still subscribed for debug saves) ─────────────────
        self.bridge = CvBridge()
        self.latest_rgb = None

        # ─── ROS 2 I/O ────────────────────────────────────────────────────────
        self.create_subscription(Image, self.image_topic, self._on_image, 1)
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
    def _on_image(self, msg: Image) -> None:
        """Cache the latest overhead frame (RGB) for on-demand costmap building."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to decode overhead image: {exc}")

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

        if self.latest_rgb is None:
            self.get_logger().warn("No overhead image received yet — cannot plan. Skipping.")
            return

        # Read the pre-inflated occupancy snapshot written by exec (map_gen.run_segmentation). exec
        # runs CLIPSeg and inflates once; translate consumes the shared file so neither step repeats.
        if not os.path.exists(_OCC_FILE):
            self.get_logger().warn(
                "Occupancy file not found — has exec published a plan yet? Skipping.")
            return
        snap = np.load(_OCC_FILE)
        pix_labels = snap["pix_labels"]
        grid = snap["grid"]
        # Post-override, pre-inflation grid — the red layer for the inflation overlay. Fall back to the
        # raw grid for older snapshots that predate the `cleared` key.
        cleared = snap["cleared"] if "cleared" in snap.files else grid
        meta = {"resolution": float(snap["resolution"]),
                "origin_x": float(snap["origin_x"]), "origin_y": float(snap["origin_y"]),
                "width": int(snap["width"]), "height": int(snap["height"])}
        ctx = {"infl": snap["infl"]}   # inflated upstream; planners never inflate

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

            world_path, dbg = planner.plan(ref, ctx, meta, self.plan_params)
            for w in dbg.get("warnings", []):
                self.get_logger().warn(f"[{name}] {w}")
            if not world_path:
                self.get_logger().warn(f"[{name}] planner produced no path; skipping.")
                continue

            self._publish(name, world_path, cur_xy)
            self.get_logger().info(
                f"[{name}] Planned path: {len(world_path)} waypoints "
                f"(grid {meta['width']}x{meta['height']} @ {meta['resolution']} m, "
                f"planner={planner_name}).")

            if self.save_debug and self.debug_dir and self.latest_rgb is not None:
                try:
                    out_dir = os.path.join(self.debug_dir, name)
                    os.makedirs(out_dir, exist_ok=True)
                    base = cv2.cvtColor(np.ascontiguousarray(self.latest_rgb), cv2.COLOR_RGB2BGR)
                    self._save_common_debug(out_dir, base, pix_labels, grid)
                    dbg["start_world"] = cur_xy
                    # Pass `cleared` (post-override occupancy) as the overlay's red layer.
                    planner.save_debug(out_dir, base, cleared, meta, ctx, dbg, self.plan_params, camera=self.camera_name)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"[{name}] Debug save failed: {exc}")

    # ------------------------------------------------------------------
    def _save_common_debug(self, out_dir, base, pix_labels, grid) -> None:
        """Shared (method-agnostic) debug: raw overhead, segmentation, grid overlay, occupancy."""
        cv2.imwrite(os.path.join(out_dir, "raw_overhead.png"), base)

        # CLIPSeg segmentation overlay (green=free, red=obstacle, gray=unknown).
        seg_color = np.zeros_like(base)
        seg_color[pix_labels == FREE] = (0, 180, 0)
        seg_color[pix_labels == OCCUPIED] = (0, 0, 200)
        seg_color[pix_labels == UNKNOWN] = (128, 128, 128)
        cv2.imwrite(os.path.join(out_dir, "segmentation.png"),
                    (0.5 * base + 0.5 * seg_color).astype(np.uint8))

        # Raw + transparent grid overlay (what the VLM sees).
        if self._grid_overlay is not None:
            g = cv2.resize(self._grid_overlay, (base.shape[1], base.shape[0]))
            if g.ndim == 3 and g.shape[2] == 4:
                a = g[..., 3:4].astype(np.float32) / 255.0
                comp = (base * (1 - a) + g[..., :3] * a).astype(np.uint8)
            else:
                comp = g[..., :3]
            cv2.imwrite(os.path.join(out_dir, "grid_overlay.png"), comp)

        # True occupancy map (white=free, black=occupied, gray=unknown), reoriented to match image.
        g2 = np.flipud(grid.T)
        occ = np.full((*g2.shape, 3), 128, np.uint8)
        occ[g2 == FREE] = (255, 255, 255)
        occ[g2 == OCCUPIED] = (0, 0, 0)
        cv2.imwrite(os.path.join(out_dir, "occ_true.png"), occ)


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
