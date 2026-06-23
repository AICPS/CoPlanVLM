#!/usr/bin/env python3
"""
simple_path_translator.py  –  v0.5 (straight‑down camera, explicit Z scale)
-------------------------------------------------------------------------
ROS 2 node that converts a JSON list of grid labels (e.g. ["L1", "M2"]) sent
on `/grid_path` into ground‑plane metres and publishes each robot's route on
`/<robot>/waypoint_path` as a `Float32MultiArray`.

**What changed in v0.5**
-----------------------
Field testing showed the intrinsics YAML still contains the *real* focal lengths
(`fx`, `fy`) rather than `fx/Z`, `fy/Z`. Therefore we restore the explicit
multiplication by the known plane height **Z** (parameter `plane_height_m`,
default ≈ 4.27 m ≃ 14 ft).

Equation
```
[u, v, 1]^T           pixel centre (homogeneous)
           K⁻¹
[x_n, y_n, 1]         normalised image coords (unit‑less)
× Z                   known plane height (metres)
[X_m, Y_m]            camera‑frame metres on the ground plane
```
Sign flips (`pixel_sign_x / pixel_sign_y`) are then applied so you can align the
final axes with the robot frame without rewriting CSV or YAML files.

Usage example
-------------
```bash
ros2 run talking_turtle simple_path_translator --ros-args \
  -p grid_csv:=/abs/path/to/grid.csv \
  -p intrinsics_yaml:=/abs/path/to/intrinsics.yaml \
  -p label_column:=cell \
  -p u_column:=center_x \
  -p v_column:=center_y \
  -p plane_height_m:=4.27            # camera ≈ 14 ft above grid
  # optional sign flips:
  -p pixel_sign_x:=-1.0 \
  -p pixel_sign_y:=-1.0
```
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
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import yaml

from coord_transform import pixel_to_world, world_to_pixel
from obs_seg import FREE, OCCUPIED, UNKNOWN
from obs_seg.segmenter import TraversabilitySegmenter
from obs_seg.occupancy import (mask_to_occupancy, inflate_occupancy,
                               world_to_cell, cell_to_world)
from grid_planner import project_to_free, astar, simplify_path_los


# ──────────────────────────────────────────────────────────────────────────────
class SimplePathTranslator(Node):
    """Pixel‑label → metre waypoint translator for a downward‑facing camera."""

    # ────────────────────────────────
    #  Initialise node and resources
    # ────────────────────────────────
    def __init__(self):
        super().__init__("simple_path_translator")

        # ─── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("grid_csv", "")
        self.declare_parameter("label_column", "cell")
        self.declare_parameter("u_column", "center_x")
        self.declare_parameter("v_column", "center_y")
        self.declare_parameter("path_topic", "/grid_path")
        # One robot per entry; each robot's plan is published to /<name>/waypoint_path.
        self.declare_parameter("robot_names", ["raph", "raph2"])

        self.declare_parameter("origin_label", "H4")
        self.declare_parameter("world_origin_x", 0.0)
        self.declare_parameter("world_origin_y", 5.5)
        self.declare_parameter("metres_per_pixel_x", 1.0 / 138.0)   # horizontal scale
        self.declare_parameter("metres_per_pixel_y", -1.0 / 152.0)  # vertical scale (inverted)

        # ─── Obstacle-aware planning params ───────────────────────────────────
        self.declare_parameter("image_topic", "/ids_overhead/image")
        self.declare_parameter("traversable_prompts", ["the floor"])
        self.declare_parameter("untraversable_prompts", [""])   # "" entries filtered out
        self.declare_parameter("threshold", 0.45)
        self.declare_parameter("resolution", 0.05)              # m / occupancy cell
        self.declare_parameter("inflation_radius", 0.4)         # m (TurtleBot4 radius ~0.17 + margin)
        self.declare_parameter("save_debug", True)
        self.declare_parameter("debug_dir", "")                 # set by launch; empty = off

        # ─── Read parameters once ─────────────────────────────────────────────
        grid_csv       = self.get_parameter("grid_csv").get_parameter_value().string_value
        self.lbl_col   = self.get_parameter("label_column").get_parameter_value().string_value
        self.u_col     = self.get_parameter("u_column").get_parameter_value().string_value
        self.v_col     = self.get_parameter("v_column").get_parameter_value().string_value
        path_topic     = self.get_parameter("path_topic").get_parameter_value().string_value
        self.robot_names = list(self.get_parameter("robot_names").value)

        self.origin_label = self.get_parameter("origin_label").value
        self.x0 = self.get_parameter("world_origin_x").value
        self.y0 = self.get_parameter("world_origin_y").value
        self.sx = self.get_parameter("metres_per_pixel_x").value
        self.sy = self.get_parameter("metres_per_pixel_y").value

        self.image_topic = self.get_parameter("image_topic").value
        self.traversable_prompts = list(self.get_parameter("traversable_prompts").value)
        self.untraversable_prompts = [p for p in self.get_parameter("untraversable_prompts").value if p]
        self.threshold = self.get_parameter("threshold").value
        self.resolution = self.get_parameter("resolution").value
        self.inflation_radius = self.get_parameter("inflation_radius").value

        self.save_debug = self.get_parameter("save_debug").value
        self.debug_dir = self.get_parameter("debug_dir").value
        self._grid_overlay = None
        if self.save_debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            try:
                gpath = os.path.join(get_package_share_directory("talking-turtle"),
                                     "config", "transparent_grid.png")
                self._grid_overlay = cv2.imread(gpath, cv2.IMREAD_UNCHANGED)  # BGRA
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Could not load grid overlay for debug: {exc}")
            self.get_logger().info(f"Debug artifacts -> {self.debug_dir}")
        elif self.save_debug:
            self.get_logger().warn("save_debug=true but debug_dir empty; debug saving disabled.")

        # ─── Load grid pixel data ─────────────────────────────────────────────
        self.grid_px: Dict[str, np.ndarray] = self._load_grid_csv(grid_csv)

        if self.origin_label in self.grid_px:
            self.u0, self.v0, _ = self.grid_px[self.origin_label]
            print(f"Origin '{self.origin_label}' at pixel ({self.u0}, {self.v0})")
        else:
            self.u0, self.v0 = 0.0, 0.0
            self.get_logger().warn(
                f"Origin label '{self.origin_label}' not found in CSV; using (0,0)."
            )

        # ─── Segmentation + overhead image ────────────────────────────────────
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.get_logger().info("Loading CLIPSeg segmenter (one-time)…")
        self.segmenter = TraversabilitySegmenter()

        # ─── ROS 2 I/O ────────────────────────────────────────────────────────
        # /ids_overhead/image is the Gazebo->ROS bridge topic; cache the latest frame
        # (depth-1, default QoS, matching node_Map_Gen / node_Path_Visualizer).
        self.create_subscription(Image, self.image_topic, self._on_image, 1)
        self.sub = self.create_subscription(String, path_topic, self._on_path_msg, 10)
        # One waypoint-path publisher per robot: /<name>/waypoint_path.
        self.world_pubs: Dict[str, object] = {
            name: self.create_publisher(Float32MultiArray, f"/{name}/waypoint_path", 10)
            for name in self.robot_names
        }

        self.get_logger().info("✓")

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        """Cache the latest overhead frame (RGB) for on-demand costmap building."""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to decode overhead image: {exc}")

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────
    def _load_grid_csv(self, csv_path: str) -> Dict[str, np.ndarray]:
        """Read CSV and build {label: [u,v,1]} dict."""
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
                        f"({self.lbl_col}, {self.u_col}, {self.v_col})."
                    )
                    break
                except ValueError:
                    self.get_logger().warn(f"Skipping malformed row: {row}")
        self.get_logger().debug(f"Loaded {len(centres)} grid centres.")
        return centres

    # ------------------------------------------------------------------
    def _pixel_to_world(self, pix: np.ndarray) -> np.ndarray:
        """Convert image pixel (u,v) → world (x,y) in metres.

        Thin wrapper around the shared coord_transform.pixel_to_world so the
        translator and the occupancy-map generator (obs_seg) can never disagree
        on the conversion.
        """
        u, v, _ = pix
        x, y = pixel_to_world(u, v, self.x0, self.y0, self.sx, self.sy, self.u0, self.v0)
        return np.array([x, y])

    # ──────────────────────────────────────────────────────────────────
    #  Subscription callback
    # ──────────────────────────────────────────────────────────────────
    def _on_path_msg(self, msg: String) -> None:
        """Plan a collision-free path for each robot named in the incoming message.

        msg.data is a JSON object {robot_name: [grid labels]}. On every new plan message the
        inflated costmap is rebuilt fresh from the latest overhead frame, then reused for all
        robots in THAT message (the segmentation is the expensive step and both robots see the
        same instant, so re-segmenting per robot would be wasteful). Each robot's labels are
        turned into world centroids -> projected onto free space -> pairwise A* -> per-segment
        line-of-sight thinning, and published to /<robot>/waypoint_path.
        """
        try:
            plans = json.loads(msg.data)
            assert isinstance(plans, dict)
        except Exception as e:
            self.get_logger().error(
                f"Bad /grid_path message (expect JSON object {{robot: [labels]}}): {e}")
            return

        if self.latest_rgb is None:
            self.get_logger().warn("No overhead image received yet — cannot plan. Skipping.")
            return

        # Rebuild the inflated costmap from the latest frame for this replan, then reuse it
        # across the robots in this message.
        pix_labels, _ = self.segmenter.classify(
            self.latest_rgb, self.traversable_prompts,
            self.untraversable_prompts, self.threshold)
        grid, meta = mask_to_occupancy(pix_labels, self.x0, self.y0,
                                       self.sx, self.sy, self.u0, self.v0, self.resolution)
        infl = inflate_occupancy(grid, self.resolution, self.inflation_radius)

        for name, labels in plans.items():
            if name not in self.world_pubs:
                self.get_logger().warn(
                    f"No publisher for robot '{name}' (not in robot_names); skipping.")
                continue
            if not isinstance(labels, list):
                self.get_logger().warn(f"[{name}] Route is not a list; skipping.")
                continue
            self._plan_one(name, labels, pix_labels, grid, infl, meta)

    # ------------------------------------------------------------------
    def _plan_one(self, name, labels, pix_labels, grid, infl, meta) -> None:
        """Plan one robot's route on the prebuilt costmap and publish /<name>/waypoint_path."""
        # 1) labels -> world centroids (also remember pixel positions for debug overlays)
        centroids: List[tuple] = []
        sel_pixels: List[tuple] = []
        for label in labels:
            if label not in self.grid_px:
                self.get_logger().warn(f"[{name}] Unknown label '{label}' – skipping.")
                continue
            u, v, _ = self.grid_px[label]
            sel_pixels.append((float(u), float(v)))
            x, y = self._pixel_to_world(self.grid_px[label])
            centroids.append((float(x), float(y)))
        if not centroids:
            self.get_logger().warn(f"[{name}] No valid waypoints in route; nothing to plan.")
            return

        # 2) project each centroid onto the nearest free cell
        anchors: List[tuple] = []
        for (x, y) in centroids:
            free_cell = project_to_free(infl, world_to_cell(x, y, meta))
            if free_cell is None:
                self.get_logger().warn(
                    f"[{name}] Waypoint ({x:.2f},{y:.2f}) has no free cell nearby; skipping.")
                continue
            anchors.append(free_cell)
        if not anchors:
            self.get_logger().warn(f"[{name}] No projectable waypoints; nothing to publish.")
            return

        # 3) pairwise A* + per-segment LOS thinning (anchors preserved).
        #    On a pathless segment, skip that anchor and continue from the last reached one.
        full_cells: List[tuple] = [anchors[0]]
        current = anchors[0]
        for nxt in anchors[1:]:
            seg = astar(infl, current, nxt)
            if seg is None:
                self.get_logger().warn(
                    f"[{name}] No A* path from {current} to {nxt}; skipping waypoint.")
                continue
            seg = simplify_path_los(infl, seg)
            full_cells.extend(seg[1:])      # drop duplicate shared endpoint
            current = nxt

        # 4) cells -> world -> publish to this robot's topic
        flat_xy: List[float] = []
        for (gx, gy) in full_cells:
            wx, wy = cell_to_world(gx, gy, meta)
            flat_xy += [float(wx), float(wy)]

        arr = Float32MultiArray()
        arr.data = flat_xy
        self.world_pubs[name].publish(arr)
        self.get_logger().info(
            f"[{name}] Planned path: {len(anchors)} anchors -> {len(full_cells)} waypoints "
            f"(grid {meta['width']}x{meta['height']} @ {meta['resolution']} m).")

        if self.save_debug and self.debug_dir and self.latest_rgb is not None:
            try:
                self._save_debug(pix_labels, sel_pixels, full_cells, grid, infl, meta, suffix=name)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"[{name}] Debug save failed: {exc}")

    # ------------------------------------------------------------------
    def _save_debug(self, pix_labels, sel_pixels, full_cells, grid, infl, meta, suffix: str = "") -> None:
        """Write the debug artifacts to self.debug_dir (overwrite in place).

        When suffix (a robot name) is given, artifacts go in a per-robot subdirectory so the
        robots planned from one /grid_path message don't overwrite each other's overlays.
        """
        d = os.path.join(self.debug_dir, suffix) if suffix else self.debug_dir
        os.makedirs(d, exist_ok=True)
        base = cv2.cvtColor(np.ascontiguousarray(self.latest_rgb), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(d, "raw_overhead.png"), base)

        # CLIPSeg segmentation result, as a colored overlay on the overhead frame.
        # pix_labels is the same shape as the image, so this is already in camera
        # orientation (no transpose needed). green=free, red=obstacle, gray=unknown.
        seg_color = np.zeros_like(base)
        seg_color[pix_labels == FREE] = (0, 180, 0)
        seg_color[pix_labels == OCCUPIED] = (0, 0, 200)
        seg_color[pix_labels == UNKNOWN] = (128, 128, 128)
        seg_overlay = (0.5 * base + 0.5 * seg_color).astype(np.uint8)
        cv2.imwrite(os.path.join(d, "segmentation.png"), seg_overlay)

        # 2) raw + transparent grid overlay (what the VLM sees)
        if self._grid_overlay is not None:
            g = cv2.resize(self._grid_overlay, (base.shape[1], base.shape[0]))
            if g.ndim == 3 and g.shape[2] == 4:
                a = g[..., 3:4].astype(np.float32) / 255.0
                comp = (base * (1 - a) + g[..., :3] * a).astype(np.uint8)
            else:
                comp = g[..., :3]
            cv2.imwrite(os.path.join(d, "grid_overlay.png"), comp)

        # 3) naive route through the selected region centroids (pixel space, orange)
        rc = base.copy()
        pts = [(int(u), int(v)) for (u, v) in sel_pixels]
        for a, b in zip(pts, pts[1:]):
            cv2.line(rc, a, b, (0, 165, 255), 2)
        for p in pts:
            cv2.circle(rc, p, 8, (255, 0, 0), -1)        # blue = centroid
        cv2.imwrite(os.path.join(d, "route_centroids.png"), rc)

        # 4) obstacle-avoiding planned route (world -> pixel, green)
        rp = base.copy()
        ppx = []
        for (gx, gy) in full_cells:
            wx, wy = cell_to_world(gx, gy, meta)
            u, v = world_to_pixel(wx, wy, self.x0, self.y0, self.sx, self.sy, self.u0, self.v0)
            ppx.append((int(u), int(v)))
        for a, b in zip(ppx, ppx[1:]):
            cv2.line(rp, a, b, (0, 200, 0), 2)
        for p in ppx:
            cv2.circle(rp, p, 5, (0, 0, 255), -1)        # red = waypoint
        cv2.imwrite(os.path.join(d, "route_planned.png"), rp)

        # 5,6) occupancy maps (white=free, black=occupied, gray=unknown).
        # The grid is in the WORLD frame, which is transposed/flipped relative to the
        # camera image (world x runs along image rows, world y along image columns).
        # Reorient the PNG with flipud(grid.T) so it visually matches the overhead
        # image — the grid data itself is left untouched (planning uses world frame).
        def viz(gmap):
            g2 = np.flipud(gmap.T)
            out = np.full((*g2.shape, 3), 128, np.uint8)
            out[g2 == FREE] = (255, 255, 255)
            out[g2 == OCCUPIED] = (0, 0, 0)
            return out
        cv2.imwrite(os.path.join(d, "occ_true.png"), viz(grid))
        cv2.imwrite(os.path.join(d, "occ_inflated.png"), viz(infl))

        # Inflated obstacles overlaid on the overhead image (pixel space) — shows how far the
        # inflation overshoots the true obstacles. Each image pixel is mapped to its occupancy
        # cell: red = true obstacle, yellow = inflation margin (inflated-occupied but not a
        # true obstacle). Pixel-space, so it lines up with the overhead image directly.
        h_img, w_img = base.shape[:2]
        uu, vv = np.meshgrid(np.arange(w_img), np.arange(h_img))
        xs, ys = pixel_to_world(uu, vv, self.x0, self.y0, self.sx, self.sy, self.u0, self.v0)
        gx = np.floor((xs - meta["origin_x"]) / meta["resolution"]).astype(np.int64)
        gy = np.floor((ys - meta["origin_y"]) / meta["resolution"]).astype(np.int64)
        inb = (gx >= 0) & (gx < meta["width"]) & (gy >= 0) & (gy < meta["height"])
        gxc = np.clip(gx, 0, meta["width"] - 1)
        gyc = np.clip(gy, 0, meta["height"] - 1)
        true_occ = inb & (grid[gyc, gxc] != FREE)      # actual obstacle / unknown
        infl_occ = inb & (infl[gyc, gxc] == OCCUPIED)
        margin = infl_occ & ~true_occ                  # cells added purely by inflation
        infl_overlay = base.copy()
        infl_overlay[margin] = (0.5 * base[margin] + 0.5 * np.array([0, 220, 220])).astype(np.uint8)    # yellow
        infl_overlay[true_occ] = (0.5 * base[true_occ] + 0.5 * np.array([0, 0, 220])).astype(np.uint8)  # red
        cv2.imwrite(os.path.join(d, "inflation_overlay.png"), infl_overlay)

        self.get_logger().info(f"Saved 8 debug artifacts to {d}")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry‑point
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
