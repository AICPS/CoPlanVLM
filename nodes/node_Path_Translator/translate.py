#!/usr/bin/env python3
"""
simple_path_translator.py  –  v0.5 (straight‑down camera, explicit Z scale)
-------------------------------------------------------------------------
ROS 2 node that converts a JSON list of grid labels (e.g. ["L1", "M2"]) sent
on `/path` into ground‑plane metres and publishes them on `/world_path` as a
`Float32MultiArray`.

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
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import yaml


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
        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("world_path_topic", "/world_path")

        self.declare_parameter("origin_label", "H5")
        self.declare_parameter("world_origin_x", 0.0)
        self.declare_parameter("world_origin_y", 5.5)
        self.declare_parameter("metres_per_pixel_x", 1.0 / 138.0)   # horizontal scale
        self.declare_parameter("metres_per_pixel_y", -1.0 / 152.0)  # vertical scale (inverted)

        # ─── Read parameters once ─────────────────────────────────────────────
        grid_csv       = self.get_parameter("grid_csv").get_parameter_value().string_value
        self.lbl_col   = self.get_parameter("label_column").get_parameter_value().string_value
        self.u_col     = self.get_parameter("u_column").get_parameter_value().string_value
        self.v_col     = self.get_parameter("v_column").get_parameter_value().string_value
        path_topic     = self.get_parameter("path_topic").get_parameter_value().string_value
        world_topic    = self.get_parameter("world_path_topic").get_parameter_value().string_value

        self.origin_label = self.get_parameter("origin_label").value
        self.x0 = self.get_parameter("world_origin_x").value
        self.y0 = self.get_parameter("world_origin_y").value
        self.sx = self.get_parameter("metres_per_pixel_x").value
        self.sy = self.get_parameter("metres_per_pixel_y").value

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

        # ─── ROS 2 I/O ────────────────────────────────────────────────────────
        self.sub = self.create_subscription(String, path_topic, self._on_path_msg, 10)
        self.pub = self.create_publisher(Float32MultiArray, world_topic, 10)

        self.get_logger().info("✓")

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
        """Convert image pixel (u,v) → world (x,y) in metres."""
        u, v, _ = pix
        x = self.x0 + (v - self.v0) * self.sy
        y = self.y0 + (u - self.u0) * self.sx
        return np.array([x, y])

    # ──────────────────────────────────────────────────────────────────
    #  Subscription callback
    # ──────────────────────────────────────────────────────────────────
    def _on_path_msg(self, msg: String) -> None:
        """Handle incoming JSON list on /path, publish Float32MultiArray."""
        try:
            labels: List[str] = json.loads(msg.data)
            assert isinstance(labels, list)
        except Exception as e:
            self.get_logger().error(f"Bad /path message (expect JSON list): {e}")
            return

        flat_xy: List[float] = []
        for label in labels:
            if label not in self.grid_px:
                self.get_logger().warn(f"Unknown label '{label}' – skipping.")
                continue
            x, y = self._pixel_to_world(self.grid_px[label])
            flat_xy += [x, y]

        arr = Float32MultiArray()
        arr.data = flat_xy
        self.pub.publish(arr)
        self.get_logger().debug(f"Sent {len(flat_xy)//2} waypoints.")


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
