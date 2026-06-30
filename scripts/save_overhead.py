#!/usr/bin/env python3
"""save_overhead.py — one-shot ROS node to capture the overhead camera frame + robot poses.

Run once with the Gazebo sim up to create test_data/ for offline pipeline testing.
The saved files are then used by test_pipeline.py without needing a running sim.

Usage (from workspace root, after 'source install/setup.bash'):
    python3 src/VLM_mission_planning/scripts/save_overhead.py
    python3 src/VLM_mission_planning/scripts/save_overhead.py --out test_data --robots raph donnie

Output:
    <out>/overhead.png   — BGR image from /ids_overhead/image (typically 1936x1216)
    <out>/poses.json     — {"raph": {"x": ..., "y": ...}, "donnie": {"x": ..., "y": ...}}
                           Raw Gazebo coordinates as received on /<robot>/pose_stamped.
                           test_pipeline.py applies gazebo_to_world() to match translate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


class OverheadSaver(Node):
    def __init__(self, out_dir: str, robot_names: list[str]) -> None:
        super().__init__("overhead_saver")
        self.out_dir = out_dir
        self.robot_names = robot_names
        self.bridge = CvBridge()
        self._image = None
        self._poses: dict[str, dict] = {}

        self.create_subscription(Image, "/ids_overhead/image", self._on_image, 1)

        best_effort_qos = QoSProfile(depth=1)
        best_effort_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        for name in robot_names:
            self.create_subscription(
                PoseStamped,
                f"/{name}/pose_stamped",
                self._make_pose_cb(name),
                best_effort_qos,
            )
        self.get_logger().info(
            f"Waiting for /ids_overhead/image and poses for {robot_names}…"
        )

    def _on_image(self, msg: Image) -> None:
        if self._image is None:
            self._image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.get_logger().info(
                f"Got image: {self._image.shape[1]}×{self._image.shape[0]}"
            )

    def _make_pose_cb(self, name: str):
        def _cb(msg: PoseStamped) -> None:
            if name not in self._poses:
                self._poses[name] = {
                    "x": msg.pose.position.x,
                    "y": msg.pose.position.y,
                }
                self.get_logger().info(
                    f"Got pose for {name}: "
                    f"({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f})"
                )
        return _cb

    def ready(self) -> bool:
        return (
            self._image is not None
            and all(n in self._poses for n in self.robot_names)
        )

    def save(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        img_path = os.path.join(self.out_dir, "overhead.png")
        poses_path = os.path.join(self.out_dir, "poses.json")
        cv2.imwrite(img_path, self._image)
        with open(poses_path, "w") as f:
            json.dump(self._poses, f, indent=2)
        print(f"Saved: {img_path}")
        print(f"Saved: {poses_path}")
        print(f"Poses (Gazebo frame): {json.dumps(self._poses)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="test_data",
                        help="Output directory (default: test_data)")
    parser.add_argument("--robots", nargs="+", default=["raph", "donnie"],
                        help="Robot names to wait for (default: raph donnie)")
    args = parser.parse_args()

    rclpy.init()
    node = OverheadSaver(out_dir=args.out, robot_names=args.robots)
    try:
        while rclpy.ok() and not node.ready():
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready():
            node.save()
        else:
            print("ROS shut down before all data was received.", file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted before all data was received.", file=sys.stderr)
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
