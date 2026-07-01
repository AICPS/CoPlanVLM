#!/usr/bin/env python3
"""Diagnostic: echo a robot's pose at each stage of the transform pipeline.

Subscribes to the raw pose topic (/<robot>/<pose_frame>/pose_stamped by default), runs it
through the SAME coord_transform functions the stack uses, and:

  * republishes the converted WORLD-frame pose as PoseStamped on /<robot>/world/pose_stamped
    so you can inspect it with `ros2 topic echo`, and
  * logs raw pose (m) -> world (m) -> pixel (u,v) so you can localise a scale/offset error
    (ned_to_world is unit-preserving, so any scale mismatch is in world_to_pixel calibration).

Run standalone (no rebuild):
    python3 src/talking_turtle/scripts/echo_world_pose.py --ros-args -p robot:=donnie
Then in another terminal:
    ros2 topic echo /donnie/world/pose_stamped
"""
import os
import sys

# Allow importing the package's coord_transform when run straight from the source tree
# (before/without a colcon build). Harmless if the installed package is already on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "nodes"))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped

import math

from tf_transformations import quaternion_from_euler
from coord_transform import (ned_to_world_pose, world_to_pixel, yaw_from_quaternion,
                             set_active_camera)


class EchoWorldPose(Node):
    def __init__(self):
        super().__init__("echo_world_pose")

        self.declare_parameter("robot", "donnie")
        self.declare_parameter("camera", "lab_test")   # world<->pixel calibration
        # Empty -> derive /<robot>/ned/pose_stamped; else use this exact topic.
        self.declare_parameter("input_topic", "")

        robot = self.get_parameter("robot").get_parameter_value().string_value
        set_active_camera(self.get_parameter("camera").get_parameter_value().string_value)
        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        if not input_topic:
            input_topic = f"/{robot}/ned/pose_stamped"

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT   # compatible with any publisher
        self.pub = self.create_publisher(PoseStamped, f"/{robot}/world/pose_stamped", qos)
        self.create_subscription(PoseStamped, input_topic, self._cb, qos)

        self.get_logger().info(
            f"echoing {input_topic} (NED) -> world, republishing /{robot}/world/pose_stamped"
        )

    def _cb(self, msg: PoseStamped):
        rx, ry = msg.pose.position.x, msg.pose.position.y
        nyaw = yaw_from_quaternion(*(getattr(msg.pose.orientation, a) for a in "xyzw"))
        wx, wy, wyaw = ned_to_world_pose(rx, ry, nyaw)
        u, v = world_to_pixel(wx, wy)

        out = PoseStamped()
        out.header = msg.header
        out.pose.position.x = float(wx)
        out.pose.position.y = float(wy)
        out.pose.position.z = msg.pose.position.z
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, wyaw)
        out.pose.orientation.x, out.pose.orientation.y = qx, qy
        out.pose.orientation.z, out.pose.orientation.w = qz, qw
        self.pub.publish(out)

        self.get_logger().info(
            f"raw({rx:+.3f}, {ry:+.3f}) m yaw {math.degrees(nyaw):+.1f}  ->  "
            f"world({wx:+.3f}, {wy:+.3f}) m yaw {math.degrees(wyaw):+.1f}  ->  px({u:.1f}, {v:.1f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EchoWorldPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
