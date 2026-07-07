#!/usr/bin/env python3
"""Dummy overhead-camera publisher for testing without the real IDS camera.

Loads a static image once and republishes it as sensor_msgs/Image on the same
topic the real overhead camera uses (/ids_overhead/image), so the rest of the
stack (node_Executive_API, node_Path_Visualizer, node_Path_Translator) sees a frame
even when no physical camera is connected.

Run standalone (no rebuild needed):
    python3 nodes/node_Dummy_Overhead/dummy_overhead.py
Or, after a colcon build, as a node:
    ros2 run talking-turtle node_Dummy_Overhead
    ros2 run talking-turtle node_Dummy_Overhead --ros-args -p image_path:=/abs/path/overhead.png
"""
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


# Default sample frame shipped for testing (workspace-level test_data dir).
DEFAULT_IMAGE = os.path.expanduser('~/projects/multi_robot_ws/test_data/overhead.png')


class DummyOverheadPublisher(Node):
    def __init__(self):
        super().__init__('node_Dummy_Overhead')

        self.declare_parameter('image_path', DEFAULT_IMAGE)
        self.declare_parameter('topic', '/ids_overhead/image')
        self.declare_parameter('rate_hz', 0.5)          # republish frequency
        self.declare_parameter('frame_id', 'ids_overhead')

        image_path = self.get_parameter('image_path').get_parameter_value().string_value
        topic = self.get_parameter('topic').get_parameter_value().string_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Load the frame once. cv2.imread returns BGR uint8 (or None on failure).
        cv_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if cv_img is None:
            raise RuntimeError(
                f"Could not read image '{image_path}'. Set the image_path parameter "
                f"to a valid file (e.g. the sample at {DEFAULT_IMAGE})."
            )

        # Pre-build the message once; we only refresh the timestamp per publish.
        self.bridge = CvBridge()
        self.msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        self.msg.header.frame_id = self.frame_id

        self.pub = self.create_publisher(Image, topic, 1)
        period = 1.0 / rate_hz if rate_hz > 0.0 else 2.0
        self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Publishing {cv_img.shape[1]}x{cv_img.shape[0]} bgr8 frame from "
            f"'{image_path}' on '{topic}' at {rate_hz} Hz"
        )

    def _tick(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main(args=None):
    rclpy.init(args=args)
    node = DummyOverheadPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
