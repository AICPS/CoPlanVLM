#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
import cv2
import numpy as np

class MapOverlaySaver(Node):
    """
    Subscribes once to /camera_image, overlays a transparent grid, and
    stores the composite as map.png in the package share directory.
    """
    def __init__(self):
        super().__init__('map_overlay_saver')

        self.declare_parameter('need_map_topic', '/need_map')   # ⇐ input
        self.declare_parameter('camera_info_topic', '/ids_overhead/camera_info')
        self.need_map_topic: str = self.get_parameter('need_map_topic').value
        self.camera_info_topic: str = self.get_parameter('camera_info_topic').value
        self.need_map_sub = self.create_subscription(Bool, self.need_map_topic, self._need_map_cb, 10)


        # Locate package assets
        self.pkg_dir = get_package_share_directory('talking-turtle')
        self.grid_path = os.path.join(self.pkg_dir, 'config', 'transparent_grid.png')
        self.out_path  = os.path.join(self.pkg_dir, 'map.png')
        self.raw_out_path  = os.path.join(self.pkg_dir, 'map_raw.png')
        self.undistorted_out_path = os.path.join(self.pkg_dir, 'map_raw_undistorted.png')

        # Load static grid once
        self.grid_img = PILImage.open(self.grid_path).convert("RGBA")

        # Bridge for ROS ↔ OpenCV
        self.bridge = CvBridge()

        self.camera_matrix = None
        self.dist_coeffs = None

        # Subscribe to the overhead camera image
        self.sub = self.create_subscription(
            Image,
            '/camera_image',
            self._camera_image_cb,
            1
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_cb,
            1,
        )

        self.get_logger().info("✓")

    def _need_map_cb(self, msg: Bool):
        if msg.data == True:
            try:
                # Convert ROS image → OpenCV → PIL
                cv_img = self.bridge.imgmsg_to_cv2(self.camera_image, desired_encoding='rgba8')
                raw_pil = PILImage.fromarray(cv_img).convert("RGBA")
                raw_pil.save(self.raw_out_path)
                self.get_logger().debug(f"Raw map image saved to {self.raw_out_path}")

                undistorted_img = self._undistort_image(cv_img)
                undistorted_pil = PILImage.fromarray(undistorted_img).convert("RGBA")
                undistorted_pil.save(self.undistorted_out_path)
                self.get_logger().debug(
                    f"Undistorted map image saved to {self.undistorted_out_path}"
                )

                # Resize grid to match map dimensions
                grid_resized = self.grid_img.resize(undistorted_pil.size)

                # Composite and save
                combined = PILImage.alpha_composite(undistorted_pil, grid_resized)
                combined.save(self.out_path)
                self.get_logger().debug(f"Grid overlay saved to {self.out_path}")

            except Exception as exc:
                self.get_logger().error(f"Failed to create overlay: {exc}")

    def _camera_image_cb(self, msg: Image):
        self.camera_image = msg

    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _undistort_image(self, image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    

def main(args=None):
    rclpy.init(args=args)
    node = MapOverlaySaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()