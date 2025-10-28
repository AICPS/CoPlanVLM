#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool

class MapOverlaySaver(Node):
    """
    Subscribes once to /raw_map, overlays a transparent grid, and
    stores the composite as map.png in the package share directory.
    """
    def __init__(self):
        super().__init__('map_overlay_saver')

        self.declare_parameter('need_map_topic', '/need_map')   # ⇐ input
        self.need_map_topic: str = self.get_parameter('need_map_topic').value
        self.need_map_sub = self.create_subscription(Bool, self.need_map_topic, self._need_map_cb, 10)


        # Locate package assets
        self.pkg_dir = get_package_share_directory('talking-turtle')
        self.grid_path = os.path.join(self.pkg_dir, 'config', 'transparent_grid.png')
        self.out_path  = os.path.join(self.pkg_dir, 'map.png')
        self.raw_out_path  = os.path.join(self.pkg_dir, 'map_raw.png')

        # Load static grid once
        self.grid_img = PILImage.open(self.grid_path).convert("RGBA")

        # Bridge for ROS ↔ OpenCV
        self.bridge = CvBridge()

        # Subscribe to raw map
        self.sub = self.create_subscription(
            Image,
            '/raw_map',
            self._raw_map_cb,
            1
        )

        self.get_logger().info("✓")

    def _need_map_cb(self, msg: Bool):
        if msg.data == True:
            try:
                # Convert ROS image → OpenCV → PIL
                cv_img = self.bridge.imgmsg_to_cv2(self.raw_map, desired_encoding='rgba8')
                raw_pil = PILImage.fromarray(cv_img).convert("RGBA")
                raw_pil.save(self.raw_out_path)
                self.get_logger().debug(f"Raw map image saved to {self.raw_out_path}")

                # Resize grid to match map dimensions
                grid_resized = self.grid_img.resize(raw_pil.size)

                # Composite and save
                combined = PILImage.alpha_composite(raw_pil, grid_resized)
                combined.save(self.out_path)
                self.get_logger().debug(f"Grid overlay saved to {self.out_path}")

            except Exception as exc:
                self.get_logger().error(f"Failed to create overlay: {exc}")

    def _raw_map_cb(self, msg: Image):
        self.raw_map = msg
    

def main(args=None):
    rclpy.init(args=args)
    node = MapOverlaySaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()