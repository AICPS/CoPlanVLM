#!/usr/bin/env python3
"""Simple ROS2 path visualizer for VLM waypoints and live robot tracking."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String

from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


class PathVisualizer(Node):
    """Visualize path overlays and live robot tracking on an overhead map."""

    def __init__(self) -> None:
        super().__init__("path_visualizer")

        self._declare_parameters()

        self.grid_csv = self.get_parameter("grid_csv").value
        self.label_column = self.get_parameter("label_column").value
        self.u_column = self.get_parameter("u_column").value
        self.v_column = self.get_parameter("v_column").value

        self.path_topic = self.get_parameter("path_topic").value
        self.world_path_topic = self.get_parameter("world_path_topic").value
        self.raw_map_topic = self.get_parameter("raw_map_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.viz_topic = self.get_parameter("viz_topic").value

        self.save_overlays = bool(self.get_parameter("save_overlays").value)
        self.window_name = self.get_parameter("window_name").value
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.line_thickness = int(self.get_parameter("line_thickness").value)
        self.circle_radius = int(self.get_parameter("circle_radius").value)

        self.start_color = tuple(int(c) for c in self.get_parameter("start_color").value)
        self.path_color = tuple(int(c) for c in self.get_parameter("path_color").value)
        self.arrowhead_color = tuple(int(c) for c in self.get_parameter("arrowhead_color").value)
        self.end_color = tuple(int(c) for c in self.get_parameter("end_color").value)
        self.world_path_color = tuple(int(c) for c in self.get_parameter("world_path_color").value)
        self.robot_color = tuple(int(c) for c in self.get_parameter("robot_color").value)

        self.origin_label = self.get_parameter("origin_label").value
        self.world_origin_x = float(self.get_parameter("world_origin_x").value)
        self.world_origin_y = float(self.get_parameter("world_origin_y").value)
        self.metres_per_pixel_x = float(self.get_parameter("metres_per_pixel_x").value)
        self.metres_per_pixel_y = float(self.get_parameter("metres_per_pixel_y").value)

        self.grid_pixels = self._load_grid_csv(self.grid_csv)
        self.origin_pixel = self.grid_pixels.get(self.origin_label)
        if self.origin_pixel is None:
            self.get_logger().warn(
                f"Origin label '{self.origin_label}' not in CSV. Using (0, 0) as fallback."
            )
            self.origin_pixel = (0.0, 0.0)

        self.bridge = CvBridge()

        self.latest_raw_map: Optional[np.ndarray] = None
        self.captured_map: Optional[np.ndarray] = None
        self.latest_path_labels: List[str] = []
        self.latest_world_path: List[Tuple[float, float]] = []
        self.latest_pose: Optional[PoseStamped] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        self.overlay_path = (
            Path(get_package_share_directory("talking-turtle")) / "path_overlay.png"
        )

        self.create_subscription(String, self.path_topic, self._path_callback, 10)
        self.create_subscription(Float32MultiArray, self.world_path_topic, self._world_path_callback, 10)
        self.create_subscription(Image, self.raw_map_topic, self._raw_map_callback, 1)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 1)
        self.create_subscription(PoseStamped, self.pose_topic, self._pose_callback, 10)

        self.viz_publisher = self.create_publisher(Image, self.viz_topic, 10)

        # Keeps the live OpenCV view updated with world path, path labels, and robot pose.
        self.create_timer(0.1, self.plot_tracking_window)

        self.get_logger().info("Path visualizer ready")

    def _declare_parameters(self) -> None:
        self.declare_parameter("grid_csv", "")
        self.declare_parameter("label_column", "cell")
        self.declare_parameter("u_column", "center_x")
        self.declare_parameter("v_column", "center_y")

        self.declare_parameter("path_topic", "/path")
        self.declare_parameter("world_path_topic", "/world_path")
        self.declare_parameter("raw_map_topic", "/raw_map")
        self.declare_parameter("camera_info_topic", "/ids_overhead/camera_info")
        self.declare_parameter("pose_topic", "/raph/sim_ground_truth_pose")
        self.declare_parameter("viz_topic", "/path_visualization")

        self.declare_parameter("save_overlays", True)
        self.declare_parameter("window_name", "Path Tracking")
        self.declare_parameter("display_scale", 0.75)
        self.declare_parameter("line_thickness", 6)
        self.declare_parameter("circle_radius", 14)

        self.declare_parameter("start_color", [0, 255, 0])
        self.declare_parameter("path_color", [165, 33, 0])
        self.declare_parameter("arrowhead_color", [22, 70, 250])
        self.declare_parameter("end_color", [0, 0, 255])
        self.declare_parameter("world_path_color", [200, 200, 200])
        self.declare_parameter("robot_color", [255, 0, 255])

        self.declare_parameter("origin_label", "H4")
        self.declare_parameter("world_origin_x", 0.0)
        self.declare_parameter("world_origin_y", 5.5)
        self.declare_parameter("metres_per_pixel_x", 1.0 / 138.0)
        self.declare_parameter("metres_per_pixel_y", -1.0 / 152.0)

    def _load_grid_csv(self, csv_path: str) -> Dict[str, Tuple[float, float]]:
        pixels: Dict[str, Tuple[float, float]] = {}
        path = Path(csv_path)
        if not csv_path or not path.exists():
            self.get_logger().warn(f"Grid CSV not found: {csv_path}")
            return pixels

        with path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    label = row[self.label_column]
                    u = float(row[self.u_column])
                    v = float(row[self.v_column])
                    pixels[label] = (u, v)
                except (KeyError, ValueError):
                    continue

        self.get_logger().info(f"Loaded {len(pixels)} grid points")
        return pixels

    def _path_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, list):
                self.latest_path_labels = [str(x) for x in data]
            else:
                self.latest_path_labels = []
        except json.JSONDecodeError:
            self.latest_path_labels = []
            self.get_logger().warn("/path is not valid JSON list")
            return

        # Capture map once and draw waypoints for a stable path snapshot.
        self.plot_waypoints_on_captured_image()

    def _world_path_callback(self, msg: Float32MultiArray) -> None:
        coords: List[Tuple[float, float]] = []
        for i in range(0, len(msg.data) - 1, 2):
            coords.append((float(msg.data[i]), float(msg.data[i + 1])))
        self.latest_world_path = coords

    def _raw_map_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to decode /raw_map: {exc}")
            return

        image = self._undistort_image(image)
        self.latest_raw_map = image
        if self.captured_map is None:
            self.captured_map = image.copy()
            self.get_logger().info("Captured base map image for waypoint overlays")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def _pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def _grid_path_pixels(self) -> List[Tuple[int, int]]:
        pixels: List[Tuple[int, int]] = []
        for label in self.latest_path_labels:
            uv = self.grid_pixels.get(label)
            if uv is None:
                continue
            pixels.append((int(round(uv[0])), int(round(uv[1]))))
        return pixels

    def _world_to_pixel(self, x_world: float, y_world: float) -> Tuple[int, int]:
        u0, v0 = self.origin_pixel
        if self.metres_per_pixel_x == 0.0 or self.metres_per_pixel_y == 0.0:
            return int(u0), int(v0)

        u = u0 + (y_world - self.world_origin_y) / self.metres_per_pixel_x
        v = v0 + (x_world - self.world_origin_x) / self.metres_per_pixel_y
        return int(round(u)), int(round(v))

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _draw_colored_arrow(
        self,
        image: np.ndarray,
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
    ) -> None:
        """Draw a navy shaft with an orange arrowhead."""
        cv2.line(image, start_pt, end_pt, self.path_color, self.line_thickness)

        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            return

        head_length = max(12.0, float(self.line_thickness) * 3.0)
        head_length = min(head_length, length * 0.75)
        head_width = max(10.0, float(self.line_thickness) * 2.6)

        ux = dx / length
        uy = dy / length
        base_x = end_pt[0] - ux * head_length
        base_y = end_pt[1] - uy * head_length
        perp_x = -uy
        perp_y = ux

        left_pt = (
            int(round(base_x + perp_x * head_width / 2.0)),
            int(round(base_y + perp_y * head_width / 2.0)),
        )
        right_pt = (
            int(round(base_x - perp_x * head_width / 2.0)),
            int(round(base_y - perp_y * head_width / 2.0)),
        )

        head = np.array([end_pt, left_pt, right_pt], dtype=np.int32)
        cv2.fillConvexPoly(image, head, self.arrowhead_color)

    def plot_waypoints_on_captured_image(self) -> None:
        """Plot waypoint arrows on the captured map image and publish/save result."""
        if self.captured_map is None:
            return

        image = self.captured_map.copy()
        path_pixels = self._grid_path_pixels()

        if len(path_pixels) >= 1:
            cv2.circle(image, path_pixels[0], self.circle_radius, self.start_color, 3)
            cv2.putText(
                image,
                "Start",
                (path_pixels[0][0] + 10, path_pixels[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.start_color,
                2,
                cv2.LINE_AA,
            )
        if len(path_pixels) >= 2:
            cv2.circle(image, path_pixels[-1], self.circle_radius, self.end_color, 3)
            cv2.putText(
                image,
                "End",
                (path_pixels[-1][0] + 10, path_pixels[-1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.end_color,
                2,
                cv2.LINE_AA,
            )

        for i in range(len(path_pixels) - 1):
            start_pt = path_pixels[i]
            end_pt = path_pixels[i + 1]
            self._draw_colored_arrow(image, start_pt, end_pt)
            cv2.putText(
                image,
                str(i + 1),
                (start_pt[0] + 8, start_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        try:
            out_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            self.viz_publisher.publish(out_msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish overlay image: {exc}")

        if self.save_overlays:
            try:
                cv2.imwrite(str(self.overlay_path), image)
            except Exception as exc:
                self.get_logger().warn(f"Failed to save overlay image: {exc}")

    def plot_tracking_window(self) -> None:
        """Show live OpenCV tracking window using world path, VLM path, and robot pose."""
        if self.latest_raw_map is None:
            return

        frame = self.latest_raw_map.copy()

        # Draw world path from /world_path (metres) converted to pixels.
        world_pixels = [self._world_to_pixel(x, y) for x, y in self.latest_world_path]
        for i in range(len(world_pixels) - 1):
            cv2.line(frame, world_pixels[i], world_pixels[i + 1], self.world_path_color, 2)

        # Draw VLM label path from /path as arrows.
        path_pixels = self._grid_path_pixels()
        if len(path_pixels) >= 1:
            cv2.circle(frame, path_pixels[0], self.circle_radius, self.start_color, 3)
            cv2.putText(
                frame,
                "S",
                (path_pixels[0][0] + 10, path_pixels[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.start_color,
                2,
                cv2.LINE_AA,
            )
        if len(path_pixels) >= 2:
            cv2.circle(frame, path_pixels[-1], self.circle_radius, self.end_color, 3)
            cv2.putText(
                frame,
                "E",
                (path_pixels[-1][0] + 10, path_pixels[-1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.end_color,
                2,
                cv2.LINE_AA,
            )
        for i in range(len(path_pixels) - 1):
            self._draw_colored_arrow(frame, path_pixels[i], path_pixels[i + 1])

        # Draw robot pose from /raph/sim_ground_truth_pose.
        if self.latest_pose is not None:
            pose = self.latest_pose.pose
            robot_u, robot_v = self._world_to_pixel(pose.position.x, pose.position.y)
            yaw = self._yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            arrow_len = 30
            tip = (
                int(robot_u + arrow_len * math.cos(yaw)),
                int(robot_v + arrow_len * math.sin(yaw)),
            )
            cv2.circle(frame, (robot_u, robot_v), 8, self.robot_color, -1)
            cv2.arrowedLine(frame, (robot_u, robot_v), tip, (255, 255, 255), 5, tipLength=0.5)

        cv2.putText(
            frame,
            f"waypoints: {len(self.latest_path_labels)}  world_pts: {len(self.latest_world_path)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if 0.0 < self.display_scale < 1.0:
            new_w = max(1, int(frame.shape[1] * self.display_scale))
            new_h = max(1, int(frame.shape[0] * self.display_scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
