#!/usr/bin/env python3
"""Simple ROS2 path visualizer for VLM waypoints and live robot tracking."""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String

from cv_bridge import CvBridge

from coord_transform import (world_to_pixel, world_to_pixel_pose, ned_to_world,
                             ned_to_world_pose, yaw_from_quaternion)

ROBOT_LABEL_SCALE = 0.8
ROBOT_LABEL_THICKNESS = 2
PLANNED_PATH_THICKNESS = 4
TRAJECTORY_THICKNESS = 4


class PathVisualizer(Node):
    """Visualize path overlays and live robot tracking on an overhead map."""

    def __init__(self) -> None:
        super().__init__("path_visualizer")

        self._declare_parameters()
        self.camera_name: str = self.get_parameter("camera").get_parameter_value().string_value

        self.grid_csv = self.get_parameter("grid_csv").value
        self.label_column = self.get_parameter("label_column").value
        self.u_column = self.get_parameter("u_column").value
        self.v_column = self.get_parameter("v_column").value

        self.vlm_plan_topic = self.get_parameter("vlm_plan_topic").value
        self.robot_names = [str(n) for n in self.get_parameter("robot_names").value]
        self.camera_image_topic = self.get_parameter("camera_image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.window_name = self.get_parameter("window_name").value
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.circle_radius = int(self.get_parameter("circle_radius").value)
        # Perpendicular spacing (px) between robots' planned paths so coincident routes render as
        # parallel tracks instead of one hiding the other. 0 disables the offset.
        self.path_offset_px = float(self.get_parameter("path_offset_px").value)

        self.start_color = tuple(int(c) for c in self.get_parameter("start_color").value)
        self.end_color = tuple(int(c) for c in self.get_parameter("end_color").value)

        # Pixel<->world calibration lives entirely in coord_transform (see _world_to_pixel).
        self.grid_pixels = self._load_grid_csv(self.grid_csv)

        self.bridge = CvBridge()

        # Distinct BGR color per robot for its path line + marker (cycled if more robots
        # than colors). raph -> magenta, donnie -> cyan by default.
        palette = [(255, 0, 255), (255, 255, 0), (0, 255, 255), (255, 128, 0)]
        self.robot_colors: Dict[str, Tuple[int, int, int]] = {
            name: palette[i % len(palette)] for i, name in enumerate(self.robot_names)
        }

        self.latest_camera_image: Optional[np.ndarray] = None
        # Per-robot latest state, keyed by robot name.
        self.latest_path_labels: Dict[str, List[str]] = {n: [] for n in self.robot_names}
        self.latest_world_path: Dict[str, List[Tuple[float, float]]] = {n: [] for n in self.robot_names}
        self.latest_pose: Dict[str, Optional[PoseStamped]] = {n: None for n in self.robot_names}
        # Per-robot trajectory trail: recent measured poses (gazebo x, y), capped.
        self.pose_history: Dict[str, Deque[Tuple[float, float]]] = {
            n: deque(maxlen=3000) for n in self.robot_names
        }
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        # /vlm_plan supplies the VLM-selected endpoints retained in the live view. The translated,
        # obstacle-aware path arrives separately on each robot's /waypoint_path topic.
        self.create_subscription(String, self.vlm_plan_topic, self._path_callback, 10)
        self.create_subscription(Image, self.camera_image_topic, self._camera_image_callback, 1)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_callback, 1)
        # Per-robot world path + pose. Pose comes from /<name>/ned/pose_stamped (PoseStamped,
        # published BEST_EFFORT by node_Odometry_To_Pose in sim, or directly by MoCap in the real
        # world) — the SAME source control.py and the translator use. The previous bug subscribed
        # to /<name>/sim_ground_truth_pose, which is nav_msgs/Odometry, as PoseStamped, so it
        # silently never connected.
        pose_qos = QoSProfile(depth=1)
        pose_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        for name in self.robot_names:
            self.create_subscription(
                Float32MultiArray, f"/{name}/waypoint_path", self._make_world_path_cb(name), 10)
            self.create_subscription(
                PoseStamped, f"/{name}/ned/pose_stamped", self._make_pose_cb(name), pose_qos)

        # Keeps the live OpenCV view updated with world path, path labels, and robot pose.
        self.create_timer(0.1, self.plot_tracking_window)

        self.get_logger().info("Path visualizer ready")

    def _declare_parameters(self) -> None:
        self.declare_parameter("grid_csv", "")
        self.declare_parameter("label_column", "cell")
        self.declare_parameter("u_column", "center_x")
        self.declare_parameter("v_column", "center_y")

        self.declare_parameter("vlm_plan_topic", "/vlm_plan")
        self.declare_parameter("robot_names", ["raph", "donnie"])
        self.declare_parameter("camera_image_topic", "/camera_image")
        self.declare_parameter("camera_info_topic", "/ids_overhead/camera_info")

        # Which overhead camera calibration to use for world<->pixel: "gazebo" (sim) or
        # "lab_test" (hardware). Set by the launch file. Poses always arrive in NED.
        self.declare_parameter("camera", "gazebo")

        self.declare_parameter("window_name", "Path Tracking")
        self.declare_parameter("display_scale", 0.5)
        self.declare_parameter("circle_radius", 14)
        self.declare_parameter("path_offset_px", 10.0)

        self.declare_parameter("start_color", [0, 255, 0])
        self.declare_parameter("end_color", [0, 0, 255])

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
        except json.JSONDecodeError:
            self.get_logger().warn("/vlm_plan is not valid JSON")
            return
        if not isinstance(data, dict):
            self.get_logger().warn("/vlm_plan is not a JSON object")
            return
        # Unwrap the {planner, routes} wrapper; a bare {robot: [labels]} object is also accepted.
        routes = data.get("routes", data) if "routes" in data else data
        if not isinstance(routes, dict):
            self.get_logger().warn("/vlm_plan 'routes' is not a {robot: [labels]} object")
            return
        for name in self.robot_names:
            labels = routes.get(name, [])
            self.latest_path_labels[name] = (
                [str(x) for x in labels] if isinstance(labels, list) else [])

    def _make_world_path_cb(self, name: str):
        def _cb(msg: Float32MultiArray) -> None:
            coords: List[Tuple[float, float]] = []
            for i in range(0, len(msg.data) - 1, 2):
                coords.append((float(msg.data[i]), float(msg.data[i + 1])))
            self.latest_world_path[name] = coords
        return _cb

    def _camera_image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to decode /camera_image: {exc}")
            return

        image = self._undistort_image(image)
        self.latest_camera_image = image

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def _make_pose_cb(self, name: str):
        def _cb(msg: PoseStamped) -> None:
            self.latest_pose[name] = msg
            self.pose_history[name].append((msg.pose.position.x, msg.pose.position.y))
        return _cb

    def _grid_path_pixels(self, labels: List[str]) -> List[Tuple[int, int]]:
        pixels: List[Tuple[int, int]] = []
        for label in labels:
            uv = self.grid_pixels.get(label)
            if uv is None:
                continue
            pixels.append((int(round(uv[0])), int(round(uv[1]))))
        return pixels

    @staticmethod
    def _offset_polyline(pts: List[Tuple[int, int]], offset: float) -> List[Tuple[int, int]]:
        """Shift each vertex perpendicular to the local path direction by `offset` px.

        Used to separate different robots' planned paths so coincident routes draw as parallel
        tracks rather than one painting over the other. The normal at each vertex uses the
        averaged direction of its adjacent segments for a smooth offset.
        """
        n = len(pts)
        if offset == 0.0 or n < 2:
            return list(pts)
        out: List[Tuple[int, int]] = []
        for i in range(n):
            if i == 0:
                dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
            elif i == n - 1:
                dx, dy = pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]
            else:
                dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                out.append((int(round(pts[i][0])), int(round(pts[i][1]))))
                continue
            nx, ny = -dy / length, dx / length          # unit normal (perpendicular)
            out.append((int(round(pts[i][0] + nx * offset)),
                        int(round(pts[i][1] + ny * offset))))
        return out

    def _robot_offset(self, index: int) -> float:
        """Symmetric per-robot perpendicular offset so paths straddle the true route evenly."""
        n = len(self.robot_names)
        return (index - (n - 1) / 2.0) * self.path_offset_px

    def _world_to_pixel(self, x_world: float, y_world: float) -> Tuple[int, int]:
        # Calibration owned by coord_transform; this wrapper just rounds to int pixels for cv2.
        u, v = world_to_pixel(x_world, y_world, camera=self.camera_name)
        return int(round(u)), int(round(v))


    def _draw_grid_endpoints(
        self,
        image: np.ndarray,
        path_pixels: List[Tuple[int, int]],
        label_prefix: str = "",
    ) -> None:
        """Draw only one VLM route's start/end markers; do not connect its centroids."""
        prefix = f"{label_prefix} " if label_prefix else ""
        if len(path_pixels) >= 1:
            cv2.circle(image, path_pixels[0], self.circle_radius, self.start_color, 3)
            cv2.putText(
                image, f"{prefix}S",
                (path_pixels[0][0] + 10, path_pixels[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.start_color, 2, cv2.LINE_AA,
            )
        if len(path_pixels) >= 2:
            cv2.circle(image, path_pixels[-1], self.circle_radius, self.end_color, 3)
            cv2.putText(
                image, f"{prefix}E",
                (path_pixels[-1][0] + 10, path_pixels[-1][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.end_color, 2, cv2.LINE_AA,
            )

    @staticmethod
    def _draw_dashed_polyline(img, pts, color, thickness=2, dash=14.0, gap=10.0):
        """Draw a dashed polyline through pts so the actual trail reads distinct from the
        SOLID planned path (dashes measured along arc length, robust to dense trail points)."""
        period = dash + gap
        acc = 0.0  # arc length consumed so far, for continuous dashing across segments
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg < 1e-9:
                continue
            steps = max(1, int(seg))
            for i in range(steps):
                if (acc + seg * (i / steps)) % period < dash:
                    x0 = int(round(a[0] + (b[0] - a[0]) * i / steps))
                    y0 = int(round(a[1] + (b[1] - a[1]) * i / steps))
                    x1 = int(round(a[0] + (b[0] - a[0]) * (i + 1) / steps))
                    y1 = int(round(a[1] + (b[1] - a[1]) * (i + 1) / steps))
                    cv2.line(img, (x0, y0), (x1, y1), color, thickness)
            acc += seg

    def plot_tracking_window(self) -> None:
        """Show live OpenCV tracking window: each robot's world path, VLM path, and pose."""
        if self.latest_camera_image is None:
            return

        frame = self.latest_camera_image.copy()

        for idx, name in enumerate(self.robot_names):
            color = self.robot_colors[name]
            # Perpendicular offset so robots' planned paths stay visible where they coincide.
            offset = self._robot_offset(idx)

            # Waypoint path from /<name>/waypoint_path (metres -> pixels), in the robot's color.
            world_pixels = [self._world_to_pixel(x, y) for x, y in self.latest_world_path[name]]
            world_pixels = self._offset_polyline(world_pixels, offset)
            for i in range(len(world_pixels) - 1):
                cv2.line(
                    frame, world_pixels[i], world_pixels[i + 1],
                    color, PLANNED_PATH_THICKNESS, cv2.LINE_AA)

            # Retain only the VLM route's start/end markers. The centroid-to-centroid route is not
            # a collision-aware path; the solid path above is the translated planner output.
            path_pixels = self._grid_path_pixels(self.latest_path_labels[name])
            path_pixels = self._offset_polyline(path_pixels, offset)
            self._draw_grid_endpoints(frame, path_pixels, name)

            # Actual measured trajectory trail (history of poses), drawn DASHED in the robot's
            # color so it reads distinct from the SOLID planned path. Poses are in NED, so
            # convert NED -> world -> pixel to overlay the planned path.
            traj = [self._world_to_pixel(*ned_to_world(hx, hy))
                    for hx, hy in self.pose_history[name]]
            self._draw_dashed_polyline(
                frame, traj, color, thickness=TRAJECTORY_THICKNESS)

            # Robot pose from /<name>/ned/pose_stamped (NED).
            pose_msg = self.latest_pose[name]
            if pose_msg is not None:
                pose = pose_msg.pose
                ned_yaw = yaw_from_quaternion(
                    pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w)
                # NED pose -> world -> pixel, all through the shared pose transforms so the arrow
                # can never disagree with the dot (each conversion carries the yaw consistently).
                wx, wy, wyaw = ned_to_world_pose(pose.position.x, pose.position.y, ned_yaw)
                pu, pv, pixel_yaw = world_to_pixel_pose(wx, wy, wyaw, camera=self.camera_name)
                robot_u, robot_v = int(round(pu)), int(round(pv))
                arrow_len = 30
                tip = (
                    int(robot_u + arrow_len * math.cos(pixel_yaw)),
                    int(robot_v + arrow_len * math.sin(pixel_yaw)),
                )
                cv2.circle(frame, (robot_u, robot_v), 10, color, -1)
                cv2.arrowedLine(frame, (robot_u, robot_v), tip, (255, 255, 255), 5, tipLength=0.5)
                cv2.circle(frame, (robot_u, robot_v), 12, (0, 0, 255), 3)   # red ring = current pose

                (text_w, text_h), baseline = cv2.getTextSize(
                    name, cv2.FONT_HERSHEY_SIMPLEX,
                    ROBOT_LABEL_SCALE, ROBOT_LABEL_THICKNESS)
                label_x = robot_u + 16
                if label_x + text_w >= frame.shape[1]:
                    label_x = robot_u - 16 - text_w
                label_x = max(0, min(label_x, frame.shape[1] - text_w - 1))
                label_y = max(
                    text_h,
                    min(robot_v + text_h // 2, frame.shape[0] - baseline - 1),
                )
                cv2.putText(
                    frame, name, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    ROBOT_LABEL_SCALE, color, ROBOT_LABEL_THICKNESS, cv2.LINE_AA,
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
