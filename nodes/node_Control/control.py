#!/usr/bin/env python

import rclpy
import random
from rclpy.node import Node
from collections import deque

import math
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy
from coord_transform import ned_to_world_pose, yaw_from_quaternion, wrap_to_pi

# ── Tunable defaults ─────────────────────────────────────────────────────────────────────
# The controller's tunable knobs, surfaced here for easy adjustment. Each is also exposed as a
# ROS parameter (same name, lower-case) so it can be overridden from the launch file or at
# runtime (`ros2 param set ...`) without editing this file.
DEFAULT_MAX_LINEAR_VEL = 0.15     # m/s   — forward speed clamp (conservative for bring-up)
DEFAULT_MAX_ANGULAR_VEL = 0.3    # rad/s — turn-rate clamp (conservative for bring-up)
DEFAULT_HEADING_GATE_DEG = 45.0   # deg   — only drive forward once heading error is within this


class ControlNode(Node):
    """Contains node to move turtlebot from the specified location into the parking space."""

    def __init__(self):
        """Attributes for the CarPark class; including sub, pub"""
        super().__init__("turtle_car")
        # Variable to hold boolean for if goal position has been reached.
        self.parked = True
        # Initialize the current pose variable to a PoseStamped() data type
        self.current_pose = PoseStamped()
        # Initialize the euler variable for later calculations.
        self.cur_yaw_euler = None
        # Initialize the current yaw variable to "None".
        self.current_yaw = None

        self.waypoint_queue = deque()

        self.goal_coordinates = None
        self.goal_yaw = None

        # The robot's pose always arrives in NED (real MoCap, or sim's odom_to_pose which now
        # emits NED). The controller converts NED -> world and runs the PID in the world frame,
        # where +yaw is CCW and matches the robot's angular.z — no per-frame switch or sign hack.

        # kP constant value.
        self.kP_val = 0.5
        self.kP_pos = 0.75

        # Tunable knobs (defaults live at the top of this file). All are ROS parameters, so they
        # can be overridden from the launch file or at runtime without a rebuild.
        # - max_linear_vel / max_angular_vel: conservative output clamps applied before publishing.
        # - heading_gate_deg: only drive forward once |heading error| is within this angle.
        self.declare_parameter("max_linear_vel", DEFAULT_MAX_LINEAR_VEL)      # m/s
        self.declare_parameter("max_angular_vel", DEFAULT_MAX_ANGULAR_VEL)    # rad/s
        self.declare_parameter("heading_gate_deg", DEFAULT_HEADING_GATE_DEG)  # deg
        self.max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self.heading_gate_rad = math.radians(
            self.get_parameter("heading_gate_deg").get_parameter_value().double_value)

        # Holds the error between the current pose & goal pose readings
        self.pose_error = None
        self.yaw_error = None

        # Creates the command in the data type of Twist for Turtlebot hardware directions
        self.command = Twist()

        self.velocity_publisher = self.create_publisher(Twist, "/cmd_vel", 1)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.mocap_subscriber = self.create_subscription(PoseStamped, "/pose_stamped", self.mocap_callback, qos)
        self.mocap_subscriber       # Prevent unused variable warning
        
        self.world_path_subscriber = self.create_subscription(Float32MultiArray, "/waypoint_path", self._world_path_cb, 1)
        self.world_path_subscriber       # Prevent unused variable warning

        self.create_timer(0.1, self.drive_to_goal)

        self.get_logger().info("✓")

    
    def _world_path_cb(self, msg: Float32MultiArray):
        # msg.data → [x1, y1, x2, y2, …]
        coords = [
            (msg.data[i], msg.data[i + 1])
            for i in range(0, len(msg.data), 2)
        ]
        self.waypoint_queue = deque(coords)  # or list(coords)
        # A new plan supersedes the old one. Drop the currently-active waypoint by re-parking so
        # the next control tick re-acquires from the new queue (whose first point is the robot's
        # current pose) instead of finishing the now-stale leg.
        self.parked = True

    def mocap_callback(self, data):
        """Gets current position information from the motion capture cameras."""
        # Sets received odom data from turtlebot, and sets to current position
        self.current_pose = data
    
    def drive_to_goal(self):
        """Function that runs the P controller algorithm."""
        if self.parked and self.waypoint_queue:
            self.goal_coordinates = self.waypoint_queue.popleft()
            self.parked = False
        if not self.parked:
            self.position_error_calc()
            self.orientation_error_calc()
            self.publish_velocity() 

    def position_error_calc(self):
        """Position error in the WORLD frame: convert the NED pose to world, diff vs the goal."""
        p = self.current_pose.pose.position
        q = self.current_pose.pose.orientation
        ned_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        # NED pose -> world (x, y, yaw). Waypoints from the translator are ALREADY world-frame,
        # so we compare directly. Cache the world heading for orientation_error_calc.
        self.robot_wx, self.robot_wy, self.current_yaw = ned_to_world_pose(p.x, p.y, ned_yaw)

        x_difference = self.goal_coordinates[0] - self.robot_wx
        y_difference = self.goal_coordinates[1] - self.robot_wy

        self.pose_error = PoseStamped()
        self.pose_error.pose.position.x = x_difference
        self.pose_error.pose.position.y = y_difference

    def orientation_error_calc(self):
        """Heading error in the WORLD frame (z-up / +yaw CCW, matching the robot's angular.z)."""
        x_difference = self.pose_error.pose.position.x
        y_difference = self.pose_error.pose.position.y

        # Bearing to the goal in world, minus the robot's world heading (cached above).
        bearing_world = math.atan2(y_difference, x_difference)
        self.yaw_error = wrap_to_pi(bearing_world - self.current_yaw)

    def publish_velocity(self):
        """Publishes the linear and angular velocity commands to the hardware."""
        # yaw_error is already wrapped to [-pi, pi]; kP_val is tuned in degrees.
        yaw_error_deg = self.yaw_error * (180 / math.pi)
        self.command.angular.z = self.kP_val * yaw_error_deg

        # Calculate and initiate forward movement of the robot.
        car = self.pose_error.pose.position
        error_distance = math.sqrt(car.x ** 2 + car.y ** 2)

        # Heading gate: only drive forward when roughly facing the goal. If the heading error
        # exceeds heading_gate_deg, command zero linear velocity and just rotate toward the goal
        # (prevents wide arcs that cut corners / clip obstacles). yaw_error is already wrapped to
        # [-pi, pi], so it reflects the true signed heading error.
        if abs(self.yaw_error) > self.heading_gate_rad:
            self.command.linear.x = 0.0
        else:
            self.command.linear.x = self.kP_pos * error_distance

        # Checks if in proximity to target (capture radius). Mark parked so the next tick pulls
        # the following waypoint; if the route is finished (queue empty) command a real stop
        # (zero linear + angular) rather than coasting on the last non-zero command. Intermediate
        # waypoints are not zeroed, so the robot flows smoothly through them.
        if error_distance < 0.1:
            if not self.waypoint_queue:
                self.command.linear.x = 0.0
                self.command.angular.z = 0.0
            self.parked = True

        # Final safety clamp: cap both commands to the conservative limits before publishing,
        # so no combination of gain * error can send the robot an unsafe velocity.
        self.command.linear.x = max(-self.max_linear_vel,
                                    min(self.max_linear_vel, self.command.linear.x))
        self.command.angular.z = max(-self.max_angular_vel,
                                     min(self.max_angular_vel, self.command.angular.z))

        # Publish the updated velocity command values to the bot.
        self.velocity_publisher.publish(self.command)

    # def publish_velocity(self):
    #     # Keep yaw error in radians
    #     yaw_error = self.yaw_error
        
    #     # Wrap to [-pi, pi]
    #     while yaw_error > math.pi:
    #         yaw_error -= 2 * math.pi
    #     while yaw_error < -math.pi:
    #         yaw_error += 2 * math.pi

    #     self.command.angular.z = self.kP_val * yaw_error

    #     car = self.pose_error.pose.position
    #     distance = math.sqrt(car.x ** 2 + car.y ** 2)
    #     self.command.linear.x = self.kP_pos * distance

    #     self.velocity_publisher.publish(self.command)

    #     if distance < 0.1:
    #         self.parked = True


def main(args=None):
    rclpy.init(args=args)

    # Create a turtle bot instance & initiate setup (goal set)
    turtle_car = ControlNode()
    
    rclpy.spin(turtle_car)
        
    turtle_car.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
