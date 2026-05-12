#!/usr/bin/env python

import rclpy
import random
from rclpy.node import Node
from collections import deque

import math
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy

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

        # kP constant value.
        self.kP_val = 0.5
        self.kP_pos = 0.75

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
        
        self.world_path_subscriber = self.create_subscription(Float32MultiArray, "/world_path", self._world_path_cb, 1)
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
        """Publishes the difference between the current position and goal position"""
        # Get current odometer reading's x & y variables.
        current_pose = self.current_pose.pose.position
        goal_coord = self.goal_coordinates

        # Calc x & y differences between pose reading & Goal position.
        x_difference = goal_coord[1] - current_pose.x
        y_difference = goal_coord[0] - current_pose.y

        # Initialize odom_error attribute to PoseStamped data type
        self.pose_error = PoseStamped()
        self.pose_error.pose.position.x = x_difference
        self.pose_error.pose.position.y = y_difference

    def orientation_error_calc(self):
        """Orients turtlebot towards the goal point."""
        # Gets current turtlebot_position
        data = self.current_pose
        # Sets y_difference and x_difference values for relative_yaw calculation
        x_difference, y_difference = self.pose_error.pose.position.x, self.pose_error.pose.position.y

        # Converts the odom data from quaternion (sensory input) into euler (yaw angular)
        ##### DO NOT CHANGE THE FOLLOW LINES OF CODE.
        ornt_quat = data.pose.orientation
        ornt_quat_list = [ornt_quat.x, ornt_quat.y, ornt_quat.z, ornt_quat.w]
        [roll, pitch, yaw] = euler_from_quaternion(ornt_quat_list)
        self.cur_yaw_euler = [roll, pitch, yaw]
        ##### END SECTION

        # Calculate the yaw error (amount to rotate)
        self.current_yaw = self.cur_yaw_euler[2]
        relative_yaw = math.atan2(y_difference, x_difference)
        self.yaw_error = relative_yaw - self.current_yaw

    def publish_velocity(self):
        """Publishes the linear and angular velocity commands to the hardware."""
        # Convert yaw error into degrees (from radians)
        yaw_error_deg = self.yaw_error * (180/math.pi)

        if yaw_error_deg > 180:
            yaw_error_deg = -(yaw_error_deg - 180)  ## TODO Computational Error here!!!
        if yaw_error_deg < -180:
            yaw_error_deg = (abs(yaw_error_deg) - 180)

        self.command.angular.z = self.kP_val * yaw_error_deg

        # Calculate and initiate forward movement of the robot.
        car = self.pose_error.pose.position
        error_distance = math.sqrt(car.x ** 2 + car.y ** 2)
        self.command.linear.x = self.kP_pos * error_distance

        # Publish the updated velocity command values to the bot
        self.velocity_publisher.publish(self.command)

        # Checks if in proximity to target.
        # if self.command.linear.x < 0.02 and self.command.linear.y < 0.02:
        #     self.parked = True
        if error_distance < 0.25:
            self.parked = True

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
