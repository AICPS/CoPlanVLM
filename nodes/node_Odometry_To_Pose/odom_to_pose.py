#!/usr/bin/env python3
"""Republish Gazebo ground-truth Odometry as an NED PoseStamped.

In sim, the ground-truth pose is in the Gazebo frame. The rest of the stack (controller,
visualizer) expects the same NED pose the real MoCap room publishes, so this node converts
Gazebo -> NED (position AND yaw) and publishes PoseStamped on /<robot>/ned/pose_stamped.
That makes sim and hardware identical downstream.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler

from coord_transform import gazebo_to_ned_pose, yaw_from_quaternion


class OdometryToPoseStamped(Node):
    def __init__(self):
        # Graph name matches the executable and the launch files' name= override, so the node is
        # called the same thing whether it is launched or run bare with `ros2 run`. The sim launch
        # runs two instances and overrides the second to node_Odometry_To_Pose_2.
        super().__init__('node_Odometry_To_Pose')

        # Parameters
        self.declare_parameter('input_topic', '/raph/sim_ground_truth_pose')
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value

        # # QoS settings
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        # Publisher and Subscriber
        self.pose_pub = self.create_publisher(PoseStamped, '/pose_stamped', qos)
        self.odom_sub = self.create_subscription(
            Odometry,
            input_topic,
            self.odom_callback,
            qos
        )

    def odom_callback(self, msg):
        """Convert Gazebo Odometry → NED PoseStamped."""
        gp = msg.pose.pose
        gq = gp.orientation
        gyaw = yaw_from_quaternion(gq.x, gq.y, gq.z, gq.w)

        # Gazebo (x, y, yaw) -> NED (x, y, yaw), yaw derived from the same transform.
        nx, ny, nyaw = gazebo_to_ned_pose(gp.position.x, gp.position.y, gyaw)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, nyaw)

        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose.position.x = nx
        pose_stamped.pose.position.y = ny
        pose_stamped.pose.position.z = gp.position.z
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        self.pose_pub.publish(pose_stamped)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryToPoseStamped()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
