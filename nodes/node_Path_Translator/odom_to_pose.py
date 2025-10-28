#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdometryToPoseStamped(Node):
    def __init__(self):
        super().__init__('node_odom_to_pose')

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
        """Convert Odometry → PoseStamped"""
        # Create PoseStamped message
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
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
