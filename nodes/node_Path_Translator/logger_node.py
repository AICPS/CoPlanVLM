import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
import csv
from datetime import datetime
import os

class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        # Track if button B is currently pressed
        self.logging_enabled = False
        self.prev_b_state = 0

        # Subscribe to PoseStamped messages
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/raph/ned/pose_stamped',
            self.pose_callback,
            10
        )

        # Subscribe to joystick messages
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Prepare CSV file for logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = f'trajectory_{timestamp}.csv'
        self.filepath = os.path.join(os.getcwd(), self.filename)
        self.file = open(self.filepath, mode='w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.csv_writer.writerow([
            'sec', 'nanosec',
            'position_x', 'position_y', 'position_z',
            'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w'
        ])
        self.get_logger().info(f"TrajectoryLogger started. Logging to: {self.filepath}")
        self.get_logger().info("Press [X] to start logging.")

    def joy_callback(self, msg: Joy):
        try:
            current_b_state = msg.buttons[2]  # Button X is index 2
            
            if current_b_state == 1 and self.prev_b_state == 0:
                self.logging_enabled = not self.logging_enabled
                state = "started" if self.logging_enabled else "stopped"
                self.get_logger().info(f"Logging {state} via B button toggle.")

            self.prev_b_state = current_b_state
        except IndexError:
            self.get_logger().warn("Received Joy message with not enough buttons.")

    def pose_callback(self, msg: PoseStamped):
        if self.logging_enabled:
            data = [
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]
            self.csv_writer.writerow(data)
            # self.get_logger().info(f"Logged pose at time {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        else:
            self.get_logger().debug("B not pressed. Skipping pose.")

    def destroy_node(self):
        self.get_logger().info("Shutting down logger. Saving file...")
        self.file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received. Exiting...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
