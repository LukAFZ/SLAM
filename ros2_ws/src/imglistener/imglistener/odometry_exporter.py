import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
from datetime import datetime
import os
import math


class OdometryExporter(Node):
    def __init__(self):
        super().__init__('odometry_exporter')
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"odometry_export_{timestamp}.csv"
        
        # Store latest odometry data from each source
        self.slam_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.wheel_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.imu_data = {'timestamp': None, 'theta': None}
        
        # Initialize CSV file with headers
        self.init_csv()
        
        # Create subscribers
        self.slam_sub = self.create_subscription(
            Odometry,
            '/serf01/odometry/project_slam',
            self.slam_callback,
            10
        )
        
        self.wheel_sub = self.create_subscription(
            Odometry,
            '/serf01/odometry/wheel',
            self.wheel_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/serf01/odometry/imu',
            self.imu_callback,
            10
        )
        
        self.get_logger().info(f"Odometry Exporter started. Logging to {self.csv_filename}")
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'slam_pose_x', 'slam_pose_y',
                'wheel_pose_x', 'wheel_pose_y',
                'slam_rotation_theta', 'imu_rotation_theta'
            ])
    
    def slam_callback(self, msg: Odometry):
        """Handle SLAM odometry data"""
        self.slam_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.slam_data['x'] = msg.pose.pose.position.x
        self.slam_data['y'] = msg.pose.pose.position.y
        self.slam_data['theta'] = self.quaternion_to_theta(msg.pose.pose.orientation)
        self.write_row()
    
    def wheel_callback(self, msg: Odometry):
        """Handle wheel odometry data"""
        self.wheel_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.wheel_data['x'] = msg.pose.pose.position.x
        self.wheel_data['y'] = msg.pose.pose.position.y
        self.wheel_data['theta'] = self.quaternion_to_theta(msg.pose.pose.orientation)
        self.write_row()
    
    def quaternion_to_theta(self, quat):
        """Convert quaternion to theta angle (rotation around z-axis)"""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        theta = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return theta
    
    def imu_callback(self, msg: Imu):
        """Handle IMU data - extract rotation from IMU orientation quaternion"""
        self.imu_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.imu_data['theta'] = self.quaternion_to_theta(msg.orientation.z)
        self.write_row()
    
    def write_row(self):
        """Write current row with latest data from all sources"""
        # Use the most recent timestamp
        slam_ts = self.slam_data['timestamp'] if self.slam_data['timestamp'] else ''
        wheel_ts = self.wheel_data['timestamp'] if self.wheel_data['timestamp'] else ''
        imu_ts = self.imu_data['timestamp'] if self.imu_data['timestamp'] else ''
        timestamp = slam_ts if slam_ts else (wheel_ts if wheel_ts else imu_ts)
        
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                self.slam_data['x'] if self.slam_data['x'] is not None else '',
                self.slam_data['y'] if self.slam_data['y'] is not None else '',
                self.wheel_data['x'] if self.wheel_data['x'] is not None else '',
                self.wheel_data['y'] if self.wheel_data['y'] is not None else '',
                self.slam_data['theta'] if self.slam_data['theta'] is not None else '',
                self.imu_data['theta'] if self.imu_data['theta'] is not None else ''
            ])


def main(args=None):
    rclpy.init(args=args)
    exporter = OdometryExporter()
    rclpy.spin(exporter)
    exporter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
