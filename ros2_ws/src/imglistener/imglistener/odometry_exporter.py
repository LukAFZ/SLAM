import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
from datetime import datetime
import math
from scipy.spatial.transform import Rotation

class OdometryExporter(Node):
    def __init__(self):
        super().__init__('odometry_exporter')
        
        # CSV Setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"odometry_export_{timestamp}.csv"
        
        # Data Storage
        self.slam_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.wheel_data = {'timestamp': None, 'x': None, 'y': None, 'theta': None}
        self.imu_data = {'timestamp': None, 'theta': None}
        
        # Calibration variables
        self.imu_start_angle = None
        self.heading_offset = None
        self.wheel_start_pos = None # (x_raw, y_raw)
        self.slam_start_pos = None  # (x_slam, y_slam)
        
        self.init_csv()
        
        # Subscribers
        self.slam_sub = self.create_subscription(Odometry, '/serf01/odometry/project_slam', self.slam_callback, 10)
        self.wheel_sub = self.create_subscription(Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/serf01/odometry/imu', self.imu_callback, 10)
        
        self.get_logger().info(f"Odometry Exporter (Full Calibration) gestartet. CSV: {self.csv_filename}")
    
    def init_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'slam_pose_x', 'slam_pose_y',
                'wheel_pose_x', 'wheel_pose_y',
                'slam_rotation_theta', 'imu_rotation_theta'
            ])
    
    def quaternion_to_theta(self, quat):
        """Konvertiert Quaternion stabil in Yaw-Winkel (Bogenmaß)"""
        try:
            q = [quat.x, quat.y, quat.z, quat.w]
            return Rotation.from_quat(q).as_euler('zyx', degrees=False)[0]
        except Exception:
            return 0.0

    def slam_callback(self, msg: Odometry):
        self.slam_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.slam_data['x'] = msg.pose.pose.position.x
        self.slam_data['y'] = msg.pose.pose.position.y
        self.slam_data['theta'] = self.quaternion_to_theta(msg.pose.pose.orientation)
        self.write_row()
    
    def wheel_callback(self, msg: Odometry):
        # 1. Raw Data without any transformation
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        raw_theta = self.quaternion_to_theta(msg.pose.pose.orientation)
        
        # 2. Inititating Calibration if not done yet
        if self.heading_offset is None:
            if self.slam_data['x'] is not None and self.slam_data['theta'] is not None:
                # Wir fixieren die Startpunkte beider Systeme
                self.slam_start_pos = (self.slam_data['x'], self.slam_data['y'])
                self.wheel_start_pos = (raw_x, raw_y)
                
                # Berechne den Winkel-Unterschied (Heading Error)
                self.heading_offset = self.slam_data['theta'] - raw_theta
                self.get_logger().info(f"Kalibrierung erfolgreich! Heading Offset: {self.heading_offset:.4f} rad")
            return # Warten bis SLAM-Referenz verfügbar ist
        
        # 3. Calculate Relative Movement in Wheel Odometry since Start
        dx_raw = raw_x - self.wheel_start_pos[0]
        dy_raw = raw_y - self.wheel_start_pos[1]
        
        # 4. Rotation to odometry frame and Heading Correction
        cos_phi = math.cos(self.heading_offset)
        sin_phi = math.sin(self.heading_offset)
        
        rotated_dx = dx_raw * cos_phi - dy_raw * sin_phi
        rotated_dy = dx_raw * sin_phi + dy_raw * cos_phi
        
        # 5. Final Position = SLAM start point + rotated relative movement
        self.wheel_data['x'] = rotated_dx + self.slam_start_pos[0]
        self.wheel_data['y'] = rotated_dy + self.slam_start_pos[1]
        
        # Correct Angle and normalize to [-pi, pi]
        w_theta = raw_theta + self.heading_offset
        self.wheel_data['theta'] = math.atan2(math.sin(w_theta), math.cos(w_theta))
        
        self.wheel_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.write_row()
    
    def imu_callback(self, msg: Imu):
        raw_theta = self.quaternion_to_theta(msg.orientation)
        
        if self.imu_start_angle is None:
            self.imu_start_angle = raw_theta
            
        # Calculate difference to start angle and normalize
        relative_theta = raw_theta - self.imu_start_angle
        # Normierung verhindert Sprünge am 180-Grad-Umbruch
        self.imu_data['theta'] = math.atan2(math.sin(relative_theta), math.cos(relative_theta))
        self.imu_data['timestamp'] = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.write_row()
    
    def write_row(self):
        # Use current timestamps of the sensor
        ts_list = [d['timestamp'] for d in [self.slam_data, self.wheel_data, self.imu_data] if d['timestamp']]
        if not ts_list: return
        
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                max(ts_list),
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
    try:
        rclpy.spin(exporter)
    except KeyboardInterrupt:
        pass
    finally:
        exporter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()