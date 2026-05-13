import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
import os
from datetime import datetime
import math

class OdometryExporter(Node):
    def __init__(self):
        super().__init__('odometry_exporter')
        
        # Speicherort: Home-Verzeichnis
        home_dir = os.path.expanduser("~")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(home_dir, f"odom_comparison_{timestamp}.csv")
        
        # Interne Datenspeicher (Werte in Grad)
        self.slam_data = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.wheel_data = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.imu_data = {'theta': 0.0}
        
        # Offsets in Radiant (werden beim ersten Callback gefüllt)
        self.offset_slam = None
        self.offset_wheel = None
        self.offset_imu = None
        
        # Datei-Handling
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'timestamp', 
            'slam_x', 'slam_y', 
            'wheel_x', 'wheel_y', 
            'slam_theta_deg', 'wheel_theta_deg', 'imu_theta_deg'
        ])
        self.csv_file.flush()

        # Subscriber
        self.create_subscription(Odometry, '/serf01/odometry/project_slam', self.slam_callback, 10)
        self.create_subscription(Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)
        self.create_subscription(Imu, '/serf01/odometry/imu', self.imu_callback, 10)
        
        self.get_logger().info(f"Logging gestartet: {self.csv_filename}")

    def get_yaw_from_quat(self, quat):
        """Extrahiert Yaw-Winkel (Radiant) aus einer Quaternion."""
        x, y, z, w = quat.x, quat.y, quat.z, quat.w
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def calculate_relative_deg(self, current_rad, offset_rad):
        """Berechnet die Differenz zum Startwert und gibt Grad zurück."""
        diff = current_rad - offset_rad
        # Normalisierung auf [-pi, pi] um Sprünge bei 180° zu vermeiden
        normalized_diff = math.atan2(math.sin(diff), math.cos(diff))
        return math.degrees(normalized_diff)

    def slam_callback(self, msg):
        raw_theta = self.get_yaw_from_quat(msg.pose.pose.orientation)
        if self.offset_slam is None:
            self.offset_slam = raw_theta
            self.get_logger().info("SLAM Nullpunkt gesetzt.")
        
        self.slam_data['x'] = msg.pose.pose.position.x
        self.slam_data['y'] = msg.pose.pose.position.y
        self.slam_data['theta'] = self.calculate_relative_deg(raw_theta, self.offset_slam)
        self.write_row(msg.header.stamp)

    def wheel_callback(self, msg):
        raw_theta = self.get_yaw_from_quat(msg.pose.pose.orientation)
        if self.offset_wheel is None:
            self.offset_wheel = raw_theta
            self.get_logger().info("Wheel Nullpunkt gesetzt.")

        self.gemessen_x = msg.pose.pose.position.x
        self.gemessen_y = msg.pose.pose.position.y
        #Drehe um -90 Grad    
        self.wheel_data['x'] = -self.gemessen_y
        self.wheel_data['y'] = self.gemessen_x
        self.wheel_data['theta'] = self.calculate_relative_deg(raw_theta, self.offset_wheel)
        self.write_row(msg.header.stamp)

    def imu_callback(self, msg):
        # Korrektur des Fehlers: Übergabe des gesamten Orientierung-Objekts
        raw_theta = self.get_yaw_from_quat(msg.orientation)
        if self.offset_imu is None:
            self.offset_imu = raw_theta
            self.get_logger().info(f"IMU Kompass-Offset gefunden: {math.degrees(raw_theta):.2f}°")
        
        self.imu_data['theta'] = self.calculate_relative_deg(raw_theta, self.offset_imu)
        self.write_row(msg.header.stamp)

    def write_row(self, stamp):
        """Schreibt die aktuelle Datenzeile in die CSV."""
        # Zeitstempel in Sekunden konvertieren
        ts = stamp.sec + stamp.nanosec / 1e9
        
        self.writer.writerow([
            f"{ts:.4f}", 
            f"{self.slam_data['x']:.4f}", f"{self.slam_data['y']:.4f}",
            f"{self.wheel_data['x']:.4f}", f"{self.wheel_data['y']:.4f}",
            f"{self.slam_data['theta']:.2f}", 
            f"{self.wheel_data['theta']:.2f}", 
            f"{self.imu_data['theta']:.2f}"
        ])
        # Sofortiges Schreiben erzwingen
        self.csv_file.flush()

    def destroy_node(self):
        self.get_logger().info("Schließe CSV-Datei...")
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = OdometryExporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()