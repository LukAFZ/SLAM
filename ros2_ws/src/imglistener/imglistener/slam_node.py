#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from nav_msgs.msg import Odometry
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation

from .slam_core import VisualSLAMCore

class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        self.bridge = CvBridge()
        
        # initialize SLAM core
        self.slam = VisualSLAMCore()
        # Anzahl der Frames, die nach einem Update übersprungen werden, um die Stabilität zu erhöhen (z.B. bei RANSAC-Updates)
        self.frame_counter = self.slam.config.frame_counter

        # Subscribe to RGB image topic
        self.subscription_rgb = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10
        )
        # Subscribe to depth image topic
        self.subscription_depth = self.create_subscription(
            Image, '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10
        )
        # Publisher for 3D pointcloud
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)
        # Odometry Publisher
        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)

        # TF initialization
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None
        self.depth_frame = None

    def lookup_static_tf(self):
        """lookup the static TF from kinect_depth to base_link and initialize the transformation matrices for coordinate transformations between the kinect frame and the robot's base frame."""
        # Initialize the transformation matrix if not already done
        if self.kinect_to_base_matrix is None:
            try:
                # get TF from kinect_depth to base_link
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, 
                    'kinect_depth', 
                    rclpy.time.Time()
                )
                
                # initialize the transformation matrix as identity
                self.kinect_to_base_matrix = np.eye(4)
                
                # store translation
                self.kinect_to_base_matrix[0, 3] = t.transform.translation.x * 1000 # convert to mm
                self.kinect_to_base_matrix[1, 3] = t.transform.translation.y * 1000 # convert to mm
                self.kinect_to_base_matrix[2, 3] = t.transform.translation.z * 1000 # convert to mm

                # store rotation (convert quaternion to rotation matrix)
                quat = [t.transform.rotation.x, t.transform.rotation.y, 
                        t.transform.rotation.z, t.transform.rotation.w]
                self.kinect_to_base_matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
                
                # calculate inverse for transforming points from kinect frame to base frame
                self.base_to_kinect_matrix = np.linalg.inv(self.kinect_to_base_matrix)
                return True
            except Exception as e:
                # If the TF is not available yet, log the error and skip processing this frame
                self.get_logger().info(f"Warte auf statischen TF... {e}")
                return False
        return True

    def listener_callback_depth(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def listener_callback_rgb(self, msg):
        if not self.lookup_static_tf() or self.depth_frame is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Slam processing in slam_core.py
        pose_updated, rx, ry, rtheta, best_map_manager = self.slam.process_frame(
            frame, self.depth_frame, 
            self.kinect_to_base_matrix, self.base_to_kinect_matrix, 
            self.frame_counter
        )

        if pose_updated:
            # Publish TF and Odometry for visualization and downstream tasks
            self.publish_tf(rx / 1000.0, ry / 1000.0, rtheta)
            self.publish_odometry_msg(rx / 1000.0, ry / 1000.0, rtheta)
            self.frame_counter = self.slam.config.frame_counter

        # Publish Landmarks as PointCloud2
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.odom_frame
        map_points = best_map_manager.get_all_points_for_msg()
        if map_points:
            self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, map_points))

        self.frame_counter -= 1

    def publish_tf(self, x, y, theta):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        euler = Rotation.from_euler('z', float(theta))
        quat = euler.as_quat(canonical=True)

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_odometry_msg(self, x, y, theta):
        msg = Odometry()
        
        # Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        
        # Child Frame ID
        msg.child_frame_id = self.base_frame

        # Pose
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        # Euler-Winkel zu Quaternion konvertieren
        r = Rotation.from_euler('z', float(theta))
        quat = r.as_quat(canonical=True)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        
        cov = [0.0] * 36
        """
        # Mapping der P-Matrix (x, y, theta) auf das ROS 6x6 Schema:
        cov[0]  = P_matrix[0, 0]/1000000 # Var(x)
        cov[1]  = P_matrix[0, 1]/1000000 # Cov(x, y)
        cov[5]  = P_matrix[0, 2]/1000 # Cov(x, theta)
        
        cov[6]  = P_matrix[1, 0]/1000000 # Cov(y, x)
        cov[7]  = P_matrix[1, 1]/1000000 # Var(y)
        cov[11] = P_matrix[1, 2]/1000 # Cov(y, theta)
        
        cov[30] = P_matrix[2, 0]/1000 # Cov(theta, x)
        cov[31] = P_matrix[2, 1]/1000 # Cov(theta, y)
        cov[35] = P_matrix[2, 2] # Var(theta)
        """
        msg.pose.covariance = cov

        # Veröffentlichen
        self.odom_publisher.publish(msg)    

def main(args=None):
    rclpy.init(args=args)
    node = SlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()