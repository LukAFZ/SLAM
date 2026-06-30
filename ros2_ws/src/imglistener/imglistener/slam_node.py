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
import cProfile
from scipy.spatial.transform import Rotation

from .slam_core import VisualSLAMCore
from .config import configurations


class SlamNode(Node):
    def __init__(self):
        super().__init__('slam_node', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        ])
        #import configurations
        self.config = configurations()

        # Initialize CvBridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()
        
        # initialize SLAM core
        self.slam = VisualSLAMCore()

        # Subscribe to RGB image topic with puffer-size 10
        self.subscription_rgb = self.create_subscription(
            Image, self.config.RGB_TOPIC, self.listener_callback_rgb, self.config.PUFFER_SIZE
        )
        # Subscribe to depth image topic with puffer-size 10
        self.subscription_depth = self.create_subscription(
            Image, self.config.DEPTH_TOPIC, self.listener_callback_depth, self.config.PUFFER_SIZE
        )
        # Publisher for 3D pointcloud with puffer-size 10
        self.pcl_publisher = self.create_publisher(PointCloud2, self.config.PCL_TOPIC, self.config.PUFFER_SIZE)
        # Odometry Publisher with puffer-size 10
        self.odom_publisher = self.create_publisher(Odometry, self.config.ODOM_TOPIC, self.config.PUFFER_SIZE)

        # TF initialization
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        # For TF listening
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Counter of frames to skip after an update to increase stability (e.g., after RANSAC updates)
        self.frame_counter = self.slam.config.FRAME_COUNTER
        self.frame_index = 0

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None
        self.depth_frame = None

        self.profiler = cProfile.Profile()
    def lookup_static_tf(self):
        """Lookup the static TF from kinect_depth to base_link and initialize the transformation matrices for coordinate transformations between the kinect frame and the robot's base frame."""
        # Initialize the transformation matrix if not already done
        if self.kinect_to_base_matrix is None:
            try:
                # get TF from kinect_depth to base_link from the TF buffer
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
        """
        Listener callback for depth images. This function saves the incoming depth image for use in the RGB callback.
        """
        # save depth frame for use in the RGB callback, convert to OpenCV format
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def listener_callback_rgb(self, msg):
        """
        Listener callback for RGB images. This function processes the incoming RGB image, performs SLAM processing, and publishes the updated pose and map if available.
        """
        if not self.lookup_static_tf() or self.depth_frame is None:
            #implement solution if fail
            return
        # Count to zero after an update to skip frames for stability
        if self.frame_counter <=0:
            # Convert ROS image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            self.profiler.enable()
            # Slam processing in slam_core.py
            pose_updated, best_pose, best_map_manager = self.slam.process_frame(
                frame, self.depth_frame, 
                self.kinect_to_base_matrix, self.base_to_kinect_matrix, 
                self.frame_index
            )
            self.profiler.disable()
            #self.profiler.print_stats(sort='cumulative')
            # Publish Landmarks as PointCloud2
            header = std_msgs.msg.Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.odom_frame

            if pose_updated:
                print(f"Pose aktualisiert: x={best_pose.x:.2f} mm, y={best_pose.y:.2f} mm, theta={best_pose.theta:.2f} rad")
                # Publish TF and Odometry for visualization and downstream tasks
                self.publish_tf(best_pose.x / 1000.0, best_pose.y / 1000.0, best_pose.theta, msg.header.stamp)
                self.publish_robots_tf_array(self.slam.robots, msg.header.stamp)
                self.publish_odometry_msg(best_pose.x / 1000.0, best_pose.y / 1000.0, best_pose.theta, msg.header.stamp)
                self.frame_counter = self.slam.config.FRAME_COUNTER
                map_points = best_map_manager.get_all_points_for_msg()
                if map_points:
                    self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, map_points))
            else:
                print("Kein Posen-Update")

        else:
            print(f"Skipping frame {self.frame_index} to increase stability. Frame counter: {self.frame_counter}")

        self.frame_counter -= 1
        self.frame_index += 1
        print(f"Frame Index: {self.frame_index}, Frame Counter: {self.frame_counter}")

    def publish_tf(self, x, y, theta, stamp):
        """
        Publish TF Message
        """
        # initialize 
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        euler = Rotation.from_euler('z', float(theta))
        quat = euler.as_quat(canonical=True)

        #Convert to ROS Quaternion format
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    #virtuelle Roboter
    def publish_robots_tf_array(self, robots_list, stamp):
        """
        Publish the array of TF messages for all virtual robots in the list. Each robot will have its own unique child frame ID based on its ID.
        """
        # Eine leere Liste für alle Transformationen erstellen
        tf_messages = []
        current_time = self.get_clock().now().to_msg()

        for robot in robots_list:
            t = TransformStamped()
            t.header.stamp = stamp if stamp is not None else current_time
            t.header.frame_id = self.odom_frame
            
            # WICHTIG: Jedem Partikel einen eindeutigen Frame-Namen geben!
            t.child_frame_id = f"virtual_robot_{robot.id}"

            # Position setzen (Achtung: Falls deine Posen im Code in mm gerechnet werden, 
            # musst du hier durch 1000.0 teilen. Wenn sie in Metern sind, lass das '/ 1000.0' weg!)
            t.transform.translation.x = robot.pose.x / 1000.0
            t.transform.translation.y = robot.pose.y / 1000.0
            t.transform.translation.z = 0.0

            # Rotation genau wie in deiner Vorlage berechnen
            theta = robot.pose.theta
            euler = Rotation.from_euler('z', float(theta))
            quat = euler.as_quat(canonical=True)

            # Convert to ROS Quaternion format
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]

            # Nachricht an die Liste anhängen
            tf_messages.append(t)

        # Alle Transformationen gesammelt als Array/Liste absenden
        if tf_messages:
            self.tf_broadcaster.sendTransform(tf_messages)

    def publish_odometry_msg(self, x, y, theta, stamp):
        """
        Publish Odometry Message
        """
        msg = Odometry()
        
        # Header
        msg.header.stamp = stamp
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
    """
    Initialize the ROS2 node
    """
    rclpy.init(args=args)
    node = SlamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()