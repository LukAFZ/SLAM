#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Path
from .extKalman_LM import *
from .config import *
from .algorithms import *
from .robot import *

from sensor_msgs.msg import PointField
import struct

class ImageSubscriber(Node):
    def __init__(self):
        self.config = configurations()
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        # Subscribe to RGB image topic
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        # Subscribe to depth image topic
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
        # Publisher for 3D pointcloud
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/serf01/nav_rgbd_1/covariance_markers', 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.kinect_to_base_matrix = None
        self.base_to_kinect_matrix = None


        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.dummy_cov = [0.1] * 36 # Dummy covariance values for pose and twist

        self.des_queue = None
        self.depth_frame = None
        # Initiate ORB detector
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Image center coordinates
        self.cu = self.config.cu
        self.cv = self.config.cv
        # Focal length 
        self.f = self.config.f

        self.frame_counter = self.config.frame_counter
        self.min_matches = self.config.min_matches

        # Min und Maximale Laenge fuer Kinect Depth
        self.min_depth = self.config.min_depth
        self.max_depth = self.config.max_depth

        # Store 3D points
        self.current_points_3d = []
        self.current_descriptors = []
        self.point_history = []

        self.ransac_iterations = self.config.ransac_iterations
        self.ransac_threshold = self.config.ransac_threshold

        # Map Management storage (Landmarks in 3D)
        self.map_landmarks = [] # List of {'pt_glob': [x,y,z], 'des': descriptor, 'seen_count': int, 'last_seen': int}
        self.seen_count_threshold = configurations().seen_count_threshold
        self.last_seen_threshold = configurations().last_seen_threshold
        self.frame_index = 0
        self.queue_index = 0
        
        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0

        #Roboterarray
        self.robots = []
        self.num_robots = self.config.num_robots
        for N in range(self.num_robots):
            self.robots.append({
                'id': N,
                'robot': robot(),
            }) 


    def publish_tf(self, x, y, theta, from_frame=None, to_frame=None):
        if from_frame is None:
            from_frame = self.odom_frame
        if to_frame is None:
            to_frame = self.base_frame

        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = from_frame
        t.child_frame_id = to_frame

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
        
    def listener_callback_rgb(self,msg):

        #Initialize the transformation matrix if not already done
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
                
               
            except Exception as e:
                # If the TF is not available yet, log the error and skip processing this frame
                self.get_logger().info(f"Warte auf statischen TF... {e}")
                return


        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        
        # find the keypoints with ORB
        kp = self.orb.detect(frame,None)
        
        kp_clean = []
        # if depth frame is available, overlay depth info on keypoints
        if(self.depth_frame is not None):
            # cycle through keypoints
            for point in kp:
                # get x,y coordinates of keypoint
                x, y = int(point.pt[0]), int(point.pt[1])
                # get depth value at keypoint location and convert to meters
                depth = self.depth_frame[y, x]
                if(self.min_depth < depth < self.max_depth): # filter out invalid depth values
                    kp_clean.append(point)
        else:
            return # skip processing if depth frame is not available
        
        # compute the descriptors with ORB
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        
        if(kp_clean is None or des_clean is None):
            return # skip processing if no valid keypoints/descriptors are found

        #fuer jeden virtuellen Roboter die Koordinaten berechnen und Map aktualisieren
        for robot in self.robots:
            self.curr_pos_x, self.curr_pos_y, self.curr_theta = robot['robot'].update_robot(kp_clean, des_clean, self.depth_frame, self.frame_index, self.frame_counter, self.kinect_to_base_matrix, self.base_to_kinect_matrix)
            
        # Publish TF and Odometry for visualization and downstream tasks
        self.publish_tf(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta)
        self.publish_odometry_msg(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta)
            
        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth",img2)
        cv2.waitKey(1)

        #Publish Landmarks as PointCloud2
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.odom_frame

        #map_points = [[lm['pt_glob'][0]/1000.0, lm['pt_glob'][1]/1000.0, lm['pt_glob'][2]/1000.0] for lm in self.map_landmarks]
        #if map_points:
        #    self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, map_points))

        self.frame_index += 1
        if self.frame_counter > 0:
            self.frame_counter -= 1
        else:
            self.frame_counter = self.config.frame_counter

    def listener_callback_depth(self, msg):
        self.depth_frame=self.bridge.imgmsg_to_cv2(msg,'passthrough')

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()