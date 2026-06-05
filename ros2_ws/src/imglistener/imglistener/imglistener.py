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
from .constants import *
from .algorithms import *


from sensor_msgs.msg import PointField
import struct

class ImageSubscriber(Node):
    def __init__(self):

        # Alle implementierten Algorithmen
        self.algorithmen = algorithms()
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
        self.orb = cv2.ORB_create(nfeatures=1500, patchSize=31)
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
        self.seen_count_threshold = self.config.seen_count_threshold
        self.last_seen_threshold = self.config.last_seen_threshold
        self.keyframes = []
        self.frame_index = 0
        self.queue_index = 0
        
        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0


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

        # 3D coordinates calculation
        local_robot_pts_3d = self.algorithmen.calculate_local_cords_from_matches(kp_clean, des_clean, self.kinect_to_base_matrix, self.depth_frame)

        # Initial map creation
        if not self.map_landmarks:
            for i in range(len(local_robot_pts_3d)):
                self.map_landmarks.append({
                    'pt_glob': local_robot_pts_3d[i], 
                    'des': des_clean[i], 
                    'seen_count': 1, 
                    'last_seen': self.frame_index,
                    'ekf': ExtKalman(np.array(local_robot_pts_3d[i]))    
                })

        # Viewing Cone / Frustum Culling
        visible_des, visible_pts_glob_2d, visible_map_indices = self.algorithmen.test_only_for_visible_landmarks(self.map_landmarks, self.curr_pos_x, self.curr_pos_y, self.curr_theta, self.base_to_kinect_matrix)
        
        delta_R = None
        delta_t = None
        delta_theta = None
        

        if(self.frame_counter == 0):
            if(len(visible_des) > 0):
                # Match visible landmarks with current frame keypoints
                matches = self.bf.match(np.array(visible_des), des_clean)
                
                if(len(matches) > self.min_matches):
                    P_local = []
                    Q_curr  = []
                    matched_curr_indices = set()
                    visible_landmarks = []

                    cos_t = math.cos(-self.curr_theta)
                    sin_t = math.sin(-self.curr_theta)

                    for match in matches:
                        map_idx = visible_map_indices[match.queryIdx]
                        pt_glob = visible_pts_glob_2d[match.queryIdx]

                        # transform global landmark position to local robot coordinates for the matched landmark
                        dx = pt_glob[0] - self.curr_pos_x
                        dy = pt_glob[1] - self.curr_pos_y
                        lx =  dx * cos_t - dy * sin_t
                        ly =  dx * sin_t + dy * cos_t

                        P_local.append([lx, ly])
                        Q_curr.append(local_robot_pts_3d[match.trainIdx][:2])
                        matched_curr_indices.add(match.trainIdx)

                        lm = self.map_landmarks[map_idx]
                        z_pt = local_robot_pts_3d[match.trainIdx][:2]
                        depth_val = local_robot_pts_3d[match.trainIdx][2]

                        lm['seen_count'] += 1
                        lm['last_seen'] = self.frame_index
                        visible_landmarks.append(lm)

                    
                    # ransac refinement to get robust transformation estimation
                    delta_R, delta_t, delta_theta = self.algorithmen.ransac_refinement(np.array(P_local), np.array(Q_curr))

                    if delta_R is not None:
                        #z_x = 0.0
                        #z_y = 0.0
                        #z_theta = 0.0
                        # relative transformation in local robot coordinates to odom frame
                        cos_c = math.cos(self.curr_theta)
                        sin_c = math.sin(self.curr_theta)
                        delta_tx_odom =  delta_t[0] * cos_c - delta_t[1] * sin_c
                        delta_ty_odom =  delta_t[0] * sin_c + delta_t[1] * cos_c

                        # update current pose with the estimated transformation
                        self.curr_pos_x += delta_tx_odom
                        self.curr_pos_y += delta_ty_odom
                        self.curr_theta += delta_theta
                        
                        P_array = np.array(P_local)
                        Q_array = np.array(Q_curr)
                        
                        # Prüfe, wo die Punkte nach der RANSAC-Drehung wirklich liegen
                        Q_transformed = (delta_R @ Q_array.T).T + delta_t
                        errors = np.linalg.norm(P_array - Q_transformed, axis=1)

                        # Dein RANSAC-Threshold war 50, mit Toleranz (1.25) = 62.5
                        for i, match in enumerate(matches):
                            if errors[i] < self.ransac_threshold*1.25: # Nur echte Inliers zulassen!
                                map_idx = visible_map_indices[match.queryIdx]
                                train_idx = match.trainIdx


                                kalman_result = []
                                lm = self.map_landmarks[map_idx]
                                depth = float(self.depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                                kalman_result, P = lm['ekf'].update(np.array(local_robot_pts_3d[train_idx]), np.array([self.curr_pos_x, self.curr_pos_y, self.curr_theta]), np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]), depth)
                                
                                self.map_landmarks[map_idx]['pt_glob'] = [kalman_result[0], kalman_result[1], kalman_result[2]]


                                z_pt = local_robot_pts_3d[train_idx][:2]
                                depth_val = local_robot_pts_3d[train_idx][2]
                                
                                
                        # Publish TF and Odometry for visualization and downstream tasks
                        self.publish_tf(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta)
                        self.publish_odometry_msg(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta)

                        # Add new landmarks
                        # add not all points but only those that where mached with the current frame, to avoid adding outliers
                        
                        for i in range(len(local_robot_pts_3d)):
                            if i not in matched_curr_indices: # Nur Punkte hinzufügen, die nicht einmal gematcht wurden (also komplett neue Punkte)
                                pt = local_robot_pts_3d[i]
                                gx = (self.curr_pos_x) + pt[0]*math.cos(self.curr_theta) - pt[1]*math.sin(self.curr_theta)
                                gy = (self.curr_pos_y) + pt[0]*math.sin(self.curr_theta) + pt[1]*math.cos(self.curr_theta)
                                self.map_landmarks.append({
                                    'pt_glob': [gx, gy, pt[2]], 
                                    'des': des_clean[i], 
                                    'seen_count': 1, 
                                    'last_seen': self.frame_index,
                                    'ekf': ExtKalman(np.array([gx, gy, pt[2]]))
                                })
                        
                        # Remove landmarks (Quality metric = seen_count)
                        # Möglicherweise Treshhold der P-matrix (als weitere Quality Metrik) hinzufügen, um nur sehr gut lokalisierte Landmarks zu behalten
                        self.map_landmarks = [lm for lm in self.map_landmarks if lm['seen_count'] > self.seen_count_threshold or (self.frame_index - lm['last_seen']) < self.last_seen_threshold]

                else:
                    print(f"Not enough matches found for RANSAC {len(matches)}")
            
            self.frame_counter = self.config.frame_counter    

        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth",img2)
        cv2.waitKey(1)

        #Publish Landmarks as PointCloud2
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.odom_frame
        map_points = [[lm['pt_glob'][0]/1000.0, lm['pt_glob'][1]/1000.0, lm['pt_glob'][2]/1000.0] for lm in self.map_landmarks]
        if map_points:
            self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, map_points))

        # empty the 3D points list for the next frame
        self.current_points_3d = []
        self.current_descriptors = []
        self.frame_index += 1
        self.frame_counter -= 1

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