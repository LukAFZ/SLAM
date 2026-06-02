#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from nav_msgs.msg import Odometry
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

from sensor_msgs.msg import PointField
import struct

class ImageSubscriber(Node):
    def __init__(self):
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
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61
        self.frame_counter = 10
        self.min_matches = 10
        self.kinect_width = 640
        self.kinect_height = 480

        # Min und Maximale Laenge fuer Kinect Depth
        self.min_depth = 400
        self.max_depth = 7500

        # Ransac Configuration
        self.ransac_iterations = 200
        self.ransac_threshold = 50

        # Store 3D points
        self.current_points_3d = []
        self.current_descriptors = []
        self.point_history = []

        # Map Management storage (Landmarks in 3D)
        self.map_landmarks = [] # List of {'pt_glob': [x,y,z], 'des': descriptor, 'seen_count': int, 'last_seen': int}
        self.seen_count_threshold = 5
        self.last_seen_threshold = 15
        self.keyframes = []
        self.frame_index = 0
        self.to_proceed_frames = 1
        self.queue_index = 0
        
        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0

        #self.kalman_filter = ExtendedKalmanFilter(State(self.curr_pos_x, self.curr_pos_y, self.curr_theta))

    def get_kapsch_2d(self, P, Q):    

        # Calculate the centroids of P and Q
        P_middle = np.mean(P, axis=0) #p_quer
        Q_middle = np.mean(Q, axis=0) #q_quer

        # Center the points by subtracting the centroids  
        P_centered = P - P_middle #p_strich
        Q_centered = Q - Q_middle #p_strich


        # Calculate the rotation angle (theta) using the Kabsch algorithm
        theta = math.atan2(sum(Q_centered[:,0]*P_centered[:,1] - Q_centered[:,1]*P_centered[:,0]), sum(Q_centered[:,0]*P_centered[:,0] + Q_centered[:,1]*P_centered[:,1]))
        #print(f"Rotation angle (theta): {math.degrees(theta):.2f} degrees")

        # Calculate the rotation matrix using the rotation angle
        Rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        # Calculate the translation vector using the centroids and the rotation matrix
        Translation = P_middle - Rotation_matrix @ Q_middle
        #print(f"Translation vector: {Translation}")
        return Rotation_matrix, Translation, theta

    def ransac_refinement(self, P, Q):
        
        max_iterations = self.ransac_iterations
        threshold = self.ransac_threshold
        best_rotation = None
        best_translation = None
        best_theta = 0
        best_inlier_count = 0
        
        if(len(P) < 5):
            # Not enough points for RANSAC, return the transformation from all points
            return best_rotation, best_translation, best_theta

        for _ in range(max_iterations):
            # Randomly select a subset of points
            P_second = []
            Q_second = []

            indices = np.random.choice(len(P), size=3, replace=False)
            P_subset = P[indices]
            Q_subset = Q[indices]

            # Estimate the transformation using the selected subset
            R_estimated, t_estimated, theta_estimated = self.get_kapsch_2d(P_subset, Q_subset)

            # Transform Q and calculate per-point errors
            Q_transformed = (R_estimated @ Q.T).T + t_estimated
            errors = np.linalg.norm(P - Q_transformed, axis=1)
            
            inlier_count = np.sum(errors < threshold)

            if inlier_count > best_inlier_count:
                
                for e, p, q in zip(errors, P, Q):
                    if e < threshold*1.25:
                        P_second.append(p)
                        Q_second.append(q)

                best_rotation, best_translation, best_theta = self.get_kapsch_2d(np.array(P_second), np.array(Q_second))

                best_inlier_count = inlier_count
                
        return best_rotation, best_translation, best_theta

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
        local_robot_pts_3d = []
        for point, des in zip(kp_clean, des_clean):
            depth = float(self.depth_frame[int(point.pt[1]), int(point.pt[0])])
            
            # X, Y, Z in camera frame
            x_c = (point.pt[0] - self.cu) * depth / self.f
            y_c = (point.pt[1] - self.cv) * depth / self.f
            z_c = depth

            self.current_points_3d.append((x_c, y_c, z_c))
            self.current_descriptors.append(des)
            # Roboterkoordinaten (X=vorne, Y=links, Z=hoch)

            pt_kinect = np.array([x_c, y_c, z_c, 1.0])
            pt_base = self.kinect_to_base_matrix @ pt_kinect
            
            local_robot_pts_3d.append([pt_base[0], pt_base[1], pt_base[2]])

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
        visible_des = []
        visible_pts_glob_2d = []
        visible_map_indices = []
        
        for idx, lm in enumerate(self.map_landmarks):
            # Calculate relative landmark position to robot
            dx = lm['pt_glob'][0] - (self.curr_pos_x)
            dy = lm['pt_glob'][1] - (self.curr_pos_y)
            # Transform to local robot coordinates
            # Rotation by -curr_theta to align with robot's current orientation
            lx = dx * math.cos(-self.curr_theta) - dy * math.sin(-self.curr_theta)
            ly = dx * math.sin(-self.curr_theta) + dy * math.cos(-self.curr_theta)
            lz = lm['pt_glob'][2]
            
            
            pt_local_base = np.array([lx, ly, lz, 1.0])
            pt_cam = self.base_to_kinect_matrix @ pt_local_base
            
            c_x = pt_cam[0]
            c_y = pt_cam[1]
            c_z = pt_cam[2]
            
            #Prüfe, ob der Punkt vor der Kamera liegt und innerhalb des gültigen Tiefenbereichs liegt
            if 0 < c_z < self.max_depth: # In front of camera and in valid depth range
                # Project to 2D image plane with pinhole camera model
                u_p = (c_x * self.f) / c_z + self.cu
                v_p = (c_y * self.f) / c_z + self.cv
                # Check if projected point is within image bounds
                if 0 <= u_p <= self.kinect_width and 0 <= v_p <= self.kinect_height:
                    visible_des.append(lm['des'])
                    visible_pts_glob_2d.append(lm['pt_glob'][:2])
                    visible_map_indices.append(idx)
        
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
                    delta_R, delta_t, delta_theta = self.ransac_refinement(np.array(P_local), np.array(Q_curr))

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
                                
                                

                        """
                        #z_dict = {}
                        #for i in matched_curr_indices:
                        #    key = des_clean[i].tobytes()
                        #    z_dict[key] = (local_robot_pts_3d[i][:2], local_robot_pts_3d[i][2]) # 2D position and depth
                        kalman_measurements = []
                        matched_curr_indices = set() # Wir überschreiben das Set, um nur ECHTE Inliers zu behalten
                        all_matched_train_indices = set()
                        
                        P_array = np.array(P_local)
                        Q_array = np.array(Q_curr)
                        
                        # Prüfe, wo die Punkte nach der RANSAC-Drehung wirklich liegen
                        Q_transformed = (delta_R @ Q_array.T).T + delta_t
                        errors = np.linalg.norm(P_array - Q_transformed, axis=1)
                        
                        # Dein RANSAC-Threshold war 50, mit Toleranz (1.25) = 62.5
                        for i, match in enumerate(matches):
                            all_matched_train_indices.add(match.trainIdx)
                            if errors[i] < self.ransac_threshold*1.25: # Nur echte Inliers zulassen!
                                map_idx = visible_map_indices[match.queryIdx]
                                train_idx = match.trainIdx
                                
                                lm = self.map_landmarks[map_idx]
                                z_pt = local_robot_pts_3d[train_idx][:2]
                                depth_val = local_robot_pts_3d[train_idx][2]
                                
                                kalman_measurements.append((lm, z_pt, depth_val))
                                matched_curr_indices.add(train_idx)

        
                        kalman_iteration_result = self.kalman_filter.kalman_iteration(Coordinate(delta_t[0], delta_t[1], 0.0), delta_theta, kalman_measurements)

                        self.curr_pos_x = kalman_iteration_result[0].x
                        self.curr_pos_y = kalman_iteration_result[0].y
                        self.curr_theta = kalman_iteration_result[0].theta
                        """
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
            
            self.frame_counter = self.to_proceed_frames    

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

        if self.map_landmarks:
            # 1. Felder definieren
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
                PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1)
            ]

            map_points_data = []
            
            # Da du im Code die originalen Bilddaten (frame) hast, 
            # müssen wir die Farbinformationen während des Matchings puffern oder auslesen.
            # Für dieses Beispiel nehmen wir an, wir packen die Farbe und den seen_count hinein.
            
            for lm in self.map_landmarks:
                # Koordinaten in Metern
                x = lm['pt_glob'][0] / 1000.0
                y = lm['pt_glob'][1] / 1000.0
                z = lm['pt_glob'][2] / 1000.0
                
                # Standardfarbe (z.B. Grün für Landmarks, da deine Keypoints im Bild auch grün sind)
                # Format: 0x00RRGGBB
                r, g, b = 0, 255, 0 
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
                
                # Qualität/Sichtungen als Intensity
                intensity = float(lm['seen_count'])
                
                # Punkt-Array anhängen
                map_points_data.append([x, y, z, rgb_packed, intensity])

        # Erzeuge die erweiterte Cloud
        pc2_msg = pcl2.create_cloud(header, fields, map_points_data)
        self.pcl_publisher.publish(pc2_msg)

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