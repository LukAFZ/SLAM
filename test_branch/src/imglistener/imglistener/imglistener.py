#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
import csv

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        
        self.subscription_rgb = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        self.subscription_depth = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
        
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)
        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.create_subscription(Odometry, '/serf01/odometry/wheel', self.wheel_callback, 10)

        # In __init__:
        #self.create_subscription(Odometry, '/serf01/odometry/imu', self.imu_callback, 10)

        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.dummy_cov = [0.1] * 36

        self.depth_frame = None
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Eigenschaften der Kamera
        self.cu = 318.525
        self.cv = 241.181
        self.f = 526.61
        self.frame_counter = 10
        self.min_matches = 10

        self.current_points_3d = []
        self.current_descriptors = []
        self.map_landmarks = [] 
        self.frame_index = 0
        self.to_proceed_frames = 1

        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0
    

    def get_kapsch_2d(self, P, Q):
        # Berechnung der Centroidenq    
        P_middle = np.mean(P, axis=0) 
        Q_middle = np.mean(Q, axis=0)

        # Zentrieren der Punkte 
        P_centered = P - P_middle 
        Q_centered = Q - Q_middle 

        # Rotationswinkelberechnung mit Kapsch Methode
        theta = math.atan2(sum(Q_centered[:,0]*P_centered[:,1] - Q_centered[:,1]*P_centered[:,0]), 
                            sum(Q_centered[:,0]*P_centered[:,0] + Q_centered[:,1]*P_centered[:,1]))

        # Rotationsmatrix und Translation berechnen
        Rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        Translation = P_middle - Rotation_matrix @ Q_middle
        return Rotation_matrix, Translation, theta

    def ransac_refinement(self, P, Q):
        max_iterations = 200
        threshold = 50
        best_rotation, best_translation, best_theta = None, None, 0
        best_inlier_count = 0
            
        if len(P) < 5: #Benötigt mindestens 5 Punkte für eine robuste Schätzung
            return best_rotation, best_translation, best_theta

        for _ in range(max_iterations):
            #Zufällige Auswahl von 3 Punkten für die Schätzung der Transformation
            indices = np.random.choice(len(P), size=3, replace=False)
            # Schätzung der Transformation mit der Kapsch Methode
            R_estimated, t_estimated, theta_estimated = self.get_kapsch_2d(P[indices], Q[indices])
            # Berechnung der Fehler für alle Punkte basierend auf der geschätzten Transformation
            Q_transformed = (R_estimated @ Q.T).T + t_estimated
            errors = np.linalg.norm(P - Q_transformed, axis=1)
            inlier_count = np.sum(errors < threshold)

            if inlier_count > best_inlier_count:
                P_second = P[errors < threshold]
                Q_second = Q[errors < threshold]
                best_rotation, best_translation, best_theta = self.get_kapsch_2d(P_second, Q_second)
                best_inlier_count = inlier_count
                    
        return best_rotation, best_translation, best_theta

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
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)

    def publish_odometry_msg(self, x, y, theta):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        
        r = Rotation.from_euler('z', float(theta))
        quat = r.as_quat(canonical=True)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = quat

        msg.pose.covariance = self.dummy_cov
        msg.twist.covariance = self.dummy_cov
        self.odom_publisher.publish(msg)

    def extract_features(self, frame):
        kp = self.orb.detect(frame, None)
        kp_clean = []
        if self.depth_frame is not None:
            for point in kp:
                x, y = int(point.pt[0]), int(point.pt[1])
                depth = self.depth_frame[y, x]
                if 400 < depth < 4500:
                    kp_clean.append(point)
            return self.orb.compute(frame, kp_clean)
        return None, None

    def compute_local_3d_points(self, kp_clean, des_clean):
        #Umrechnung der 2D-Feature-Punkte in 3D-Punkte in das lokale Kinect System
        local_robot_pts_3d = []
        for point, des in zip(kp_clean, des_clean):
            depth = float(self.depth_frame[int(point.pt[1]), int(point.pt[0])])
            x_c = (point.pt[0] - self.cu) * depth / self.f
            y_c = (point.pt[1] - self.cv) * depth / self.f
            z_c = depth
            self.current_points_3d.append((x_c, y_c, z_c))
            self.current_descriptors.append(des)
            #Drehe auf das Kinect Koordinatensystem
            local_robot_pts_3d.append([z_c, -x_c, -y_c])
        return local_robot_pts_3d

    def get_visible_landmarks(self):
        visible_des, visible_pts_glob_2d, visible_map_indices = [], [], []
        for idx, lm in enumerate(self.map_landmarks):
            dx = lm['pt_glob'][0] - self.curr_pos_x
            dy = lm['pt_glob'][1] - self.curr_pos_y
            lx = dx * math.cos(-self.curr_theta) - dy * math.sin(-self.curr_theta)
            ly = dx * math.sin(-self.curr_theta) + dy * math.cos(-self.curr_theta)
            lz = lm['pt_glob'][2]
            
            if lx > 0:
                #Umgekehrter Strahlensatz, um zu pruefen, ob die Landmarke im Sichtfeld der Kamera liegt
                u_p = (-ly * self.f) / lx + self.cu
                v_p = (-lz * self.f) / lx + self.cv
                if 0 <= u_p <= 640 and 0 <= v_p <= 480:
                    #Hinzufuegen der sichtbaren Landmarken
                    visible_des.append(lm['des'])
                    visible_pts_glob_2d.append(lm['pt_glob'][:2])
                    visible_map_indices.append(idx)
        return visible_des, visible_pts_glob_2d, visible_map_indices

    def update_map(self, local_robot_pts_3d, des_clean, matched_curr_indices):
        # Add new landmarks
        for i in range(len(local_robot_pts_3d)):
            if i not in matched_curr_indices:
                pt = local_robot_pts_3d[i]
                gx = self.curr_pos_x + pt[0]*math.cos(self.curr_theta) - pt[1]*math.sin(self.curr_theta)
                gy = self.curr_pos_y + pt[0]*math.sin(self.curr_theta) + pt[1]*math.cos(self.curr_theta)
                self.map_landmarks.append({
                    'pt_glob': [gx, gy, pt[2]], 
                    'des': des_clean[i], 
                    'seen_count': 1, 
                    'last_seen': self.frame_index
                })
        # Quality-based removal
        self.map_landmarks = [lm for lm in self.map_landmarks if lm['seen_count'] > 2 or (self.frame_index - lm['last_seen']) < 15]

    def listener_callback_rgb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        kp_clean, des_clean = self.extract_features(frame)
        
        #Berechnung nur möglich, wenn kp und des nicht leer
        if kp_clean is None or des_clean is None:
            return

        #Umrechnung in das lokale Kinect Koordinatensystem
        local_robot_pts_3d = self.compute_local_3d_points(kp_clean, des_clean)

        #Initialisierung der Karte mit den ersten Landmarken, wenn die Karte noch leer ist
        if not self.map_landmarks:
            for i in range(len(local_robot_pts_3d)):
                self.map_landmarks.append({'pt_glob': local_robot_pts_3d[i], 
                    'des': des_clean[i], 
                    'seen_count': 1, 
                    'last_seen': self.frame_index
                })

        #Filterung der sichtbaren Landmarken basierend auf dem sichtbaren Bereich der Kamera
        visible_des, visible_pts_glob_2d, visible_map_indices = self.get_visible_landmarks()

        #Matchen über die Sichtbaren Landmarken und die aktuellen Features, Berechnung der Pose mit RANSAC und Aktualisierung der Karte
        if self.frame_counter == 0:
            if len(visible_des) > 0:
                matches = self.bf.match(np.array(visible_des), des_clean) #Matcht die sichtbaren Landmarken mit den aktuellen Features
                if len(matches) > self.min_matches:
                    P_glob, Q_curr = [], []
                    matched_curr_indices = set()
                    for match in matches:
                        map_idx = visible_map_indices[match.queryIdx] # Index der Landmarke in der Karte, die mit dem aktuellen Feature gematcht wurde
                        P_glob.append(visible_pts_glob_2d[match.queryIdx])
                        Q_curr.append(local_robot_pts_3d[match.trainIdx][:2]) #Nur die x und y Koordinaten für die Pose Berechnung verwenden
                        matched_curr_indices.add(match.trainIdx)
                        self.map_landmarks[map_idx]['seen_count'] += 1
                        self.map_landmarks[map_idx]['last_seen'] = self.frame_index

                    R, t, theta = self.ransac_refinement(np.array(P_glob), np.array(Q_curr)) #RANSAC, Kapsch Transformation berechnen, um die Pose zu bestimmen und Ausreißer zu entfernen
                    if R is not None:

                        #Kalman Filter

                        self.curr_pos_x, self.curr_pos_y, self.curr_theta = t[0], t[1], theta
                        self.publish_tf(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta) #Publish TF
                        self.publish_odometry_msg(self.curr_pos_x / 1000.0, self.curr_pos_y / 1000.0, self.curr_theta) #Publish Odometry
                        self.update_map(local_robot_pts_3d, des_clean, matched_curr_indices) #Publish Map Update


                else:
                    print(f"Not enough matches: {len(matches)}")
            self.frame_counter = self.to_proceed_frames

        # Visualization and Cleanup
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)

        #PointCloud Publishing Landmarks
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.odom_frame
        map_points = [[lm['pt_glob'][0]/1000.0, lm['pt_glob'][1]/1000.0, lm['pt_glob'][2]/1000.0] for lm in self.map_landmarks]
        if map_points:
            self.pcl_publisher.publish(pcl2.create_cloud_xyz32(header, map_points))

        self.current_points_3d, self.current_descriptors = [], []
        self.frame_index += 1
        self.frame_counter -= 1

    def listener_callback_depth(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

def main():
    rclpy.init()
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()