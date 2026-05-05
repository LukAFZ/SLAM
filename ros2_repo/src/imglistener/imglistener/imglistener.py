#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
import time
from scipy.spatial.transform import Rotation

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
        # Publisher
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        #self.landmark_publisher = self.create_publisher(MarkerArray, '/serf01/viz/landmarks', 10)
        
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.dummy_cov = [0.1] * 36 # Dummy covariance values for pose and twist

        self.des_queue = None
        self.points_queue = None
        self.depth_frame = None
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61
        self.frame_counter = 10
        self.min_matches = 30

        # Store 3D points
        self.current_points_3d = []
        self.current_descriptors = []

        self.point_list = []
        self.frame_index = 0
        self.to_proceed_frames = 1
        self.queue_index = 0
        
        self.curr_pos_x = 0
        self.curr_pos_y = 0
        self.curr_theta = 0

        self.landmark_id_counter = 0

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
        
        max_iterations= 200
        threshold= 50
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
                    if e < threshold:
                        P_second.append(p)
                        Q_second.append(q)

                best_rotation, best_translation, best_theta = self.get_kapsch_2d(np.array(P_second), np.array(Q_second))

                best_inlier_count = inlier_count
                

        #print("Inliner percentage: {:.2f}%".format(best_inlier_count / len(P) * 100))


        return best_rotation, best_translation, best_theta

# odom berechnen
# odom publisher und in RVIZ2 anschauen
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
        """Erstellt und sendet die Nachricht basierend auf nav_msgs/Odometry.msg"""
        msg = Odometry()
        
        # 1. Header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        
        # 2. Child Frame ID
        msg.child_frame_id = self.base_frame

        # 3. Pose (Position & Orientation)
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
        
        # Pose Covariance
        msg.pose.covariance = self.dummy_cov

        # 4. Twist (Geschwindigkeit - hier Dummy 0.0)
        msg.twist.twist.linear.x = 0.0
        msg.twist.twist.angular.z = 0.0
        msg.twist.covariance = self.dummy_cov

        # Veröffentlichen
        self.odom_publisher.publish(msg)    


    def listener_callback_rgb(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        
        # find the keypoints with ORB
        kp = self.orb.detect(frame,None)
        
        matrix_zero_3d = []
        matrix_second_3d = []
        
        
        kp_clean = []
        # if depth frame is available, overlay depth info on keypoints
        if(self.depth_frame is not None):
            # cycle through keypoints
            for point in kp:
                # get x,y coordinates of keypoint
                x, y = int(point.pt[0]), int(point.pt[1])
                # get depth value at keypoint location and convert to meters
                depth = self.depth_frame[y, x]
                if(400 < depth < 4500): # filter out invalid depth values
                    text = f"{depth:.1f} mm"
                    cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    kp_clean.append(point)
        else:
            return # skip processing if depth frame is not available
        
        #Erstelle Liste mit im Sichtkegel befindlichen Landmarks

        # compute the descriptors with ORB
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        # for i in range(len(kp_clean)):
        #   point = kp_clean[i]
        #   des = des_clean[i]

        # for point, des in zip(kp_clean, des_clean):
        #   ...
        if(kp_clean is None or des_clean is None):
            return # skip processing if no valid keypoints/descriptors are found

        for point, des in zip(kp_clean, des_clean):
            depth = self.depth_frame[int(point.pt[1]), int(point.pt[0])]
            u = self.cu - point.pt[0]
            v = point.pt[1]-self.cv


            # 3D coordinates calculation
            x = (point.pt[0] - self.cu) * depth / self.f
            y = (point.pt[1] - self.cv) * depth / self.f
            z = depth

            #control_u = self.f*(x/z)
            #control_v = self.f*(y/z)
            #print(f"Real coordinates: ({(u):.2f}, {(v):.2f}), Control commands: ({control_u:.2f}, {control_v:.2f})")
            self.current_points_3d.append((x, y, z))
            self.current_descriptors.append(des)
        points_np = np.array(self.current_points_3d)
        des_np = np.array(self.current_descriptors)     


        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth",img2)
        cv2.waitKey(1)
        # publish 3D points as PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'kinect_depth'
        points_in_meters = [(p[0]/1000, p[1]/1000, p[2]/1000) for p in self.current_points_3d] # convert from mm to meters
        pointcloud_msg = pcl2.create_cloud_xyz32(header, points_in_meters) # only publish x,y,z coordinates, ignore descriptors
        self.pcl_publisher.publish(pointcloud_msg)
        # TF tree missing
        # ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link feature_points

        # empty the 3D points list for the next frame
        self.current_points_3d = []
        self.current_descriptors = []
    

        if(self.frame_counter == 0):
            if(self.des_queue is not None):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.des_queue, des_clean)
                if(len(matches) > self.min_matches):
                    for match in matches:
                        idx1 = match.queryIdx
                        idx2 = match.trainIdx
                        matrix_zero_3d.append(self.points_queue[idx1])
                        matrix_second_3d.append(points_np[idx2])
                    
                    matrix_zero = np.delete(matrix_zero_3d, 1, axis=1) # remove y coordinate
                    matrix_second = np.delete(matrix_second_3d, 1, axis=1) # remove y coordinate
                    R, t, theta = self.ransac_refinement(matrix_zero, matrix_second)


                    t = np.array(t) / 1000 # convert from mm to meters
                    t_rot = np.array([[0, 1], [-1, 0]]) @ t

                    print(f"Estimated rotation (theta): {math.degrees(theta):.2f} degrees")
                    print(f"Estimated translation: {t_rot[0]:.2f}, {t_rot[1]:.2f}")

                    self.curr_pos_x += t_rot[0]*math.cos(self.curr_theta) - t_rot[1]*math.sin(self.curr_theta)
                    self.curr_pos_y += t_rot[0]*math.sin(self.curr_theta) + t_rot[1]*math.cos(self.curr_theta)
                    self.curr_theta += theta

                    # publish tf
                    self.publish_tf(self.curr_pos_x, self.curr_pos_y, self.curr_theta)
                    # publish odometry message
                    self.publish_odometry_msg(self.curr_pos_x, self.curr_pos_y, self.curr_theta)

                    #Speichere die Landmarken in der Karte
                    for pt_local, des in zip(points_np, des_np):
                        # Umrechnung mm -> m
                        lx, ly, lz = pt_local[0]/1000, pt_local[1]/1000, pt_local[2]/1000
                        
                        # Transformation in Welt-Koordinaten (Rotation + Translation)
                        # Wir nutzen hier die 2D-Rotation für x und y
                        world_x = self.curr_pos_x + lx * math.cos(self.curr_theta) - ly * math.sin(self.curr_theta)
                        world_y = self.curr_pos_y + lx * math.sin(self.curr_theta) + ly * math.cos(self.curr_theta)
                        world_z = lz # Z bleibt hier vereinfacht gleich

                        #Speichere Relative Koordinaten und Welt Koordinaten (Landmarken)
                        self.point_list.append({
                            'des': des_np,
                            'world_coordinates': (world_x, world_y, world_z),
                            'hits': None
                        })
                    
                else:
                    print(f"Not enough matches found for RANSAC {len(matches)}")

            self.points_queue = points_np
            self.des_queue = des_clean
            self.queue_index = self.frame_index
            self.frame_counter = self.to_proceed_frames    

        self.frame_index += 1
        self.frame_counter -= 1
        
    def listener_callback_depth(self, msg):
        self.depth_frame=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        #cv2.imshow("Depth",self.depth_frame)
        #cv2.waitKey(1)
    

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()