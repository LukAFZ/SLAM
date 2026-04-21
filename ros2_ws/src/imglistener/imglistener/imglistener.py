#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
import time

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
        
        
        self.des_queue = None
        self.depth_frame = None
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61

        # Store 3D points
        self.current_points_3d = []
        self.current_descriptors = []
        self.point_history = []

        self.keyframes = []
        self.frame_index = 0
        self.frame_counter = 10
        self.to_proceed_frames = 1
        self.queue_index = 0

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
                if(depth > 400 and depth < 4000): # filter out invalid depth values
                    text = f"{depth:.1f} mm"
                    cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    kp_clean.append(point)
        else:
            return # skip processing if depth frame is not available
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

        # store keyframe data
        self.keyframes.append({
            'points_3d': points_np,
            'des': des_np
        })

        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth",img2)
        cv2.waitKey(1)
        # publish 3D points as PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'kinect_depth'
        pointcloud_msg = pcl2.create_cloud_xyz32(header, self.current_points_3d) # only publish x,y,z coordinates, ignore descriptors
        self.pcl_publisher.publish(pointcloud_msg)
        # TF tree missing
        # ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link feature_points

        # empty the 3D points list for the next frame
        self.current_points_3d = []
        self.current_descriptors = []
        

        # if(self.des_queue is not None):
        #     # create BFMatcher object
        #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #     # Match descriptors.
        #     matches = bf.match(self.des_queue, des_clean)
        #     # Sort them in the order of their distance.
        #     matches = sorted(matches, key = lambda x:x.distance)
        #     # Draw first 10 matches.
        #     #img3 = cv2.drawMatches(frame, kp_clean, frame, kp_clean, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     #cv2.imshow("Matches",img3)
        #     #cv2.waitKey(1)

        

        if(self.frame_counter == 0):
            if(self.des_queue is not None):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(self.des_queue, des_clean)

                for match in matches:
                    idx1 = match.queryIdx
                    idx2 = match.trainIdx
                    matrix_zero_3d.append(self.keyframes[self.queue_index]['points_3d'][idx1])
                    matrix_second_3d.append(points_np[idx2])
                
                matrix_zero = np.delete(matrix_zero_3d, 1, axis=1) # remove y coordinate
                matrix_second = np.delete(matrix_second_3d, 1, axis=1) # remove y coordinate
                R, t, theta = self.ransac_refinement(matrix_zero, matrix_second)
                print(f"Estimated rotation (theta): {math.degrees(theta):.2f} degrees")
                print(f"Estimated translation: {t}")

                tf = TransformStamped()
                tf.header.stamp = self.get_clock().now().to_msg()
                tf.header.frame_id = 'odom'
                tf.child_frame_id = 'kinect_depth'
                tf.transform.translation.x = t[0]
                tf.transform.translation.y = 0
                tf.transform.translation.z = t[1]
                tf.transform.rotation = euler_to_quaternion(0, theta, 0)

                # kinect_depth [x, 0, z]
                # rotation = euler_to_quaternion(0, theta, 0)



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