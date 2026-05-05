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

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
        self.pcl_publisher = self.create_publisher(PointCloud2, '/serf01/nav_rgbd_1/pointcloud', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_publisher = self.create_publisher(Odometry, '/serf01/odometry/project_slam', 10)
        
        # Neue Landmarken-Struktur
        self.map_landmarks = [] # [{'pos': [x,y,z], 'des': descriptor, 'hits': int}]
        
        self.odom_frame = 'odom'
        self.base_frame = 'base_link'
        self.dummy_cov = [0.1] * 36
        self.depth_frame = None
        self.orb = cv2.ORB_create()
        self.cu, self.cv, self.f = 318.525, 241.181, 526.61
        self.frame_counter = 10
        self.min_matches = 30
        self.curr_pos_x, self.curr_pos_y, self.curr_theta = 0.0, 0.0, 0.0
        self.frame_index = 0
        self.to_proceed_frames = 1
        self.des_queue, self.points_queue = None, None

    def get_kapsch_2d(self, P, Q):    
        P_middle, Q_middle = np.mean(P, axis=0), np.mean(Q, axis=0)
        P_c, Q_c = P - P_middle, Q - Q_middle
        theta = math.atan2(sum(Q_c[:,0]*P_c[:,1] - Q_c[:,1]*P_c[:,0]), sum(Q_c[:,0]*P_c[:,0] + Q_c[:,1]*P_c[:,1]))
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        return R, P_middle - R @ Q_middle, theta

    def ransac_refinement(self, P, Q):
        best_R, best_t, best_theta, best_inliers = None, None, 0, 0
        if len(P) < 5: return best_R, best_t, best_theta
        for _ in range(200):
            idx = np.random.choice(len(P), size=3, replace=False)
            R_est, t_est, th_est = self.get_kapsch_2d(P[idx], Q[idx])
            err = np.linalg.norm(P - ((R_est @ Q.T).T + t_est), axis=1)
            inliers = np.sum(err < 50)
            if inliers > best_inliers:
                best_inliers = inliers
                mask = err < 50
                best_R, best_t, best_theta = self.get_kapsch_2d(P[mask], Q[mask])
        return best_R, best_t, best_theta

    def publish_tf(self, x, y, theta):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = self.get_clock().now().to_msg(), self.odom_frame, self.base_frame
        t.transform.translation.x, t.transform.translation.y = x, y
        quat = Rotation.from_euler('z', float(theta)).as_quat(canonical=True)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)

    def publish_odometry_msg(self, x, y, theta):
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = self.get_clock().now().to_msg(), self.odom_frame, self.base_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        quat = Rotation.from_euler('z', float(theta)).as_quat(canonical=True)
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = quat
        msg.pose.covariance = self.dummy_cov
        self.odom_publisher.publish(msg)    

    def listener_callback_rgb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.depth_frame is None: return
        
        kp = self.orb.detect(frame, None)
        kp_clean = [p for p in kp if 400 < self.depth_frame[int(p.pt[1]), int(p.pt[0])] < 4500]
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        if des_clean is None: return

        # Viewing Cone Filterung & Matching (image_99bb7d.png Anforderung)
        in_view_idx = []
        in_view_des = []
        fov = 2 * math.atan(640 / (2 * self.f))
        for i, lm in enumerate(self.map_landmarks):
            dx, dy = lm['pos'][0] - self.curr_pos_x, lm['pos'][1] - self.curr_pos_y
            if math.sqrt(dx**2+dy**2) < 4.5 and abs((math.atan2(dy, dx)-self.curr_theta+math.pi)%(2*math.pi)-math.pi) < fov/2:
                in_view_idx.append(i)
                in_view_des.append(lm['des'])

        current_pts_3d = []
        for p in kp_clean:
            d = self.depth_frame[int(p.pt[1]), int(p.pt[0])]
            current_pts_3d.append([(p.pt[0]-self.cu)*d/self.f, (p.pt[1]-self.cv)*d/self.f, d])
        
        points_np = np.array(current_pts_3d)

        if self.frame_counter == 0:
            if len(in_view_des) > 0:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING, True).match(np.array(in_view_des), des_clean)
                for m in matches: self.map_landmarks[in_view_idx[m.queryIdx]]['hits'] += 1
                
            if self.des_queue is not None:
                matches = cv2.BFMatcher(cv2.NORM_HAMMING, True).match(self.des_queue, des_clean)
                if len(matches) > self.min_matches:
                    P = np.delete([self.points_queue[m.queryIdx] for m in matches], 1, axis=1)
                    Q = np.delete([points_np[m.trainIdx] for m in matches], 1, axis=1)
                    R, t, theta = self.ransac_refinement(P, Q)
                    if t is not None:
                        t_rot = np.array([[0, 1], [-1, 0]]) @ (np.array(t)/1000)
                        self.curr_pos_x += t_rot[0]*math.cos(self.curr_theta) - t_rot[1]*math.sin(self.curr_theta)
                        self.curr_pos_y += t_rot[0]*math.sin(self.curr_theta) + t_rot[1]*math.cos(self.curr_theta)
                        self.curr_theta += theta
                        self.publish_tf(self.curr_pos_x, self.curr_pos_y, self.curr_theta)
                        self.publish_odometry_msg(self.curr_pos_x, self.curr_pos_y, self.curr_theta)

            # Neue Landmarken hinzufügen & Map bereinigen (Punkt 1 & 2)
            for pt, des in zip(points_np, des_clean):
                wx = self.curr_pos_x + (pt[0]/1000)*math.cos(self.curr_theta) - (pt[1]/1000)*math.sin(self.curr_theta)
                wy = self.curr_pos_y + (pt[0]/1000)*math.sin(self.curr_theta) + (pt[1]/1000)*math.cos(self.curr_theta)
                self.map_landmarks.append({'pos': np.array([wx, wy, pt[2]/1000]), 'des': des, 'hits': 1})
            
            if len(self.map_landmarks) > 1000: self.map_landmarks = [lm for lm in self.map_landmarks if lm['hits'] >= 2]
            
            self.points_queue, self.des_queue = points_np, des_clean
            self.frame_counter = self.to_proceed_frames    
        
        self.frame_counter -= 1
        cv2.imshow("Feature", cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0)))
        cv2.waitKey(1)

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