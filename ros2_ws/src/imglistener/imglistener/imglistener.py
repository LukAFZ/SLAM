#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import math
import numpy as np

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


    def listener_callback_rgb(self,msg):
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
                depth = self.depth_frame[y, x]/1000.0
                if(depth > 0.4 and depth < 4): # filter out invalid depth values
                    text = f"{depth:.2f} m"
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
        
        for point, des in zip(kp_clean, des_clean):
            depth = self.depth_frame[int(point.pt[1]), int(point.pt[0])] / 1000.0
            u = self.cu - point.pt[0]
            v = point.pt[1]-self.cv

            #phi1 = math.atan2(u, self.f)
            #phi2 = math.atan2(v, self.f)
            #x = depth * math.tan(phi1)
            #y = depth * math.tan(phi2)
            #z = depth

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
        header.frame_id = 'feature_points'
        pointcloud_msg = pcl2.create_cloud_xyz32(header, self.current_points_3d) # only publish x,y,z coordinates, ignore descriptors
        self.pcl_publisher.publish(pointcloud_msg)
        # TF tree missing
        # ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link feature_points

        # empty the 3D points list for the next frame
        self.current_points_3d = []
        self.current_descriptors = []
        

        if(self.des_queue is not None):
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(self.des_queue, des_clean)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 10 matches.
            #img3 = cv2.drawMatches(frame, kp_clean, frame, kp_clean, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            #cv2.imshow("Matches",img3)
            #cv2.waitKey(1)


        if(self.frame_index == 20):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.keyframes[0]['des'], des_clean)
            matches = sorted(matches, key = lambda x:x.distance)
            print(f"Number of matches: {len(matches)}")
            #Kabsch Algorithmus




        

        self.frame_index += 1

        # store current descriptors for next frame matching
        self.des_queue = des_clean
        

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