#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.last_frame = None
        self.last_kp = None
        self.last_des = None
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        #self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)

    def listener_callback_rgb(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp1 = orb.detect(frame,None)
        # compute the descriptors with ORB
        kp1, des1 = orb.compute(frame, kp1)
        # draw only keypoints location,not size and orientation
        img3 = cv2.drawKeypoints(frame, kp1, None, color=(0,255,0), flags=0)
        if(self.last_des is not None):
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.last_des, des1)
            matches = sorted(matches, key = lambda x:x.distance)
            img3 = cv2.drawMatches(self.last_frame, self.last_kp, frame, kp1, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Image", img3)
        self.last_frame = frame
        self.last_kp = kp1
        self.last_des = des1
        #cv2.imshow("Image",img2)
        cv2.waitKey(1)
        
    #def listener_callback_depth(self,msg):
    #    frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
    #    orb = cv.ORB_create()
    #    cv2.imshow("Image",frame)
    #    cv2.waitKey(1)

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()