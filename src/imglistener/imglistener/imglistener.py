#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2 as cv
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        self.subscription = self.create_subscription(Image, '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)

    def listener_callback_rgb(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(frame,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(frame, kp)
        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        frame.imshow(img2), frame.show()
        cv2.imshow("Image",frame)
        cv2.waitKey(1)
        
    def listener_callback_depth(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        cv2.imshow("Image",frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()