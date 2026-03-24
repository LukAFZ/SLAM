#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
        self.des_queue = None




    def listener_callback_rgb(self,msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'bgr8')
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(frame,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(frame, kp)

        if(self.des_queue is not None):
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(self.des_queue, des)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 10 matches.
            img3 = cv2.drawMatches(frame, kp, frame, kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches",img3)
            cv2.waitKey(1)

        #img2 = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        #cv2.imshow("Image",img2)
        #cv2.waitKey(1)
        self.des_queue = des
        

    def listener_callback_depth(self, msg):
        frame=self.bridge.imgmsg_to_cv2(msg,'passthrough')
        cv2.imshow("Depth",frame)
        cv2.waitKey(1)
    

def main():
    rclpy.init()
    node=ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()