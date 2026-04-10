#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/rgb/image_raw', self.listener_callback_rgb, 10)
        self.subscription = self.create_subscription(Image,
            '/serf01/nav_rgbd_1/depth/image_raw', self.listener_callback_depth, 10)
        self.des_queue = None
        self.depth_frame = None
        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61


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

        # compute the descriptors with ORB
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)

        for point in kp_clean:
            depth = self.depth_frame[int(point.pt[1]), int(point.pt[0])]
            u = self.cu - point.pt[0]
            v = point.pt[1]-self.cv

            phi1 = math.atan2(u, self.f)
            phi2 = math.atan2(v, self.f)
            x = depth * math.tan(phi1)
            y = depth * math.tan(phi2)
            z = depth
            
            control_u = self.f*(x/z)
            control_v = self.f*(y/z)
    
            print(f"Real coordinates: ({(u):.2f}, {(v):.2f}), Control commands: ({control_u:.2f}, {control_v:.2f}), Angles: ({math.degrees(phi1):.2f} deg, {math.degrees(phi2):.2f} deg)")
        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth",img2)
        cv2.waitKey(1)

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