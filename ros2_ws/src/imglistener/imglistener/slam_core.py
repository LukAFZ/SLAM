import cv2
import math
import numpy as np
from .algorithms import algorithms
from .config import configurations
from .map_manager import MapManager
from .robot import *

class VisualSLAMCore:
    def __init__(self):
        self.config = configurations()
        self.algo = algorithms()
        self.map_manager = MapManager(self.config)
        
        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=1500, patchSize=31)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Roboter-Pose & Index-Tracking
        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0
        self.frame_index = 0

        self.robots = []
        self.num_robots = self.config.num_robots
        for N in range(self.num_robots):
            self.robots.append({
                'id': N,
                'pose': np.array([0.0, 0.0, 0.0]), # x, y, theta
                'robot': Robot(),
                'likelihood': 0.0
            }) 

    def process_frame(self, frame, depth_frame, kinect_to_base_matrix, base_to_kinect_matrix, frame_counter):
        """
        compute the robot pose and update the map based on the current RGB and Depth frame, as well as the current pose estimation and the map state.
        return pose_updated, x, y, theta
        """
        # find the keypoints with ORB
        kp = self.orb.detect(frame, None)
        
        kp_clean = []
        # if depth frame is available, overlay depth info on keypoints
        if depth_frame is not None:
            # cycle through keypoints
            for point in kp:
                # get x,y coordinates of keypoint
                x, y = int(point.pt[0]), int(point.pt[1])
                # get depth value at keypoint location and convert to meters
                depth = depth_frame[y, x]
                if self.config.min_depth < depth < self.config.max_depth: # filter out invalid depth values
                    kp_clean.append(point)
        else:
            return False, self.curr_pos_x, self.curr_pos_y, self.curr_theta # skip processing if depth frame is not available
        
        # compute the descriptors with ORB
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        
        if kp_clean is None or des_clean is None:
            return False, self.curr_pos_x, self.curr_pos_y, self.curr_theta # skip processing if no valid keypoints/descriptors are found

        # 3D coordinates calculation
        local_robot_pts_3d = self.algo.calculate_local_cords_from_matches(
            kp_clean, des_clean, kinect_to_base_matrix, depth_frame
        )

        log_robot_likelihood = []
        for robot in self.robots:

            #best robot selection to be implemented here
            pose_updated, robot['pose'][0], robot['pose'][1], robot['pose'][2], log_r_l= robot['robot'].update_robot(kp_clean, des_clean, depth_frame, self.frame_index, frame_counter, kinect_to_base_matrix, base_to_kinect_matrix)
            log_robot_likelihood.append(log_r_l)

        max_log_l = max(log_robot_likelihood)

        # 2. Ziehe das Maximum ab, bevor du die Exponentialfunktion anwendest.
        # Das verschiebt den besten Partikel auf log(w) = 0 -> w = e^0 = 1.0.
        # Alle anderen Partikel skalieren sich relativ dazu, was Underflows unmöglich macht!
        for idx, robot in enumerate(self.robots):
            robot['likelihood'] = np.exp(log_robot_likelihood[idx] - max_log_l)

        # 3. Jetzt wie gewohnt normalisieren, damit die Summe aller Gewichte 1 ergibt
        total_weight = sum(robot['likelihood'] for robot in self.robots)
        if total_weight > 0:
            for robot in self.robots:
                robot['likelihood'] /= total_weight

        max_likelihood_robot = max(self.robots, key=lambda r: r['likelihood'])
        self.curr_pos_x, self.curr_pos_y, self.curr_theta = max_likelihood_robot['pose']
        best_map_manager = max_likelihood_robot['robot'].map_manager


        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)

        self.frame_index += 1
        return pose_updated, self.curr_pos_x, self.curr_pos_y, self.curr_theta, best_map_manager