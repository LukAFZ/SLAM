import cv2
import math
import numpy as np
from .algorithms import algorithms
from .config import configurations
from .map_manager import MapManager
from .robot import *
from .data_types import RobotOdom2D

@dataclass
class Robots:
    id: int
    pose: RobotOdom2D
    robot: Robot
    likelihood: float = 0.0

class VisualSLAMCore:
    def __init__(self):
        self.config = configurations()
        self.algo = algorithms()
        self.map_manager = MapManager(self.config)
        
        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=1500, patchSize=31)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Roboter-Pose & Index-Tracking
        self.best_pose = RobotOdom2D(x=0.0, y=0.0, theta=0.0)

        # Anzahl der Frames, die nach einem Update übersprungen werden, um die Stabilität zu erhöhen (z.B. bei RANSAC-Updates)
        #self.frame_counter = self.slam.config.frame_counter
        #self.frame_index = 0

        self.robots = []
        self.num_robots = self.config.num_robots
        if self.num_robots <= 0:
            self.num_robots = 1 # mindestens ein Roboter, um die Karte zu erstellen
        for N in range(self.num_robots):
            self.robots.append(Robots(
                id=N,
                pose=RobotOdom2D(x=0.0, y=0.0, theta=0.0),
                robot=Robot(),
                likelihood=0.0
            ))

    def process_frame(self, frame, depth_frame, kinect_to_base_matrix, base_to_kinect_matrix, frame_index):
        """
        Berechnung der Pose des Roboters und Aktualisierung der Karte basierend auf dem aktuellen RGB- und Tiefenbild, sowie der aktuellen Pose-Schätzung und dem Kartenstatus.
        Rückgabe von pose_updated, x, y, theta
        """
        
        # find the keypoints with ORB
        kp = self.orb.detect(frame, None)
        
        kp_clean = []
        # if depth frame is available, filter keypoints based on depth values to remove outliers and points that are too close or too far
        if depth_frame is not None:
            # cycle through keypoints
            for point in kp:
                # get x,y coordinates of keypoint in pixel space
                x, y = int(point.pt[0]), int(point.pt[1])
                # get depth value at keypoint location and convert to meters
                depth = depth_frame[y, x]
                if self.config.min_depth < depth < self.config.max_depth: # filter out invalid depth values
                    kp_clean.append(point)
        else:
            return False, self.best_pose, self.map_manager # skip processing if depth frame is not available
        
        # compute the descriptors with ORB only for the filtered keypoints
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        
        if kp_clean is None or des_clean is None:
            return False, self.best_pose, self.map_manager# skip processing if no valid keypoints/descriptors are found

        # 3D coordinates calculation from perspective of the robot based on the depth values and the camera intrinsics
        local_robot_pts_3d = self.algo.calculate_local_cords_from_matches(
            kp_clean, des_clean, kinect_to_base_matrix, depth_frame
        )

        log_robot_likelihood = []
        for selected_robot in self.robots:

            #best robot selection to be implemented here
            pose_updated, selected_robot.pose, log_r_l = selected_robot.robot.update_robot(kp_clean, des_clean, depth_frame, frame_index, base_to_kinect_matrix, local_robot_pts_3d)
            #if len(des_clean) > 0:
            #    log_r_l = log_r_l / len(des_clean) # normalize log likelihood by number of descriptors to avoid bias towards frames with more features
            #else:
            #    log_r_l = -700
            log_robot_likelihood.append(log_r_l)


        max_log_l = max(log_robot_likelihood)


        #BERECHNUNG STIMMT NOCH NICHT
        # 2. Ziehe das Maximum ab, bevor du die Exponentialfunktion anwendest.
        # Das verschiebt den besten Partikel auf log(w) = 0 -> w = e^0 = 1.0.
        for idx, robot in enumerate(self.robots):
            print(f"Robot ID: {robot.id}, Log_Likelihood-Maximum: {log_robot_likelihood[idx] - max_log_l}")
            robot.likelihood = np.exp(log_robot_likelihood[idx] - max_log_l)

        # 3. Normarlisieren, damit die Summe aller Gewichte 1 ergibt
        total_weight = sum(robot.likelihood for robot in self.robots)
        if total_weight > 0:
            for robot in self.robots:
                robot.likelihood /= total_weight
        else:
            # Falls alle Likelihoods 0 sind, setze sie gleichmäßig auf 1/N
            for robot in self.robots:
                robot.likelihood = 1.0 / self.num_robots

        max_likelihood_robot = max(self.robots, key=lambda r: r.likelihood)
        print(f"Best robot ID: {max_likelihood_robot.id}, Max_Likelihood: {max_likelihood_robot.likelihood}")

        self.best_pose = max_likelihood_robot.pose
        best_map_manager = max_likelihood_robot.robot.map_manager
        pose_updated = True

        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)
            
        return pose_updated, self.best_pose, best_map_manager