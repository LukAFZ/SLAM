import cv2
import numpy as np
from .algorithms import algorithms
from .config import configurations
from .map_manager import MapManager
from .robot import *
from .data_types import RobotOdom2D

@dataclass(slots=True)
class Robots:
    id: int
    pose: RobotOdom2D
    robot: Robot
    likelihood: float = 0.0
    log_weight: float = 0.0



class VisualSLAMCore:
    def __init__(self):
        self.config = configurations()
        self.algo = algorithms()
        self.map_manager = MapManager(self.config)
        
        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.config.ORB_NFEATURES, 
                                  patchSize=self.config.ORB_PATCH_SIZE,
                                  edgeThreshold=self.config.ORB_EDGE_THRESHOLD,
                                  fastThreshold=self.config.ORB_FAST_THRESHOLD,
                                  WTA_K=self.config.ORB_WTA_K,
                                  nlevels=self.config.ORB_NLEVELS,
                                  scaleFactor=self.config.ORB_SCALE_FACTOR)
        # Initiate Brute-Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Roboter-Pose & Index-Tracking
        self.best_pose = RobotOdom2D(x=0.0, y=0.0, theta=0.0)

        # Initialize a list to hold multiple robot instances
        self.robots = []
        self.num_robots = self.config.NUM_ROBOTS
        if self.num_robots <= 0:
            self.num_robots = 1 # mindestens ein Roboter, um die Karte zu erstellen
        for N in range(self.num_robots):
            self.robots.append(Robots(
                id=N,
                pose=RobotOdom2D(x=0.0, y=0.0, theta=0.0),
                robot=Robot(),
                likelihood=0.0
            ))

        # Initialize previous descriptors and 3D points for frame-to-frame matching
        self.previous_des = None
        self.previous_local_pts_3d = None


    def process_frame(self, frame, depth_frame, kinect_to_base_matrix, base_to_kinect_matrix, frame_index):
        """
        calculate the pose of the robot and update the map based on the current grayscale and depth image, as well as the current pose estimate and map status.
        Returns pose_updated, x, y, theta
        """
        # Initialize variables
        pose_updated = False
        best_map_manager = None
        kp_clean = []

        # find the keypoints with ORB
        kp = self.orb.detect(frame, None)
        
      

        occupied_cells = set()
        if depth_frame is not None:
            
            #Sort Keypoints by their response (strength)
            kp = sorted(kp, key=lambda x: x.response, reverse=True)
            
            # cycle through keypoints
            for point in kp:
                # get x,y coordinates of keypoint in pixel space
                x, y = int(point.pt[0]), int(point.pt[1])
                
                # 1. Calculate depth value
                depth = depth_frame[y, x]
                if self.config.MIN_DEPTH < depth < self.config.MAX_DEPTH:
                    
                    # Calculate grid cell ID
                    cell_x = x // self.config.GRID_SIZE
                    cell_y = y // self.config.GRID_SIZE
                    cell_id = (cell_x, cell_y)
                    
                    # If Region is not occupied, add keypoint and mark region as occupied
                    if cell_id not in occupied_cells:
                        kp_clean.append(point)
                        occupied_cells.add(cell_id) # Block Region for other keypoints
                        
        else:
            return False, self.best_pose, self.map_manager
        # compute the descriptors with ORB only for the filtered keypoints
        kp_clean, des_clean = self.orb.compute(frame, kp_clean)
        
        if kp_clean is None or des_clean is None:
            return False, self.best_pose, self.map_manager# skip processing if no valid keypoints/descriptors are found

        # 3D coordinates calculation of the keypoints to base coordinates
        local_robot_pts_3d = self.algo.calculate_local_cords_from_matches(
            kp_clean, des_clean, kinect_to_base_matrix, depth_frame
        )

        delta_R, delta_t, delta_theta = None, None, None
        # Matching from Frame to Frame
        if self.previous_des is not None and len(self.previous_des) > 0 and len(des_clean) > 0:
            f2f_matches = self.bf.match(self.previous_des, des_clean)
            if len(f2f_matches) > self.config.MIN_MATCHES:
                P_prev = np.array([self.previous_local_pts_3d[m.queryIdx][:2] for m in f2f_matches])
                Q_curr = np.array([local_robot_pts_3d[m.trainIdx][:2] for m in f2f_matches])
                delta_R, delta_t, delta_theta = self.algo.ransac_refinement(P_prev, Q_curr)

            #best robot selection
            for selected_robot in self.robots:
                
                # Update the robot's pose and map based on the current frame and the previous frame's data
                pose_updated, selected_robot.pose, log_r_l = selected_robot.robot.update_robot(kp_clean, des_clean, depth_frame, frame_index, base_to_kinect_matrix, kinect_to_base_matrix, local_robot_pts_3d, delta_R, delta_t, delta_theta)

                # save the log likelihood for the selected robot
                selected_robot.log_weight = log_r_l/len(des_clean)


            # Find the robot with the maximum log likelihood for numerical stability
            max_log_l = max(robot.log_weight for robot in self.robots) # find maximum log likelihood among all robots for numerical stability (underflow prevention)

            # Subtract the maximum log likelihood from each robot's log likelihood to prevent numerical underflow when exponentiating.
            # This shifts the best particle to log(w) = 0 -> w = e^0 = 1.0.
            for idx, robot in enumerate(self.robots):
                robot.likelihood = np.exp(robot.log_weight - max_log_l)

            # Normalize the likelihoods to ensure they sum to 1.0
            total_weight = sum(robot.likelihood for robot in self.robots)
            if total_weight > 0:
                for robot in self.robots:
                    robot.likelihood /= total_weight
                    print(f"Robot ID: {robot.id}, Log_Likelihood-Maximum des Roboters: {robot.log_weight}, Partikel Weight: {robot.likelihood}")
            else:
                # If all likelihoods are zero (which shouldn't happen), reset to uniform distribution
                for robot in self.robots:
                    robot.likelihood = 1.0 / self.num_robots

            
            # Select the robot with the maximum likelihood as the best estimate
            max_likelihood_robot = max(self.robots, key=lambda r: r.likelihood)



            self.best_pose = max_likelihood_robot.pose
            best_map_manager = max_likelihood_robot.robot.map_manager
            pose_updated = True

        # save this frame
        self.previous_des = des_clean
        self.previous_local_pts_3d = local_robot_pts_3d

        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)
            
        return pose_updated, self.best_pose, best_map_manager