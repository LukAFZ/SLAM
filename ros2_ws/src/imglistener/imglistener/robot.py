import numpy as np
import math

from cv_bridge import CvBridge
import cv2

from .extKalman_LM import *
from .algorithms import *
from .config import *
from .map_manager import *
from .data_types import RobotOdom2D

class Robot():
    def __init__(self):
        
        self.config = configurations()

        self.map_manager = MapManager(self.config)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Initialize Local Objects from Config (faster)
        self.min_matches = self.config.MIN_MATCHES

        self.ransac_iterations = self.config.RANSAC_ITERATIONS
        self.ransac_threshold = self.config.RANSAC_THRESHOLD
        self.ransac_max_deviation_delta = self.config.RANSAC_MAX_DEVIATION_DELTA
        self.ransac_max_deviation_theta = self.config.RANSAC_MAX_DEVIATION_THETA

        self.sigma_x = self.config.SIGMA_X
        self.sigma_y = self.config.SIGMA_Y
        self.sigma_theta = self.config.SIGMA_THETA

        self.seen_count_threshold = self.config.SEEN_COUNT_THRESHOLD
        self.last_seen_threshold = self.config.LAST_SEEN_THRESHOLD

        self.fatal_error = self.config.PARTICLE_FILTER_FAIL_STANDARD_ERROR

        # Store 3D points
        self.point_history = []

        self.pose = RobotOdom2D(x=0.0, y=0.0, theta=0.0)

        self.algo = algorithms()
        self.map_initialized = False


    def update_robot(self, kp_clean, des_clean, depth_frame, frame_index, base_to_kinect_matrix, kinect_to_base_matrix, local_robot_pts_3d, delta_R, delta_t, delta_theta):
        """
        Update of a robot based on the current frame and the map using RANSAC, Kabsch algorithm, and EKF updates for landmarks.
        """
        # Initial map creation
        #Calculate in Local Robot Coordinates
        if self.map_manager.is_empty() and not self.map_initialized:
            self.map_manager.initialize_map(local_robot_pts_3d, des_clean, frame_index, kinect_to_base_matrix)
            self.map_initialized = True
 

        # Viewing Cone - filter landmarks that are in the field of view of the robot
        visible_des, visible_pts_glob_2d, visible_map_indices = self.algo.test_only_for_visible_landmarks(self.map_manager.landmarks, self.pose, base_to_kinect_matrix)
        
        log_robot_likelihood = 0.0
        pose_updated = False

        if len(visible_des) > 0:
            # Match visible landmarks with current frame keypoints
            matches = self.bf.match(np.array(visible_des), des_clean)
            
            if len(matches) > self.min_matches:
                # Prepare matched points for RANSAC
                P_local = []
                Q_curr  = []
                matched_curr_indices = set()
  

                for match in matches:
                    map_idx = visible_map_indices[match.queryIdx] # Landmark index in the global map
                    pt_glob = visible_pts_glob_2d[match.queryIdx] # Landmark position in global coordinates (odom)

                    # transform global landmark position to local robot coordinates for the matched landmark
                    lx, ly = self.algo.transform_delta_odom_to_local_robot_coords(Coordinate(pt_glob[0], pt_glob[1], z=0), self.pose)

                    P_local.append([lx, ly])
                    Q_curr.append(local_robot_pts_3d[match.trainIdx][:2])
                    matched_curr_indices.add(match.trainIdx)

                    lm = self.map_manager.landmarks[map_idx]


                    lm.seen_count += 1
                    lm.last_seen = frame_index



                #Calculate in ODOM
                if delta_R is not None:
                    # relative transformation from local robot coordinates to odom frame
                    delta_tx_odom, delta_ty_odom = self.algo.matrix_from_local_robot_to_odom_coords(Coordinate(delta_t[0], delta_t[1], z=0), self.pose)

                    # Add Gaussian noise to the estimated transformation based on the configured standard deviations
                    sigma_x = self.sigma_x
                    sigma_y = self.sigma_y
                    sigma_theta = self.sigma_theta

                    epsilon_x = np.random.normal(0, sigma_x)
                    epsilon_y = np.random.normal(0, sigma_y)
                    epsilon_theta = np.random.normal(0, sigma_theta)

                    # update current pose with the estimated transformation + noise
                    self.pose.x += delta_tx_odom + epsilon_x
                    self.pose.y += delta_ty_odom + epsilon_y
                    self.pose.theta += delta_theta + epsilon_theta

                    normalized_theta = self.algo.normalize_angle(self.pose.theta)
                    self.pose.theta = normalized_theta
                    pose_updated = True
                    
                    P_array = np.array(P_local)
                    Q_array = np.array(Q_curr)
                    
                    # Check if the estimated transformation is reasonable to avoid outliers
                    Q_transformed = (delta_R @ Q_array.T).T + delta_t
                    errors = np.linalg.norm(P_array - Q_transformed, axis=1)


                    for i, match in enumerate(matches):
                        if errors[i] < self.ransac_threshold: # Only allow true insliers to update the map

                            map_idx = visible_map_indices[match.queryIdx]
                            train_idx = match.trainIdx

                            kalman_result = []
                            lm = self.map_manager.landmarks[map_idx]
                            depth = float(depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                            #Update Kalman Filter für Landmarke mit aktuellem Messwert
                            kalman_result, P, log_likelihood = lm.ekf.update(
                                np.array(local_robot_pts_3d[train_idx]), 
                                self.pose, 
                                np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]), # x and y pixel coordinates of the matched keypoint
                                depth
                            )
                            
                            # Accumulate the log likelihood for the robot's pose based on the matched landmarks
                            log_robot_likelihood += log_likelihood
                            lm.pt_glob = Coordinate(x=kalman_result[0], y=kalman_result[1], z=kalman_result[2])
                        else:
                            #Match is rejected as outlier by RANSAC, penalize likelihood
                            log_robot_likelihood += self.fatal_error # penalize outliers in the likelihood calculation
                    # add new landmarks to the map based on the current frame and the updated pose estimation
                    self.map_manager.add_new_landmarks(
                        local_robot_pts_3d, des_clean, matched_curr_indices, 
                        self.pose, frame_index, kinect_to_base_matrix
                    )

                    # remove old landmarks that are not seen anymore
                    self.map_manager.clean_map(frame_index)

                    #not valid if delta is too high, likely an outlier
                    if (np.linalg.norm(delta_t) > self.ransac_max_deviation_delta or
                            abs(delta_theta) > self.ransac_max_deviation_theta):
                        log_robot_likelihood = self.fatal_error

                else:
                    print("RANSAC failed to find a valid transformation.")
                    log_robot_likelihood = self.fatal_error # very low likelihood if RANSAC fails to discourage this pose update
            else:
                print(f"Not enough matches: {len(matches)}")
                log_robot_likelihood = self.fatal_error # very low likelihood if not enough matches are found to discourage this pose update
        else:
            print("No visible landmarks to match with.")
            log_robot_likelihood = self.fatal_error # very low likelihood if no visible landmarks are found to discourage this pose update
        return pose_updated, self.pose, log_robot_likelihood