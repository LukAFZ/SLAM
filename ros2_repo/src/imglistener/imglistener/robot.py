import numpy as np
import math

from cv_bridge import CvBridge
import cv2

from .extKalman_LM import *
from .algorithms import *
from .config import *
from .map_manager import *
from .constants import State

class Robot():
    def __init__(self):
        
        self.config = configurations()

        self.map_manager = MapManager(self.config)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = self.config.min_matches

        self.ransac_iterations = self.config.ransac_iterations
        self.ransac_threshold = self.config.ransac_threshold

        # Store 3D points
        self.point_history = []

        self.pose = State(x=0.0, y=0.0, theta=0.0)

        self.seen_count_threshold = configurations().seen_count_threshold
        self.last_seen_threshold = configurations().last_seen_threshold

        self.algorithmen = algorithms()

    def update_robot(self, kp_clean, des_clean, depth_frame, frame_index, base_to_kinect_matrix, local_robot_pts_3d):

        # Initial map creation
        if self.map_manager.is_empty():
            self.map_manager.initialize_map(local_robot_pts_3d, des_clean, frame_index)
 

        # Viewing Cone / Frustum Culling
        visible_des, visible_pts_glob_2d, visible_map_indices = self.algorithmen.test_only_for_visible_landmarks(self.map_manager.landmarks, self.pose, base_to_kinect_matrix)
        
        delta_R = None
        delta_t = None
        delta_theta = None
        
        log_robot_likelihood = 0.0
        pose_updated = False
        if len(visible_des) > 0:
            # Match visible landmarks with current frame keypoints
            matches = self.bf.match(np.array(visible_des), des_clean)
            
            if len(matches) > self.config.min_matches:
                P_local = []
                Q_curr  = []
                matched_curr_indices = set()
                #visible_landmarks = []

                cos_t = math.cos(-self.pose.theta)
                sin_t = math.sin(-self.pose.theta)

                for match in matches:
                    map_idx = visible_map_indices[match.queryIdx]
                    pt_glob = visible_pts_glob_2d[match.queryIdx]

                    # transform global landmark position to local robot coordinates for the matched landmark
                    dx = pt_glob[0] - self.pose.x
                    dy = pt_glob[1] - self.pose.y
                    lx =  dx * cos_t - dy * sin_t
                    ly =  dx * sin_t + dy * cos_t

                    P_local.append([lx, ly])
                    Q_curr.append(local_robot_pts_3d[match.trainIdx][:2])
                    matched_curr_indices.add(match.trainIdx)

                    lm = self.map_manager.landmarks[map_idx]
                    #z_pt = local_robot_pts_3d[match.trainIdx][:2]
                    #depth_val = local_robot_pts_3d[match.trainIdx][2]

                    lm.seen_count += 1
                    lm.last_seen = frame_index
                    #visible_landmarks.append(lm)

                # ransac refinement to get robust transformation estimation
                delta_R, delta_t, delta_theta = self.algorithmen.ransac_refinement(np.array(P_local), np.array(Q_curr))


                if delta_R is not None:
                    #z_x = 0.0
                    #z_y = 0.0
                    #z_theta = 0.0
                    # relative transformation in local robot coordinates to odom frame
                    cos_c = math.cos(self.pose.theta)
                    sin_c = math.sin(self.pose.theta)
                    delta_tx_odom =  delta_t[0] * cos_c - delta_t[1] * sin_c
                    delta_ty_odom =  delta_t[0] * sin_c + delta_t[1] * cos_c

                    sigma_x = self.config.sigma_x
                    sigma_y = self.config.sigma_y
                    sigma_theta = self.config.sigma_theta
                    
                    epsilon_x = np.random.normal(0, sigma_x)
                    epsilon_y = np.random.normal(0, sigma_y)
                    epsilon_theta = np.random.normal(0, sigma_theta)


                    # update current pose with the estimated transformation
                    self.pose.x += delta_tx_odom + epsilon_x
                    self.pose.y += delta_ty_odom + epsilon_y
                    self.pose.theta += delta_theta + epsilon_theta
                    pose_updated = True
                    
                    P_array = np.array(P_local)
                    Q_array = np.array(Q_curr)
                    
                    # Prüfe, wo die Punkte nach der RANSAC-Drehung wirklich liegen
                    Q_transformed = (delta_R @ Q_array.T).T + delta_t
                    errors = np.linalg.norm(P_array - Q_transformed, axis=1)


                    for i, match in enumerate(matches):
                        if errors[i] < self.config.ransac_threshold: # Nur echte Inliers zulassen!
                            map_idx = visible_map_indices[match.queryIdx]
                            train_idx = match.trainIdx

                            kalman_result = []
                            lm = self.map_manager.landmarks[map_idx]
                            depth = float(depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                            #Update Kalman Filter für Landmarke mit aktuellem Messwert
                            kalman_result, P, log_likelihood = lm.ekf.update(
                                np.array(local_robot_pts_3d[train_idx]), 
                                np.array([self.pose.x, self.pose.y, self.pose.theta]), 
                                np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]), 
                                depth
                            )

                            log_robot_likelihood += log_likelihood
                            #self.map_manager.landmarks[map_idx]['pt_glob'] = [kalman_result[0], kalman_result[1], kalman_result[2]]
                            self.map_manager.landmarks[map_idx]
                            lm.pt_glob = Coordinate(x=kalman_result[0], y=kalman_result[1], z=kalman_result[2])

                            #z_pt = local_robot_pts_3d[train_idx][:2]
                            #depth_val = local_robot_pts_3d[train_idx][2]

                    # add new landmarks to the map based on the current frame and the updated pose estimation
                    self.map_manager.add_new_landmarks(
                        local_robot_pts_3d, des_clean, matched_curr_indices, 
                        self.pose, frame_index
                    )
                    
                    # remove old landmarks that are not seen anymore
                    self.map_manager.clean_map(frame_index)

                    #Unguelitg wenn Delta zu groß ist, um Ausreißer zu vermeiden
                    if (np.linalg.norm(delta_t) > self.config.ransac_max_deviation_delta or
                            abs(delta_theta) > self.config.ransac_max_deviation_theta):
                        print(f"RANSAC sehr schlecht: |Δt|={np.linalg.norm(delta_t):.1f}mm, "
                            f"Δθ={np.degrees(delta_theta):.1f}°")
                        log_robot_likelihood = self.config.partical_filter_fail_standart_error

                else:
                    print("RANSAC failed to find a valid transformation.")
                    log_robot_likelihood = self.config.partical_filter_fail_standart_error # very low likelihood if RANSAC fails to discourage this pose update
            else:
                print(f"Not enough matches found for RANSAC {len(matches)}")
                log_robot_likelihood = self.config.partical_filter_fail_standart_error # very low likelihood if not enough matches are found to discourage this pose update
        else:
            print("No visible landmarks to match with.")
            log_robot_likelihood = self.config.partical_filter_fail_standart_error # very low likelihood if no visible landmarks are found to discourage this pose update
        return pose_updated, self.pose, log_robot_likelihood