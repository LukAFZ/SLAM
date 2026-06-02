import numpy as np
import math

from cv_bridge import CvBridge
import cv2

from .extKalman_LM import *
from .algorithms import *
from .config import *

class robot():
    def __init__(self):
        
        self.config = configurations()

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_matches = self.config.min_matches

        self.ransac_iterations = self.config.ransac_iterations
        self.ransac_threshold = self.config.ransac_threshold

        # Store 3D points
        self.point_history = []

        self.curr_pos_x = 0.0
        self.curr_pos_y = 0.0
        self.curr_theta = 0.0

        self.map_landmarks = []
        self.seen_count_threshold = configurations().seen_count_threshold
        self.last_seen_threshold = configurations().last_seen_threshold

        self.algorithmen = algorithms()

    def update_robot(self, kp_clean, des_clean, depth_frame, frame_index, frame_counter, kinect_to_base_matrix, base_to_kinect_matrix):
        # 3D coordinates calculation
        local_robot_pts_3d = self.algorithmen.calculate_local_cords_from_matches(kp_clean, des_clean, kinect_to_base_matrix, depth_frame)

        # Initial map creation
        if not self.map_landmarks:
            for i in range(len(local_robot_pts_3d)):
                self.map_landmarks.append({
                    'pt_glob': local_robot_pts_3d[i], 
                    'des': des_clean[i], 
                    'seen_count': 1, 
                    'last_seen': frame_index,
                    'ekf': ExtKalman(np.array(local_robot_pts_3d[i]))    
                })

        # Viewing Cone / Frustum Culling
        visible_des, visible_pts_glob_2d, visible_map_indices = self.algorithmen.test_only_for_visible_landmarks(self.map_landmarks, self.curr_pos_x, self.curr_pos_y, self.curr_theta, base_to_kinect_matrix)
        
        delta_R = None
        delta_t = None
        delta_theta = None
        

        if(frame_counter == 0):
            if(len(visible_des) > 0):
                # Match visible landmarks with current frame keypoints
                matches = self.bf.match(np.array(visible_des), des_clean)
                
                if(len(matches) > self.min_matches):
                    P_local = []
                    Q_curr  = []
                    matched_curr_indices = set()
                    visible_landmarks = []

                    cos_t = math.cos(-self.curr_theta)
                    sin_t = math.sin(-self.curr_theta)

                    for match in matches:
                        map_idx = visible_map_indices[match.queryIdx]
                        pt_glob = visible_pts_glob_2d[match.queryIdx]

                        # transform global landmark position to local robot coordinates for the matched landmark
                        dx = pt_glob[0] - self.curr_pos_x
                        dy = pt_glob[1] - self.curr_pos_y
                        lx =  dx * cos_t - dy * sin_t
                        ly =  dx * sin_t + dy * cos_t

                        P_local.append([lx, ly])
                        Q_curr.append(local_robot_pts_3d[match.trainIdx][:2])
                        matched_curr_indices.add(match.trainIdx)

                        lm = self.map_landmarks[map_idx]
                        z_pt = local_robot_pts_3d[match.trainIdx][:2]
                        depth_val = local_robot_pts_3d[match.trainIdx][2]

                        lm['seen_count'] += 1
                        lm['last_seen'] = frame_index
                        visible_landmarks.append(lm)

                    
                    # ransac refinement to get robust transformation estimation
                    delta_R, delta_t, delta_theta = self.algorithmen.ransac_refinement(np.array(P_local), np.array(Q_curr))

                    if delta_R is not None:
                        #z_x = 0.0
                        #z_y = 0.0
                        #z_theta = 0.0
                        # relative transformation in local robot coordinates to odom frame
                        cos_c = math.cos(self.curr_theta)
                        sin_c = math.sin(self.curr_theta)
                        delta_tx_odom =  delta_t[0] * cos_c - delta_t[1] * sin_c
                        delta_ty_odom =  delta_t[0] * sin_c + delta_t[1] * cos_c

                        # update current pose with the estimated transformation
                        self.curr_pos_x += delta_tx_odom
                        self.curr_pos_y += delta_ty_odom
                        self.curr_theta += delta_theta
                        
                        P_array = np.array(P_local)
                        Q_array = np.array(Q_curr)
                        
                        # Prüfe, wo die Punkte nach der RANSAC-Drehung wirklich liegen
                        Q_transformed = (delta_R @ Q_array.T).T + delta_t
                        errors = np.linalg.norm(P_array - Q_transformed, axis=1)

                        # Dein RANSAC-Threshold war 50, mit Toleranz (1.25) = 62.5
                        for i, match in enumerate(matches):
                            if errors[i] < self.ransac_threshold*1.25: # Nur echte Inliers zulassen!
                                map_idx = visible_map_indices[match.queryIdx]
                                train_idx = match.trainIdx


                                kalman_result = []
                                lm = self.map_landmarks[map_idx]
                                depth = float(depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                                kalman_result, P = lm['ekf'].update(np.array(local_robot_pts_3d[train_idx]), np.array([self.curr_pos_x, self.curr_pos_y, self.curr_theta]), np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]), depth)
                                
                                self.map_landmarks[map_idx]['pt_glob'] = [kalman_result[0], kalman_result[1], kalman_result[2]]


                                z_pt = local_robot_pts_3d[train_idx][:2]
                                depth_val = local_robot_pts_3d[train_idx][2]

                        # Add new landmarks
                        # add not all points but only those that where mached with the current frame, to avoid adding outliers
                        
                        for i in range(len(local_robot_pts_3d)):
                            if i not in matched_curr_indices: # Nur Punkte hinzufügen, die nicht einmal gematcht wurden (also komplett neue Punkte)
                                pt = local_robot_pts_3d[i]
                                gx = (self.curr_pos_x) + pt[0]*math.cos(self.curr_theta) - pt[1]*math.sin(self.curr_theta)
                                gy = (self.curr_pos_y) + pt[0]*math.sin(self.curr_theta) + pt[1]*math.cos(self.curr_theta)
                                self.map_landmarks.append({
                                    'pt_glob': [gx, gy, pt[2]], 
                                    'des': des_clean[i], 
                                    'seen_count': 1, 
                                    'last_seen': frame_index,
                                    'ekf': ExtKalman(np.array([gx, gy, pt[2]]))
                                })
                        
                        # Remove landmarks (Quality metric = seen_count)
                        # Möglicherweise Treshhold der P-matrix (als weitere Quality Metrik) hinzufügen, um nur sehr gut lokalisierte Landmarks zu behalten
                        self.map_landmarks = [lm for lm in self.map_landmarks if lm['seen_count'] > self.seen_count_threshold or (frame_index - lm['last_seen']) < self.last_seen_threshold]

                else:
                    print("Not enough matches for RANSAC refinement: ", len(matches))
        return self.curr_pos_x, self.curr_pos_y, self.curr_theta#, delta_R, delta_t, delta_theta, visible_landmarks