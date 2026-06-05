import cv2
import math
import numpy as np
from .algorithms import algorithms
from .config import configurations
from .map_manager import MapManager

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

        # Initial map creation
        if self.map_manager.is_empty():
            self.map_manager.initialize_map(local_robot_pts_3d, des_clean, self.frame_index)
            self.frame_index += 1
            return False, self.curr_pos_x, self.curr_pos_y, self.curr_theta

        # Viewing Cone / Frustum Culling
        visible_des, visible_pts_glob_2d, visible_map_indices = self.algo.test_only_for_visible_landmarks(
            self.map_manager.landmarks, self.curr_pos_x, self.curr_pos_y, self.curr_theta, base_to_kinect_matrix
        )

        pose_updated = False

        if frame_counter == 0:
            if len(visible_des) > 0:
                # Match visible landmarks with current frame keypoints
                matches = self.bf.match(np.array(visible_des), des_clean)
                
                if len(matches) > self.config.min_matches:
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

                        lm = self.map_manager.landmarks[map_idx]
                        z_pt = local_robot_pts_3d[match.trainIdx][:2]
                        depth_val = local_robot_pts_3d[match.trainIdx][2]

                        lm['seen_count'] += 1
                        lm['last_seen'] = self.frame_index
                        visible_landmarks.append(lm)

                    # ransac refinement to get robust transformation estimation
                    delta_R, delta_t, delta_theta = self.algo.ransac_refinement(np.array(P_local), np.array(Q_curr))

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
                        pose_updated = True
                        
                        P_array = np.array(P_local)
                        Q_array = np.array(Q_curr)
                        
                        # Prüfe, wo die Punkte nach der RANSAC-Drehung wirklich liegen
                        Q_transformed = (delta_R @ Q_array.T).T + delta_t
                        errors = np.linalg.norm(P_array - Q_transformed, axis=1)

                        # Dein RANSAC-Threshold war 50, mit Toleranz (1.25) = 62.5
                        for i, match in enumerate(matches):
                            if errors[i] < self.config.ransac_threshold * 1.25: # Nur echte Inliers zulassen!
                                map_idx = visible_map_indices[match.queryIdx]
                                train_idx = match.trainIdx

                                kalman_result = []
                                lm = self.map_manager.landmarks[map_idx]
                                depth = float(depth_frame[int(kp_clean[train_idx].pt[1]), int(kp_clean[train_idx].pt[0])])

                                kalman_result, P = lm['ekf'].update(
                                    np.array(local_robot_pts_3d[train_idx]), 
                                    np.array([self.curr_pos_x, self.curr_pos_y, self.curr_theta]), 
                                    np.array([kp_clean[train_idx].pt[0], kp_clean[train_idx].pt[1]]), 
                                    depth
                                )
                                
                                self.map_manager.landmarks[map_idx]['pt_glob'] = [kalman_result[0], kalman_result[1], kalman_result[2]]

                                z_pt = local_robot_pts_3d[train_idx][:2]
                                depth_val = local_robot_pts_3d[train_idx][2]

                        # add new landmarks to the map based on the current frame and the updated pose estimation
                        self.map_manager.add_new_landmarks(
                            local_robot_pts_3d, des_clean, matched_curr_indices, 
                            (self.curr_pos_x, self.curr_pos_y, self.curr_theta), self.frame_index
                        )
                        
                        # remove old landmarks that are not seen anymore
                        self.map_manager.clean_map(self.frame_index)

                else:
                    print(f"Not enough matches found for RANSAC {len(matches)}")

        # draw keypoints in green
        img2 = cv2.drawKeypoints(frame, kp_clean, None, color=(0,255,0), flags=0)
        cv2.imshow("Feature + Depth", img2)
        cv2.waitKey(1)

        self.frame_index += 1
        return pose_updated, self.curr_pos_x, self.curr_pos_y, self.curr_theta