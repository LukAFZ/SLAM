import numpy as np
import math
from .config import configurations
from .constants import Coordinate, State

class algorithms:

    def __init__(self):
        self.config = configurations()
        self.ransac_iterations = self.config.ransac_iterations
        self.ransac_threshold = self.config.ransac_threshold

        self.cu = self.config.cu
        self.cv = self.config.cv
        self.f = self.config.f
        self.kinect_height = self.config.kinect_height
        self.kinect_width = self.config.kinect_width
        self.min_depth = self.config.min_depth
        self.max_depth = self.config.max_depth

    def ransac_refinement(self, P, Q):
        
        max_iterations = self.ransac_iterations
        threshold = self.ransac_threshold
        best_rotation = None
        best_translation = None
        best_theta = 0
        best_inlier_count = 0
        
        if(len(P) < 5):
            # Not enough points for RANSAC, return the transformation from all points
            return best_rotation, best_translation, best_theta

        for _ in range(max_iterations):
            # Randomly select a subset of points
            P_second = []
            Q_second = []

            indices = np.random.choice(len(P), size=3, replace=False)
            P_subset = P[indices]
            Q_subset = Q[indices]

            # Estimate the transformation using the selected subset
            R_estimated, t_estimated, theta_estimated = self.get_kapsch_2d(P_subset, Q_subset)

            # Transform Q and calculate per-point errors
            Q_transformed = (R_estimated @ Q.T).T + t_estimated
            errors = np.linalg.norm(P - Q_transformed, axis=1)
            
            inlier_count = np.sum(errors < threshold)

            if inlier_count > best_inlier_count:
                
                for e, p, q in zip(errors, P, Q):
                    if e < threshold:
                        P_second.append(p)
                        Q_second.append(q)

                best_rotation, best_translation, best_theta = self.get_kapsch_2d(np.array(P_second), np.array(Q_second))

                best_inlier_count = inlier_count
                
        return best_rotation, best_translation, best_theta
    
    def get_kapsch_2d(self, P, Q):    

        # Calculate the centroids of P and Q
        P_middle = np.mean(P, axis=0) #p_quer
        Q_middle = np.mean(Q, axis=0) #q_quer

        # Center the points by subtracting the centroids  
        P_centered = P - P_middle #p_strich
        Q_centered = Q - Q_middle #p_strich


        # Calculate the rotation angle (theta) using the Kabsch algorithm
        theta = math.atan2(sum(Q_centered[:,0]*P_centered[:,1] - Q_centered[:,1]*P_centered[:,0]), sum(Q_centered[:,0]*P_centered[:,0] + Q_centered[:,1]*P_centered[:,1]))
        #print(f"Rotation angle (theta): {math.degrees(theta):.2f} degrees")

        # Calculate the rotation matrix using the rotation angle
        Rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                    [math.sin(theta), math.cos(theta)]])
        # Calculate the translation vector using the centroids and the rotation matrix
        Translation = P_middle - Rotation_matrix @ Q_middle
        #print(f"Translation vector: {Translation}")
        return Rotation_matrix, Translation, theta
    
    def calculate_local_cords_from_matches(self, kp_clean, des_clean, kinect_to_base_matrix, depth_frame):
        #current_points_3d = []
        #current_descriptors = []
        local_robot_pts_3d = []
        for point, des in zip(kp_clean, des_clean):
            depth = float(depth_frame[int(point.pt[1]), int(point.pt[0])])
            
            # X, Y, Z in camera frame
            x_c = (point.pt[0] - self.cu) * depth / self.f
            y_c = (point.pt[1] - self.cv) * depth / self.f
            z_c = depth

            #current_points_3d.append((x_c, y_c, z_c))
            #current_descriptors.append(des)
            # Roboterkoordinaten (X=vorne, Y=links, Z=hoch)

            pt_kinect = np.array([x_c, y_c, z_c, 1.0])
            pt_base = kinect_to_base_matrix @ pt_kinect
            
            local_robot_pts_3d.append([pt_base[0], pt_base[1], pt_base[2]])

        return local_robot_pts_3d
    
    def test_only_for_visible_landmarks(self, map_landmarks, robot_pose: State, base_to_kinect_matrix):
         
        visible_des = []
        visible_pts_glob_2d = []
        visible_map_indices = []

        for idx, lm in enumerate(map_landmarks):
            # Calculate relative landmark position to robot
            delta_x = lm.pt_glob.x - (robot_pose.x)
            delta_y = lm.pt_glob.y - (robot_pose.y)
            # Transform to local robot coordinates
            # Rotation by -robot_pose.theta to align with robot's current orientation
            lx = delta_x * math.cos(-robot_pose.theta) - delta_y * math.sin(-robot_pose.theta)
            ly = delta_x * math.sin(-robot_pose.theta) + delta_y * math.cos(-robot_pose.theta)
            lz = lm.pt_glob.z


            pt_local_base = np.array([lx, ly, lz, 1.0])
            pt_cam = base_to_kinect_matrix @ pt_local_base
            
            c_x = pt_cam[0]
            c_y = pt_cam[1]
            c_z = pt_cam[2]
            
            #Prüfe, ob der Punkt vor der Kamera liegt und innerhalb des gültigen Tiefenbereichs liegt
            if 0 < c_z < self.max_depth: # In front of camera and in valid depth range
                # Project to 2D image plane with pinhole camera model
                u_p = (c_x * self.f) / c_z + self.cu
                v_p = (c_y * self.f) / c_z + self.cv
                # Check if projected point is within image bounds
                if 0 <= u_p <= self.kinect_width and 0 <= v_p <= self.kinect_height:
                    visible_des.append(lm.des)
                    visible_pts_glob_2d.append((lm.pt_glob.x, lm.pt_glob.y))
                    visible_map_indices.append(idx)

        return visible_des, visible_pts_glob_2d, visible_map_indices