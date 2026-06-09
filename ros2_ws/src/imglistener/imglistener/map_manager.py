import math
import numpy as np
from .extKalman_LM import ExtKalman

class MapManager:
    def __init__(self, config):
        self.config = config
        # Map Management storage (Landmarks in 3D)
        # List of {'pt_glob': [x,y,z], 'des': descriptor, 'seen_count': int, 'last_seen': int}
        self.landmarks = []
        
    def is_empty(self):
        return len(self.landmarks) == 0

    def initialize_map(self, local_pts_3d, descriptors, frame_index):
        """create initial map with the first frame's keypoints"""
        for i in range(len(local_pts_3d)):
            self.landmarks.append({
                'pt_glob': local_pts_3d[i], 
                'des': descriptors[i], 
                'seen_count': 1, 
                'last_seen': frame_index,
                'ekf': ExtKalman(np.array(local_pts_3d[i]))    
            })

    def add_new_landmarks(self, local_pts_3d, descriptors, matched_curr_indices, curr_pose, frame_index):
        """add new, unmatched points to the global map"""
        curr_pos_x, curr_pos_y, curr_theta = curr_pose
        
        # Add new landmarks
        # add not all points but only those that where mached with the current frame, to avoid adding outliers
        for i in range(len(local_pts_3d)):
            if i not in matched_curr_indices: # Nur Punkte hinzufügen, die nicht einmal gematcht wurden (also komplett neue Punkte)
                pt = local_pts_3d[i]
                gx = (curr_pos_x) + pt[0]*math.cos(curr_theta) - pt[1]*math.sin(curr_theta)
                gy = (curr_pos_y) + pt[0]*math.sin(curr_theta) + pt[1]*math.cos(curr_theta)
                self.landmarks.append({
                    'pt_glob': [gx, gy, pt[2]], 
                    'des': descriptors[i], 
                    'seen_count': 1, 
                    'last_seen': frame_index,
                    'ekf': ExtKalman(np.array([gx, gy, pt[2]]))
                })

    def clean_map(self, frame_index):
        """delete landmarks that are not seen for a long time or have a low seen_count (quality metric)"""
        # Remove landmarks (Quality metric = seen_count)
        self.landmarks = [
            lm for lm in self.landmarks 
            if lm['seen_count'] > self.config.seen_count_threshold 
            or (frame_index - lm['last_seen']) < self.config.last_seen_threshold
        ]

    def get_all_points_for_msg(self):
        """get all landmark points in the format for PointCloud2 message"""
        return [[lm['pt_glob'][0]/1000.0, lm['pt_glob'][1]/1000.0, lm['pt_glob'][2]/1000.0] for lm in self.landmarks]