class configurations():

    def __init__(self):

        #Puffer Size for Ros Subscriber and Publisher
        self.puffer_size = 10
        #Subscritption Paths
        self.rgb_topic = '/serf01/nav_rgbd_1/rgb/image_raw'
        self.depth_topic = '/serf01/nav_rgbd_1/depth/image_raw'
        self.pcl_topic = '/serf01/nav_rgbd_1/pointcloud'
        self.odom_topic = '/serf01/odometry/project_slam'

        self.num_robots = 20 #Count of Robots to be initialized in the system, if 0 or less, only one robot will be initialized (for single-robot SLAM)

        #ORB Configuration
        self.orb_nfeatures = 1000
        self.orb_patchSize = 31

        # Ransac Configuration
        self.ransac_iterations = 100 # Count of RANSAC iterations for robust pose estimation
        self.ransac_threshold = 40 # in mm, maximum distance for a point to be considered as an inlier in RANSAC
        self.ransac_max_deviation_delta = 400 # in mm maximale erlaubte Abweichung von Δt, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        self.ransac_max_deviation_theta = 2.35 # in radians maximale erlaubte Abweichung von Δtheta, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        
        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61
        self.kinect_width = 640
        self.kinect_height = 480

        self.grid_size = 5 # in pixels, minimum distance between keypoints in pixel space (e.g., 7 means one keypoint per 7x7 pixel area)

        # min and max length for kinect depth values to filter out outliers and points that are too close or too far
        self.min_depth = 400
        self.max_depth = 7500

        self.seen_count_threshold = 5 # Minimum Count of times a landmark has been seen to be considered valid (quality metric)
        self.last_seen_threshold = 15 # Count of the number of frames after which a landmark is considered outdated if it hasn't been seen again

        self.frame_counter = 1 # Count of frames to skip after a pose update to increase stability
        self.min_matches = 10 # Minimum Count of matches for a landmark update to be considered valid (to avoid outliers)

        self.sigma_x = 1.5 # in mm
        self.sigma_y = 1.5 # in mm
        self.sigma_theta = 0.002 # in radians

        #Sigma R-Approximation
        self.a = 0.001477 #Coefficient a for depth error approximation (constant offset)
        self.b = 0.002294 #Coefficient b for depth error approximation (quadratic term)
        self.s_x = 0.8/3 # Lateral Error

        self.partical_filter_fail_standart_error = -70000 #Error for Log-Likelyhood to penaltize particle filter updates that fail (e.g., due to too few matches or RANSAC failure)