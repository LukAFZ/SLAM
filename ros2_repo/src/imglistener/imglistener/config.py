class configurations():

    def __init__(self):

        #Puffer Size for Ros Subscriber and Publisher
        self.PUFFER_SIZE = 10
        #Subscritption Paths
        self.RGB_TOPIC = '/serf01/nav_rgbd_1/rgb/image_raw'
        self.DEPTH_TOPIC = '/serf01/nav_rgbd_1/depth/image_raw'
        self.PCL_TOPIC = '/serf01/nav_rgbd_1/pointcloud'
        self.ODOM_TOPIC = '/serf01/odometry/project_slam'

        self.NUM_ROBOTS = 20 #Count of Robots to be initialized in the system, if 0 or less, only one robot will be initialized (for single-robot SLAM)

        #ORB Configuration
        self.ORB_NFEATURES = 1000 #Standard 500
        self.ORB_PATCH_SIZE = 31 #Standard 31
        self.ORB_EDGE_THRESHOLD = 31 #Standard 31
        self.ORB_SCALE_FACTOR = 1.2 #Standard 1.2
        self.ORB_NLEVELS = 8 #Standard 8
        self.ORB_WTA_K = 2 #Standard 2
        self.ORB_FIRST_LEVEL = 0
        self.ORB_FAST_THRESHOLD = 50 #Standard 10

        # Ransac Configuration
        self.RANSAC_ITERATIONS = 100 # Count of RANSAC iterations for robust pose estimation
        self.RANSAC_THRESHOLD = 40 # in mm, maximum distance for a point to be considered as an inlier in RANSAC
        self.RANSAC_MAX_DEVIATION_DELTA = 400 # in mm maximale erlaubte Abweichung von Δt, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        self.RANSAC_MAX_DEVIATION_THETA = 2.35 # in radians maximale erlaubte Abweichung von Δtheta, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        
        # Image center coordinates
        self.CU = 318.525
        self.CV = 241.181
        # Focal length 
        self.F = 526.61
        self.KINECT_WIDTH = 640
        self.KINECT_HEIGHT = 480

        self.GRID_SIZE = 5 # in pixels, minimum distance between keypoints in pixel space (e.g., 7 means one keypoint per 7x7 pixel area)

        # min and max length for kinect depth values to filter out outliers and points that are too close or too far
        self.MIN_DEPTH = 400
        self.MAX_DEPTH = 7500

        self.SEEN_COUNT_THRESHOLD = 5 # Minimum Count of times a landmark has been seen to be considered valid (quality metric)
        self.LAST_SEEN_THRESHOLD = 15 # Count of the number of frames after which a landmark is considered outdated if it hasn't been seen again

        self.FRAME_COUNTER = 1 # Count of frames to skip after a pose update to increase stability
        self.MIN_MATCHES = 10 # Minimum Count of matches for a landmark update to be considered valid (to avoid outliers)

        self.SIGMA_X = 1.5 # in mm
        self.SIGMA_Y = 1.5 # in mm
        self.SIGMA_THETA = 0.002 # in radians

        #Sigma R-Approximation
        self.A = 0.001477 #Coefficient a for depth error approximation (constant offset)
        self.B = 0.002294 #Coefficient b for depth error approximation (quadratic term)
        self.S_X = 0.8/3 # Lateral Error

        self.PARTICLE_FILTER_FAIL_STANDARD_ERROR = -70000 #Error for Log-Likelyhood to penaltize particle filter updates that fail (e.g., due to too few matches or RANSAC failure)