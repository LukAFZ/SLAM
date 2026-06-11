class configurations():

    def __init__(self):

        self.num_robots = 5
        # Ransac Configuration
        self.ransac_iterations = 100
        self.ransac_threshold = 40
        self.ransac_max_deviation_delta = 400 # in mm
        self.ransac_max_deviation_theta = 2.35 # in radians
        
        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61
        self.kinect_width = 640
        self.kinect_height = 480

        # Min und Maximale Laenge fuer Kinect Depth
        self.min_depth = 400
        self.max_depth = 7500

        self.seen_count_threshold = 5
        self.last_seen_threshold = 15

        self.frame_counter = 20 #Anzahl der Frames - 1, die nach einem Update übersprungen werden, um die Stabilität zu erhöhen (z.B. bei RANSAC-Updates)
        self.min_matches = 30 # Minimum der Anzahl von Matches, damit ein RANSAC-Update durchgeführt wird

        self.sigma_x = 1.5 # in mm
        self.sigma_y = 1.5 # in mm
        self.sigma_theta = 0.002 # in radians

        self.partical_filter_fail_standart_error = -700