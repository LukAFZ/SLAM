class configurations():

    def __init__(self):

        self.num_robots = 1

        # Ransac Configuration
        self.ransac_iterations = 200
        self.ransac_threshold = 50
        
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

        self.frame_counter = 5 #Anzahl der Frames, die nach einem Update übersprungen werden, um die Stabilität zu erhöhen (z.B. bei RANSAC-Updates)
        self.min_matches = 10

        self.sigma_x = 5.0
        self.sigma_y = 5.0
        self.sigma_theta = 0.01