class configurations():

    def __init__(self):

        #Puffer Size für die ROS Subscriber und Publisher
        self.puffer_size = 10
        #Subscritption Paths
        self.rgb_topic = '/serf01/nav_rgbd_1/rgb/image_raw'
        self.depth_topic = '/serf01/nav_rgbd_1/depth/image_raw'
        self.pcl_topic = '/serf01/nav_rgbd_1/pointcloud'
        self.odom_topic = '/serf01/odometry/project_slam'

        self.num_robots = 1 #Anzahl virueller Roboter, die gleichzeitig in der Karte verfolgt werden sollen

        # Ransac Configuration
        self.ransac_iterations = 100 # Anzahl der Iterationen für RANSAC
        self.ransac_threshold = 40 # in mm, maximaler Abstand, um als Inlier zu gelten
        self.ransac_max_deviation_delta = 400 # in mm maximale erlaubte Abweichung von Δt, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        self.ransac_max_deviation_theta = 2.35 # in radians maximale erlaubte Abweichung von Δtheta, damit ein RANSAC-Update als gültig angesehen wird (zur Vermeidung von Ausreißern)
        
        # Image center coordinates
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61
        self.kinect_width = 640
        self.kinect_height = 480

        # Min und Maximale Laenge fuer Kinect in der Tiefenwerte als realistisch angesehen werden (in mm)
        self.min_depth = 400
        self.max_depth = 7500

        self.seen_count_threshold = 5 # Anzahl der Sichtungen, die eine Landmarke mindestens haben muss, um als stabil zu gelten
        self.last_seen_threshold = 15 # Anzahl der Frames, die seit der letzten Sichtung einer Landmarke vergangen sein müssen, damit sie als "verloren" gilt

        self.frame_counter = 1 #Anzahl der Frames - 1, die nach einem Update übersprungen werden, um die Stabilität zu erhöhen (z.B. bei RANSAC-Updates)
        self.min_matches = 30 # Minimum der Anzahl von Matches, damit ein RANSAC-Update durchgeführt wird

        self.sigma_x = 1.5 # in mm
        self.sigma_y = 1.5 # in mm
        self.sigma_theta = 0.002 # in radians

        #Sigma R-Approximation
        self.a = 0.001477
        self.b = 0.002294
        self.s_x = 0.8/3 #Lateraler Fehler

        self.partical_filter_fail_standart_error = -700 #Fehler für die Log-Likelihood, um Fehlberechnungen bspw. beim Ransac zu bestrafen