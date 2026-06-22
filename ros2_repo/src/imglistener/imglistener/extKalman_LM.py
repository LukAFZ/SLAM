from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from .data_types import RobotOdom2D



class ExtKalman:
    def __init__(self, x, config):

        self.config = config

        self.x = x
        self.state_func = 0
        self.meas_func = 0

  
        self.JF = np.eye(3)
        self.Q=np.zeros((3,3))
        self.P=None
        # for R calculation
        # initiate the values directly from the config to avoid having to access the config object multiple times (faster)
        self.cu = self.config.CU
        self.cv = self.config.CV
        # Focal length 
        self.f = self.config.F
        self.a = self.config.A
        self.b = self.config.B
        self.s_x = self.config.S_X
        self.fatal_error = self.config.PARTICLE_FILTER_FAIL_STANDARD_ERROR

    # set Jacobi matrix of the state transition
    def set_JF(self, JF):
        self.JF = JF
        
    # set Jacobi Matrix of the measurement function
    def set_JH(self, c, s):
        """
        The Jacobian matrix JH represents the differentiated measurement function that maps the landmark's position from world coordinates (Odom frame) to the robot's local coordinates (Base frame). 
        It is used to project the state covariance matrix P into the measurement space (Base frame), accounting for the current rotation of the robot.
        """
        #Transformierte JH Matrix
        self.JH = np.array([[c, s, 0.0],
                            [-s, c, 0.0],
                            [0.0, 0.0, 1.0]])

    # set measurement noise -- eg. for EKF
    def set_R(self, pt, depth_value, c, s):
        """
        Calculate the measurement noise covariance R based on the depth value and the camera intrinsics. The measurement noise in pixel space is approximated as a function of the depth, 
        and then transformed to 3D space (Base frame) using the Jacobian of the measurement function and the rotation from Kinect to base coordinates.
        """
        s_z, s_x = self.sigma_R_approximation(depth_value)

        #Guaranteed minimum error in 2D pixel space, transformed to 3D space by the Jacobian of the measurement function
        R_sigma_pixel = np.array([[s_x**2,  0.0,    0.0],
                                [   0.0,   s_x**2, 0.0],
                                [   0.0,   0.0,    s_z**2]])

        #Transformation of the measurement noise from 2D pixel space to 3D space in the Kinect coordinate system, and then to the base coordinate system.
        #This accounts for the fact that the measurement noise in pixel space translates to different noise characteristics in 3D space depending on the depth and the camera intrinsics.
        #Differentiated intercept theorem
        J_pixel = np.array([[depth_value/self.f,       0.0,        (pt[0]-self.cu)/self.f],
                            [        0.0,      depth_value/self.f, (pt[1]-self.cv)/self.f],
                            [        0.0,               0.0,                 1.0]])

        #Rotation_Matrix from Kinect coordinates to base coordinates
        R_rot_kb = np.array([[ c, -s, 0.0], 
                             [s, c, 0.0],
                             [0.0  , 0.0, 1.0]])

        R_sigma_kinect = J_pixel@R_sigma_pixel@J_pixel.T #Base Transformation of the measurement noise from pixel space to 3D space in the Kinect coordinate system

        R_sigma_base = R_rot_kb@R_sigma_kinect@R_rot_kb.T #Base Transformation of the measurement noise from Kinect to 3D space in the base coordinate system

        self.R = R_sigma_base

    # set model noise -- eg. for EKF
    def set_Q(self, Q):
        self.Q = Q

    def predict_state(self):
        """
        Predict the state and the covariance of the landmark EKF. Since the landmark is static, the state prediction is just the current state, 
        and the covariance prediction is the current covariance transformed by the Jacobian of the state transition
        """
        #pstate = self.state_func(self.x)
        pstate = self.x
        pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
        return pstate, pP

    # return measurement prediction (\hat z_{t|t-1})
    def predict_measurement(self, curr_pose, c, s):
        """
        Predict the Measurement by substracting the pose position from the landmark position and transforming it to base coordinates
        """
        #Calculate Matrix R_T (to base) * (Current Landmark position - Robot position)
        #R^1 = R.T because R is orthogonal
		#R = ([[c ,  -s, 0.0],       [x_l - x_r]
		#     [s  ,   c, 0.0],   ^T  [y_l - y_r]
	    #	  [0.0, 0.0, 1.0]])      [z_l - z_r]

        pmeas = np.array([c*(self.x[0]-curr_pose.x)+s*(self.x[1]-curr_pose.y),
                         -s*(self.x[0]-curr_pose.x)+c*(self.x[1]-curr_pose.y),
                         self.x[2]                 -0]) # Only Rotation 
        return pmeas
    
    # return matrix K
    def compute_kalman_gain(self):
        """
        Compute the Kalman Gain
        """
        PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
        HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
        HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
        K = np.matmul(PHT, HPHTpRi)
        return K

    def setP(self, P):
        self.P=P


    # Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
    def update(self, z, curr_pose: RobotOdom2D, pt, depth_value):
        """
        Update step of the Landmark EKF
        """

        c = cos(curr_pose.theta)
        s = sin(curr_pose.theta)

        self.set_R(pt, depth_value, c, s)

        if self.P is None:
            # First Measurement: P = Measurementcovariance
            self.P = self.R.copy()

        #print("State:", self.x)
        x_tt1, P_tt1 = self.predict_state()
        #print("Predicted state:", x_tt1)
        self.set_JH(c, s)
        
        self.P = P_tt1

        z_tt1 = self.predict_measurement(curr_pose, c, s)
        #print("Predicted measurement:", z_tt1)
        #print("Actual measurement:", z)
        K = self.compute_kalman_gain()
        self.x = self.x + np.matmul(K, (z-z_tt1))
        self.P = self.P - np.matmul(K, np.matmul(self.JH, self.P))
        

        likelihood = self.compute_measurement_likelihood(z, z_tt1)
        return self.x, self.P, likelihood
    
    def sigma_R_approximation(self, depth_value: float):
        """
        Approximate the measurement noise covariance R based on the depth value. The approximation is based on empirical observations of the Kinect sensor's noise characteristics
        Return the approximated noise covariance matrix for depth and lateral measurements in 3D space, given the depth value in millimeters.
        """
        #a-> constant offset (noise at minimum distance)
        #b→ quadratic offset
        a = self.a
        b = self.b

        depth_m = depth_value / 1000.0 # convert to meters
         # Error for depth (d^2 für Kinect structured light)
        s_z_m = a + b * (depth_m - 0.4)**2
        s_z = s_z_m * 1000 # in mm
        # Lateral error (~0.086° Resolution)
        s_x = self.s_x


        return s_z, s_x

    def compute_measurement_likelihood(self, z, z_tt1):
        """
        Calculate the probability of the measurement z given the predicted measurement z_tt1 and the measurement noise covariance R.
        The calculation is based on the multivariate normal distribution, using the innovation (z - z_tt1) and the covariance R to determine the likelihood.
        """
        innovation = (z - z_tt1)

        try:
            PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
            HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
            S = HPHT + self.R  # Innovation covariance
            S_inv = np.linalg.inv(S)
            S_det = max(np.linalg.det(S), 1e-12)  # Avoid very small determinant for numerical stability

            exponent = -0.5 * innovation.T @ S_inv @ innovation

            #likelihood = (1.0 / np.sqrt(((2 * np.pi) ** 3) * S_det)) * np.exp(exponent)
            log_likelihood = -0.5 * (3 * np.log(2 * np.pi) + np.log(S_det)) + exponent

            #print(f"Landmark Likelihood: {log_likelihood:.6f}")
            #print(f"innovation={z-z_tt1}, sqrt(diag(S))={np.sqrt(np.diag(S))}, P_diag={np.diag(self.P)}")
            return log_likelihood
        except np.linalg.LinAlgError:
            # Fallback block to guard against zero or singular determinant matrix crashes
            return self.fatal_error # very low likelihood in case of numerical issues to discourage this measurement update
        
    def clone(self):
        """Fast clone of the EKF, including state, covariance, and Jacobians."""
        new_ekf = ExtKalman(self.x.copy(), self.config)
        if self.P is not None:
            new_ekf.P = self.P.copy()
        new_ekf.JF = self.JF.copy()
        if hasattr(self, 'JH') and self.JH is not None:
            new_ekf.JH = self.JH.copy()
        if hasattr(self, 'R') and self.R is not None:
            new_ekf.R = self.R.copy()
        return new_ekf