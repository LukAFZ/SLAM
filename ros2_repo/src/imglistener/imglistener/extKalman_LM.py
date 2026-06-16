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
        self.cu = self.config.cu
        self.cv = self.config.cv
        # Focal length 
        self.f = self.config.f

    # set Jacobi matrix of the state transition
    def set_JF(self, JF):
        self.JF = JF
        
    # set Jacobi Matrix of the measurement function
    def set_JH(self, c, s):
        #Transformierte JH Matrix
        self.JH = np.array([[c, s, 0.0],
                            [-s, c, 0.0],
                            [0.0, 0.0, 1.0]])

    # set measurement noise -- eg. for EKF
    def set_R(self, pt, depth_value, c, s):
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

    def predict_State(self):
        #pstate = self.state_func(self.x)
        pstate = self.x
        pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
        return pstate, pP

    # return measurement prediction (\hat z_{t|t-1})
    def predict_Measurement(self, curr_pose, c, s):
        #Calculate Matrix R_T (to odom) * (Current Landmark position - Robot position)
		#R = ([[c ,  -s, 0.0],       
		#     [s  ,   c, 0.0],   ^T 
	    #	  [0.0, 0.0, 1.0]])

        pmeas = np.array([c*(self.x[0]-curr_pose.x)+s*(self.x[1]-curr_pose.y),
                         -s*(self.x[0]-curr_pose.x)+c*(self.x[1]-curr_pose.y),
                         self.x[2]                 -0]) # Only Rotation 
        return pmeas
    
    # return matrix K
    def compute_Kalman_Gain(self):
        
        
        PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
        HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
        HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
        K = np.matmul(PHT, HPHTpRi)
        return K

    def setP(self, P):
        self.P=P


    # Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
    def update(self, z, curr_pose: RobotOdom2D, pt, depth_value):

        c = cos(curr_pose.theta)
        s = sin(curr_pose.theta)

        self.set_R(pt, depth_value, c, s)

        if self.P is None:
            # Erste Beobachtung: P = Messunsicherheit (in den passenden Koordinaten)
            self.P = self.R.copy()

        #print("State:", self.x)
        x_tt1, P_tt1 = self.predict_State()
        #print("Predicted state:", x_tt1)
        self.set_JH(c, s)
        
        self.P = P_tt1

        z_tt1 = self.predict_Measurement(curr_pose, c, s)
        #print("Predicted measurement:", z_tt1)
        #print("Actual measurement:", z)
        K = self.compute_Kalman_Gain()
        self.x = self.x + np.matmul(K, (z-z_tt1))
        self.P = self.P - np.matmul(K, np.matmul(self.JH, self.P))
        

        likelihood = self.compute_measurement_likelihood(z, z_tt1)
        return self.x, self.P, likelihood
    
    def sigma_R_approximation(self, depth_value: float):
        #a→ konstanter Offset (Rauschen bei minimalem Abstand)
        #b→ quadratischer Koeffizient (fitted)
        a = self.config.a
        b = self.config.b

        depth_m = depth_value / 1000.0 # convert to meters
         # Tiefenfehler (d^2 für Kinect structured light)
        s_z_m = a + b * (depth_m - 0.4)**2
        s_z = s_z_m * 1000 # in mm
        # Lateraler Fehler (Bogenlänge, ~0.086° Auflösung)
        s_x = self.config.s_x


        return s_z, s_x

    def compute_measurement_likelihood(self, z, z_tt1):
        """
        Berechne die Wahrscheinlichkeit der Messung z gegeben der vorhergesagten Messung z_tt1 und der Messrauschen-Kovarianz R.
        Die Berechnung basiert auf der multivariaten Normalverteilung, wobei die Innovation (z - z_tt1) und die Kovarianz R verwendet werden, um die Likelihood zu bestimmen.
        """
        innovation = (z - z_tt1)

        try:
            PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
            HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
            S = HPHT + self.R  # Innovation covariance
            S_inv = np.linalg.inv(S)
            S_det = np.linalg.det(S)

            exponent = -0.5 * innovation.T @ S_inv @ innovation

            #likelihood = (1.0 / np.sqrt(((2 * np.pi) ** 3) * S_det)) * np.exp(exponent)
            log_likelihood = -0.5 * (3 * np.log(2 * np.pi) + np.log(S_det)) + exponent

            #print(f"Landmark Likelihood: {log_likelihood:.6f}")
            #print(f"innovation={z-z_tt1}, sqrt(diag(S))={np.sqrt(np.diag(S))}, P_diag={np.diag(self.P)}")
            return log_likelihood
        except np.linalg.LinAlgError:
            # Fallback block to guard against zero or singular determinant matrix crashes
            return self.config.partical_filter_fail_standart_error # very low likelihood in case of numerical issues to discourage this measurement update