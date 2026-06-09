from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from .config import configurations



class ExtKalman:
    def __init__(self, x):

        self.config = configurations()

        self.x = x
        self.state_func = 0
        self.meas_func = 0

  
        self.JF = np.eye(3)
        self.Q=np.eye(3)
        self.P=np.eye(3)
        # for R calculation
        self.cu = configurations().cu
        self.cv = configurations().cv
        # Focal length 
        self.f = configurations().f

    # set Jacobi matrix of the state transition
    def setJF(self, JF):
        self.JF = JF
        
    # set Jacobi Matrix of the measurement function
    def setJH(self, rob_curr):

        self.JH = np.array([[cos(rob_curr[2]), sin(rob_curr[2]), 0.0],
                            [-sin(rob_curr[2]), cos(rob_curr[2]), 0.0],
                            [   0.0   ,           0.0,            1.0]])

    # set measurement noise -- eg. for EKF
    def setR(self, pt, depth_value, rob_curr):
        s_z, s_x = self.sigma_R_approximation(depth_value)

        R_sigma_pixel = np.array([[s_x**2,  0.0,    0.0],
                                [   0.0,   s_x**2, 0.0],
                                [   0.0,   0.0,    s_z**2]])

        J_pixel = np.array([[depth_value/self.f,       0.0,        (pt[0]-self.cu)/self.f],
                            [        0.0,      depth_value/self.f, (pt[1]-self.cv)/self.f],
                            [        0.0,               0.0,                 1.0]])

        R_rot_kb = np.array([[cos(rob_curr[2]), sin(rob_curr[2]), 0.0],
                             [-sin(rob_curr[2]), cos(rob_curr[2]), 0.0],
                             [   0.0   ,           0.0,            1.0]])

        R_sigma_kinect = J_pixel@R_sigma_pixel@J_pixel.T

        R_sigma_base = R_rot_kb.T@R_sigma_kinect@R_rot_kb

        self.R = R_sigma_base

    # set model noise -- eg. for EKF
    def setQ(self, Q):
        self.Q = Q

    def predictState(self):
        #pstate = self.state_func(self.x)
        pstate = self.x
        pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q
        return pstate, pP

    # return measurement prediction (\hat z_{t|t-1})
    def predictMeasurement(self, rob_curr):
        pmeas = np.array([cos(rob_curr[2])*(self.x[0]-rob_curr[0])+sin(rob_curr[2])*(self.x[1]-rob_curr[1]),
                           -sin(rob_curr[2])*(self.x[0]-rob_curr[0])+cos(rob_curr[2])*(self.x[1]-rob_curr[1]),
                           self.x[2]])
        return pmeas
    
    # return matrix K
    def computeKalmanGain(self):
        
        
        PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
        HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
        HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
        K = np.matmul(PHT, HPHTpRi)
        return K

    def setP(self, P):
        self.P=P


    # Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
    def update(self, z, rob_curr, pt, depth_value):

        self.setR(pt, depth_value, rob_curr)
        self.setP(self.R)
        #print("State:", self.x)
        x_tt1, P_tt1 = self.predictState()
        #print("Predicted state:", x_tt1)
        self.setJH(rob_curr)
        
        self.P = P_tt1

        z_tt1 = self.predictMeasurement(rob_curr)
        #print("Predicted measurement:", z_tt1)
        #print("Actual measurement:", z)
        K = self.computeKalmanGain()
        self.x = self.x + np.matmul(K, (z-z_tt1))
        self.P = self.P - np.matmul(K, np.matmul(self.JH, self.P))

        likelihood = self.compute_measurement_likelihood(z, z_tt1)
        return self.x, self.P, likelihood
    
    def sigma_R_approximation(self, depth_value: float):
        #a→ konstanter Offset (Rauschen bei minimalem Abstand)
        #b→ quadratischer Koeffizient (fitted)
        a = 0.001477
        b = 0.002294

        depth_m = depth_value / 1000.0 # convert to meters
         # Tiefenfehler (d^2 für Kinect structured light)
        s_z_m = a + b * (depth_m - 0.4)**2
        s_z = s_z_m * 1000 # in mm
        # Lateraler Fehler (Bogenlänge, ~0.086° Auflösung)
        s_x = 0.8/3


        return s_z, s_x

    def compute_measurement_likelihood(self, z, z_tt1):
        # compute the likelihood of the measurement given the current state estimate
        # using the measurement noise covariance R and the innovation (z - z_tt1)
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
            return log_likelihood
        except np.linalg.LinAlgError:
            # Fallback block to guard against zero or singular determinant matrix crashes
            return self.config.partical_filter_fail_standart_error # very low likelihood in case of numerical issues to discourage this measurement update