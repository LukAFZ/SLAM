from random import *
from math import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats



class ExtKalman:
    def __init__(self, x):
        self.x = x
        self.state_func = lambda x: x.copy() # Identity function for state transition (static landmark)
        self.meas_func = 0
        self.JF = np.eye(3) # Identity matrix for state transition Jacobian
  
        self.P = np.eye(3) * 100.0 # Initial covariance (high uncertainty)
        self.Q = np.eye(3) * 0.0001 # Process noise covariance (small, since we assume static landmarks)

        # for R calculation
        self.cu = 318.525
        self.cv = 241.181
        # Focal length 
        self.f = 526.61

    # set Jacobi matrix of the state transition
    def setJF(self, JF):
        self.JF = JF
        
    # set Jacobi Matrix of the measurement function
    def setJH(self, c, s):
        self.JH = np.array([[c, s, 0.0],
                            [-s, c, 0.0],
                            [0.0, 0.0, 1.0]])

    # set measurement noise -- eg. for EKF
    def setR(self, pt, depth_value, c, s):
        s_z, s_x = self.sigma_R_approximation(depth_value)

        R_sigma_pixel = np.array([[s_x**2,  0.0,    0.0],
                                [   0.0,   s_x**2, 0.0],
                                [   0.0,   0.0,    s_z**2]])

        J_pixel = np.array([[depth_value/self.f,       0.0,        (pt[0]-self.cu)/self.f],
                            [        0.0,      depth_value/self.f, (pt[1]-self.cv)/self.f],
                            [        0.0,               0.0,                 1.0]])

        R_rot_kb = np.array([[c, s, 0.0],
                             [-s, c, 0.0],
                             [0.0, 0.0, 1.0]])

        R_sigma_kinect = J_pixel@R_sigma_pixel@J_pixel.T

        R_sigma_base = R_rot_kb.T@R_sigma_kinect@R_rot_kb

        self.R = R_sigma_base

    # set model noise -- eg. for EKF
    def setQ(self, Q):
        self.Q = Q 

    def predictState(self):
        pstate = self.state_func(self.x)
        pP = np.matmul(self.JF, np.matmul(self.P, self.JF.transpose()))+self.Q #JF P JF^\top + Q
        #print("Predicted state:", pstate)
        return pstate, pP

    # return measurement prediction (\hat z_{t|t-1})
    def predictMeasurement(self, rob_curr, c, s):
        pmeas = np.array([c*(self.x[0]-rob_curr[0])+s*(self.x[1]-rob_curr[1]),
                           -s*(self.x[0]-rob_curr[0])+c*(self.x[1]-rob_curr[1]),
                           self.x[2]])
        return pmeas
    
    # return matrix K
    def computeKalmanGain(self):
        
        
        PHT = np.matmul(self.P, self.JH.transpose())         # PH^\top
        HPHT = np.matmul(self.JH, PHT)                      # HPH^\top
        #HPHTpRi = np.linalg.inv(HPHT + self.R)             # (HPH^\top + R)^{-1}
        #K = np.matmul(PHT, HPHTpRi)
        K = np.linalg.solve((HPHT + self.R).T, PHT.T).T
        return K

    def setP(self, P):
        self.P=P



    # Update self.x and self.P, return tuple (x_{t|t}, P_{t_t})
    def update(self, z, rob_curr, pt, depth_value):
        #print("State:", self.x)
        x_tt1, P_tt1 = self.predictState()
        self.x = x_tt1
        self.P = P_tt1
        #print("Predicted state:", x_tt1)
        c, s = cos(rob_curr[2]), sin(rob_curr[2]) #Berechne nur 1-Mal
        self.setJH(c, s)
        self.setR(pt, depth_value, c, s)
        #self.setP(self.R)
        

        z_tt1 = self.predictMeasurement(rob_curr, c, s)
        #print("Predicted measurement:", z_tt1)
        #print("Actual measurement:", z)
        K = self.computeKalmanGain()
        self.x = self.x + np.matmul(K, (z-z_tt1)) #Innovation
        self.P = self.P - np.matmul(K, np.matmul(self.JH, self.P)) #Kovarianzupdate
        return self.x, self.P
    
    def sigma_R_approximation(self, depth_value: float):
        #a→ konstanter Offset (Rauschen bei minimalem Abstand)
        #b→ quadratischer Koeffizient (fitted)
        a = 0.001477
        b = 0.002294

        depth_m = depth_value / 1000.0 # convert to meters
         # Tiefenfehler (d^2 für Kinect structured light)
        s_z_m = a + b * (depth_m - 0.4)**2
        s_z = s_z_m * 1000 * 0.5 # in mm und mit Faktor 0.5 für realistischere Werte
        # Lateraler Fehler (Bogenlänge, ~0.086° Auflösung)
        s_x = 0.8/3


        return s_z, s_x