# kapsch_und_ransac.py
import numpy as np
import math

def get_kapsch_2d(self, P, Q):
    # Berechnung der Centroidenq    
    P_middle = np.mean(P, axis=0) 
    Q_middle = np.mean(Q, axis=0)

    # Zentrieren der Punkte 
    P_centered = P - P_middle 
    Q_centered = Q - Q_middle 

    # Rotationswinkelberechnung mit Kapsch Methode
    theta = math.atan2(sum(Q_centered[:,0]*P_centered[:,1] - Q_centered[:,1]*P_centered[:,0]), 
                        sum(Q_centered[:,0]*P_centered[:,0] + Q_centered[:,1]*P_centered[:,1]))

    # Rotationsmatrix und Translation berechnen
    Rotation_matrix = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])
    Translation = P_middle - Rotation_matrix @ Q_middle
    return Rotation_matrix, Translation, theta

def ransac_refinement(self, P, Q):
    max_iterations = 200
    threshold = 50
    best_rotation, best_translation, best_theta = None, None, 0
    best_inlier_count = 0
        
    if len(P) < 5: #Benötigt mindestens 5 Punkte für eine robuste Schätzung
        return best_rotation, best_translation, best_theta

    for _ in range(max_iterations):
        #Zufällige Auswahl von 3 Punkten für die Schätzung der Transformation
        indices = np.random.choice(len(P), size=3, replace=False)
        # Schätzung der Transformation mit der Kapsch Methode
        R_estimated, t_estimated, theta_estimated = self.get_kapsch_2d(P[indices], Q[indices])
        # Berechnung der Fehler für alle Punkte basierend auf der geschätzten Transformation
        Q_transformed = (R_estimated @ Q.T).T + t_estimated
        errors = np.linalg.norm(P - Q_transformed, axis=1)
        inlier_count = np.sum(errors < threshold)

        if inlier_count > best_inlier_count:
            P_second = P[errors < threshold]
            Q_second = Q[errors < threshold]
            best_rotation, best_translation, best_theta = self.get_kapsch_2d(P_second, Q_second)
            best_inlier_count = inlier_count
                
    return best_rotation, best_translation, best_theta