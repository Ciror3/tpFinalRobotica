import numpy as np
from geometry_msgs.msg import Twist
import math


def yaw_from_quat(q):
    """ Convierte cuaternión a Euler Yaw """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class PurePursuit:
    def __init__(self):
        # PARÁMETROS A SINTONIZAR
        self.lookahead_dist = 0.15  # La "zanahoria" (m)
        self.linear_v = 0.08        # Velocidad constante (m/s)
        self.max_w = 0.5            # Límite de velocidad angular (rad/s)
        self.goal_tolerance = 0.10  # Distancia para considerar que llegó

    def get_target_point(self, current_pose, path):
        """
        Busca el punto del camino que está a 'lookahead_dist' del robot.
        """
        if not path:
            return None

        my_pos = np.array([current_pose.pose.position.x, current_pose.pose.position.y])
        final_point = np.array(path[-1])
        
        # 1. Chequeo de final de camino (Si estamos muy cerca del final, la meta es el final)
        dist_to_final = np.linalg.norm(final_point - my_pos)
        if dist_to_final < self.lookahead_dist:
            return path[-1]

        # 2. Búsqueda del Lookahead Point
        # Buscamos el primer punto que esté MÁS LEJOS que la distancia de lookahead
        # (Esto asegura que el robot siempre tenga un objetivo por delante)
        target = None
        
        # Optimización: Empezar a buscar desde el punto más cercano hacia adelante
        closest_idx = 0
        min_d = float('inf')
        
        # Encontrar índice más cercano
        for i, p in enumerate(path):
            d = np.linalg.norm(np.array(p) - my_pos)
            if d < min_d:
                min_d = d
                closest_idx = i

        # Buscar hacia adelante desde el más cercano
        for i in range(closest_idx, len(path)):
            p = path[i]
            d = np.linalg.norm(np.array(p) - my_pos)
            if d > self.lookahead_dist:
                target = p
                break
        
        # Si no encontramos ninguno más lejos (raro si ya chequeamos el final), usamos el último
        if target is None:
            target = path[-1]
            
        return target
    
    def compute_command(self, current_pose, path):
        cmd = Twist()
        theta = yaw_from_quat(current_pose.pose.orientation)
        
        # 1. Obtener objetivo
        target = self.get_target_point(current_pose, path)
        if target is None:
            return cmd

        angle_to_start = math.atan2(current_pose.pose.position.y - path[0][1], current_pose.pose.position.y - path[0][0])
        heading_error = angle_to_start - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # if abs(heading_error) > (math.pi / 2):
        #     cmd.linear.x = 0.0

        #     cmd.angular.z = 0.5 * heading_error
        #     max_rot_vel = 1.0
        #     cmd.angular.z = np.clip(cmd.angular.z, -max_rot_vel, max_rot_vel)
            
        #     return cmd

        dx = target[0] - current_pose.pose.position.x
        dy = target[1] - current_pose.pose.position.y

        target_y_local = -dx * np.sin(theta) + dy * np.cos(theta)
        
        L = np.hypot(dx, dy)

        if L < self.goal_tolerance:
            return cmd 
        
        curvature = 2.0 * target_y_local / (L**2)
        
        cmd.linear.x = self.linear_v
        cmd.angular.z = self.linear_v * curvature
        
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_w, self.max_w)
        
        if abs(cmd.angular.z) > 0.8:
            cmd.linear.x *= 0.5

        return cmd
