# import numpy as np
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# import heapq
# import math

# def yaw_from_quat(q):
#     """ Convierte cuaternión a ángulo Yaw (Euler) """
#     t3 = +2.0 * (q.w * q.z + q.x * q.y)
#     t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
#     return math.atan2(t3, t4)

# def wrap_angle(angle):
#     """ Normaliza el ángulo entre -pi y pi """
#     return (angle + np.pi) % (2 * np.pi) - np.pi

# class State5D:
#     def __init__(self, x, y, theta, v, w, g=0.0, parent=None, control=None):
#         self.x = x
#         self.y = y
#         self.theta = theta
#         self.v = v
#         self.w = w
#         self.g = g  
#         self.f = 0.0  
#         self.parent = parent
#         self.control = control  

#     def __lt__(self, other):
#         return self.f < other.f
    
#     def to_key(self):
#         # f"{valor:.2f}" redondea a 2 decimales. 
#         # Ej: 0.023 -> "0.02", 0.028 -> "0.03"
#         return "{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(
#             self.x, self.y, self.theta, self.v, self.w
#         )

# class ChannelPlanner(Node): 
#     def __init__(self):
#         super().__init__("path_follower") 

#         self.max_v = 0.5
#         self.min_v = 0.0
#         self.max_w = 2.5
#         self.max_accel = 0.2
#         self.max_alpha = 1.0
        
#         # Simulación
#         self.dt = 0.2         # Paso de tiempo (0.2s)
#         self.horizon = 10    
        
#         self.acc_samples = [-0.1, 0.0, 0.1]
#         self.alpha_samples = [-0.5, 0.0, 0.5]

#         self.get_logger().info("5D")

    
#     def motion_model(self, s, acc, alpha):
#         theta_n = s.theta + s.w * self.dt
#         theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi 
        
#         x_n = s.x + s.v * np.cos(s.theta) * self.dt
#         y_n = s.y + s.v * np.sin(s.theta) * self.dt
        
#         v_n = np.clip(s.v + acc * self.dt, self.min_v, self.max_v)
#         w_n = np.clip(s.w + alpha * self.dt, -self.max_w, self.max_w)
        
#         return State5D(x_n, y_n, theta_n, v_n, w_n, parent=s, control=(acc, alpha))

#     def create_channel(self, start_state, path, obstacle_list):
#         # Intentamos con diferentes distancias de Lookahead.
#         # Si falla el de 0.4m, probamos con 0.25m, luego 0.15m.
#         # Esto permite que el robot "doble la esquina" pegadito si es necesario.
#         lookahead_candidates = [0.40, 0.25, 0.15] 
        
#         best_cmd = Twist() # Por defecto quieto
#         path_found = False

#         for dist in lookahead_candidates:
#             # 1. Obtenemos meta local con esa distancia
#             local_goal = self.get_target_point(start_state, path, dist) # <--- Modificar get_target_point para aceptar dist
#             if local_goal is None: continue
            
#             # 2. Ejecutamos el A* (Extraje la lógica a una función interna para poder llamarla varias veces)
#             cmd, success = self.run_astar(start_state, local_goal, obstacle_list)
            
#             if success:
#                 return cmd # ¡Éxito! Encontramos camino con este lookahead
            
#             # Si fallamos, el bucle continua e intenta con una distancia MENOR

#         # Si llegamos aquí, fallaron todas las distancias. Estamos realmente bloqueados.
#         self.get_logger().warn("BLOQUEADO TOTAL: Iniciando giro de recuperación")
#         best_cmd.linear.x = 0.0
#         best_cmd.angular.z = 0.3 
#         return best_cmd

#     # --- Modifica get_target_point para aceptar override de distancia ---
    # def get_target_point(self, current_pose, path, lookahead_dist=0.30):
    #     my_pos = np.array([current_pose.x, current_pose.y])
    #     if not path: return None
    #     final_point = path[-1] 
        
    #     # Lógica de cierre (si estamos cerca del final)
    #     dist_to_final = np.linalg.norm(np.array(final_point) - my_pos)
    #     if dist_to_final < lookahead_dist: return final_point

    #     target = final_point 
    #     for point in path:
    #         dist = np.linalg.norm(np.array(point) - my_pos)
    #         if dist > lookahead_dist:
    #             target = point
    #             break
    #     return target

#     # --- Mueve todo tu bucle While a esta función auxiliar ---
#     def run_astar(self, start_state, local_goal, obstacle_list):
#         goal_x, goal_y = local_goal
        
#         open_set = []
#         heapq.heappush(open_set, start_state)
#         visited = set()
        
#         # Score inicial
#         dist_start = np.hypot(start_state.x - goal_x, start_state.y - goal_y)
#         angle_to_goal_start = math.atan2(goal_y - start_state.y, goal_x - start_state.x)
#         head_err_start = abs(wrap_angle(angle_to_goal_start - start_state.theta))
#         min_score = dist_start + (head_err_start * 0.5)
        
#         best_node = start_state
#         max_iters = 2000 # Un poco menos para que los reintentos sean rápidos
#         iters = 0
        
#         found_better = False

#         while len(open_set) > 0 and iters < max_iters:
#             iters += 1
#             current = heapq.heappop(open_set)
            
#             dist_to_goal = np.hypot(current.x - goal_x, current.y - goal_y)
#             angle_to_goal = math.atan2(goal_y - current.y, goal_x - current.x)
#             heading_error = abs(wrap_angle(angle_to_goal - current.theta))
#             current_score = dist_to_goal + (heading_error * 0.5)

#             if current_score < min_score:
#                 min_score = current_score
#                 best_node = current
#                 found_better = True # Marcamos que encontramos algo mejor que el inicio

#             if dist_to_goal < 0.10:
#                 best_node = current
#                 found_better = True
#                 break

#             if current.to_key() in visited: continue
#             visited.add(current.to_key())

#             for acc in self.acc_samples:
#                 for alpha in self.alpha_samples:
#                     neighbor = self.motion_model(current, acc, alpha)
                    
#                     # COLISIONES
#                     collision = False
#                     for ox, oy in obstacle_list:
#                         # Radio ajustado a 0.15 para ser un poco más permisivo en esquinas
#                         if np.hypot(neighbor.x - ox, neighbor.y - oy) < 0.15: 
#                             collision = True
#                             break
#                     if collision: continue
                    
#                     # COSTOS
#                     step_cost = 0.1
#                     cte_cost = dist_to_goal * 3.0 
#                     heading_cost = heading_error * 0.8
#                     smooth_cost = 0
#                     if current.control:
#                         smooth_cost = abs(acc - current.control[0]) + abs(alpha - current.control[1])

#                     neighbor.g = current.g + step_cost + cte_cost + heading_cost + smooth_cost
#                     neighbor.f = neighbor.g 
#                     heapq.heappush(open_set, neighbor)

#         # Reconstrucción
#         chain = []
#         curr = best_node
#         while curr.parent:
#             chain.append(curr)
#             curr = curr.parent
        
#         cmd = Twist()
#         # ÉXITO si hay cadena Y el mejor nodo no es el inicio (nos movimos)
#         if chain and found_better:
#             step = chain[-1] 
#             cmd.linear.x = float(step.v)
#             cmd.angular.z = float(step.w)
#             return cmd, True
        
#         return cmd, False


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
        self.linear_v = 0.10        # Velocidad constante (m/s)
        self.max_w = 1.5            # Límite de velocidad angular (rad/s)
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