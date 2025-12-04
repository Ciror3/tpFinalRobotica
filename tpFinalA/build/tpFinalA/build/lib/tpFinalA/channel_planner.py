import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import heapq
import math

def yaw_from_quat(q):
    """ Convierte cuaternión a ángulo Yaw (Euler) """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

def wrap_angle(angle):
    """ Normaliza el ángulo entre -pi y pi """
    return (angle + np.pi) % (2 * np.pi) - np.pi

class State5D:
    def __init__(self, x, y, theta, v, w, g=0.0, parent=None, control=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.w = w
        self.g = g  
        self.f = 0.0  
        self.parent = parent
        self.control = control  

    def __lt__(self, other):
        return self.f < other.f
    
    def to_key(self):
        # f"{valor:.2f}" redondea a 2 decimales. 
        # Ej: 0.023 -> "0.02", 0.028 -> "0.03"
        return "{:.2f}_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(
            self.x, self.y, self.theta, self.v, self.w
        )

class ChannelPlanner(Node): 
    def __init__(self):
        super().__init__("path_follower") 

        self.max_v = 0.22
        self.min_v = 0.0
        self.max_w = 1.5
        self.max_accel = 0.2
        self.max_alpha = 1.0
        
        # Simulación
        self.dt = 0.2         # Paso de tiempo (0.2s)
        self.horizon = 10    
        
        self.acc_samples = [-0.1, 0.0, 0.1]
        self.alpha_samples = [-0.5, 0.0, 0.5]

        self.get_logger().info("5D")

    def get_target_point(self, current_pose, path):
        lookahead_dist = 0.40
        my_pos = np.array([current_pose.x,current_pose.y])
        target = path[-1]
        

        for point in path:
            dist = np.linalg.norm(np.array(point) - my_pos)
            if dist > lookahead_dist:
                target = point
                break
        
        return target
    
    def motion_model(self, s, acc, alpha):
        theta_n = s.theta + s.w * self.dt
        theta_n = (theta_n + np.pi) % (2 * np.pi) - np.pi 
        
        x_n = s.x + s.v * np.cos(s.theta) * self.dt
        y_n = s.y + s.v * np.sin(s.theta) * self.dt
        
        v_n = np.clip(s.v + acc * self.dt, self.min_v, self.max_v)
        w_n = np.clip(s.w + alpha * self.dt, -self.max_w, self.max_w)
        
        return State5D(x_n, y_n, theta_n, v_n, w_n, parent=s, control=(acc, alpha))

    def create_channel(self, start_state, path, obstacle_list):
        # 1. Definir meta local
        local_goal = self.get_target_point(start_state, path)
        if local_goal is None: 
            return Twist()
        goal_x, goal_y = local_goal

        # 2. Inicializar A*
        open_set = []
        heapq.heappush(open_set, start_state)
        
        visited = set()
        # visited.add(start_state.to_key())
        
        best_node = start_state
        min_heuristic = float('inf')
        max_iters = 300
        iters = 0

        while iters < max_iters:
            iters += 1
            current = heapq.heappop(open_set)
            
            # Distancia a la meta local (Heurística)
            dist_to_goal = np.hypot(current.x - goal_x, current.y - goal_y)

            # Guardamos el nodo que más se acercó (aunque no llegue exacto)
            if dist_to_goal < min_heuristic:
                min_heuristic = dist_to_goal
                best_node = current

            # Si llegamos lo suficientemente cerca, terminamos
            if dist_to_goal < 0.1:
                best_node = current
                break

            # Podar si nos pasamos del horizonte de tiempo
            if current.g > self.horizon:
                continue

            # Expandir vecinos (Acciones cinemáticas)
            for acc in self.acc_samples:
                for alpha in self.alpha_samples:
                    neighbor = self.motion_model(current, acc, alpha)
                    if neighbor.to_key() in visited: 
                        continue

                    # --- Chequeo de Colisiones ---
                    collision = False
                    for ox, oy in obstacle_list:
                        # Si el robot se traba, prueba bajar 0.20 a 0.15
                        if np.hypot(neighbor.x - ox, neighbor.y - oy) < 0.0:
                            collision = True
                            break
                    
                    if collision: 
                        continue
                    
                    # --- Costos ---
                    cte_cost = dist_to_goal * 2.0
                    smooth_cost = 0
                    if current.control:
                        smooth_cost = abs(acc - current.control[0]) + abs(alpha - current.control[1])

                    neighbor.g = current.g + 1.0 + cte_cost + smooth_cost
                    neighbor.f = neighbor.g # Greedy search

                    visited.add(neighbor.to_key())
                    heapq.heappush(open_set, neighbor)

        # --- CORRECCIÓN 2: Reconstrucción del Camino (Backtracking) ---
        # Sin esto, la función no devuelve nada útil.
        
        # Recuperamos la cadena de padres desde el mejor nodo hasta el inicio
        chain = []
        curr = best_node
        while curr.parent:
            chain.append(curr)
            curr = curr.parent
        
        cmd = Twist()
        
        # Si hay cadena, tomamos el último elemento (que es el primer paso desde el start)
        if chain:
            step = chain[-1] 
            cmd.linear.x = float(step.v)
            cmd.angular.z = float(step.w)
        else:
            # Si no hay cadena, significa que no pudimos movernos del start (atrapado)
            # Frenar el robot
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        return cmd