import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
import heapq
import json
import os

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner")

        # --- 1. CONFIGURACIÓN DEL MAPA ---
        # Ruta absoluta o relativa al archivo
        self.map_path = '/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas/mi_mapa_grid_test.txt'
        
        try:
            self.occ_map = np.loadtxt(self.map_path)
            # NOTA: Si el mapa se ve "patas arriba" en RViz, comenta o descomenta la siguiente linea:
            self.occ_map = np.flipud(self.occ_map) 
            self.get_logger().info(f"Mapa cargado: {self.occ_map.shape}")
        except Exception as e:
            self.get_logger().error(f"Error cargando mapa TXT: {e}")
            self.occ_map = np.zeros((100, 100))

        self.rows, self.cols = self.occ_map.shape

        # Parámetros ajustados para mapa de 5x5 metros (100 celdas)
        self.map_res = 0.10        
        self.map_origin_x = -5.0  
        self.map_origin_y = -5.0
        self.map_width = 100      # Coincide con tu archivo txt
        self.map_height = 100

        # --- 2. CARGAR Y FUSIONAR LANDMARKS ---
        # Asumimos que el json se llama igual pero con terminación _landmarks.json
        # o puedes poner la ruta directa abajo.
        json_path = self.map_path.replace('_grid_test.txt', '_landmarks.json')
        # json_path = '/home/ciror/.../mi_mapa_landmarks.json' # Ruta manual si falla la automatica
        
        self.load_and_inflate_landmarks(json_path)

        # --- 3. ROS SUBSCRIBERS & PUBLISHERS ---
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "/fpose", self.pose_callback, 10)
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)
        
        # Publisher para ver el mapa con landmarks en RViz
        self.map_vis_pub = self.create_publisher(OccupancyGrid, "/planning_map", 10)
        self.create_timer(2.0, self.publish_visual_map) # Publicar mapa cada 2s

        self.current_pose = None
        self.get_logger().info("Path Planner con Landmarks listo.")

    def load_and_inflate_landmarks(self, filepath):
        """ Lee el JSON y marca los landmarks como obstáculos (100) en el mapa """
        if not os.path.exists(filepath):
            self.get_logger().warn(f"NO se encontró archivo de landmarks en: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                landmarks = json.load(f)
            
            self.get_logger().info(f"Cargando {len(landmarks)} landmarks...")
            
            # Radio de seguridad (en celdas). 3 celdas * 5cm = 15cm de radio
            radius = 3 
            
            for lm in landmarks:
                lx, ly = lm['x'], lm['y']
                gy, gx = self.world_to_grid(lx, ly)
                
                # Dibujar un cuadrado alrededor del landmark
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = gy + dy, gx + dx
                        # Verificar limites
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            self.occ_map[ny, nx] = 100 # OBSTÁCULO SOLIDO
                            
            self.get_logger().info("Landmarks inyectados en el mapa de costos.")
            
        except Exception as e:
            self.get_logger().error(f"Error parseando JSON: {e}")

    def publish_visual_map(self):
        """ Publica el mapa interno para ver en RViz dónde cree el robot que están los obstáculos """
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.map_res
        msg.info.width = self.cols
        msg.info.height = self.rows
        msg.info.origin.position.x = float(self.map_origin_x)
        msg.info.origin.position.y = float(self.map_origin_y)
        msg.info.origin.orientation.w = 1.0
        
        # Convertir a lista int plana
        data = self.occ_map.flatten().astype(int).tolist()
        msg.data = data
        self.map_vis_pub.publish(msg)


    def pose_callback(self, msg):
        self.current_pose = msg

    def world_to_grid(self, x_world, y_world):
        ix = int((x_world - self.map_origin_x) / self.map_res)
        iy = int((y_world - self.map_origin_y) / self.map_res)
        ix = max(0, min(self.cols - 1, ix))
        iy = max(0, min(self.rows - 1, iy))
        return np.array([iy, ix]) 

    def grid_to_world(self, iy, ix):
        x_world = ix * self.map_res + self.map_origin_x
        y_world = iy * self.map_res + self.map_origin_y
        return x_world, y_world

    def get_neighborhood(self, node):
        y, x = node
        neighbors = []
        # (dy, dx, costo)
        directions = [
            (0, 1, 1), (0, -1, 1), (1, 0, 1), (-1, 0, 1), 
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        for dy, dx, cost in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                # Consideramos transitable si el valor < 50
                if self.occ_map[ny, nx] < 50: 
                    neighbors.append(((ny, nx), cost))
        return neighbors

    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def plan_path(self, start, goal):
        if np.array_equal(start, goal): return None
        if self.occ_map[tuple(goal)] >= 50:
            self.get_logger().warn("¡Meta inválida! Cae en obstáculo/landmark.")
            return None

        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        
        came_from = {}
        g_score = {tuple(start): 0}
        
        path_found = False
        goal_tuple = tuple(goal)
        
        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal_tuple:
                path_found = True
                break

            for neighbor, move_cost in self.get_neighborhood(current):
                tentative_g = g_score[current] + move_cost
                
                # Penalización extra si está cerca de obstáculo (opcional)
                cell_val = self.occ_map[neighbor]
                penalty = (cell_val / 10.0) # Pequeña penalización por terreno 'sucio'
                
                if neighbor not in g_score or (tentative_g + penalty) < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g + penalty
                    f = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        if not path_found:
            return None

        # Reconstruir
        path = []
        curr = goal_tuple
        while curr in came_from:
            path.append(curr)
            curr = came_from[curr]
        path.append(tuple(start))
        path.reverse()
        return np.array(path)
    
    def goal_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("Esperando pose del robot...")
            return
        
        goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        start = self.world_to_grid(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        
        self.get_logger().info(f"Planificando ruta...")
        path = self.plan_path(start, goal)
        
        if path is not None:
            self.publish_path(path)
        else:
            self.get_logger().warn("No se encontró camino.")

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for cell in path:
            x, y = self.grid_to_world(int(cell[0]), int(cell[1]))
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Ruta publicada ({len(path)} puntos).")

def main():
    rclpy.init()
    node = PathPlannerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()