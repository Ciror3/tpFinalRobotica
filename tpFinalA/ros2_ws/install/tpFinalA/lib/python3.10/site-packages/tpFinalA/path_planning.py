import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
import numpy as np
import heapq
import json
import os
import cv2 

class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner")

        # --- 1. CONFIGURACIÓN DEL MAPA (Basado en tu YAML) ---
        # Ajusta la ruta a tu carpeta
        self.base_path = '/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas'
        self.map_pgm_path = self.base_path + '/mapa_final.pgm'
        self.landmarks_json_path = self.base_path + '/mapa_landmarks_clasificados.json'
        
        # PARÁMETROS DEL YAML (Copiados de tu input)
        self.map_res = 0.05        # 5 cm por pixel
        self.map_origin_x = -10.0  # Metros
        self.map_origin_y = -10.0  # Metros
        
        # Cargar mapa PGM
        self.occ_map = self.load_pgm_map(self.map_pgm_path)
        self.rows, self.cols = self.occ_map.shape
        self.map_width = self.cols
        self.map_height = self.rows

        # --- 2. FUSIONAR LANDMARKS ---
        self.load_and_inflate_landmarks(self.landmarks_json_path)

        # Suscriptores y Publicadores
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, "/amcl_pose", self.pose_callback, 10)

        self.path_pub = self.create_publisher(Path, "/planned_path", 10)
        self.map_vis_pub = self.create_publisher(OccupancyGrid, "/planning_map", 10)
        # Timer visualización
        self.create_timer(1.0, self.publish_visual_map)

        self.current_pose = None
        self.get_logger().info(f"Path Planner Listo. Mapa: {self.cols}x{self.rows} px, Res: {self.map_res}")

    def load_pgm_map(self, filepath):
        """ Carga la imagen PGM y la convierte a matriz de costos 0-100 """
        if not os.path.exists(filepath):
            self.get_logger().error(f"¡No se encuentra el mapa PGM!: {filepath}")
            # Retornar mapa vacío de seguridad (20x20m / 0.05 = 400px)
            return np.zeros((400, 400), dtype=int)

        try:
            # Leer imagen en escala de grises
            # OpenCV carga [fila, col] -> [y, x]
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                raise ValueError("cv2.imread devolvió None. Formato inválido.")

            # --- CONVERSIÓN DE VALORES ---
            # En PGM generado por GridMapper:
            # 0 (Negro) = Ocupado -> Costo 100
            # 254 (Blanco) = Libre -> Costo 0
            # 205 (Gris) = Desconocido -> Costo 100 (evitar zonas desconocidas) o 50
            
            grid = np.zeros_like(img, dtype=int)
            
            # Umbralización para A*
            # Tratamos como obstáculo (100) todo lo que no sea espacio libre seguro
            grid[img < 250] = 100  # Paredes y desconocidos son obstáculos
            grid[img >= 250] = 0   # Espacio libre (blanco) es caminable
            

            grid = np.flipud(grid)

            self.get_logger().info("Mapa PGM cargado y procesado.")
            return grid

        except Exception as e:
            self.get_logger().error(f"Error cargando PGM: {e}")
            return np.zeros((400, 400), dtype=int)

    def load_and_inflate_landmarks(self, filepath):
        """ Lee JSON e inyecta obstáculos (Landmarks) en el mapa """
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath, 'r') as f:
                landmarks = json.load(f)
            
            radius_px = 5
            
            for lm in landmarks:
                lx, ly = lm['x'], lm['y']
                gy, gx = self.world_to_grid(lx, ly)
                
                for dy in range(-radius_px, radius_px + 1):
                    for dx in range(-radius_px, radius_px + 1):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            self.occ_map[ny, nx] = 100 
                            
            self.get_logger().info(f"Landmarks inyectados ({len(landmarks)}).")
        except Exception as e:
            self.get_logger().error(f"Error JSON: {e}")

    def pose_callback(self, msg):
        self.current_pose = msg

    def goal_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("Esperando pose del robot...")
            return
        
        start_world = (self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        goal_world = (msg.pose.position.x, msg.pose.position.y)
        
        start_grid = self.world_to_grid(*start_world)
        goal_grid = self.world_to_grid(*goal_world)
        
        if not (0 <= start_grid[0] < self.rows and 0 <= start_grid[1] < self.cols):
            self.get_logger().warn("El robot está FUERA del mapa.")
            return

        self.get_logger().info(f"Planificando: {start_grid} -> {goal_grid}")
        path_grid = self.astar(start_grid, goal_grid)
        
        if path_grid is not None:
            self.publish_path(path_grid)
        else:
            self.get_logger().warn("No se encontró camino.")

    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x) / self.map_res)
        gy = int((y - self.map_origin_y) / self.map_res)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))
        return (gy, gx)

    def grid_to_world(self, gy, gx):
        wx = (gx * self.map_res) + self.map_origin_x
        wy = (gy * self.map_res) + self.map_origin_y
        return wx, wy

    def astar(self, start, goal, dynamic_obstacles=[]):
        dynamic_obs =  set(tuple(p) for p in dynamic_obstacles)
        if self.occ_map[goal] >= 50:
            self.get_logger().warn("Meta en obstáculo.")
            return None

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, cost in self.get_neighbors(current,dynamic_obs):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        return None

    def get_neighbors(self, node, dynamic_obstacles):
        y, x = node
        neighbors = []
        moves = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,1.41), (1,-1,1.41), (-1,1,1.41), (-1,-1,1.41)]
        
        for dy, dx, cost in moves:
            ny, nx = y + dy, x + dx
            if self.occ_map[ny, nx] >= 50:
                    continue
            
            if (ny, nx) in dynamic_obstacles:
                continue
                
            neighbors.append(((ny, nx), cost))
        return neighbors

    def heuristic(self, a, b):
        D = 1
        D2 = np.sqrt(2)
        cy,cx = a
        gy,gx = b
        dx = abs(cx - gx)
        dy = abs(cy - gy)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def publish_path(self, grid_path):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for (gy, gx) in grid_path:
            wx, wy = self.grid_to_world(gy, gx)
            pose = PoseStamped()
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def publish_visual_map(self):
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.map_res
        msg.info.width = self.cols
        msg.info.height = self.rows
        msg.info.origin.position.x = float(self.map_origin_x)
        msg.info.origin.position.y = float(self.map_origin_y)
        
        data = self.occ_map.flatten().astype(np.int8)
        msg.data = data.tolist()
        self.map_vis_pub.publish(msg)

def main():
    rclpy.init()
    node = PathPlannerNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()
