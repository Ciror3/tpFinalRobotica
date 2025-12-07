import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from custom_msgs.msg import DeltaOdom 
import numpy as np
import math

# IMPORTANTE: Asegúrate de que este import apunte a tu nuevo archivo
from tpFinalA.channel_planner import PurePursuit
from tpFinalA.d_star_lite import DStarLite  

def yaw_from_quat(q):
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class Controller5DNode(Node):
    def __init__(self):
        super().__init__("controller")

        self.frequency = 10.0
        
        # --- Configuración del Mapa ---
        # AJUSTAR RUTAS SEGÚN TU PC
        base_path = '/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas'
        pgm_path = base_path + '/mapa_final.pgm'
        json_path = base_path + '/mapa_landmarks_clasificados.json'
        
        # Inicializar D* Lite (Carga mapa estático + landmarks)
        self.dstar = DStarLite(pgm_path, json_path, origin_x=-10.0, origin_y=-10.0, res=0.05)
        
        # Inicializar Pure Pursuit
        self.planner = PurePursuit()
        
        # Estado Robot
        self.curr_x = 0.0; self.curr_y = 0.0; self.curr_theta = 0.0
        self.curr_v = 0.0; self.curr_w = 0.0  
        self.last_odom_time = self.get_clock().now()
        
        self.global_path = [] 
        self.scan_obstacles_local = []
        self.landmark_obstacles = []
        self.got_pose = False
        self.goal_received = False

        # --- Suscripciones ---
        self.sub_path = self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/amcl_pose", self.pose_callback, 10)
        self.sub_delta = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.sub_markers = self.create_subscription(MarkerArray, "/localization/markers", self.landmarks_callback, 10)

        # --- Publicadores ---
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/global_path", 10)
        self.inflated_pub = self.create_publisher(Marker, "/inflated_obstacles", 10)

        self.timer = self.create_timer(1.0/self.frequency, self.control_loop)
        
        self.get_logger().info("Controlador D* Lite + PurePursuit Iniciado.")

    def path_callback(self, msg):
        """ Recibe el objetivo global inicial (normalmente de RViz o planificador global) """
        if not msg.poses: return
        
        final_pose = msg.poses[-1].pose.position
        goal_world = (final_pose.x, final_pose.y)
        
        if self.got_pose:
            start_world = (self.curr_x, self.curr_y)
            self.dstar.set_goal(start_world, goal_world)
            
            # Generar primer camino
            path = self.dstar.get_path_world()
            if path:
                self.global_path = path
                self.publish_path(path)
                self.goal_received = True
                self.get_logger().info("Meta recibida. Navegación iniciada.")

    def pose_callback(self, msg):
        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_theta = yaw_from_quat(msg.pose.orientation)
        self.pose = msg
        self.got_pose = True

    def delta_odom_callback(self, msg: DeltaOdom):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_odom_time).nanoseconds / 1e9
        if dt > 0.0001:
            self.curr_v = msg.dt / dt
            self.curr_w = (msg.dr1 + msg.dr2) / dt
            self.last_odom_time = current_time

    def landmarks_callback(self, msg):
        lm_points = []
        for marker in msg.markers:
            if marker.type == Marker.SPHERE:
                lm_points.append((marker.pose.position.x, marker.pose.position.y))
        self.landmark_obstacles = lm_points

    # --- PROCESAMIENTO LÁSER (CORREGIDO CON TF) ---
    def scan_callback(self, msg):
        if not self.got_pose: return
        
        # CALIBRACIÓN EXACTA (Datos de tu TF)
        LASER_X_OFFSET = -0.032  # 3.2 cm atrás
        LASER_ROTATION_OFFSET = 0.0 
        FOV_LIMIT = np.deg2rad(90) # Ver solo al frente/lados

        local_points = []
        raw_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

        for i, r in enumerate(msg.ranges):
            if not (0.1 < r < 2.0): continue

            # A. Ángulo Local Normalizado
            local_angle = raw_angles[i] + LASER_ROTATION_OFFSET
            local_angle = (local_angle + np.pi) % (2 * np.pi) - np.pi

            if abs(local_angle) > FOV_LIMIT: continue

            # B. Polar -> Cartesiano (Frame Sensor)
            x_sensor = r * np.cos(local_angle)
            y_sensor = r * np.sin(local_angle)

            # C. Sensor -> Base Robot (Solo offset físico)
            x_base = x_sensor + LASER_X_OFFSET
            y_base = y_sensor 
            
            local_points.append((x_base, y_base))
        
        self.scan_obstacles_local = local_points

    def get_inflated_grid_set(self, radius_cells=1):
        """ 
        Calcula la grilla de obstáculos inflados para D* usando los puntos locales y la pose actual del robot.
        """
        inflated_set = set()
        
        # 1. Transformación Local -> Global (Just-in-Time)
        rob_cos = np.cos(self.curr_theta)
        rob_sin = np.sin(self.curr_theta)
        
        # Combinar obstáculos Lidar + Landmarks visuales
        # Nota: Landmarks ya vienen en global, lidar viene en local.
        
        # A. Procesar Lidar
        for lx, ly in self.scan_obstacles_local:
            gx = self.curr_x + (lx * rob_cos - ly * rob_sin)
            gy = self.curr_y + (lx * rob_sin + ly * rob_cos)
            
            iy, ix = self.dstar.world_to_grid(gx, gy)
            self._inflate_cell(iy, ix, radius_cells, inflated_set)

        # B. Procesar Landmarks (Ya globales)
        for gx, gy in self.landmark_obstacles:
            iy, ix = self.dstar.world_to_grid(gx, gy)
            self._inflate_cell(iy, ix, radius_cells, inflated_set)
            
        return inflated_set

    def _inflate_cell(self, iy, ix, radius, output_set):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy*dy + dx*dx <= radius*radius:
                    ny, nx = iy + dy, ix + dx
                    # Bounds check
                    if 0 <= ny < self.dstar.rows and 0 <= nx < self.dstar.cols:
                        output_set.add((ny, nx))

    def publish_viz_grid(self, grid_set):
        if not grid_set: return
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "dstar_inflation"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05; marker.scale.y = 0.05; marker.scale.z = 0.05
        marker.color.r = 0.6; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.5
        marker.pose.orientation.w = 1.0

        for iy, ix in grid_set:
            wx, wy = self.dstar.grid_to_world(iy, ix)
            p = Point(); p.x = float(wx); p.y = float(wy); p.z = 0.0
            marker.points.append(p)
        self.inflated_pub.publish(marker)

    def publish_path(self, path_list):
        if not path_list: return
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path_list:
            pose = PoseStamped()
            pose.pose.position.x = float(x); pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    # --- BUCLE PRINCIPAL ---
    def control_loop(self):
        if not self.got_pose or not self.goal_received:
            return

        # 1. Calcular obstáculos dinámicos actuales (inflados)
        current_obstacles_set = self.get_inflated_grid_set(radius_cells=3)
        
        # Visualizar lo que ve el robot (Debug)
        self.publish_viz_grid(current_obstacles_set)

        # 2. Actualizar D* Lite (El cerebro mágico)
        # Le pasamos dónde estamos y qué vemos. Él decide si recalcular.
        self.dstar.update_obstacles((self.curr_x, self.curr_y), current_obstacles_set)
        
        # 3. Obtener el mejor camino actual
        path = self.dstar.get_path_world()
        
        if path:
            # FIX: Grid Snapping (Saldar línea al centro del robot)
            if len(path) > 0:
                path[0] = (self.curr_x, self.curr_y)
            
            self.global_path = path
            self.publish_path(path)
            
            # 4. Pure Pursuit
            cmd = self.planner.compute_command(self.pose, self.global_path)
            self.pub_vel.publish(cmd)
        else:
            self.get_logger().error("D* Bloqueado: Deteniendo robot.")
            self.pub_vel.publish(Twist())

def main():
    rclpy.init()
    node = Controller5DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.pub_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()