import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path
# from nav_msgs.msg import Odometry  <-- ELIMINADO
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from custom_msgs.msg import DeltaOdom # <-- AGREGADO
import numpy as np
import math
from tpFinalA.channel_planner import PurePursuit
from tpFinalA.path_planning import PathPlannerNode

def yaw_from_quat(q):
    """ Convierte cuaternión a Euler Yaw """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class Controller5DNode(Node):
    def __init__(self):
        super().__init__("controller")

        # --- 1. CONFIGURACIÓN ---
        self.frequency = 10.0  # Hz
        self.planner = PurePursuit()
        self.astar_planner = PathPlannerNode()
        
        # Estado actual del robot (5D)
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_theta = 0.0
        self.curr_v = 0.0  
        self.curr_w = 0.0  

        # Variables para calcular velocidad desde DeltaOdom
        self.last_odom_time = self.get_clock().now()

        self.global_path = [] 
        self.obstacles = []   
        self.got_pose = False

        # Suscripciones
        self.sub_path = self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        self.sub_pose = self.create_subscription(PoseStamped, "/amcl_pose", self.pose_callback, 10) # Tu pose estimada del SLAM
        # En el __init__ de tu nodo principal
        self.inflated_pub = self.create_publisher(Marker, "/inflated_obstacles", 10)
        # --- CAMBIO AQUÍ: Suscripción a DeltaOdom en lugar de Odom ---
        self.sub_delta = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.sub_markers = self.create_subscription(MarkerArray, "/fastslam/markers", self.landmarks_callback, 10)

        # Publicadores
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/global_path", 10)
        self.pub_local_goal = self.create_publisher(Point, "/local_goal_carrot", 10)

        self.timer = self.create_timer(1.0/self.frequency, self.control_loop)
        self.state = "movement"
        
        self.get_logger().info("Controlador 5D Iniciado (Usando DeltaOdom). Esperando ruta...")

    def path_callback(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"Ruta global recibida: {len(self.global_path)} puntos")

    def pose_callback(self, msg):
        # Esta es la pose absoluta (x, y, theta) que viene del FastSLAM o Localization
        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_theta = yaw_from_quat(msg.pose.orientation)
        self.pose = msg
        self.got_pose = True

    def delta_odom_callback(self, msg: DeltaOdom):
        """
        Calcula la velocidad actual (v, w) basándose en el incremento de posición
        y el tiempo transcurrido entre mensajes.
        """
        current_time = self.get_clock().now()
        dt_seconds = (current_time - self.last_odom_time).nanoseconds / 1e9
        
        if dt_seconds > 0.0001:
            self.curr_v = msg.dt / dt_seconds
            
            self.curr_w = (msg.dr1 + msg.dr2) / dt_seconds
            self.last_odom_time = current_time

    def scan_callback(self, msg):
        if not self.got_pose: return
        
        # --- 1. CALIBRACIÓN EXACTA (SEGÚN TF) ---
        LASER_X_OFFSET = -0.032  # Negativo porque está atrás
        LASER_ROTATION_OFFSET = 0.0 # Apunta al frente (0 grados)
        
        points = []
        
        # Pre-calcular trigonometría del ROBOT (Global)
        rob_cos = np.cos(self.curr_theta)
        rob_sin = np.sin(self.curr_theta)

        for i, r in enumerate(msg.ranges):
            
            if not (0.1 < r < 2.0):
                continue

            # --- A. ÁNGULO ---
            sensor_angle = msg.angle_min + i * msg.angle_increment
            
            # Aplicar rotación (que es 0.0, así que no cambia nada, pero lo dejamos por rigor)
            final_local_angle = sensor_angle + LASER_ROTATION_OFFSET
            
            # Normalizar
            final_local_angle = (final_local_angle + np.pi) % (2 * np.pi) - np.pi

            # --- FILTRO FOV (Opcional) ---
            if abs(final_local_angle) > np.deg2rad(70): 
                continue

            # --- B. POLAR -> CARTESIANO (LOCAL ROBOT) ---
            # x_local apunta al frente
            x_local = r * np.cos(final_local_angle)
            y_local = r * np.sin(final_local_angle)

            # --- C. SUMAR OFFSET DE POSICIÓN (AQUÍ ESTÁ LA MAGIA) ---
            # Al sumar un número negativo (-0.032), estamos moviendo el origen del rayo hacia atrás
            x_base = x_local + LASER_X_OFFSET
            y_base = y_local 

            # --- D. LOCAL -> GLOBAL (MAPA) ---
            # Rotación estándar 2D
            x_global = (x_base * rob_cos - y_base * rob_sin) + self.curr_x
            y_global = (x_base * rob_sin + y_base * rob_cos) + self.curr_y
            
            points.append((x_global, y_global))
        
        self.scan_obstacles = points
    
    def landmarks_callback(self, msg):
        lm_points = []
        for marker in msg.markers:
            if marker.type == Marker.SPHERE: 
                lm_points.append((marker.pose.position.x, marker.pose.position.y))
        self.landmark_obstacles = lm_points

    def is_path_blocked(self, dynamic_obstacles, safety_radius=0.45):
        """
        Revisa si algun punto del camino futuro choca con los nuevos obstaculos.
        """
        if self.global_path is None: return

        horizon = min(len(self.global_path),20)
        for i in range(horizon):
            px,py = self.global_path[i]
            for ox,oy in dynamic_obstacles:
                dist =  np.hypot(px-ox,py-oy)
                if dist < safety_radius:
                    return True
        return False

    def filter_dynamic_obstacles(self, raw_obstacles):
        """
        Filtra los puntos del láser.
        - Si caen en una celda que YA es obstáculo en el mapa estático -> Lo ignora.
        - Si caen en una celda LIBRE del mapa estático -> Es un obstáculo nuevo (dinámico).
        """
        real_dynamic_obstacles = []
        
        for ox, oy in raw_obstacles:
            iy, ix = self.astar_planner.world_to_grid(ox, oy)
            
            if 0 <= iy < self.astar_planner.rows and 0 <= ix < self.astar_planner.cols:
                if self.astar_planner.occ_map[iy, ix] < 50:
                    real_dynamic_obstacles.append((ox, oy))
            else:
                real_dynamic_obstacles.append((ox, oy))
                
        return real_dynamic_obstacles

    def control_loop(self):
        if not self.got_pose or not self.global_path:
            return

        raw_scan_obstacles = getattr(self, 'scan_obstacles', [])
        dynamic_obstacles = self.filter_dynamic_obstacles(raw_scan_obstacles)
        
        all_dynamic_obstacles = dynamic_obstacles + getattr(self, 'landmark_obstacles', [])

        if (self.is_path_blocked( all_dynamic_obstacles)):
            self.state = "safety"
            self.get_logger().warn("¡Objeto NUEVO en el camino! Recalculando A*...")
            
            # Frenar por seguridad
            stop_msg = Twist()
            self.pub_vel.publish(stop_msg)
            
            start = (self.curr_x, self.curr_y)
            goal = self.global_path[-1] 
            
            start_grid = self.astar_planner.world_to_grid(start[0], start[1])
            goal_grid = self.astar_planner.world_to_grid(goal[0], goal[1])
            
            dyn_obs_grid = []
            INFLATION_RADIUS = 2 

            dyn_obs_grid = set() 
            for ox, oy in all_dynamic_obstacles:
                center_y, center_x = self.astar_planner.world_to_grid(oy, ox)          
                for dy in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                    for dx in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                        if dy*dy + dx*dx <= INFLATION_RADIUS*INFLATION_RADIUS:
                            dyn_obs_grid.add((center_y + dy, center_x + dx))
            self.publish_inflated_grid(dyn_obs_grid)
            path_grid = self.astar_planner.astar(start_grid, goal_grid, dyn_obs_grid)
            
            if path_grid:
                new_path_world = []
                for iy, ix in path_grid:
                    wx, wy = self.astar_planner.grid_to_world(iy, ix)
                    new_path_world.append((wx, wy))
                new_path_world[0] = (self.curr_x, self.curr_y)
                
                self.global_path = new_path_world
                self.publish_path(self.global_path )
                self.get_logger().info("¡Nuevo camino calculado con éxito!")
            # else:
            #     self.get_logger().error("A* falló: No hay ruta alternativa. Robot detenido.")
            #     return

        cmd = self.planner.compute_command(self.pose, self.global_path)
        
        self.pub_vel.publish(cmd)

    def publish_inflated_grid(self, grid_set):
        if not grid_set:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "inflation"
        marker.id = 999
        marker.type = Marker.POINTS  # Usamos POINTS para eficiencia
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        
        # Escala: Que coincida con la resolución de tu mapa (ej. 0.05m)
        # Si tienes guardada la resolución en astar_planner, úsala:
        res = getattr(self.astar_planner, 'resolution', 0.05)
        marker.scale.x = res
        marker.scale.y = res
        marker.scale.z = 0.05 # Altura visual

        # Color: Violeta semitransparente para diferenciar de la pared real
        marker.color.r = 0.6
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.6

        # Convertir cada celda de la grilla a punto en el mundo
        for iy, ix in grid_set:
            wx, wy = self.astar_planner.grid_to_world(iy, ix)
            p = Point()
            p.x = float(wx)
            p.y = float(wy)
            p.z = 0.0
            marker.points.append(p)

        self.inflated_pub.publish(marker)

    def publish_path(self, path_list):
        if not path_list:
            return

        msg = Path()
        msg.header.frame_id = "map" # O el frame que estés usando
        msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path_list:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0 # Orientación neutra
            msg.poses.append(pose)
            
        self.path_pub.publish(msg)

def main():
    rclpy.init()
    node = Controller5DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()