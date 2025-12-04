import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from tpFinalA.channel_planner import ChannelPlanner, State5D 

def yaw_from_quat(q):
    """ Convierte cuaternión a Euler Yaw """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class Controller5DNode(Node):
    def __init__(self):
        super().__init__("controller_5d")

        # --- 1. CONFIGURACIÓN ---
        self.frequency = 10.0  # Hz (Ejecutar control 10 veces por segundo)
        self.planner = ChannelPlanner() # Instancia de tu lógica 5D
        
        # Estado actual del robot (5D)
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_theta = 0.0
        self.curr_v = 0.0  # Velocidad lineal actual
        self.curr_w = 0.0  # Velocidad angular actual

        self.global_path = [] # Lista de tuplas (x, y)
        self.obstacles = []   # Lista de tuplas (x, y)
        self.got_pose = False

        # --- 2. SUSCRIPCIONES ---
        
        # Path Global (del A* estático)
        self.sub_path = self.create_subscription(Path, "/planned_path", self.path_callback, 10)
        
        # Pose Estimada (FastSLAM/MCL - CRITICO: Usar esta, no odom cruda)
        self.sub_pose = self.create_subscription(PoseStamped, "/fpose", self.pose_callback, 10)
        
        # Velocidad Actual (Para saber la inercia). 
        # Si tu /fpose no trae velocidad, usamos /odom solo para sacar v y w.
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        
        # Obstáculos (Lidar + Landmarks)
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.sub_markers = self.create_subscription(MarkerArray, "/fastslam/markers", self.landmarks_callback, 10)

        # --- 3. PUBLICADORES ---
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Visualización: Publicamos la trayectoria local que el 5D está pensando
        self.pub_local_path = self.create_publisher(Path, "/local_plan_5d", 10)
        self.pub_local_goal = self.create_publisher(Point, "/local_goal_carrot", 10)

        # Timer de Control
        self.timer = self.create_timer(1.0/self.frequency, self.control_loop)
        
        self.get_logger().info("Controlador 5D Iniciado. Esperando ruta...")

    # --- CALLBACKS ---

    def path_callback(self, msg):
        # Convertimos a lista ligera de tuplas para el planner
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"Ruta global recibida: {len(self.global_path)} puntos")

    def pose_callback(self, msg):
        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_theta = yaw_from_quat(msg.pose.orientation)
        self.got_pose = True

    def odom_callback(self, msg):
        # Solo leemos velocidades (la posición la sacamos de /fpose que es más precisa)
        self.curr_v = msg.twist.twist.linear.x
        self.curr_w = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        # Convertir Lidar a obstáculos (x,y) globales simples para evasión
        # Nota: Para hacerlo eficiente, el planner 5D puede chequear colisión 
        # solo contra los puntos más cercanos. Aquí hacemos un filtro básico.
        if not self.got_pose: return
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = []
        
        # Downsampling para velocidad (tomar 1 de cada 5 rayos)
        for i in range(0, len(msg.ranges), 5):
            r = msg.ranges[i]
            if 0.1 < r < 3.0: # Ignorar ruido y puntos lejanos
                # Proyección local a global
                a = angles[i] + self.curr_theta
                ox = self.curr_x + r * np.cos(a)
                oy = self.curr_y + r * np.sin(a)
                points.append((ox, oy))
        
        # Guardamos en una lista temporal (se mezcla con landmarks luego)
        self.scan_obstacles = points

    def landmarks_callback(self, msg):
        lm_points = []
        for marker in msg.markers:
            if marker.type == Marker.SPHERE: # Centros de landmarks
                lm_points.append((marker.pose.position.x, marker.pose.position.y))
        self.landmark_obstacles = lm_points

    # --- BUCLE DE CONTROL PRINCIPAL ---

    def control_loop(self):
        # Seguridad: Si no hay pose o ruta, paramos
        if not self.got_pose or not self.global_path:
            return

        # 1. Fusionar obstáculos (Lidar + Landmarks)
        # (Si no has recibido scan aún, usa lista vacía)
        current_obstacles = getattr(self, 'scan_obstacles', []) + getattr(self, 'landmark_obstacles', [])

        # 2. Construir el Estado Inicial 5D
        # Aquí es donde le decimos al planner: "Partimos de esta situación física"
        start_state = State5D(
            x=self.curr_x,
            y=self.curr_y,
            theta=self.curr_theta,
            v=self.curr_v,  # Importante: Inercia actual
            w=self.curr_w
        )

        # 3. EJECUTAR PLANIFICADOR LOCAL
        # Buscamos la mejor acción inmediata
        best_cmd = self.planner.create_channel(
            start_state, 
            self.global_path, 
            current_obstacles
        )

        # 4. Publicar comando
        # self.get_logger().info(f"channel: {best_cmd} ")

        self.pub_vel.publish(best_cmd)

        # 5. Visualización (Opcional pero muy útil para debugging)
        # Queremos ver qué "Local Goal" eligió el planner
        carrot = self.planner.get_target_point(start_state, self.global_path)
        if carrot:
            p = Point()
            p.x, p.y = carrot
            self.pub_local_goal.publish(p)
        
        # Nota: Si quisieras ver la trayectoria curva completa (los puntitos verdes en RViz),
        # tendrías que modificar el planner para que devuelva la lista de nodos 'path'
        # en lugar de solo el comando 'Twist'.

def main():
    rclpy.init()
    node = Controller5DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Parada de emergencia al cerrar
        node.pub_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()