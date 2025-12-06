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
        self.frequency = 10.0  # Hz
        self.planner = ChannelPlanner()
        
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
        
        # --- CAMBIO AQUÍ: Suscripción a DeltaOdom en lugar de Odom ---
        self.sub_delta = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.sub_markers = self.create_subscription(MarkerArray, "/fastslam/markers", self.landmarks_callback, 10)

        # Publicadores
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_local_path = self.create_publisher(Path, "/local_plan_5d", 10)
        self.pub_local_goal = self.create_publisher(Point, "/local_goal_carrot", 10)

        self.timer = self.create_timer(1.0/self.frequency, self.control_loop)
        
        self.get_logger().info("Controlador 5D Iniciado (Usando DeltaOdom). Esperando ruta...")

    def path_callback(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.get_logger().info(f"Ruta global recibida: {len(self.global_path)} puntos")

    def pose_callback(self, msg):
        # Esta es la pose absoluta (x, y, theta) que viene del FastSLAM o Localization
        self.curr_x = msg.pose.position.x
        self.curr_y = msg.pose.position.y
        self.curr_theta = yaw_from_quat(msg.pose.orientation)
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
        
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = []
        
        for i in range(0, len(msg.ranges), 5):
            r = msg.ranges[i]
            if 0.1 < r < 3.0: 
                a = angles[i] + self.curr_theta
                ox = self.curr_x + r * np.cos(a)
                oy = self.curr_y + r * np.sin(a)
                points.append((ox, oy))
        
        self.scan_obstacles = points

    def landmarks_callback(self, msg):
        lm_points = []
        for marker in msg.markers:
            if marker.type == Marker.SPHERE: 
                lm_points.append((marker.pose.position.x, marker.pose.position.y))
        self.landmark_obstacles = lm_points

    # --- BUCLE DE CONTROL PRINCIPAL ---

    def control_loop(self):
        if not self.got_pose or not self.global_path:
            # self.get_logger().info(f"vel: ")

            return

        current_obstacles = getattr(self, 'scan_obstacles', []) + getattr(self, 'landmark_obstacles', [])

        # Estado 5D usando las velocidades calculadas desde DeltaOdom
        start_state = State5D(
            x=self.curr_x,
            y=self.curr_y,
            theta=self.curr_theta,
            v=self.curr_v,  
            w=self.curr_w
        )

        best_cmd = self.planner.create_channel(
            start_state, 
            self.global_path, 
            current_obstacles
        )
        self.get_logger().info(f"vel: {best_cmd}")

        self.pub_vel.publish(best_cmd)
        carrot = self.planner.get_target_point(start_state, self.global_path)
        if carrot:
            p = Point()
            p.x, p.y = carrot
            self.pub_local_goal.publish(p)

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