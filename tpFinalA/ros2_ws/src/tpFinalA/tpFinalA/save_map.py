import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
from numba import njit
import math
import os

@njit(fastmath=True)
def fast_grid_update(map_grid, x0, y0, x1_arr, y1_arr, lo_occ, lo_free, max_val, min_val):
    height, width = map_grid.shape
    for i in range(len(x1_arr)):
        r_x, r_y = x0, y0
        end_x, end_y = x1_arr[i], y1_arr[i]
        
        dx = abs(end_x - r_x); dy = abs(end_y - r_y)
        sx = 1 if r_x < end_x else -1; sy = 1 if r_y < end_y else -1
        err = dx - dy
        
        while True:
            if 0 <= r_x < width and 0 <= r_y < height:
                if r_x == end_x and r_y == end_y:
                    val = map_grid[r_y, r_x] + lo_occ
                    if val > max_val: val = max_val
                    map_grid[r_y, r_x] = val
                    break
                else:
                    val = map_grid[r_y, r_x] + lo_free
                    if val < min_val: val = min_val
                    map_grid[r_y, r_x] = val
            else:
                break
            if r_x == end_x and r_y == end_y: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; r_x += sx
            if e2 < dx: err += dx; r_y += sy
    return map_grid

class GridMapper:
    def __init__(self, width_m=20.0, height_m=20.0, resolution=0.05):
        self.res = resolution
        self.w = int(width_m / resolution)
        self.h = int(height_m / resolution)
        self.origin_x = -width_m / 2.0
        self.origin_y = -height_m / 2.0
        
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)
        
        self.LO_OCC = 0.85
        self.LO_FREE = -0.4
        self.MAX = 5.0
        self.MIN = -5.0

    def update(self, rx, ry, rth, ranges, angle_min, angle_inc, r_min, r_max):
        # 1. Filtros
        valid = (ranges >= r_min) & (ranges <= r_max) & (~np.isnan(ranges))
        dists = ranges[valid]
        angles = rth + angle_min + np.where(valid)[0] * angle_inc
        
        # 2. Coordenadas Mundo
        hit_wx = rx + dists * np.cos(angles)
        hit_wy = ry + dists * np.sin(angles)
        
        # 3. Coordenadas Grid
        robot_gx = int((rx - self.origin_x) / self.res)
        robot_gy = int((ry - self.origin_y) / self.res)
        
        hits_gx = ((hit_wx - self.origin_x) / self.res).astype(np.int32)
        hits_gy = ((hit_wy - self.origin_y) / self.res).astype(np.int32)
        
        # 4. Numba Update
        self.grid = fast_grid_update(
            self.grid, robot_gx, robot_gy, hits_gx, hits_gy,
            self.LO_OCC, self.LO_FREE, self.MAX, self.MIN
        )

    def get_msg(self, header):
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = self.res
        msg.info.width = self.w
        msg.info.height = self.h
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        
        # Convertir Log-Odds a Probabilidad (0-100)
        # Optimizacion visual:
        probs = np.full(self.grid.shape, -1, dtype=np.int8) # Desconocido
        probs[self.grid > 0.5] = 100 # Ocupado
        probs[self.grid < -0.5] = 0  # Libre
        
        msg.data = probs.flatten().tolist()
        return msg

    def save_map_to_file(self, file_base="mi_mapa_grid"):
        try:
            # Generar PGM
            pgm_data = np.full(self.grid.shape, 205, dtype=np.uint8) # 205=Gris
            pgm_data[self.grid < -0.5] = 254 # Blanco (Libre)
            pgm_data[self.grid > 0.5] = 0    # Negro (Ocupado)
            pgm_data = np.flipud(pgm_data)   # Flip para visualizadores de imagen

            filename_pgm = f"{file_base}.pgm"
            with open(filename_pgm, 'wb') as f:
                header = f"P5\n{self.w} {self.h}\n255\n".encode()
                f.write(header)
                f.write(pgm_data.tobytes())

            # Generar YAML
            filename_yaml = f"{file_base}.yaml"
            with open(filename_yaml, 'w') as f:
                f.write(f"image: {filename_pgm}\n")
                f.write(f"resolution: {self.res}\n")
                f.write(f"origin: [{self.origin_x}, {self.origin_y}, 0.0]\n")
                f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

            return True, f"{filename_pgm}"
        except Exception as e:
            return False, str(e)

def get_yaw_from_pose(pose):
    q = pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

# ==============================================================================
# 3. NODO ROS 2
# ==============================================================================
class GridMapperNode(Node):
    def __init__(self):
        super().__init__('grid_mapper_node')
        
        # Configuración del mapa (20x20 metros, 5cm px)
        self.mapper = GridMapper(width_m=20.0, height_m=20.0, resolution=0.05)
        
        self.get_logger().info("Grid Mapper Iniciado. Esperando /scan y /amcl_pose...")

        # Suscripción Sincronizada
        self.scan_sub = Subscriber(self, LaserScan, "/scan")
        self.pose_sub = Subscriber(self, PoseStamped, "/amcl_pose")

        # ApproximateTimeSynchronizer: Crucial para alinear láser con posición
        # slop=0.1 significa que permite hasta 100ms de diferencia entre mensajes
        self.ts = ApproximateTimeSynchronizer([self.pose_sub, self.scan_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10)

    def sync_callback(self, pose_msg, scan_msg):
        # 1. Extraer Pose
        rx = pose_msg.pose.position.x
        ry = pose_msg.pose.position.y
        rth = get_yaw_from_pose(pose_msg.pose)

        # 2. Actualizar Mapa
        self.mapper.update(
            rx, ry, rth,
            np.array(scan_msg.ranges),
            scan_msg.angle_min,
            scan_msg.angle_increment,
            scan_msg.range_min,
            scan_msg.range_max
        )

        # 3. Publicar
        # Usamos el timestamp del laser para consistencia
        msg = self.mapper.get_msg(scan_msg.header)
        self.map_pub.publish(msg)

    def save(self):
        self.get_logger().info("Guardando mapa de ocupación...")
        ok, name = self.mapper.save_map_to_file("mapa_final")
        if ok:
            self.get_logger().info(f"¡Mapa guardado!: {os.getcwd()}/{name}")
        else:
            self.get_logger().error(f"Error guardando: {name}")

def main(args=None):
    rclpy.init(args=args)
    node = GridMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # --- AQUÍ SE GUARDA AL PRESIONAR CTRL+C ---
        node.save()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()