import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import json
from visualization_msgs.msg import MarkerArray


class mapSaverNode(Node):
    def __init__(self):
        super().__init__("full_saver_node")
        
        # Suscripciones
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.marker_sub = self.create_subscription(MarkerArray, "/fastslam/markers", self.marker_callback, 10)
        
        self.latest_map_msg = None
        self.latest_landmarks = {}
        self.map_received = False
        
        self.get_logger().info("Nodo Saver Iniciado. Presiona Ctrl+C para guardar y salir.")

    def map_callback(self, msg):
        self.latest_map_msg = msg
        self.map_received = True

    def marker_callback(self, msg):
        for marker in msg.markers:
            if marker.type == 2: # SPHERE
                
                # El tipo viene directo en el namespace gracias a slam.py
                lm_type_str = marker.ns 
                
                real_id = marker.id 
                
                self.latest_landmarks[real_id] = {
                    "id": real_id,
                    "x": marker.pose.position.x,
                    "y": marker.pose.position.y,
                    "type": lm_type_str  # Guardamos "segment" o "cluster"
                }

    def save_data(self):
        if not self.map_received:
            self.get_logger().warn("No se recibió mapa. Nada que guardar.")
            return

        self.get_logger().info("Guardando datos...")
        
        # 1. GUARDAR MAPA (Formato Matriz TXT)
        msg = self.latest_map_msg
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Convertir y voltear (Flip) para coordenadas correctas
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        data = np.flipud(data)
        
        # Guardamos datos de configuración en el header del txt para no perderlos
        header = f"{res},{origin_x},{origin_y},{width},{height}"
        np.savetxt("mi_mapa_grid.txt", data, fmt='%d', header=header)
        
        # 2. GUARDAR LANDMARKS (Formato JSON)
        with open("mi_mapa_landmarks.json", "w") as f:
            json.dump(self.latest_landmarks, f, indent=4)

        self.get_logger().info(f"¡Éxito! Guardados: mi_mapa_grid.txt y mi_mapa_landmarks.json")
        self.get_logger().info(f"Landmarks guardados: {len(self.latest_landmarks)}")

def main():
    rclpy.init()
    node = mapSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Al presionar Ctrl+C, guardamos antes de morir
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()