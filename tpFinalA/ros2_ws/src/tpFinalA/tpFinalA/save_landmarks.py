import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
import json
import os

class LandmarkSaverNode(Node):
    def __init__(self):
        super().__init__("landmark_saver_node")
        
        # Suscripción solo a los marcadores
        self.marker_sub = self.create_subscription(
            MarkerArray, 
            "/fastslam/markers", 
            self.marker_callback, 
            10
        )
        
        self.latest_landmarks = {} # Usamos dict para evitar duplicados por ID
        self.landmarks_received = False
        
        self.get_logger().info("Landmark Saver Iniciado. Esperando datos...")
        self.get_logger().info("Presiona Ctrl+C para guardar el JSON.")

    def marker_callback(self, msg):
        """ 
        Extrae landmarks. 
        Asume que marker.ns contiene el tipo ('segment' o 'cluster').
        """
        count = 0
        for marker in msg.markers:
            # Filtramos: Solo esferas (centros) y acción ADD (0)
            if marker.type == Marker.SPHERE and marker.action == Marker.ADD:
                
                # Clasificar según el namespace enviado por el SLAM
                lm_type = marker.ns if marker.ns else "unknown"
                
                lm_data = {
                    "id": marker.id,
                    "x": round(marker.pose.position.x, 4),
                    "y": round(marker.pose.position.y, 4),
                    "type": lm_type  # Aquí guardamos si es segmento o cluster
                }
                
                # Guardamos en diccionario usando ID como clave para actualizar
                self.latest_landmarks[marker.id] = lm_data
                count += 1
        
        if count > 0:
            self.landmarks_received = True
            # Loguear solo de vez en cuando para no saturar
            # self.get_logger().info(f"Recibidos {len(self.latest_landmarks)} landmarks únicos.")

    def save_data(self):
        if not self.landmarks_received:
            self.get_logger().warn("No se han recibido landmarks. Archivo vacío.")
            return

        filename = "mapa_landmarks_clasificados.json"
        self.get_logger().info(f"Guardando {len(self.latest_landmarks)} landmarks en {filename}...")
        
        # Convertir diccionario a lista para el JSON
        lista_final = list(self.latest_landmarks.values())
        
        try:
            with open(filename, "w") as f:
                json.dump(lista_final, f, indent=4)
            self.get_logger().info("¡Guardado exitoso!")
        except Exception as e:
            self.get_logger().error(f"Error guardando archivo: {e}")

def main():
    rclpy.init()
    node = LandmarkSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()