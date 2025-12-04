import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from custom_msgs.msg import DeltaOdom
from tf2_ros import TransformBroadcaster
import math
import json
import os
import threading

# --- UTILIDADES ---
def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q

def wrap(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

class LineSegment:
    def __init__(self, m, n, p0, v, e1=None, e2=None):
        self.m, self.n = m, n
        self.p0, self.v = p0, v
        self.e1, self.e2 = e1, e2

class Particle():
    def __init__(self, x=0, y=0, theta=0, weight=1.0):
        self.x = x
        self.y = y
        self.orientation = theta
        self.weight = weight

    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, alpha):
        """ Modelo de movimiento odometrico """
        dist  = odom[0]       
        delta_rot1  = odom[1]
        delta_rot2 = odom[2]

        # Ruido dependiente del movimiento
        varDeltaRot1 = alpha[0]*np.abs(delta_rot1) + alpha[1]*dist
        varDeltaRot2 = alpha[0]*np.abs(delta_rot2) + alpha[1]*dist
        varDeltaTrans = alpha[3]*(np.abs(delta_rot1)+np.abs(delta_rot2)) + alpha[2]*dist

        deltaRot1 = delta_rot1 + np.random.normal(0, np.sqrt(varDeltaRot1 + 1e-9))
        deltaRot2 = delta_rot2 + np.random.normal(0, np.sqrt(varDeltaRot2 + 1e-9))
        deltaRotT = dist + np.random.normal(0, np.sqrt(varDeltaTrans + 1e-9))

        self.x += deltaRotT * np.cos(self.orientation + deltaRot1)
        self.y += deltaRotT * np.sin(self.orientation + deltaRot1)
        self.orientation = wrap(self.orientation + deltaRot1 + deltaRot2)

    def calculate_weight(self, observed_landmarks, map_landmarks, R_cov):
        """
        Compara los landmarks observados (lista de [r, theta]) 
        con el mapa conocido (diccionario) para actualizar el peso.
        """
        total_prob = 1.0
        
        # Si no vemos nada, el peso se mantiene (o decae ligeramente)
        if len(observed_landmarks) == 0:
            return 1.0

        # Para cada observación, buscamos el landmark del mapa más cercano (Data Association)
        # Nota: En localización pura, asumimos que la posición de los landmarks del mapa es exacta.
        
        inv_R = np.linalg.inv(R_cov)
        det_R = np.linalg.det(R_cov)
        norm_const = 1.0 / (2.0 * np.pi * math.sqrt(det_R))

        for obs in observed_landmarks:
            r_obs, th_obs = obs
            
            # Convertir observación local a coordenada global aproximada
            # basada en la posición actual de la partícula
            a_global = self.orientation + th_obs
            lx_obs = self.x + r_obs * np.cos(a_global)
            ly_obs = self.y + r_obs * np.sin(a_global)

            # Buscar vecino más cercano en el mapa (Nearest Neighbor)
            best_dist = float('inf')
            best_nu = None

            for lid, lm in map_landmarks.items():
                mx, my = lm['x'], lm['y']
                dist = (lx_obs - mx)**2 + (ly_obs - my)**2
                
                if dist < best_dist:
                    best_dist = dist
                    # Calculamos el error en el espacio de medida (r, theta)
                    # Predicción esperada si la partícula estuviera correcta:
                    dx = mx - self.x
                    dy = my - self.y
                    q = dx*dx + dy*dy
                    r_pred = math.sqrt(q)
                    th_pred = wrap(math.atan2(dy, dx) - self.orientation)
                    
                    nu = np.array([r_obs - r_pred, wrap(th_obs - th_pred)])
                    best_nu = nu

            # Umbral de asociación: Si el landmark está muy lejos (ej. > 1 metro), es "ruido" o un obstáculo nuevo
            if best_dist < 1.0 and best_nu is not None:
                # Gaussiana multivariada
                exponent = -0.5 * (best_nu.T @ inv_R @ best_nu)
                prob = norm_const * np.exp(exponent)
                total_prob *= prob
            else:
                # Penalización suave por característica no explicada
                total_prob *= 0.8

        return total_prob

    def copy(self):
        return Particle(self.x, self.y, self.orientation, self.weight)

class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")

        # --- CONFIGURACIÓN ---
        # RUTA AL MAPA JSON (AJUSTAR ESTA RUTA)
        self.landmarks_file = '/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas/mi_mapa_landmarks_test.json'
        
        self.num_particles = 200 # Más partículas ayudan a recuperar la posición al inicio
        self.R = np.diag([0.05, 0.03]) # Ruido de medida (r, theta)
        self.noise_odom = [0.02, 0.02, 0.01, 0.01] # Ruido movimiento

        # Subscripciones y Publicadores
        self.delta_sub = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseStamped, "/locpose", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "/localization/particles", 10)
        self.map_vis_pub = self.create_publisher(MarkerArray, "/localization/map_landmarks", 10) # Visualizar el mapa cargado
        self.segments_pub = self.create_publisher(MarkerArray, "extracted_segments", 10)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.lock = threading.Lock()

        # Cargar Mapa
        self.map_landmarks = self.load_landmarks()
        
        # Inicializar Partículas
        # Asumimos inicio en (0,0) o cerca. Si quieres "Secuestro", aumenta el ruido inicial.
        self.particles = [Particle(0,0,0) for _ in range(self.num_particles)]

        # Parámetros Extracción Características (Idéntico a SLAM)
        self.Pmin = 18; self.Lmin = 0.34; self.eps = 0.03; self.Snum = 8; self.delta = 0.1
        self.corner_thresh = 0.5; self.angle_thresh = np.deg2rad(30)

        # Publicar el mapa estático una vez al principio para verlo en RViz
        self.publish_static_map()
        self.get_logger().info(f"Localización Iniciada con {len(self.map_landmarks)} landmarks conocidos.")

    def load_landmarks(self):
        if not os.path.exists(self.landmarks_file):
            self.get_logger().error(f"NO SE ENCONTRÓ EL MAPA EN: {self.landmarks_file}")
            return {}
        
        try:
            with open(self.landmarks_file, 'r') as f:
                data = json.load(f)
            
            landmarks = {}
            # Manejar formato lista o dict
            iterable = data.values() if isinstance(data, dict) else data
            for item in iterable:
                lid = item.get('id', len(landmarks))
                # Soporte para claves 'mu' o 'x'/'y'
                if 'mu' in item:
                    x, y = item['mu'][0], item['mu'][1]
                else:
                    x, y = item['x'], item['y']
                landmarks[lid] = {'x': x, 'y': y}
            return landmarks
        except Exception as e:
            self.get_logger().error(f"Error parseando JSON: {e}")
            return {}

    def delta_odom_callback(self, msg: DeltaOdom):
        with self.lock:
            d = [msg.dt, msg.dr1, msg.dr2]
            for p in self.particles:
                p.move_odom(d, self.noise_odom)

    def scan_callback(self, data):
        with self.lock:
            # 1. Extracción de Características (Landmarks Observados)
            ranges = np.array(data.ranges)
            angle_min = data.angle_min
            angle_inc = data.angle_increment
            valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
            indices = np.where(valid)[0]
            if len(indices) < self.Pmin: return
            
            ranges = ranges[indices]
            angles = angle_min + indices * angle_inc
            xs = ranges * np.cos(angles)
            ys = ranges * np.sin(angles)
            points = np.stack([xs, ys], axis=1)

            # Extraer segmentos y puntos
            segments, used_mask = self.extract_segments(points, angles)
            
            # (Opcional) Publicar segmentos para ver qué ve el robot
            self.publish_segments_vis(data, segments)

            corner_landmarks = self.segments_to_landmarks(segments)
            point_landmarks = self.extract_point_landmarks(points, used_mask, segments)

            # Juntar todo lo observado
            observed = []
            if len(corner_landmarks) > 0: observed.append(corner_landmarks)
            if len(point_landmarks) > 0: observed.append(point_landmarks)
            
            if len(observed) > 0:
                observed = np.vstack(observed)
            else:
                observed = np.empty((0, 2))

            # 2. Actualización MCL (Pesos)
            if len(observed) > 0 and len(self.map_landmarks) > 0:
                for p in self.particles:
                    w = p.calculate_weight(observed, self.map_landmarks, self.R)
                    p.weight *= w

            # 3. Resampling
            self.normalize_and_resample()

            # 4. Publicar Resultado
            best_particle = max(self.particles, key=lambda p: p.weight)
            self.publish_tf(best_particle)
            self.publish_pose(best_particle)
            # self.publish_particle_cloud() # Descomentar si quieres ver la nube (consume recursos)

    # --- LÓGICA DE EXTRACCIÓN (REUTILIZADA DE SLAM.PY) ---
    def extract_segments(self, points, angles):
        segments = []
        N = len(points)
        used_mask = np.zeros(N, dtype=bool) 
        start = 0
        while start < N - self.Pmin:
            res = self.seed_segment_detection(points[start:], angles[start:], self.eps, self.delta, self.Snum, self.Pmin)
            if res is None: break
            i_seed, j_seed, p0, v = res
            i_seed += start; j_seed += start
            seg_dict = self.grow_region(points, i_seed, j_seed, self.eps, self.Pmin, self.Lmin)
            if seg_dict is None:
                start = j_seed + 1
                continue
            used_mask[seg_dict["Pb"] : seg_dict["Pf"] + 1] = True 
            # Crear segmento
            t1 = np.dot(points[seg_dict["Pb"]] - seg_dict["p0"], seg_dict["v"])
            t2 = np.dot(points[seg_dict["Pf"]]   - seg_dict["p0"], seg_dict["v"])
            e1 = seg_dict["p0"] + t1 * seg_dict["v"]
            e2 = seg_dict["p0"] + t2 * seg_dict["v"]
            segments.append(LineSegment(seg_dict["Pb"], seg_dict["Pf"], seg_dict["p0"], seg_dict["v"], e1, e2))
            start = seg_dict["Pf"] + 1
        return segments, used_mask

    def seed_segment_detection(self, points, angles, eps, delta, S, P):
        Np = len(points)
        for i in range(Np-P):
            j = i + S
            if j >= Np: break
            p0, v = self.fit_line(points[i:j+1])
            flag = True
            for k in range(i, j+1):
                pk_prime = self.predict_point(p0, v, angles[k])
                if np.linalg.norm(points[k] - pk_prime) > delta or self.dist_point_line(points[k], p0, v) > eps:
                    flag = False; break
            if flag: return i, j, p0, v
        return None
    
    def grow_region(self, points, i, j, eps, Pmin, Lmin):
        Np = len(points); Pb, Pf = i, j
        p0, v = self.fit_line(points[Pb:Pf+1])
        # Adelante
        k = Pf + 1
        while k < Np and self.dist_point_line(points[k], p0, v) < eps:
            Pf = k; p0, v = self.fit_line(points[Pb:Pf+1]); k+=1
        # Atras
        k = Pb - 1
        while k >= 0 and self.dist_point_line(points[k], p0, v) < eps:
            Pb = k; p0, v = self.fit_line(points[Pb:Pf+1]); k-=1
        if np.linalg.norm(points[Pf]-points[Pb]) >= Lmin and (Pf-Pb+1) >= Pmin:
            return {"Pb": Pb, "Pf": Pf, "p0": p0, "v": v}
        return None

    def fit_line(self, pts):
        m = np.mean(pts, axis=0)
        centered = pts - m
        _, _, vt = np.linalg.svd(centered)
        return m, vt[0]

    def dist_point_line(self, p, p0, v):
        w = p - p0
        return abs(w[0]*v[1] - w[1]*v[0])
    
    def predict_point(self, p0, v, theta):
        denom = v[0]*np.sin(theta) - v[1]*np.cos(theta)
        if abs(denom) < 1e-6: return p0
        t = (p0[1]*np.cos(theta) - p0[0]*np.sin(theta)) / denom
        return p0 + t * v

    def segments_to_landmarks(self, segments):
        lms = []
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                s1, s2 = segments[i], segments[j]
                if abs(np.dot(s1.v, s2.v)) > np.cos(self.angle_thresh): continue # Paralelos
                # Intersección
                A = np.array([[s1.v[0], -s2.v[0]], [s1.v[1], -s2.v[1]]])
                b = s2.p0 - s1.p0
                try:
                    x = np.linalg.solve(A, b)
                    inter = s1.p0 + x[0]*s1.v
                    # Verificar cercanía
                    d1 = min(np.linalg.norm(inter-s1.e1), np.linalg.norm(inter-s1.e2))
                    d2 = min(np.linalg.norm(inter-s2.e1), np.linalg.norm(inter-s2.e2))
                    if d1 < self.corner_thresh and d2 < self.corner_thresh:
                        lms.append([np.hypot(inter[0], inter[1]), np.arctan2(inter[1], inter[0])])
                except: pass
        return np.array(lms) if lms else np.empty((0, 2))

    def extract_point_landmarks(self, points, used_mask, segments):
        # Simplificación de extracción de puntos aislados (clusters pequeños)
        leftover = points[~used_mask]
        if len(leftover) < 3: return np.empty((0, 2))
        
        # Clustering simple por distancia
        clusters = []; curr = [leftover[0]]
        for i in range(1, len(leftover)):
            if np.linalg.norm(leftover[i] - leftover[i-1]) < 0.2:
                curr.append(leftover[i])
            else:
                if 3 <= len(curr) <= 10: clusters.append(np.mean(curr, axis=0))
                curr = [leftover[i]]
        if 3 <= len(curr) <= 10: clusters.append(np.mean(curr, axis=0))
        
        lms = []
        for c in clusters:
            # Filtrar si está muy cerca de una pared (segmento)
            isolated = True
            for s in segments:
                # Distancia punto-segmento
                ab = s.e2 - s.e1
                t = np.dot(c - s.e1, ab) / np.dot(ab, ab)
                closest = s.e1 + np.clip(t, 0, 1) * ab
                if np.linalg.norm(c - closest) < 0.35:
                    isolated = False; break
            if isolated:
                lms.append([np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])])
        return np.array(lms) if lms else np.empty((0, 2))

    # --- PARTICLE FILTER CORE ---
    def normalize_and_resample(self):
        weights = np.array([p.weight for p in self.particles])
        if weights.sum() == 0: weights[:] = 1.0
        weights /= weights.sum()
        
        # Actualizar pesos en objetos
        for i, p in enumerate(self.particles): p.weight = weights[i]

        # Effective sample size
        neff = 1.0 / np.sum(weights**2)
        if neff < self.num_particles / 2.0:
            self.resample_sus(weights)

    def resample_sus(self, weights):
        # Stochastic Universal Sampling
        cumulative = np.cumsum(weights)
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        indices = []
        ptr = start
        idx = 0
        for _ in range(self.num_particles):
            while ptr > cumulative[idx]:
                idx = min(idx + 1, len(weights) - 1)
            indices.append(idx)
            ptr += step
        
        new_particles = []
        for i in indices:
            p_old = self.particles[i]
            # Copia con ligero jitter para evitar colapso total
            new_p = p_old.copy()
            new_particles.append(new_p)
        self.particles = new_particles

    # --- PUBLICADORES ---
    def publish_tf(self, particle):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        
        # CORRECCION AQUI: Casting explicito a float
        t.transform.translation.x = float(particle.x)
        t.transform.translation.y = float(particle.y)
        t.transform.translation.z = 0.0
        
        t.transform.rotation = yaw_to_quaternion(float(particle.orientation))
        self.tf_broadcaster.sendTransform(t)

    def publish_pose(self, particle):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        
        # CORRECCION AQUI: Casting explicito a float
        ps.pose.position.x = float(particle.x)
        ps.pose.position.y = float(particle.y)
        ps.pose.orientation = yaw_to_quaternion(float(particle.orientation))
        
        self.pose_pub.publish(ps)

    def publish_static_map(self):
        """ Visualiza los landmarks cargados en VERDE """
        ma = MarkerArray()
        self.get_logger().info(f"landmarks pub.")

        for lid, val in self.map_landmarks.items():
            m = Marker() 
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg() 
            m.id = int(lid) 
            m.type = Marker.SPHERE 
            m.action = Marker.ADD 
            m.pose.position.x = float(val['x']) 
            m.pose.position.y = float(val['y'])
            m.scale.x = 0.15; m.scale.y = 0.15; m.scale.z = 0.15 
            m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0 
            ma.markers.append(m)
        self.map_vis_pub.publish(ma)

    def publish_segments_vis(self, data, segments):
        msg = MarkerArray()

        for idx, seg in enumerate(segments):
            m = Marker()
            m.header.frame_id = data.header.frame_id
            m.header.stamp = data.header.stamp
            m.id = idx + 1000
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.05
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0
            p1 = Point(x=float(seg.e1[0]), y=float(seg.e1[1]), z=0.0)
            p2 = Point(x=float(seg.e2[0]), y=float(seg.e2[1]), z=0.0)
            m.points = [p1, p2]
            msg.markers.append(m)
        self.segments_pub.publish(msg)

def main():
    rclpy.init()
    node = LocalizationNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()