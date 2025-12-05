import numpy as np    
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from custom_msgs.msg import DeltaOdom 
import math
import json
import os
import copy

# --- FUNCIONES MATEMÁTICAS ---
def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5); q.x = 0.0; q.y = 0.0; q.z = math.sin(yaw * 0.5)
    return q

def wrap(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

# --- CLASE SEGMENTO ---
class LineSegment:
    def __init__(self, m, n, p0, v, e1=None, e2=None):
        self.m = m; self.n = n; self.p0 = p0; self.v = v; self.e1 = e1; self.e2 = e2

# --- CLASE PARTÍCULA (MODO LOCALIZACIÓN) ---
class Particle():
    def __init__(self, static_map_ref=None):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.weight = 1.0
        
        # Referencia al mapa estático (Solo lectura)
        self.landmarks = static_map_ref if static_map_ref else {}

    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, alpha):
        dist, dr1, dr2 = odom
        # Modelo de movimiento odometrico con ruido
        varRot1 = alpha[0]*abs(dr1) + alpha[1]*dist
        varRot2 = alpha[0]*abs(dr2) + alpha[1]*dist
        varTrans = alpha[3]*(abs(dr1)+abs(dr2)) + alpha[2]*dist
        
        dr1_hat = dr1 + np.random.normal(0, np.sqrt(varRot1))
        dr2_hat = dr2 + np.random.normal(0, np.sqrt(varRot2))
        dt_hat  = dist + np.random.normal(0, np.sqrt(varTrans))
        
        self.x += dt_hat * np.cos(self.orientation + dr1_hat)
        self.y += dt_hat * np.sin(self.orientation + dr1_hat)
        self.orientation += dr1_hat + dr2_hat

    def get_likelihood(self, z, landmark_mu, landmark_sigma, R):
        """
        Calcula la probabilidad P(z | mapa, x) usando Mahalanobis.
        NO actualiza el mapa.
        """
        # Predicción: Dónde debería estar el landmark según mi posición
        dx = landmark_mu[0] - self.x
        dy = landmark_mu[1] - self.y
        q = max(dx*dx + dy*dy, 1e-12)
        r_pred = math.sqrt(q)
        
        # Ángulo esperado
        angle_pred = wrap(math.atan2(dy, dx) - self.orientation)
        zhat = np.array([r_pred, angle_pred])
        
        # Jacobiano H
        H = np.array([[ dx/r_pred, dy/r_pred ], 
                      [-dy/q,      dx/q      ]], dtype=float)
        
        # Innovación
        nu = z - zhat
        nu[1] = wrap(nu[1])
        
        # Covarianza de la innovación (Incertidumbre Landmark + Sensor)
        # En localización pura, landmark_sigma suele ser pequeño/fijo.
        Q = H @ landmark_sigma @ H.T + R
        
        try:
            det_Q = np.linalg.det(Q)
            if det_Q <= 1e-15: return 1e-12
            
            inv_Q = np.linalg.inv(Q)
            exponent = -0.5 * (nu.T @ inv_Q @ nu)
            coeff = 1.0 / (2.0 * np.pi * math.sqrt(det_Q))
            return float(coeff * np.exp(exponent))
        except np.linalg.LinAlgError:
            return 1e-12

    def copy_particle(self):
        p = Particle(self.landmarks) # Comparten la referencia al mapa
        p.x = self.x
        p.y = self.y
        p.orientation = self.orientation
        p.weight = self.weight
        return p

# --- NODO PRINCIPAL ---
class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        
        # 1. CARGAR MAPA
        # Asegúrate de que este archivo existe en la carpeta donde corres el nodo
        self.known_landmarks = self.load_map("/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas/mapa_landmarks_clasificados.json")
        if not self.known_landmarks:
            self.get_logger().error("¡ERROR CRÍTICO: No se encontró 'mi_mapa_landmarks.json'!")
        else:
            self.get_logger().info(f"Mapa cargado correctamente: {len(self.known_landmarks)} landmarks.")

        # 2. Configuración
        self.num_particles = 300 
        self.particles = [Particle(self.known_landmarks) for _ in range(self.num_particles)]
        
        # Inicialización (asumimos 0,0,0 o dispersión pequeña)
        for p in self.particles:
            p.x = np.random.normal(0, 0.1)
            p.y = np.random.normal(0, 0.1)
            p.orientation = np.random.normal(0, 0.05)

        self.R = np.diag([0.1, 0.05]) # Ruido del sensor (Radio, Ángulo)

        # Suscripciones
        self.delta_sub = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        
        # Publicadores
        self.pose_pub = self.create_publisher(PoseStamped, "/amcl_pose", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "/localization/markers", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Parámetros de Extracción (Mismos que SLAM)
        self.Pmin = 15; self.Lmin = 0.3; self.eps = 0.03; self.Snum = 6; self.delta = 0.1
        self.corner_thresh = 0.5; self.angle_thresh = np.deg2rad(30)

    def load_map(self, filename):
        if not os.path.exists(filename): return {}
        with open(filename, 'r') as f:
            data = json.load(f)
        landmarks = {}
        for item in data:
            lid = item['id']
            mu = np.array([item['x'], item['y']])
            # Matriz de covarianza fija para el mapa (confianza alta)
            sigma = np.eye(2) * 0.005 
            lm_type = item.get('type', 'unknown')
            landmarks[lid] = {'mu': mu, 'sigma': sigma, 'type': lm_type}
        return landmarks

    def delta_odom_callback(self, msg: DeltaOdom):
        # Ruido de movimiento [rot1, trans, trans, rot2] (ajustar según robot)
        noise = [0.05, 0.05, 0.05, 0.05] 
        delta_odom = [msg.dt, msg.dr1, msg.dr2]
        if abs(msg.dt) > 0.001 or abs(msg.dr1) > 0.001:
           self.get_logger().info(f"Moviendo: {msg.dt:.3f}, {msg.dr1:.3f}")
        for p in self.particles:
            p.move_odom(delta_odom, noise)
        self.publish_results()

    def scan_callback(self, data):
        # --- 1. EXTRACCIÓN DE CARACTERÍSTICAS (FEATURE EXTRACTION) ---
        ranges = np.array(data.ranges)
        angle_min = data.angle_min; angle_inc = data.angle_increment
        valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
        indices = np.where(valid)[0]
        
        if len(indices) < self.Pmin: return
        
        ranges = ranges[indices]; angles = angle_min + indices * angle_inc
        xs = ranges * np.cos(angles); ys = ranges * np.sin(angles)
        points = np.stack([xs, ys], axis=1)
        
        segments, used_mask = self.extract_segments(points, angles)
        
        # Obtener landmarks observados ahora
        obs_segments = self.segments_to_landmarks(segments)
        obs_clusters = self.extract_point_landmarks(points, used_mask, segments)
        
        # Lista unificada de observaciones: [(r, theta, type), ...]
        observations = []
        if len(obs_segments) > 0:
            for r, th in obs_segments: observations.append((r, th, "segment"))
        if len(obs_clusters) > 0:
            for r, th in obs_clusters: observations.append((r, th, "cluster"))

        # --- 2. ACTUALIZACIÓN DE PESOS (DATA ASSOCIATION) ---
        if observations:
            for part in self.particles:
                self.update_particle_weight(part, observations)

        # --- 3. RESAMPLING Y PUBLICACIÓN ---
        self.normalize_and_resample()
        self.publish_results()

    def update_particle_weight(self, part, observations):
        """
        Compara las observaciones actuales con el mapa cargado.
        Asocia datos por Cercanía + Tipo.
        """
        total_weight = 1.0
        
        # Iteramos sobre lo que el robot ve
        for r, theta, obs_type in observations:
            
            # 1. Proyectar al mundo global según esta partícula
            obs_x = part.x + r * np.cos(part.orientation + theta)
            obs_y = part.y + r * np.sin(part.orientation + theta)
            
            best_dist_sq = float('inf')
            best_landmark = None

            # 2. Buscar en el mapa cargado el landmark más cercano del mismo tipo
            for lid, lm in part.landmarks.items():
                if lm['type'] != 'unknown' and lm['type'] != obs_type:
                    continue
                
                dx = lm['mu'][0] - obs_x
                dy = lm['mu'][1] - obs_y
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_landmark = lm

            # 3. Gating (Umbral de aceptación)
            # Si el landmark más cercano está a > 1.5 metros, no es match.
            MAX_MATCH_DIST_SQ = 1.5**2 

            if best_landmark and best_dist_sq < MAX_MATCH_DIST_SQ:
                prob = part.get_likelihood(
                    np.array([r, theta]), 
                    best_landmark['mu'], 
                    best_landmark['sigma'], 
                    self.R
                )
                total_weight *= prob
            else:
                # Penalización suave por objeto no mapeado o espurio
                total_weight *= 0.1 
        
        part.weight *= total_weight

    # --- MÉTODOS DE EXTRACCIÓN (LÓGICA COPIADA DE SLAM) ---
    def get_distance_to_segment(self, point, seg):
        p, a, b = point, seg.e1, seg.e2
        ab, ap = b - a, p - a
        len_sq = np.dot(ab, ab)
        if len_sq == 0: return np.linalg.norm(ap)
        t = max(0.0, min(1.0, np.dot(ap, ab) / len_sq))
        return np.linalg.norm(p - (a + t * ab))

    def extract_point_landmarks(self, points, used_mask, segments):
        leftover_indices = np.where(~used_mask)[0]
        if len(leftover_indices) == 0: return np.empty((0, 2))
        candidates = []; current_cluster = [points[leftover_indices[0]]]
        for i in range(1, len(leftover_indices)):
            idx, prev_idx = leftover_indices[i], leftover_indices[i-1]
            pt, prev_pt = points[idx], points[prev_idx]
            if (idx - prev_idx < 5) and (np.linalg.norm(pt - prev_pt) < 0.1): # Cluster thresh 0.1
                current_cluster.append(pt)
            else:
                if 3 <= len(current_cluster) <= 10: candidates.append(np.array(current_cluster))
                current_cluster = [pt]
        if 3 <= len(current_cluster) <= 10: candidates.append(np.array(current_cluster))

        landmarks = []
        for cluster_pts in candidates:
            centroid = np.mean(cluster_pts, axis=0)
            is_isolated = True
            for seg in segments:
                if self.get_distance_to_segment(centroid, seg) < 0.35: # Isolation thresh
                    is_isolated = False; break 
            if is_isolated:
                landmarks.append([np.hypot(centroid[0], centroid[1]), np.arctan2(centroid[1], centroid[0])])
        return np.array(landmarks) if landmarks else np.empty((0, 2))
    
    def extract_segments(self, points, angles):
        segments = []; N = len(points); used_mask = np.zeros(N, dtype=bool); start = 0
        while start < N - self.Pmin:
            res = self.seed_segment_detection(points[start:], angles[start:], self.eps, self.delta, self.Snum, self.Pmin)
            if res is None: break
            i_seed, j_seed, _, _ = res; i_seed += start; j_seed += start
            seg_dict = self.grow_region(points, i_seed, j_seed, self.eps, self.Pmin, self.Lmin)
            if seg_dict is None: start = j_seed + 1; continue
            used_mask[seg_dict["Pb"] : seg_dict["Pf"] + 1] = True 
            p0, v = seg_dict["p0"], seg_dict["v"]
            p_start, p_end = points[seg_dict["Pb"]], points[seg_dict["Pf"]]
            e1 = p0 + np.dot(p_start - p0, v) * v; e2 = p0 + np.dot(p_end - p0, v) * v
            segments.append(LineSegment(seg_dict["Pb"], seg_dict["Pf"], p0, v, e1, e2))
            start = seg_dict["Pf"] + 1
        return self.process_overlaps(points, segments), used_mask

    def seed_segment_detection(self, points, angles, eps, delta, S, P):
        Np = len(points)
        for i in range(Np-P):
            j = i + S
            if j >= Np: break
            p0, v = self.fit_seed(points,i,j); flag = True
            for k in range(i,j+1):
                pk_prime = self.predict_point_from_bearing(p0, v, angles[k])
                if np.linalg.norm(points[k] - pk_prime) > delta or self.point_to_line_distance(points[k], p0, v) > eps:
                    flag = False; break
            if flag: return i,j,p0,v
        return None
    
    def point_to_line_distance(self, p, p0, v): return abs((p-p0)[0]*v[1] - (p-p0)[1]*v[0])
    def predict_point_from_bearing(self, p0, v, theta):
        denom = v[0]*np.sin(theta) - v[1]*np.cos(theta)
        return p0 + ((p0[1]*np.cos(theta) - p0[0]*np.sin(theta))/denom) * v if denom != 0 else p0
    def fit_seed(self, points, i, j):
        pts = points[i:j+1]; p0 = np.mean(pts,axis=0)
        _,_,vt = np.linalg.svd(pts - p0); v = vt[0] / np.linalg.norm(vt[0])
        return p0,v
    def grow_region(self, points, i, j, eps, Pmin, Lmin):
        Np = len(points); Pb, Pf = i, j; p0, v = self.fit_seed(points, Pb, Pf)
        k = Pf + 1
        while k < Np and self.point_to_line_distance(points[k],p0,v) < eps: Pf = k; p0,v = self.fit_seed(points,Pb,Pf); k+=1
        k = Pb - 1
        while k >= 0 and self.point_to_line_distance(points[k],p0,v) < eps: Pb = k; p0,v = self.fit_seed(points,Pb,Pf); k-=1
        if (np.linalg.norm(points[Pf] - points[Pb]) >= Lmin) and (Pf - Pb + 1 >= Pmin): return {"Pb": Pb, "Pf": Pf, "p0": p0, "v": v}
        return None
    def process_overlaps(self, points, segments):
        if len(segments) <= 1: return segments
        for i in range(len(segments)-1):
            l_i, l_j = segments[i], segments[i + 1]
            if l_j.m <= l_i.n:
                split_k = l_j.m
                for k in range(l_j.m, l_i.n + 1):
                    if self.point_to_line_distance(points[k], l_j.p0, l_j.v) < self.point_to_line_distance(points[k], l_i.p0, l_i.v): split_k = k; break
                l_i.n = split_k - 1; l_j.m = split_k
                if l_i.n > l_i.m: l_i.p0, l_i.v = self.fit_seed(points, l_i.m, l_i.n)
                if l_j.n > l_j.m: l_j.p0, l_j.v = self.fit_seed(points, l_j.m, l_j.n)
        return segments

    def segments_to_landmarks(self, segments):
        landmarks = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1, s2 = segments[i], segments[j]
                if np.abs(np.dot(s1.v, s2.v)) > np.cos(self.angle_thresh): continue 
                # Intersección
                A = np.array([[s1.v[0], -s2.v[0]], [s1.v[1], -s2.v[1]]])
                b = s2.p0 - s1.p0
                try: x = np.linalg.solve(A, b); intersection = s1.p0 + x[0] * s1.v
                except: intersection = None
                
                if intersection is not None:
                    d1 = min(np.linalg.norm(intersection-s1.e1), np.linalg.norm(intersection-s1.e2))
                    d2 = min(np.linalg.norm(intersection-s2.e1), np.linalg.norm(intersection-s2.e2))
                    if d1 < self.corner_thresh and d2 < self.corner_thresh:
                        landmarks.append([np.hypot(intersection[0], intersection[1]), np.arctan2(intersection[1], intersection[0])])
        return np.array(landmarks) if landmarks else np.empty((0, 2))

    # --- FINALIZACIÓN Y VISUALIZACIÓN ---
    def normalize_and_resample(self):
        weights = np.array([p.weight for p in self.particles])
        if np.sum(weights) == 0: weights[:] = 1.0 / len(weights)
        else: weights /= np.sum(weights)
        for i, p in enumerate(self.particles): p.weight = weights[i]
        
        neff = 1.0 / np.sum(np.square(weights))
        if neff < 0.5 * self.num_particles:
            step = 1.0 / self.num_particles; start = np.random.uniform(0, step)
            pointers = [start + i*step for i in range(self.num_particles)]
            indices = []; i = 0; cumsum = np.cumsum(weights)
            for p_ptr in pointers:
                while p_ptr > cumsum[i]: i += 1
                indices.append(i)
            self.particles = [self.particles[i].copy_particle() for i in indices]

    def publish_results(self):
        best_particle = max(self.particles, key=lambda p: p.weight)

        
        # Publicar TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"; t.child_frame_id = "base_link"
        t.transform.translation.x = best_particle.x; t.transform.translation.y = best_particle.y
        t.transform.rotation = yaw_to_quaternion(best_particle.orientation)
        self.tf_broadcaster.sendTransform(t)
        
        # Publicar Pose
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = best_particle.x
        ps.pose.position.y = best_particle.y
        ps.pose.orientation = t.transform.rotation
        self.pose_pub.publish(ps)
        
        # Visualizar Mapa Cargado (Estático)
        ma = MarkerArray()
        for lid, lm in self.known_landmarks.items():
            m = Marker(); m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.SPHERE; m.action = Marker.ADD; m.id = lid
            m.pose.position.x = float(lm['mu'][0]); m.pose.position.y = float(lm['mu'][1])
            m.scale.x = 0.2; m.scale.y = 0.2; m.scale.z = 0.2
            
            if lm['type'] == 'segment': m.color.r = 1.0; m.color.a = 0.8 # Rojo: Paredes
            elif lm['type'] == 'cluster': m.color.g = 1.0; m.color.a = 0.8 # Verde: Objetos
            else: m.color.b = 1.0; m.color.a = 0.5
            ma.markers.append(m)
        self.markers_pub.publish(ma)

def main():
    rclpy.init()
    node = LocalizationNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()