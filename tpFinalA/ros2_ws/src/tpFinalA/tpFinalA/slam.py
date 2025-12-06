import numpy as np    
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Point, PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from custom_msgs.msg import DeltaOdom 
import math
from numba import njit

@njit(fastmath=True)
def fast_mahalanobis_sq(px, py, ptheta, lm_mu, lm_sigma, z_r, z_theta, R):
    """
    Calcula la distancia de Mahalanobis al cuadrado entre una medida y un landmark.
    Reemplaza h_and_H y np.linalg.inv para máxima velocidad.
    """
    # --- 1. Predicción de Medida (h) y Jacobianos (H) ---
    dx = lm_mu[0] - px
    dy = lm_mu[1] - py
    q = dx*dx + dy*dy
    
    if q < 1e-12: q = 1e-12 
    r_pred = math.sqrt(q)
    
    # Ángulo esperado con wrap manual de (-pi, pi)
    angle_pred = math.atan2(dy, dx) - ptheta
    while angle_pred <= -np.pi: angle_pred += 2*np.pi
    while angle_pred > np.pi: angle_pred -= 2*np.pi

    # Jacobiano H (2x2) construido manualmente
    h00 = dx / r_pred; h01 = dy / r_pred
    h10 = -dy / q;     h11 = dx / q

    # --- 2. Covarianza de Innovación: S = H @ Sigma @ H.T + R ---
    s00 = lm_sigma[0, 0]; s01 = lm_sigma[0, 1]
    s10 = lm_sigma[1, 0]; s11 = lm_sigma[1, 1]

    # Multiplicación H @ Sigma (Intermedio 2x2)
    tmp00 = h00 * s00 + h01 * s10
    tmp01 = h00 * s01 + h01 * s11
    tmp10 = h10 * s00 + h11 * s10
    tmp11 = h10 * s01 + h11 * s11

    # S resultante (2x2)
    S00 = (tmp00 * h00 + tmp01 * h01) + R[0, 0]
    S01 = (tmp00 * h10 + tmp01 * h11) 
    S10 = (tmp10 * h00 + tmp11 * h01) 
    S11 = (tmp10 * h10 + tmp11 * h11) + R[1, 1]

    # --- 3. Inversa de S (Determinante 2x2) ---
    det_S = S00 * S11 - S01 * S10
    if det_S <= 1e-15: 
        return 9999.0 # Singular

    invDet = 1.0 / det_S
    invS00 =  S11 * invDet; invS01 = -S01 * invDet
    invS10 = -S10 * invDet; invS11 =  S00 * invDet

    # --- 4. Innovación (nu) ---
    nu_r = z_r - r_pred
    nu_th = z_theta - angle_pred
    while nu_th <= -np.pi: nu_th += 2*np.pi
    while nu_th > np.pi: nu_th -= 2*np.pi

    # --- 5. Distancia Mahalanobis: nu.T @ invS @ nu ---
    t0 = nu_r * invS00 + nu_th * invS10
    t1 = nu_r * invS01 + nu_th * invS11
    d2 = t0 * nu_r + t1 * nu_th
    
    return d2


def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5); q.x = 0.0; q.y = 0.0; q.z = math.sin(yaw * 0.5)
    return q

def wrap(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

class LineSegment:
    def __init__(self, m, n, p0, v, e1=None, e2=None):
        self.m = m; self.n = n; self.p0 = p0; self.v = v; self.e1 = e1; self.e2 = e2

class Particle():
    def __init__(self):
        self.x = 0; self.y = 0; self.orientation = 0
        self.weight = 1.0
        self.landmarks = {}

    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x); self.y = float(new_y); self.orientation = float(new_orientation)

    def move_odom(self, odom, alpha):
        dist, dr1, dr2 = odom
        # Modelo Odometría Probabilístico
        varRot1 = alpha[0]*abs(dr1) + alpha[1]*dist
        varRot2 = alpha[0]*abs(dr2) + alpha[1]*dist
        varTrans = alpha[3]*(abs(dr1)+abs(dr2)) + alpha[2]*dist

        dr1_hat = dr1 + np.random.normal(0, np.sqrt(varRot1))
        dr2_hat = dr2 + np.random.normal(0, np.sqrt(varRot2))
        dt_hat  = dist + np.random.normal(0, np.sqrt(varTrans))

        self.x += dt_hat * np.cos(self.orientation + dr1_hat)
        self.y += dt_hat * np.sin(self.orientation + dr1_hat)
        self.orientation += dr1_hat + dr2_hat

    def h_and_H(self, landmark_mu):
        dx = landmark_mu[0] - self.x
        dy = landmark_mu[1] - self.y
        q = max(dx*dx + dy*dy, 1e-12)
        r = math.sqrt(q)
        zhat = np.array([r, wrap(math.atan2(dy, dx) - self.orientation)])
        H = np.array([[ dx/r, dy/r ], [-dy/q, dx/q]], dtype=float)
        return zhat, H
      
    def gaussian_weight(self, nu, Q):
        try:
            det_Q = np.linalg.det(Q)
            if det_Q <= 1e-15: return 1e-12
            inv_Q = np.linalg.inv(Q)
            exponent = -0.5 * (nu.T @ inv_Q @ nu)
            coeff = 1.0 / (2.0 * np.pi * math.sqrt(det_Q))
            return float(coeff * np.exp(exponent))
        except: return 1e-12
    
    def update_landmarks(self, lid, z, R):
        mu = self.landmarks[lid]['mu']
        sigma = self.landmarks[lid]['sigma']
        zhat, H = self.h_and_H(mu)
        nu = np.array([z[0] - zhat[0], wrap(z[1]-zhat[1])], dtype=float)
        
        Q = H @ sigma @ H.T + R
        K = sigma @ H.T @ np.linalg.inv(Q) # Ganancia Kalman
        
        mu_new = mu + K @ nu
        I = np.eye(2)
        Sigma_new = (I - K @ H) @ sigma
        
        # Simetrizar covarianza para evitar errores numéricos
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        
        self.landmarks[lid]['mu'] = mu_new
        self.landmarks[lid]['sigma'] = Sigma_new
        return self.gaussian_weight(nu, Q)
    
    def create_landmark(self, r, ang, R, lm_type="unknown"):
        a = self.orientation + ang
        mx = self.x + r * np.cos(a)
        my = self.y + r * np.sin(a)
        mu = np.array([mx, my], dtype=float)

        J = np.array([[ np.cos(a), -r * np.sin(a)], 
                      [ np.sin(a),  r * np.cos(a)]], dtype=float)
        sigma = J @ R @ J.T
        
        lid = max(self.landmarks.keys()) + 1 if self.landmarks else 0
        
        # GUARDAMOS EL TIPO AQUI
        self.landmarks[lid] = {'mu': mu, 'sigma': sigma, 'type': lm_type}
        return mu, sigma
    
    def copy_particle(self):
        # Copia manual optimizada (más rápido que deepcopy)
        p = Particle()
        p.x = self.x; p.y = self.y; p.orientation = self.orientation; p.weight = self.weight
        for lid, val in self.landmarks.items():
            p.landmarks[lid] = {
                'mu': val['mu'].copy(),
                'sigma': val['sigma'].copy(),
                'type': val.get('type', 'unknown')
            }
        return p

class fastslamNode(Node):
    def __init__(self, num_particles=100):
        super().__init__("fastslamNode")
        
        # Suscripciones
        self.delta_sub = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        
        # Publicadores
        self.markers_pub = self.create_publisher(MarkerArray, "/fastslam/markers", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/fpose", 10)
        self.segments_pub = self.create_publisher(MarkerArray, "extracted_segments", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Parámetros EKF y Features
        self.R = np.diag([0.05, 0.03]) # Ruido de medida (r, theta)
        self.Pmin = 18; self.Lmin = 0.34; self.eps = 0.03; self.Snum = 8; self.delta = 0.1
        self.corner_thresh = 0.5; self.angle_thresh = np.deg2rad(30)

        # Inicialización de Partículas
        self.num_particles = num_particles
        self.particles = [Particle() for _ in range(self.num_particles)]
        self.get_logger().info(f"FastSLAM iniciado con {num_particles} partículas.")
        
    def delta_odom_callback(self, msg: DeltaOdom):
        # Mover partículas y publicar TF inmediatamente para suavidad
        noise = [0.05, 0.05, 0.05, 0.05] 
        delta_odom = [msg.dt, msg.dr1, msg.dr2]
        
        for part in self.particles:
            part.move_odom(delta_odom, noise)
            
        # Publicamos TF con tiempo actual (odometría es rápida)
        self.publish_tf(stamp=None) 

    def scan_callback(self, data):
        # 1. Extracción de Features
        ranges = np.array(data.ranges)
        angle_min = data.angle_min; angle_inc = data.angle_increment
        valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
        indices = np.where(valid)[0]
        if len(indices) < self.Pmin: return
        
        ranges = ranges[indices]; angles = angle_min + indices * angle_inc
        xs = ranges * np.cos(angles); ys = ranges * np.sin(angles)
        points = np.stack([xs, ys], axis=1)
        
        # Extraer segmentos y visualizar
        segments, used_mask = self.extract_segments(points, angles)
        self.publish_vector_map(segments) # Paredes instantáneas

        # 2. Identificar Landmarks (Esquinas y Clusters)
        corner_landmarks = self.segments_to_landmarks(segments)
        point_landmarks = self.extract_point_landmarks(points, used_mask, segments)

        # 3. Actualizar SLAM (Corrección con Numba)
        if corner_landmarks.shape[0] > 0:
            for part in self.particles:
                for r, th in corner_landmarks:
                    self.update_particles(part, r, th, lm_type="segment", thresh=6.0)

        if point_landmarks.shape[0] > 0:
            for part in self.particles:
                for r, th in point_landmarks:
                    self.update_particles(part, r, th, lm_type="cluster", thresh=6.0)

        # 4. Resample y Publicar Resultados
        self.normalize_and_resample()
        self.publish_fastslam(stamp=data.header.stamp) # Usar tiempo del láser para sincronizar
        
        # Publicar TF corregida (con tiempo del láser para evitar TF_OLD_DATA)
        self.publish_tf(stamp=data.header.stamp)

    def update_particles(self, part: Particle, r, theta, lm_type, thresh=6.0):
        # Preparar variables simples para Numba
        px, py, pth = part.x, part.y, part.orientation
        
        if not part.landmarks:
            part.create_landmark(r, theta, self.R, lm_type=lm_type)
            return

        best_id = None
        best_dist = float('inf')

        # --- BUCLE OPTIMIZADO CON NUMBA ---
        for lid, lm in part.landmarks.items():
            # 1. Filtro Rápido en Python (Evita Numba si el tipo no coincide)
            if lm.get('type') != lm_type:
                continue

            # 2. Cálculo Pesado en Numba
            d2 = fast_mahalanobis_sq(
                px, py, pth, 
                lm['mu'], lm['sigma'], 
                r, theta, 
                self.R
            )

            if d2 < best_dist:
                best_dist = d2; best_id = lid

        # --- ACTUALIZACIÓN EKF ---
        if best_id is not None and best_dist < thresh:
            w = part.update_landmarks(best_id, np.array([r, theta]), self.R)
            part.weight *= w
        else:
            part.create_landmark(r, theta, self.R, lm_type=lm_type)
            part.weight *= 0.5 

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
            if (idx - prev_idx < 5) and (np.linalg.norm(pt - prev_pt) < 0.1): 
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
                if self.get_distance_to_segment(centroid, seg) < 0.35: 
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

    def make_landmark_marker(self, idx, x, y, lm_type="unknown"): 
        m = Marker(); m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
        m.id = idx; m.type = Marker.SPHERE; m.action = Marker.ADD 
        m.ns = lm_type # Namespace para el Saver
        m.pose.position.x = x; m.pose.position.y = y; m.scale.x = 0.15; m.scale.y = 0.15; m.scale.z = 0.15 
        
        if lm_type == "segment": m.color.r = 1.0; m.color.a = 1.0 # Rojo (Pared)
        elif lm_type == "cluster": m.color.g = 1.0; m.color.a = 1.0 # Verde (Objeto)
        else: m.color.b = 1.0; m.color.a = 1.0
        return m

    def publish_fastslam(self, stamp):
        if not self.particles: return
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        ps = PoseStamped(); ps.header.frame_id = "map"; ps.header.stamp = stamp
        ps.pose.position.x = float(best_particle.x); ps.pose.position.y = float(best_particle.y)
        ps.pose.orientation = yaw_to_quaternion(float(best_particle.orientation))
        self.pose_pub.publish(ps)

        ma = MarkerArray()    
        for lid, lm in best_particle.landmarks.items():
            mu = lm["mu"]; ltype = lm.get("type", "unknown")
            if np.linalg.det(lm["sigma"]) > 1e-12: 
                ma.markers.append(self.make_landmark_marker(lid * 2, mu[0], mu[1], lm_type=ltype))
        self.markers_pub.publish(ma)

    def publish_vector_map(self, segments):
        if not self.particles: return
        best_particle = max(self.particles, key=lambda p: p.weight)
        px, py, pth = best_particle.x, best_particle.y, best_particle.orientation
        
        wall_marker = Marker()
        wall_marker.header.frame_id = "map"; wall_marker.header.stamp = self.get_clock().now().to_msg()
        wall_marker.ns = "walls"; wall_marker.id = 1; wall_marker.type = Marker.LINE_LIST; wall_marker.action = Marker.ADD
        wall_marker.scale.x = 0.05; wall_marker.color.r = 0.0; wall_marker.color.g = 1.0; wall_marker.color.b = 1.0; wall_marker.color.a = 1.0

        c, s = np.cos(pth), np.sin(pth); rot_matrix = np.array([[c, -s], [s, c]]); robot_pos = np.array([px, py])
        for seg in segments:
            p1_global = np.dot(rot_matrix, seg.e1) + robot_pos
            p2_global = np.dot(rot_matrix, seg.e2) + robot_pos
            pp1 = Point(); pp1.x = float(p1_global[0]); pp1.y = float(p1_global[1]); pp1.z = 0.0
            pp2 = Point(); pp2.x = float(p2_global[0]); pp2.y = float(p2_global[1]); pp2.z = 0.0
            wall_marker.points.append(pp1); wall_marker.points.append(pp2)
        
        self.markers_pub.publish(MarkerArray(markers=[wall_marker]))

    def publish_tf(self, stamp=None):
        if not self.particles: return
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        t = TransformStamped()
        t.header.stamp = stamp if stamp else self.get_clock().now().to_msg()
        t.header.frame_id = "map"; t.child_frame_id = "base_link" 
        t.transform.translation.x = float(best_particle.x); t.transform.translation.y = float(best_particle.y); t.transform.translation.z = 0.0
        t.transform.rotation = yaw_to_quaternion(float(best_particle.orientation))
        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init(); node = fastslamNode(num_particles=200) 
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__": main()