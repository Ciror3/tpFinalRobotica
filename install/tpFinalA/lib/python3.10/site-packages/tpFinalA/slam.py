import numpy as np    
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
import math
from custom_msgs.msg import DeltaOdom
from geometry_msgs.msg import Point, PoseStamped
import copy

def yaw_to_quaternion(yaw):
    """Convert a yaw angle (in radians) into a Quaternion message."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    return q
def wrap(a): 
    return (a + np.pi) % (2*np.pi) - np.pi

class Particle():
    def __init__(self):
        self.x = 0  # initial x position
        self.y = 0  # initial y position
        self.orientation = 0 # initial orientation
        self.weight = 1.0
        self.landmarks = {}

    def set(self, new_x, new_y, new_orientation):
        '''
        set: sets a robot coordinate, including x, y and orientation
        '''
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self,odom,alpha):
        '''
        move_odom: Takes in Odometry data and moves the robot based on the odometry data
        
        Devuelve una particula (del robot) actualizada
        '''      
        dist  = odom[0]       
        delta_rot1  = odom[1]
        delta_rot2 = odom[2]

        x_new = self.x
        y_new = self.y
        theta_new = self.orientation
        varDeltaRot1 = alpha[0]*np.abs(delta_rot1)+alpha[1]*dist
        varDeltaRot2 = alpha[0]*np.abs(delta_rot2)+alpha[1]*dist
        varDeltaTrans = alpha[3]*(np.abs(delta_rot1)+np.abs(delta_rot2))+alpha[2]*dist

        deltaRot1 = delta_rot1 + np.random.normal(0,np.sqrt(varDeltaRot1))
        deltaRot2 = delta_rot2 + np.random.normal(0,np.sqrt(varDeltaRot2))
        deltaRotT = dist + np.random.normal(0,np.sqrt(varDeltaTrans))

        x_new = self.x + deltaRotT*np.cos(self.orientation+deltaRot1)
        y_new = self.y + deltaRotT*np.sin(self.orientation+deltaRot1)
        theta_new = self.orientation + deltaRot1 + deltaRot2

        self.set(x_new, y_new,theta_new)

    def set_weight(self, weight):
        '''
        set_weights: sets the weight of the particles
        '''
        #noise parameters
        self.weight  = float(weight)

    def h_and_H(self, landmark_mu):
        dx = landmark_mu[0] - self.x
        dy = landmark_mu[1] - self.y
        q = max(dx*dx + dy*dy, 1e-12)
        r = np.sqrt(q)
        r  = math.sqrt(q)
        zhat = np.array([r, wrap(math.atan2(dy, dx) - self.orientation)])
        H = np.array([[ dx/r,      dy/r     ],
                      [-dy/q,      dx/q     ]], dtype=float)
        return zhat, H
      
    def gaussian_weight(self, nu, Q):
        det_Q = np.linalg.det(Q)
        if det_Q <= 0 or not np.isfinite(det_Q):
            return 1e-12
        inv_Q = np.linalg.inv(Q)
        exponent = -0.5 * (nu.T @ inv_Q @ nu)
        coeff = 1.0 / (2.0 * np.pi * math.sqrt(det_Q))
        return float(coeff * np.exp(exponent))
    
    def update_landmarks(self,lid,z,R):
        mu = self.landmarks[lid]['mu']
        sigma = self.landmarks[lid]['sigma']
        zhat,H = self.h_and_H(mu)
        nu = np.array([z[0] - zhat[0], wrap(z[1]-zhat[1])],dtype=float)
        Q = H @ sigma @ H.T + R
        Q = 0.5 * (Q + Q.T) + 1e-8 * np.eye(2)

        K = sigma @ H.T
        K = np.linalg.solve(Q, K.T).T

        mu_new = mu + K @ nu

        I = np.eye(2)
        Sigma_new = (I - K @ H) @ sigma @ (I - K @ H).T + K @ R @ K.T
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        self.landmarks[lid]['mu'] = mu_new
        self.landmarks[lid]['sigma'] = Sigma_new

        return self.gaussian_weight(nu, Q)
    
    def create_landmark(self, r, ang, R):
        a = self.orientation + ang
        mx = self.x + r * np.cos(a)
        my = self.y + r * np.sin(a)
        mu = np.array([mx, my], dtype=float)

        J = np.array([
            [ np.cos(a), -r * np.sin(a)],
            [ np.sin(a),  r * np.cos(a)]
        ], dtype=float)

        sigma = J @ R @ J.T
        lid = 0
        if self.landmarks:
            lid = max(self.landmarks.keys()) + 1
            
        self.landmarks[lid] = {'mu': mu, 'sigma': sigma}
        return mu, sigma
      
class LineSegment:
    def __init__(self, m, n, p0, v, e1=None, e2=None):
        self.m = m      
        self.n = n      
        self.p0 = p0    
        self.v = v 
        self.e1 = e1 
        self.e2 = e2
        
# class fastslamNodse(Node):
#     def __init__(self, num_particles=1000):
#         super().__init__("fastslamNode")
#         self.delta_sub = self.create_subscription(
#             DeltaOdom, "/delta", self.delta_odom_callback, 10
#         )
#         self.scan_sub = self.create_subscription(
#             LaserScan, "/scan", self.scan_callback, 10
#         )
#         self.markers_pub = self.create_publisher(MarkerArray, "/fastslam/markers", 10)
#         self.pose_pub = self.create_publisher(PoseStamped, "/fpose", 10)
#         self.segments_pub = self.create_publisher(MarkerArray, "extracted_segments", 10)

#         self.R = np.diag([0.1, 0.05])
        
#         # Parámetros del paper / ajustados
#         self.Pmin  = 8          
#         self.Lmin  = 0.25       
#         self.eps   = 0.03       # Threshold distancia punto-recta
#         self.Snum  = 6          # Puntos semilla
#         self.delta = 0.1        # Threshold predicción
        
#         # Parámetros para detección de esquinas
#         self.corner_thresh = 0.5 # Distancia máxima intersección-segmento
#         self.angle_thresh = np.deg2rad(30) # Ángulo mínimo entre paredes

#         self.num_particles = num_particles
#         if num_particles != 0:
#             self.particles = [Particle() for _ in range(self.num_particles)]
#             self.get_logger().info(f"Se crearon {num_particles} partículas")

#     def delta_odom_callback(self,msg: DeltaOdom):
#         delta_odom = [msg.dt,msg.dr1,msg.dr2]
#         self.move_particles(delta_odom)

#     def scan_callback(self, data):
#         ranges = np.array(data.ranges)
#         angle_min = data.anlge_min
#         angle_inc = data.angle_increment

#         valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
#         indices = np.where(valid)[0]
#         if len(indices) < self.Pmin:
#             return
        
#         ranges = ranges[indices]
#         angles = angle_min + indices * angle_inc

#         xs = ranges * np.cos(angles)
#         ys = ranges * np.sin(angles)
#         points = np.stack([xs,ys],axis=1)

#         segments = self.extract_segments(points, angles)
#         self.publish_segments(data, segments)
#         landmarks = self.segments_to_landmarks(segments)

#         if landmarks is not None and len(landmarks) > 0:
#             for part in self.particles:
#                 for r, th in landmarks:
#                     self.update_particles(part, r, th, thresh=3.0) 

#         self.normalize_and_resample()
#         self.publish_fastslam()


#     def extract_segments(self,points,angles):
#         segments = []
#         N = len(points)
#         start = 0

#         while start < N - self.Pmin:
#             res = self.seed_segment_detection(points[start:],angles, eps=self.eps, delta=self.delta,  S=self.Snum, P=self.Pmin)
#             i_seed, j_seed, _, _ = res
#             i_seed += start
#             j_seed += start

#             seg_dict = self.grow_region(points, i_seed, j_seed, eps=self.eps, Pmin=self.Pmin, Lmin=self.Lmin)

#             if seg_dict is None:
#                 start = j_seed + 1
#                 continue

#             # [cite_start]Calcular endpoints geométricos (proyección ortogonal) [cite: 7]
#             # p0 + t*v
#             p_start = points[seg_dict["Pb"]]
#             p_end   = points[seg_dict["Pf"]]
#             t1 = np.dot(p_start - seg_dict["p0"], seg_dict["v"])
#             t2 = np.dot(p_end   - seg_dict["p0"], seg_dict["v"])
#             e1 = seg_dict["p0"] + t1 * seg_dict["v"]
#             e2 = seg_dict["p0"] + t2 * seg_dict["v"]

#             segments.append(
#                 LineSegment(
#                     m=seg_dict["Pb"],
#                     n=seg_dict["Pf"],
#                     p0=seg_dict["p0"],
#                     v=seg_dict["v"],
#                     e1=e1,
#                     e2=e2)
#             )
#             start = seg_dict["Pf"] + 1

#         segments = self.process_overlaps(points, segments)
#         return segments

#     def segments_to_landmarks(self, segments):
#         """
#         Genera landmarks detectando ESQUINAS (intersecciones) entre segmentos.
#         """
#         landmarks = []
#         N = len(segments)


#     def seed_segment_detection(self, points, angles, eps, delta, S, P):
#         Np = len(points)
#         for i in range(Np-P):
#             j = i + S
#             p0, v =  self.fit_seed(points,i,j)
#             flag = True
            
#             for k in range(i,j):
#                 pk = points[k]
#                 theta_k = angles[k]
#                 pk_prime = self.predict_point_from_bearing(p0, v, theta_k)
#                 d1 = np.linalg.norm(pk - pk_prime)
#                 if d1 > delta:
#                     flag = False
#                     break

#                 d2 = self.point_to_line_distance(pk, p0, v)
#                 if d2 > eps:
#                     flag = False
#                     break

#             if flag:
#                 return i,j,p0,v
            
#         return None
    
#     def point_to_line_distance(self, p, p0, v):
#         w = p - p0
#         return abs(w[0]*v[1] - w[1]*v[0])

#     def predict_point_from_bearing(self, p0, v, theta):
#         denom = v[0]*np.sin(theta) - v[1]*np.cos(theta)
#         num = p0[1]*np.cos(theta) - p0[0]*np.sin(theta)
#         t = num/denom
#         return p0 + t * v

#     def fit_seed(self, points, i, j):
#         segment_points = points[i:j+1]
#         p0 = np.mean(segment_points,axis=0)
#         centered = segment_points - p0
#         _,_,vt = np.linalg.svd(centered)
#         v = vt[0]
#         v = v / np.linalg.norm(v)
#         return p0,v

#     def grow_region(self, points, i, j, eps, Pmin, Lmin):
#         Np = len(points)
#         Pb, Pf = i, j
#         p0, v = self.fit_seed(points, Pb, Pf)

#         #adelante
#         k = Pf + 1
#         while k < Np:
#             dist = self.point_to_line_distance(points[k],p0,v)
#             if dist>= eps:
#                 break

#             Pf = k
#             p0,v = self.fit_seed(points,Pb,Pf)
#             k+=1

#         #atras
#         k = Pb - 1
#         while k >= 0:
#             dist = self.point_to_line_distance(points[k],p0,v)
#             if dist>= eps:
#                 break

#             Pb = k
#             p0,v = self.fit_seed(points,Pb,Pf)
#             k-=1

#         length = np.linalg.norm(points[Pf] - points[Pb])
#         num_points = Pf - Pb + 1

#         if (length >= Lmin) and (num_points >= Pmin):
#             return {"Pb": Pb, "Pf": Pf, "p0": p0, "v": v}

#         return None

#     def process_overlaps(self, points, segments):
#         Nl = len(segments)
#         if Nl <= 1: return segments

#         for i in range(Nl-1):
#             j = i+1
#             line_i = segments[i]
#             line_j = segments[i + 1]

#             m2 = line_j.m
#             n1 = line_i.n
#             if m2 <= n1:
#                 split_k = m2
#                 for k in range(m2, n1 + 1):
#                     pk = points[k]
#                     d_i = self.point_to_line_distance(pk, line_i.p0, line_i.v)
#                     d_j = self.point_to_line_distance(pk, line_j.p0, line_j.v)
#                     if d_j < d_i:
#                         split_k = k
#                         break
#                     split_k = k
#                 line_i.n = split_k - 1
#                 line_j.m = split_k
                
#                 if line_i.n > line_i.m:
#                     p0, v = self.fit_seed(points, line_i.m, line_i.n)
#                     line_i.p0, line_i.v = p0, v
                
#                 if line_j.n > line_j.m:
#                     p0, v = self.fit_seed(points, line_j.m, line_j.n)
#                     line_j.p0, line_j.v = p0, v

#         return segments

#     def update_particles(self, part: Particle, r, theta, thresh=10.0):
#         z = np.array([r, theta])
        
#         if not part.landmarks:
#             part.create_landmark(r, theta, self.R)
#             return

#         best_id = None
#         best_dist = float('inf')

#         for lid, lm in part.landmarks.items():
#             zhat, H = part.h_and_H(lm['mu'])
#             nu = z - zhat
#             nu[1] = wrap(nu[1]) 
            
#             S = H @ lm['sigma'] @ H.T + self.R
#             try:
#                 # Mahalanobis distance
#                 d2 = nu.T @ np.linalg.inv(S) @ nu
#             except np.linalg.LinAlgError:
#                 continue

#             if d2 < best_dist:
#                 best_dist = d2
#                 best_id = lid

#         if best_id is not None and best_dist < thresh:
#             w = part.update_landmarks(best_id, z, self.R)
#             part.weight *= w
#         else:
#             # Nuevo landmark
#             part.create_landmark(r, theta, self.R)
#             part.weight *= 0.5 # Penalización pequeña por agregar complejidad

#     def move_particles(self, deltas):
#         noise = [0.05, 0.05, 0.01, 0.01] 
#         for part in self.particles:
#             part.move_odom(deltas, noise)

#     def normalize_and_resample(self):
#         weights = np.array([p.weight for p in self.particles])
#         weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
#         s = weights.sum()
        
#         if s > 0: weights /= s
#         else: weights[:] = 1.0 / len(weights)
            
#         for i, p in enumerate(self.particles):
#             p.weight = float(weights[i])

#         neff = 1.0 / np.sum(np.square(weights))
#         if neff < 0.5 * self.num_particles:
#             self.particles = self.sus(self.num_particles, weights)

#     def sus(self, size, weights):
#         """ Stochastic Universal Sampling """
#         ranges = np.cumsum(weights)
#         step = 1.0 / size
#         start = np.random.uniform(0, step)
#         pointers = [start + i * step for i in range(size)]
#         indices = []
#         i = 0
#         for p in pointers:
#             while p > ranges[i]:
#                 i += 1
#             indices.append(i)
        
#         return [copy.deepcopy(self.particles[i]) for i in indices]

#     def make_landmark_marker(self, idx, x, y): 
#         m = Marker() 
#         m.header.frame_id = "map"
#         m.header.stamp = self.get_clock().now().to_msg() 
#         m.id = idx 
#         m.type = Marker.SPHERE 
#         m.action = Marker.ADD 
#         m.pose.position.x = x 
#         m.pose.position.y = y 
#         m.scale.x = 0.15; m.scale.y = 0.15; m.scale.z = 0.15 
#         m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0 
#         return m

#     def make_covariance_marker(self, idx, x, y, cov):
#         vals, vecs = np.linalg.eigh(cov)
#         order = vals.argsort()[::-1]
#         vals, vecs = vals[order], vecs[:, order]
#         angle = np.arctan2(vecs[1, 0], vecs[0, 0])
        
#         m = Marker()
#         m.header.frame_id = "map"
#         m.header.stamp = self.get_clock().now().to_msg()
#         m.id = idx  
#         m.type = Marker.CYLINDER 
#         m.pose.position.x = float(x); m.pose.position.y = float(y)
#         m.pose.orientation = yaw_to_quaternion(angle)
#         m.scale.x = max(0.05, 2 * np.sqrt(vals[0]))
#         m.scale.y = max(0.05, 2 * np.sqrt(vals[1]))
#         m.scale.z = 0.01
#         m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; m.color.a = 0.3
#         return m

#     def publish_fastslam(self):
#         if not self.particles: return
#         best_particle = max(self.particles, key=lambda p: p.weight)
        
#         # Publicar pose
#         ps = PoseStamped()
#         ps.header.frame_id = "map"
#         ps.header.stamp = self.get_clock().now().to_msg()
#         ps.pose.position.x = float(best_particle.x)
#         ps.pose.position.y = float(best_particle.y)
#         ps.pose.orientation = yaw_to_quaternion(float(best_particle.orientation))
#         self.pose_pub.publish(ps)

#         # Publicar landmarks
#         ma = MarkerArray()    
#         for lid, lm in best_particle.landmarks.items():
#             mu, sigma = lm["mu"], lm["sigma"]
#             # Visualizar solo landmarks válidos
#             if np.linalg.det(sigma) > 1e-12: 
#                 ma.markers.append(self.make_landmark_marker(lid * 2, mu[0], mu[1]))
#                 ma.markers.append(self.make_covariance_marker(lid * 2 + 1, mu[0], mu[1], sigma))
#         self.markers_pub.publish(ma)

#     def publish_segments(self, data, segments):
#         """ Visualiza los segmentos extraídos """
#         msg = MarkerArray()
#         now = data.header.stamp
#         frame = data.header.frame_id

#         for idx, seg in enumerate(segments):
#             m = Marker()
#             m.header.frame_id = frame
#             m.header.stamp = now
#             m.id = idx
#             m.type = Marker.LINE_STRIP
#             m.scale.x = 0.05
#             m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0

#             p1 = Point(x=float(seg.e1[0]), y=float(seg.e1[1]), z=0.0)
#             p2 = Point(x=float(seg.e2[0]), y=float(seg.e2[1]), z=0.0)
#             m.points = [p1, p2]
#             msg.markers.append(m)

#         self.segments_pub.publish(msg)



class fastslamNode(Node):
    def __init__(self, num_particles=1000):
        super().__init__("fastslamNode")
        self.delta_sub = self.create_subscription(
            DeltaOdom, "/delta", self.delta_odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.markers_pub = self.create_publisher(MarkerArray, "/fastslam/markers", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/fpose", 10)
        self.segments_pub = self.create_publisher(MarkerArray, "extracted_segments", 10)

        self.R = np.diag([0.1, 0.05])
        
        # Parámetros del paper / ajustados
        self.Pmin  = 8          
        self.Lmin  = 0.25       
        self.eps   = 0.03       # Threshold distancia punto-recta
        self.Snum  = 6          # Puntos semilla
        self.delta = 0.1        # Threshold predicción
        
        # Parámetros para detección de esquinas
        self.corner_thresh = 0.5 # Distancia máxima intersección-segmento
        self.angle_thresh = np.deg2rad(30) # Ángulo mínimo entre paredes

        self.num_particles = num_particles
        if num_particles != 0:
            self.particles = [Particle() for _ in range(self.num_particles)]
            self.get_logger().info(f"Se crearon {num_particles} partículas")

    def delta_odom_callback(self, msg: DeltaOdom):
        delta_odom = [msg.dt, msg.dr1, msg.dr2]
        self.move_particles(delta_odom)

    def scan_callback(self, data):
        ranges = np.array(data.ranges)
        angle_min = data.angle_min
        angle_inc = data.angle_increment
        
        valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
        indices = np.where(valid)[0]

        if len(indices) < self.Pmin:  
            return

        ranges = ranges[indices]
        angles = angle_min + indices * angle_inc

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        points = np.stack([xs, ys], axis=1)  

        # 1. Extraer Segmentos
        segments = self.extract_segments(points, angles)
        self.publish_segments(data, segments) # Modificado para no requerir points raw

        # 2. Extraer Landmarks (Esquinas)
        landmarks = self.segments_to_landmarks(segments)

        # 3. Actualizar SLAM
        if landmarks is not None and len(landmarks) > 0:
            for part in self.particles:
                for r, th in landmarks:
                    self.update_particles(part, r, th, thresh=3.0) 

        self.normalize_and_resample()
        self.publish_fastslam()
        
    def extract_segments(self, points, angles):
        segments = []
        N = len(points)
        start = 0

        while start < N - self.Pmin:
            res = self.seed_segment_detection(points[start:], angles[start:], 
                                              eps=self.eps, delta=self.delta, 
                                              S=self.Snum, P=self.Pmin)
            if res is None:
                break

            i_seed, j_seed, p0_seed, v_seed = res
            # Ajustar índices relativos a globales
            i_seed += start
            j_seed += start

            seg_dict = self.grow_region(points, i_seed, j_seed, eps=self.eps, Pmin=self.Pmin, Lmin=self.Lmin)
            
            if seg_dict is None:
                start = j_seed + 1
                continue

            # [cite_start]Calcular endpoints geométricos (proyección ortogonal) [cite: 7]
            # p0 + t*v
            p_start = points[seg_dict["Pb"]]
            p_end   = points[seg_dict["Pf"]]
            t1 = np.dot(p_start - seg_dict["p0"], seg_dict["v"])
            t2 = np.dot(p_end   - seg_dict["p0"], seg_dict["v"])
            e1 = seg_dict["p0"] + t1 * seg_dict["v"]
            e2 = seg_dict["p0"] + t2 * seg_dict["v"]

            segments.append(
                LineSegment(
                    m=seg_dict["Pb"],
                    n=seg_dict["Pf"],
                    p0=seg_dict["p0"],
                    v=seg_dict["v"],
                    e1=e1,
                    e2=e2
                )
            )
            start = seg_dict["Pf"] + 1

        segments = self.process_overlaps(points, segments)
        return segments

    def segments_to_landmarks(self, segments):
        """
        Genera landmarks detectando ESQUINAS (intersecciones) entre segmentos.
        """
        landmarks = []
        N = len(segments)
        
        # Iterar sobre pares para buscar esquinas
        for i in range(N):
            for j in range(i + 1, N):
                seg1 = segments[i]
                seg2 = segments[j]

                # 1. Verificar si son paralelos (producto punto cercano a 1)
                dot_prod = np.abs(np.dot(seg1.v, seg2.v))
                if dot_prod > np.cos(self.angle_thresh):
                    continue 

                # 2. Calcular intersección matemática
                intersection = self.find_intersection(seg1.p0, seg1.v, seg2.p0, seg2.v)
                if intersection is None: continue

                # 3. Validar si la intersección está cerca de los segmentos (forma una L o V)
                valid_1 = self.is_point_near_segment(intersection, seg1, self.corner_thresh)
                valid_2 = self.is_point_near_segment(intersection, seg2, self.corner_thresh)

                if valid_1 and valid_2:
                    r = np.hypot(intersection[0], intersection[1])
                    th = np.arctan2(intersection[1], intersection[0])
                    landmarks.append([r, th])

        # (Opcional) Si no hay esquinas, se podría agregar lógica para endpoints sueltos aquí
        return np.array(landmarks) if landmarks else np.empty((0, 2))

    def find_intersection(self, p1, v1, p2, v2):
        """ Intersección de dos rectas paramétricas """
        A = np.array([[v1[0], -v2[0]], [v1[1], -v2[1]]])
        b = p2 - p1
        try:
            x = np.linalg.solve(A, b) # x = [t, u]
            return p1 + x[0] * v1
        except np.linalg.LinAlgError:
            return None

    def is_point_near_segment(self, point, seg, dist_thresh):
        """ Verifica si el punto está cerca de alguno de los extremos del segmento """
        d1 = np.linalg.norm(point - seg.e1)
        d2 = np.linalg.norm(point - seg.e2)
        return (d1 < dist_thresh) or (d2 < dist_thresh)

    def fit_seed(self, points, i, j):
        segment_points = points[i:j+1]
        p0 = np.mean(segment_points, axis=0)
        centered = segment_points - p0
        _, _, vt = np.linalg.svd(centered)
        v = vt[0] # Dirección principal
        v = v / np.linalg.norm(v)
        return p0, v
    
    def predict_point_from_bearing(self, p0, v, theta):
        """ Intersección geométrica optimizada (Eq 6 del paper) """
        denom = v[1] * np.cos(theta) - v[0] * np.sin(theta)
        if abs(denom) < 1e-10: return p0
        
        numerator = p0[0] * np.sin(theta) - p0[1] * np.cos(theta)
        t = numerator / denom
        return p0 + t * v
    
    def point_to_segment_distance(self, p, a, b):
        """ Distancia punto a segmento finito """
        ab = b - a
        ab2 = np.dot(ab, ab)
        if ab2 == 0: return np.linalg.norm(p - a)
        t = np.dot(p - a, ab) / ab2  
        t = np.clip(t, 0.0, 1.0)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def point_to_line_distance(self, p, p0, v):
        """ Distancia perpendicular punto a recta infinita (Eq 1) """
        # v es unitario, p0 está en la recta.
        # vector de p0 a p
        w = p - p0
        # proyección ortogonal: |w x v| en 2D es el determinante
        return abs(w[0]*v[1] - w[1]*v[0])

    def seed_segment_detection(self, points, angles, eps, delta, S, P):
        Np = len(points)
        # Límite corregido según paper: Np - Pmin
        for i in range(Np - P):
            j = i + S
            if j >= Np: break
            
            # Ajuste de semilla
            p0, v = self.fit_seed(points, i, j)
            
            flag = True
            for k in range(i, j + 1):
                pk = points[k]
                theta_k = angles[k]

                # [cite_start]Condición 1: Distancia a predicción [cite: 171]
                pk_prime = self.predict_point_from_bearing(p0, v, theta_k)
                d1 = np.linalg.norm(pk - pk_prime)
                if d1 > delta:
                    flag = False; break

                # [cite_start]Condición 2: Distancia a la línea [cite: 119]
                d2 = self.point_to_line_distance(pk, p0, v)
                if d2 > eps:
                    flag = False; break
            
            if flag:
                return i, j, p0, v
            
        return None

    def grow_region(self, points, i, j, eps, Pmin, Lmin):
        Np = len(points)
        Pb, Pf = i, j
        p0, v = self.fit_seed(points, Pb, Pf)

        # Crecer adelante
        k = Pf + 1
        while k < Np:
            d = self.point_to_line_distance(points[k], p0, v)
            if d >= eps: break
            Pf = k
            # Refit (opcional por performance, paper lo sugiere)
            p0, v = self.fit_seed(points, Pb, Pf)
            k += 1
        
        # Crecer atrás
        k = Pb - 1
        while k >= 0:
            d = self.point_to_line_distance(points[k], p0, v)
            if d >= eps: break
            Pb = k
            p0, v = self.fit_seed(points, Pb, Pf)   
            k -= 1

        length = np.linalg.norm(points[Pf] - points[Pb])
        num_points = Pf - Pb + 1

        if length >= Lmin and num_points >= Pmin:
            return {"Pb": Pb, "Pf": Pf, "p0": p0, "v": v}

        return None
    
    def process_overlaps(self, points, segments):
        """ Resuelve disputas de puntos entre segmentos adyacentes """
        Nl = len(segments)
        if Nl <= 1: return segments

        for i in range(Nl - 1):
            line_i = segments[i]
            line_j = segments[i + 1]

            m2 = line_j.m
            n1 = line_i.n

            if m2 <= n1: # Hay superposición
                # Buscar punto de corte óptimo
                split_k = m2
                for k in range(m2, n1 + 1):
                    pk = points[k]
                    d_i = self.point_to_line_distance(pk, line_i.p0, line_i.v)
                    d_j = self.point_to_line_distance(pk, line_j.p0, line_j.v)
                    if d_j < d_i:
                        split_k = k
                        break
                    split_k = k # si nunca gana j, el corte se mueve al final
                
                # Reasignar índices
                line_i.n = split_k - 1
                line_j.m = split_k
                
                # Recalcular endpoints y parámetros con nuevos índices
                if line_i.n > line_i.m:
                    p0, v = self.fit_seed(points, line_i.m, line_i.n)
                    line_i.p0, line_i.v = p0, v
                    # update endpoints e1, e2... (simplificado aquí)
                
                if line_j.n > line_j.m:
                    p0, v = self.fit_seed(points, line_j.m, line_j.n)
                    line_j.p0, line_j.v = p0, v

        return segments

    def update_particles(self, part: Particle, r, theta, thresh=10.0):
        z = np.array([r, theta])
        
        if not part.landmarks:
            part.create_landmark(r, theta, self.R)
            return

        best_id = None
        best_dist = float('inf')

        # Data Association: Nearest Neighbor
        for lid, lm in part.landmarks.items():
            zhat, H = part.h_and_H(lm['mu'])
            nu = z - zhat
            nu[1] = wrap(nu[1]) 
            
            S = H @ lm['sigma'] @ H.T + self.R
            try:
                # Mahalanobis distance
                d2 = nu.T @ np.linalg.inv(S) @ nu
            except np.linalg.LinAlgError:
                continue

            if d2 < best_dist:
                best_dist = d2
                best_id = lid

        if best_id is not None and best_dist < thresh:
            w = part.update_landmarks(best_id, z, self.R)
            part.weight *= w
        else:
            # Nuevo landmark
            part.create_landmark(r, theta, self.R)
            part.weight *= 0.5 # Penalización pequeña por agregar complejidad

    def normalize_and_resample(self):
        weights = np.array([p.weight for p in self.particles])
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        s = weights.sum()
        
        if s > 0: weights /= s
        else: weights[:] = 1.0 / len(weights)
            
        for i, p in enumerate(self.particles):
            p.weight = float(weights[i])

        neff = 1.0 / np.sum(np.square(weights))
        if neff < 0.5 * self.num_particles:
            self.particles = self.sus(self.num_particles, weights)

    def sus(self, size, weights):
        """ Stochastic Universal Sampling """
        ranges = np.cumsum(weights)
        step = 1.0 / size
        start = np.random.uniform(0, step)
        pointers = [start + i * step for i in range(size)]
        indices = []
        i = 0
        for p in pointers:
            while p > ranges[i]:
                i += 1
            indices.append(i)
        
        return [copy.deepcopy(self.particles[i]) for i in indices]


    def move_particles(self, deltas):
        # Ruido de odometría (ajustar según robot)
        noise = [0.05, 0.05, 0.01, 0.01] 
        for part in self.particles:
            part.move_odom(deltas, noise)

    def publish_fastslam(self):
        if not self.particles: return
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        # Publicar pose
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(best_particle.x)
        ps.pose.position.y = float(best_particle.y)
        ps.pose.orientation = yaw_to_quaternion(float(best_particle.orientation))
        self.pose_pub.publish(ps)

        # Publicar landmarks
        ma = MarkerArray()    
        for lid, lm in best_particle.landmarks.items():
            mu, sigma = lm["mu"], lm["sigma"]
            # Visualizar solo landmarks válidos
            if np.linalg.det(sigma) > 1e-12: 
                ma.markers.append(self.make_landmark_marker(lid * 2, mu[0], mu[1]))
                ma.markers.append(self.make_covariance_marker(lid * 2 + 1, mu[0], mu[1], sigma))
        self.markers_pub.publish(ma)

    def make_landmark_marker(self, idx, x, y): 
        m = Marker() 
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg() 
        m.id = idx 
        m.type = Marker.SPHERE 
        m.action = Marker.ADD 
        m.pose.position.x = x 
        m.pose.position.y = y 
        m.scale.x = 0.15; m.scale.y = 0.15; m.scale.z = 0.15 
        m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0 
        return m

    def make_covariance_marker(self, idx, x, y, cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.arctan2(vecs[1, 0], vecs[0, 0])
        
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = idx  
        m.type = Marker.CYLINDER 
        m.pose.position.x = float(x); m.pose.position.y = float(y)
        m.pose.orientation = yaw_to_quaternion(angle)
        m.scale.x = max(0.05, 2 * np.sqrt(vals[0])) # Elipse 2 sigma
        m.scale.y = max(0.05, 2 * np.sqrt(vals[1]))
        m.scale.z = 0.01
        m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0; m.color.a = 0.3
        return m

    def publish_segments(self, data, segments):
        """ Visualiza los segmentos extraídos """
        msg = MarkerArray()
        now = data.header.stamp
        frame = data.header.frame_id

        for idx, seg in enumerate(segments):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = now
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.05
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0

            p1 = Point(x=float(seg.e1[0]), y=float(seg.e1[1]), z=0.0)
            p2 = Point(x=float(seg.e2[0]), y=float(seg.e2[1]), z=0.0)
            m.points = [p1, p2]
            msg.markers.append(m)

        self.segments_pub.publish(msg)

def main():
    rclpy.init()
    node = fastslamNode(num_particles=100) 
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()