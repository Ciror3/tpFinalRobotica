import numpy as np    
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion, Point, PoseStamped, TransformStamped, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data
from custom_msgs.msg import DeltaOdom 
import math
import json
import os
import copy

def yaw_to_quaternion(yaw):
    """
    Converts a yaw angle (in radians) to a geometry_msgs/Quaternion.
    
    Args:
        yaw (float): Angle in radians.
        
    Returns:
        Quaternion: The resulting quaternion.
    """
    q = Quaternion()
    q.w = math.cos(yaw * 0.5); q.x = 0.0; q.y = 0.0; q.z = math.sin(yaw * 0.5)
    return q

def quaternion_to_yaw(q: Quaternion) -> float:
    """
    Converts a geometry_msgs/Quaternion to a yaw angle (in radians).
    
    Args:
        q (Quaternion): The quaternion to convert.
        
    Returns:
        float: The yaw angle in radians.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap(a): 
    """
    Normalizes an angle to the range [-pi, pi].
    
    Args:
        a (float): The angle in radians.
        
    Returns:
        float: The normalized angle.
    """
    return (a + np.pi) % (2*np.pi) - np.pi

class LineSegment:
    def __init__(self, m, n, p0, v, e1=None, e2=None):
        self.m = m; self.n = n; self.p0 = p0; self.v = v; self.e1 = e1; self.e2 = e2

class Particle():
    def __init__(self, static_map_ref=None):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.weight = 1.0
        
        self.landmarks = static_map_ref if static_map_ref else {}

    def set(self, new_x, new_y, new_orientation):
        """
        Sets the particle's state.
        
        Args:
            new_x (float): New X position.
            new_y (float): New Y position.
            new_orientation (float): New orientation in radians.
        """
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def move_odom(self, odom, alpha):
        """
        Updates the particle's pose based on odometry delta measurements and a noise model.
        
        Args:
            odom (list): A list [dist, dr1, dr2] representing displacement and rotations.
            alpha (list): Noise parameters for the motion model.
        """
        dist, dr1, dr2 = odom
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
        Calculates the measurement likelihood of an observation given a landmark.
        
        Args:
            z (np.array): The observation vector [range, bearing].
            landmark_mu (np.array): The landmark's mean position [x, y].
            landmark_sigma (np.array): The landmark's covariance matrix.
            R (np.array): The sensor noise covariance matrix.
            
        Returns:
            float: The probability density of the observation.
        """
        dx = landmark_mu[0] - self.x
        dy = landmark_mu[1] - self.y
        q = max(dx*dx + dy*dy, 1e-12)
        r_pred = math.sqrt(q)
        
        angle_pred = wrap(math.atan2(dy, dx) - self.orientation)
        zhat = np.array([r_pred, angle_pred])
        
        H = np.array([[ dx/r_pred, dy/r_pred ], 
                      [-dy/q,      dx/q      ]], dtype=float)
        
        nu = z - zhat
        nu[1] = wrap(nu[1])
        
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
        """
        Creates a deep copy of the particle.
        
        Returns:
            Particle: A new particle instance with identical state.
        """
        p = Particle(self.landmarks) 
        p.x = self.x
        p.y = self.y
        p.orientation = self.orientation
        p.weight = self.weight
        return p

class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")
        
        self.known_landmarks = self.load_map("/home/ciror/Desktop/robotica/tps/tpFinalRobotica/tpFinalA/ros2_ws/src/tpFinalA/mapas/mapa_landmarks_clasificados.json")
        if not self.known_landmarks:
            self.get_logger().error("¡ERROR CRÍTICO: No se encontró el mapa json!")
        else:
            self.get_logger().info(f"Mapa cargado correctamente: {len(self.known_landmarks)} landmarks.")

        self.num_particles = 100 
        self.particles = [Particle(self.known_landmarks) for _ in range(self.num_particles)]
        
        for p in self.particles:
            p.x = 0.0
            p.y = 0.0
            p.orientation = 0.0

        self.R = np.diag([0.1, 0.05])

        self.delta_sub = self.create_subscription(DeltaOdom, "/delta", self.delta_odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data)
        
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            10
        )

        self.pose_pub = self.create_publisher(PoseStamped, "/amcl_pose", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "/localization/markers", 10)
        self.segments_pub = self.create_publisher(MarkerArray, "/localization/extracted_segments", 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)

        self.Pmin = 18; self.Lmin = 0.34; self.eps = 0.03; self.Snum = 8; self.delta = 0.1
        self.corner_thresh = 0.5; self.angle_thresh = np.deg2rad(30)

        self.received_initial_pose = False

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)

        sigma_x = 0.1   
        sigma_y = 0.1  
        sigma_yaw = 0.05 

        self.get_logger().info(
            f"Recibido initialpose de RViz: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad"
        )

        for p in self.particles:
            p.x = np.random.normal(x, sigma_x)
            p.y = np.random.normal(y, sigma_y)
            p.orientation = wrap(np.random.normal(yaw, sigma_yaw))
            p.weight = 1.0 / self.num_particles

        self.received_initial_pose = True

    def load_map(self, filename):
        """
        Loads landmarks from a JSON file.
        
        Args:
            filename (str): Path to the .json file.
            
        Returns:
            dict: A dictionary of landmarks keyed by ID.
        """
        if not os.path.exists(filename): return {}
        with open(filename, 'r') as f:
            data = json.load(f)
        landmarks = {}
        for item in data:
            lid = item['id']
            mu = np.array([item['x'], item['y']])
            sigma = np.eye(2) * 0.005 
            lm_type = item.get('type', 'unknown')
            landmarks[lid] = {'mu': mu, 'sigma': sigma, 'type': lm_type}
        return landmarks

    def delta_odom_callback(self, msg: DeltaOdom):
        noise = [0.05, 0.05, 0.05, 0.05] 
        delta_odom = [msg.dt, msg.dr1, msg.dr2]
        if abs(msg.dt) > 0.001 or abs(msg.dr1) > 0.001:
           pass 
        for p in self.particles:
            p.move_odom(delta_odom, noise)
        self.publish_results()

    def scan_callback(self, data):
        ranges = np.array(data.ranges)
        angle_min = data.angle_min; angle_inc = data.angle_increment
        valid = (ranges >= data.range_min) & (ranges <= data.range_max) & (~np.isnan(ranges))
        indices = np.where(valid)[0]
        
        if len(indices) < self.Pmin: return
        
        ranges = ranges[indices]; angles = angle_min + indices * angle_inc
        xs = ranges * np.cos(angles); ys = ranges * np.sin(angles)
        points = np.stack([xs, ys], axis=1)
        
        segments, used_mask = self.extract_segments(points, angles)
        self.publish_extracted_segments(segments, data.header)

        obs_segments = self.segments_to_landmarks(segments)
        obs_clusters = self.extract_point_landmarks(points, used_mask, segments)
        
        observations = []
        if len(obs_segments) > 0:
            for r, th in obs_segments: observations.append((r, th, "segment"))
        if len(obs_clusters) > 0:
            for r, th in obs_clusters: observations.append((r, th, "cluster"))

        if observations:
            for part in self.particles:
                self.update_particle_weight(part, observations)

        self.normalize_and_resample()
        self.publish_results()

    def publish_extracted_segments(self, segments, header):
        """
        Publishes extracted line segments as markers for visualization.
        
        Args:
            segments (list): List of LineSegment objects.
            header (std_msgs/Header): Header info from the scan message.
        """
        ma = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        
        for i, seg in enumerate(segments):
            m = Marker()
            m.header.frame_id = header.frame_id 
            m.header.stamp = current_time       
            
            m.ns = "raw_segments"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0 
            
            p1 = Point(x=float(seg.e1[0]), y=float(seg.e1[1]), z=0.0)
            p2 = Point(x=float(seg.e2[0]), y=float(seg.e2[1]), z=0.0)
            m.points = [p1, p2]
            
            ma.markers.append(m)
        
        self.segments_pub.publish(ma)

    def update_particle_weight(self, part, observations):
        """
        Updates the weight of a particle based on the likelihood of observations.
        
        Args:
            part (Particle): The particle to update.
            observations (list): A list of observations [(r, theta, type), ...].
        """
        total_weight = 1.0
        
        sensor_x = part.x + 0.1 * np.cos(part.orientation)
        sensor_y = part.y + 0.1 * np.sin(part.orientation)

        for r, theta, obs_type in observations:
            obs_x = sensor_x + r * np.cos(part.orientation + theta)
            obs_y = sensor_y + r * np.sin(part.orientation + theta)
            
            best_dist_sq = float('inf')
            best_landmark = None

            for lid, lm in part.landmarks.items():
                if lm['type'] != 'unknown' and lm['type'] != obs_type:
                    continue
                
                dx = lm['mu'][0] - obs_x
                dy = lm['mu'][1] - obs_y
                dist_sq = dx*dx + dy*dy
                
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_landmark = lm

            GATE_DIST_SQ = 0.5**2 

            if best_landmark and best_dist_sq < GATE_DIST_SQ:
                prob = part.get_likelihood(
                    np.array([r, theta]), 
                    best_landmark['mu'], 
                    best_landmark['sigma'], 
                    self.R
                )
                total_weight *= prob
            else:
                pass 
        
        part.weight *= total_weight

    def get_distance_to_segment(self, point, seg):
        """
        Calculates the perpendicular distance from a point to a line segment.
        
        Args:
            point (np.array): The point coordinates [x, y].
            seg (LineSegment): The line segment object.
            
        Returns:
            float: The distance.
        """
        p, a, b = point, seg.e1, seg.e2
        ab, ap = b - a, p - a
        len_sq = np.dot(ab, ab)
        if len_sq == 0: return np.linalg.norm(ap)
        t = max(0.0, min(1.0, np.dot(ap, ab) / len_sq))
        return np.linalg.norm(p - (a + t * ab))

    def extract_point_landmarks(self, points, used_mask, segments):
        """
        Extracts point-like landmarks from scan points that were not part of any segment.
        
        Args:
            points (np.array): Array of all scan points in Cartesian coordinates.
            used_mask (np.array): Boolean mask indicating which points belong to segments.
            segments (list): List of extracted segments to check isolation.
            
        Returns:
            np.array: Array of landmarks in polar coordinates [r, theta].
        """
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
        """
        Extracts line segments from scan data using a seed-growing algorithm (Split-and-Merge variant).
        
        Args:
            points (np.array): Array of scan points.
            angles (np.array): Array of corresponding scan angles.
            
        Returns:
            tuple: (list of LineSegment, boolean mask of used points).
        """
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
        """
        Finds a seed segment within a subset of points.
        
        Args:
            points (np.array): Points to search.
            angles (np.array): Angles corresponding to points.
            eps (float): Maximum distance from line.
            delta (float): Maximum gap between predicted and actual points.
            S (int): Seed length.
            P (int): Minimum points for a segment.
            
        Returns:
            tuple: (start_idx, end_idx, origin, direction_vector) or None.
        """
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
        """
        Expands a seed segment to include collinear points.
        
        Args:
            points (np.array): All points.
            i (int): Start index of seed.
            j (int): End index of seed.
            eps (float): Tolerance.
            Pmin (int): Min points.
            Lmin (float): Min length.
            
        Returns:
            dict: Segment details or None.
        """
        Np = len(points); Pb, Pf = i, j; p0, v = self.fit_seed(points, Pb, Pf)
        k = Pf + 1
        while k < Np and self.point_to_line_distance(points[k],p0,v) < eps: Pf = k; p0,v = self.fit_seed(points,Pb,Pf); k+=1
        k = Pb - 1
        while k >= 0 and self.point_to_line_distance(points[k],p0,v) < eps: Pb = k; p0,v = self.fit_seed(points,Pb,Pf); k-=1
        if (np.linalg.norm(points[Pf] - points[Pb]) >= Lmin) and (Pf - Pb + 1 >= Pmin): return {"Pb": Pb, "Pf": Pf, "p0": p0, "v": v}
        return None
    def process_overlaps(self, points, segments):
        """
        Resolves overlapping segments by assigning points to the closest line.
        
        Args:
            points (np.array): All points.
            segments (list): List of detected segments.
            
        Returns:
            list: Refined list of segments.
        """
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
        """
        Converts intersecting segments into corner landmarks.
        
        Args:
            segments (list): List of segments.
            
        Returns:
            np.array: Array of corner landmarks in polar coordinates.
        """
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

    def normalize_and_resample(self):
        """
        Normalizes particle weights and performs resampling if effective particle count is low.
        """
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
        """
        Publishes the best particle pose, map transformation, and landmark markers.
        """
        best_particle = max(self.particles, key=lambda p: p.weight)
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"; t.child_frame_id = "base_link"
        t.transform.translation.x = best_particle.x; t.transform.translation.y = best_particle.y
        t.transform.rotation = yaw_to_quaternion(best_particle.orientation)
        self.tf_broadcaster.sendTransform(t)
        
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = best_particle.x
        ps.pose.position.y = best_particle.y
        ps.pose.orientation = t.transform.rotation
        self.pose_pub.publish(ps)
        
        ma = MarkerArray()
        for lid, lm in self.known_landmarks.items():
            m = Marker(); m.header.frame_id = "map"; m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.SPHERE; m.action = Marker.ADD; m.id = lid
            m.pose.position.x = float(lm['mu'][0]); m.pose.position.y = float(lm['mu'][1])
            m.scale.x = 0.2; m.scale.y = 0.2; m.scale.z = 0.2
            
            if lm['type'] == 'segment': m.color.r = 1.0; m.color.a = 0.8 
            elif lm['type'] == 'cluster': m.color.g = 1.0; m.color.a = 0.8 
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