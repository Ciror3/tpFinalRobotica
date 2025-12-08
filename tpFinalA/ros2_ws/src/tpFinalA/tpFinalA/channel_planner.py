import numpy as np
from geometry_msgs.msg import Twist
import math


def yaw_from_quat(q):
    """
    Converts a geometry_msgs/Quaternion to a Euler yaw angle.
    
    Args:
        q (Quaternion): The quaternion message.
        
    Returns:
        float: The yaw angle in radians.
    """
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(t3, t4)

class PurePursuit:
    def __init__(self):
        self.lookahead_dist = 0.15  
        self.linear_v = 0.08        
        self.max_w = 0.5            
        self.goal_tolerance = 0.10  

    def get_target_point(self, current_pose, path):
        """
        Finds the lookahead point on the path relative to the robot's current position.
        
        Args:
            current_pose (PoseStamped): The current pose of the robot.
            path (list): List of (x, y) tuples representing the path.
            
        Returns:
            tuple or None: The target point (x, y) or None if the path is invalid.
        """
        if not path:
            return None

        my_pos = np.array([current_pose.pose.position.x, current_pose.pose.position.y])
        final_point = np.array(path[-1])
        
        dist_to_final = np.linalg.norm(final_point - my_pos)
        if dist_to_final < self.lookahead_dist:
            return path[-1]

        target = None
        closest_idx = 0
        min_d = float('inf')
        
        for i, p in enumerate(path):
            d = np.linalg.norm(np.array(p) - my_pos)
            if d < min_d:
                min_d = d
                closest_idx = i

        for i in range(closest_idx, len(path)):
            p = path[i]
            d = np.linalg.norm(np.array(p) - my_pos)
            if d > self.lookahead_dist:
                target = p
                break
        
        if target is None:
            target = path[-1]
            
        return target
    
    def compute_command(self, current_pose, path):
        """
        Computes the velocity command (Twist) to follow the path using Pure Pursuit logic.
        
        Args:
            current_pose (PoseStamped): The current pose of the robot.
            path (list): List of (x, y) tuples representing the path.
            
        Returns:
            Twist: The velocity command message.
        """
        cmd = Twist()
        theta = yaw_from_quat(current_pose.pose.orientation)
        
        target = self.get_target_point(current_pose, path)
        if target is None:
            return cmd

        angle_to_start = math.atan2(current_pose.pose.position.y - path[0][1], current_pose.pose.position.y - path[0][0])
        heading_error = angle_to_start - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        dx = target[0] - current_pose.pose.position.x
        dy = target[1] - current_pose.pose.position.y

        target_y_local = -dx * np.sin(theta) + dy * np.cos(theta)
        
        L = np.hypot(dx, dy)

        if L < self.goal_tolerance:
            return cmd 
        
        curvature = 2.0 * target_y_local / (L**2)
        
        cmd.linear.x = self.linear_v
        cmd.angular.z = self.linear_v * curvature
        
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_w, self.max_w)
        
        if abs(cmd.angular.z) > 0.8:
            cmd.linear.x *= 0.5

        return cmd