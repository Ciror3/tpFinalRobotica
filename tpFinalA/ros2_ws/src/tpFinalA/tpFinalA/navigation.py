import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quat(q):
    yaw = np.arctan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return yaw, cos_yaw, sin_yaw


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class my_node(Node): 
    def __init__(self):
        super().__init__("RobotMovement") 

        self.turn_angle = 160 * np.pi/180
        self.scan_msg = None
        self.odom_msg = None
        self.rotating = False
        self.yaw_ref = None
        self.turn_speed = 0.8
        self.opening_angle = np.pi/4
        self.sub_belief = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.sub_odom = self.create_subscription(
            Odometry, "/calc_odom", self.odom_callback, 10
        )

        self.sub_landmarks = self.create_subscription(
            MarkerArray, "/fastslam/markers", self.landmarks_callback, 10
        )

        self.publish_vel = self.create_publisher(
            Twist, "/cmd_vel", 10
        )


        self.timer = self.create_timer(0.1, self.state_machine)

        self.get_logger().info("Iniciando el movimiento")   

    def scan_callback(self, msg:LaserScan):
        self.scan_msg = msg
    
    def odom_callback(self, msg:Odometry):
        self.yaw, self.cos_yaw, self.sin_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def landmarks_callback(self, msg:MarkerArray):
        obstacles = []

        for marker in msg.markers:           
            if marker.type == Marker.CYLINDER:
                continue

            if marker.type in [Marker.SPHERE, Marker.CUBE]:
                obstacles.append([marker.pose.position.x, marker.pose.position.y])

            elif marker.type in [Marker.SPHERE_LIST, Marker.POINTS]:
                for p in marker.points:
                    obstacles.append([p.x, p.y])
            
            elif marker.type in [Marker.LINE_LIST, Marker.LINE_STRIP]:
                for p in marker.points:
                    obstacles.append([p.x, p.y])

        self.obstacles_np = np.array(obstacles)
        self.state_machine()


    def state_machine(self):
        scan = self.scan_msg
        if scan is None:
            return

        if self.rotating and self.yaw_ref is not None:
            err = wrap(self.yaw_ref - self.yaw)
            if abs(err) < 0.05:
                self.rotating = False
                self.yaw_ref = None
                self.publish_cmd_vel()
            else:
                self.publish_rotation(sign=1.0 if err > 0 else -1.0)
            return

        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        mask = (angles >= -self.opening_angle) & (angles <= self.opening_angle)
        ranges = ranges[mask]

        for d in ranges:
            if d < scan.range_min or d > scan.range_max:
                continue
            if d < 0.5:
                self.yaw_ref = wrap(self.yaw + self.turn_angle)
                self.rotating = True
                self.publish_rotation(sign=1.0)  
                return
            else:  
                self.publish_cmd_vel()
                return
        self.publish_cmd_vel()



    def publish_rotation(self, sign=1.0):
        msg = Twist()
        msg.angular.z = sign * self.turn_speed
        self.publish_vel.publish(msg)

    def publish_cmd_vel(self):
        msg = Twist()
        msg.linear.x  = 0.2
        msg.angular.z = 0.0
        self.publish_vel.publish(msg)

def main():
    rclpy.init()
    node = my_node()
    rclpy.spin(node)
    rclpy.shutdown()