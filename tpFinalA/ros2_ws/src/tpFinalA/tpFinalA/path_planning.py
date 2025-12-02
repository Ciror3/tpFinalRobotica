import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class PathPlannerNode(Node):

    def __init__(self):
        super().__init__("path_planner")

        self.occ_map = np.loadtxt('slam_map.txt')
        self.map_res = 0.05        
        self.map_origin_x = -20.0  
        self.map_origin_y = -20.0
        
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)

        self.current_pose = None
        self.pose_sub = self.create_subscription(PoseStamped, "/fpose", self.pose_callback, 10)

    def pose_callback(self, msg):
        self.current_pose = msg

    def world_to_grid(self, x_world, y_world):
        """Convertir coordenadas del mundo (metros) a índices de grilla"""
        ix = int((x_world - self.map_origin_x) / self.map_res)
        iy = int((y_world - self.map_origin_y) / self.map_res)
        return np.array([iy, ix]) 

    def grid_to_world(self, iy, ix):
        """Convertir índices de grilla a coordenadas del mundo """
        x_world = ix * self.map_res + self.map_origin_x
        y_world = iy * self.map_res + self.map_origin_y
        return x_world, y_world
    

    def get_neighborhood(self, cell, occ_map_shape):
        neighbors = []
        y, x = cell
        ny, nx = occ_map_shape

        deltas = [
            (-1,-1), (-1,0), (-1,+1),
            (0,-1),          (0,+1),
            (+1,-1), (+1,0), (+1,+1)
        ]

        for dy, dx in deltas:
            ny_cell = y + dy
            nx_cell = x + dx

            if 0<= ny_cell < ny and 0 <= nx_cell < nx:
                neighbors.append((ny_cell, nx_cell))

        return neighbors

    def get_edge_cost(self, parent, child, occ_map):
        edge_cost = 0

        py, px = parent
        cy, cx = child

        if occ_map[cy, cx] > 0.5: 
            return np.inf

        dy = cy - py
        dx = cx - px

        if abs(dx) + abs(dy) == 1:
            base_cost = 1              
        else:
            base_cost = np.sqrt(2)    

        occ_penalty = 1 + 5 * occ_map[cy, cx]
        edge_cost = base_cost * occ_penalty

        return edge_cost

    def get_heuristic(self, cell, goal):

        cy, cx = cell
        gy, gx = goal

        dy = gy - cy
        dx = gx - cx

        heuristic = np.sqrt(dx*dx + dy*dy)

        return heuristic


    def plan_path(self, start, goal):
        costs = np.ones(self.occ_map.shape) * np.inf
        closed_flags = np.zeros(self.occ_map.shape)
        predecessors = -np.ones(self.occ_map.shape + (2,), dtype=int)

        heuristic = np.zeros(self.occ_map.shape)
        for x in range(self.occ_map.shape[0]):
            for y in range(self.occ_map.shape[1]):
                heuristic[x, y] = self.get_heuristic([x, y], goal)

        parent = start
        costs[start[0], start[1]] = 0

        while not np.array_equal(parent, goal):
            open_costs = np.where(closed_flags==1, np.inf, costs) + heuristic
            x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)
            
            if open_costs[x, y] == np.inf:
                break
            
            parent = np.array([x, y])
            closed_flags[x, y] = 1
            
            neighbors = self.get_neighborhood(parent, self.occ_map.shape)  

            for child in neighbors:
                cy, cx = child

                if closed_flags[cy, cx] == 1:
                    continue

                edge = self.get_edge_cost(parent, child, self.occ_map) 
                if edge == np.inf:
                    continue

                new_cost = costs[parent[0], parent[1]] + edge

                if new_cost < costs[cy, cx]:
                    costs[cy, cx] = new_cost
                    predecessors[cy, cx] = parent
        
        path_list = []
        if np.array_equal(parent, goal):
            while predecessors[parent[0], parent[1]][0] >= 0:
                path_list.append(parent.copy())  
                parent = predecessors[parent[0], parent[1]]
            path_list.append(start)
            path_list.reverse()
            
        return np.array(path_list) if path_list else None
    
    def goal_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().warn("No robot pose yet!")
            return
        
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        goal = self.world_to_grid(goal_x, goal_y)
        
        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y
        start = self.world_to_grid(start_x, start_y)
        
        path = self.plan_path(start, goal)
        
        if path is not None:
            self.publish_path(path)
        else:
            self.get_logger().warn("No path found!")

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for cell in path:
            x_world, y_world = self.grid_to_world(int(cell[0]), int(cell[1]))
            
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(x_world)
            pose.pose.position.y = float(y_world)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)
        self.get_logger().info("Path Published.")

def main():
    rclpy.init()
    node = PathPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
  main()
