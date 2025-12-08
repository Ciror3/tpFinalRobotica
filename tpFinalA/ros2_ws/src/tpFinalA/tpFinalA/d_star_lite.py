import numpy as np
import heapq
import math
import cv2
import json
import os

class DStarLite:
    def __init__(self, map_pgm_path, landmarks_json_path, origin_x, origin_y, res):
        self.map_res = res
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        
        self.static_map = self.load_pgm_map(map_pgm_path)
        self.rows, self.cols = self.static_map.shape
        self.inject_landmarks(landmarks_json_path)
        
        self.cost_map = self.static_map.copy()
        
        self.g = {}
        self.rhs = {}
        self.OPEN = []        
        self.open_set = {}    
        self.km = 0
        self.start = None
        self.goal = None
        self.last_start = None
        
        self.active_dynamic_obstacles = set()
        
        self.moves = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), 
                      (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]

    def load_pgm_map(self, filepath):
        """
        Loads a PGM map file and converts it into a binary occupancy grid.
        
        Args:
            filepath (str): Path to the .pgm map file.
            
        Returns:
            np.ndarray: A 2D numpy array where 0 represents free space and 100 represents obstacles.
        """
        if not os.path.exists(filepath): return np.zeros((400, 400), dtype=int)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((400, 400), dtype=int)
        
        grid = np.zeros_like(img, dtype=int)
        grid[img < 250] = 100 
        grid[img >= 250] = 0  
        return np.flipud(grid)

    def inject_landmarks(self, filepath):
        """
        Reads a JSON file containing landmark positions and marks them as obstacles on the static map.
        
        Args:
            filepath (str): Path to the .json landmarks file.
        """
        if not os.path.exists(filepath): return
        try:
            with open(filepath, 'r') as f: landmarks = json.load(f)
            radius_px = 1
            for lm in landmarks:
                gy, gx = self.world_to_grid(lm['x'], lm['y'])
                for dy in range(-radius_px, radius_px + 1):
                    for dx in range(-radius_px, radius_px + 1):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < self.rows and 0 <= nx < self.cols:
                            self.static_map[ny, nx] = 100
        except: pass

    def heuristic(self, s1, s2):
        """
        Calculates the Octile distance heuristic between two grid points.
        
        Args:
            s1 (tuple): Coordinates (y, x) of the first point.
            s2 (tuple): Coordinates (y, x) of the second point.
            
        Returns:
            float: The heuristic distance.
        """
        dy = abs(s1[0] - s2[0])
        dx = abs(s1[1] - s2[1])
        return (dx + dy) + (1.414 - 2) * min(dx, dy)

    def calculate_key(self, s):
        """
        Calculates the priority key for a node in the D* Lite priority queue.
        
        Args:
            s (tuple): The grid coordinates (y, x) of the node.
            
        Returns:
            tuple: A tuple (k1, k2) used for sorting in the priority queue.
        """
        g_val = self.g.get(s, float('inf'))
        rhs_val = self.rhs.get(s, float('inf'))
        min_val = min(g_val, rhs_val)
        return (min_val + self.heuristic(self.start, s) + self.km, min_val)

    def get_neighbors(self, s):
        """
        Retrieves valid neighboring nodes for a given grid cell.
        
        Args:
            s (tuple): The grid coordinates (y, x) of the current node.
            
        Returns:
            list: A list of tuples ((y, x), cost) for valid, traversable neighbors.
        """
        y, x = s
        neighbors = []
        for dy, dx, cost in self.moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                if self.cost_map[ny, nx] < 100: 
                    neighbors.append(((ny, nx), cost))
        return neighbors

    def update_vertex(self, u):
        """
        Updates the consistency of a node and its status in the priority queue.
        
        Args:
            u (tuple): The grid coordinates (y, x) of the node to update.
        """
        if u != self.goal:
            min_rhs = float('inf')
            for sprime, cost in self.get_neighbors(u):
                g_sprime = self.g.get(sprime, float('inf'))
                if g_sprime < float('inf'):
                    temp = cost + g_sprime
                    if temp < min_rhs: min_rhs = temp
            self.rhs[u] = min_rhs
        
        if u in self.open_set:
            del self.open_set[u] 
        
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            k = self.calculate_key(u)
            heapq.heappush(self.OPEN, (k, u))
            self.open_set[u] = k 

    def compute_shortest_path(self):
        """
        Main D* Lite loop that processes the priority queue to compute the optimal path.
        """
        while self.OPEN:
            k_old, u = heapq.heappop(self.OPEN)
            
            if u not in self.open_set or self.open_set[u] != k_old:
                continue
            
            k_new = self.calculate_key(u)
            start_key = self.calculate_key(self.start)
            
            if k_old < start_key or self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf')):
                pass 
            else:
                heapq.heappush(self.OPEN, (k_old, u)) 
                break

            if k_old < k_new:
                heapq.heappush(self.OPEN, (k_new, u))
                self.open_set[u] = k_new
            
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                if u in self.open_set: del self.open_set[u]
                for s, _ in self.get_neighbors(u): self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s, _ in self.get_neighbors(u): self.update_vertex(s)

    def set_goal(self, start_world, goal_world):
        """
        Sets a new start and goal position in world coordinates and initializes the search.
        
        Args:
            start_world (tuple): The starting position (x, y) in meters.
            goal_world (tuple): The goal position (x, y) in meters.
        """
        self.start = self.world_to_grid(*start_world)
        self.goal = self.world_to_grid(*goal_world)
        self.last_start = self.start
        
        self.g.clear()
        self.rhs.clear()
        self.OPEN.clear()
        self.open_set.clear() 
        self.km = 0
        
        self.rhs[self.goal] = 0
        k = self.calculate_key(self.goal)
        heapq.heappush(self.OPEN, (k, self.goal))
        self.open_set[self.goal] = k
        
        self.compute_shortest_path()

    def update_obstacles(self, robot_pos_world, new_dynamic_set):
        """
        Updates the map with new dynamic obstacles and triggers a path repair.
        
        Args:
            robot_pos_world (tuple): Current robot position (x, y) in meters.
            new_dynamic_set (set): A set of grid coordinates (y, x) representing currently detected obstacles.
        """
        if self.start is None or self.goal is None: return

        new_start = self.world_to_grid(*robot_pos_world)
        if self.start != new_start:
            self.km += self.heuristic(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start

        removed_obstacles = self.active_dynamic_obstacles - new_dynamic_set
        added_obstacles = new_dynamic_set - self.active_dynamic_obstacles
        
        if not removed_obstacles and not added_obstacles and self.km == 0:
            return

        for (y, x) in removed_obstacles:
            if self.static_map[y, x] == 0:
                self.cost_map[y, x] = 0
                self.update_vertex((y, x))
                for dy, dx, _ in self.moves:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        self.update_vertex((ny, nx))

        for (y, x) in added_obstacles:
            if self.cost_map[y, x] == 0: 
                self.cost_map[y, x] = 100
                self.g[(y,x)] = float('inf')
                self.rhs[(y,x)] = float('inf')
                self.update_vertex((y, x))
                for dy, dx, _ in self.moves:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        self.update_vertex((ny, nx))

        self.active_dynamic_obstacles = new_dynamic_set.copy()

        self.compute_shortest_path()

    def get_path_world(self):
        """
        Reconstructs the optimal path from start to goal in world coordinates.
        
        Returns:
            list: A list of (x, y) tuples representing the path in meters, or None if no path exists.
        """
        if self.start is None or self.goal is None: return None
        if self.g.get(self.start, float('inf')) == float('inf'): return None

        path_world = []
        curr = self.start
        path_world.append(self.grid_to_world(*curr))
        
        max_steps = 2000 
        steps = 0
        
        while curr != self.goal and steps < max_steps:
            steps += 1
            min_c = float('inf')
            best_n = None
            
            for n, cost in self.get_neighbors(curr):
                c = cost + self.g.get(n, float('inf'))
                if c < min_c:
                    min_c = c
                    best_n = n
            
            if best_n is None: break
            curr = best_n
            path_world.append(self.grid_to_world(*curr))
            
        return path_world

    def world_to_grid(self, x, y):
        """
        Converts world coordinates (meters) to grid indices.
        
        Args:
            x (float): X coordinate in meters.
            y (float): Y coordinate in meters.
            
        Returns:
            tuple: (row, col) grid indices.
        """
        gx = int((x - self.map_origin_x) / self.map_res)
        gy = int((y - self.map_origin_y) / self.map_res)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))
        return (gy, gx)

    def grid_to_world(self, gy, gx):
        """
        Converts grid indices to world coordinates (meters).
        
        Args:
            gy (int): Grid row index.
            gx (int): Grid column index.
            
        Returns:
            tuple: (x, y) coordinates in meters.
        """
        wx = (gx * self.map_res) + self.map_origin_x
        wy = (gy * self.map_res) + self.map_origin_y
        return wx, wy