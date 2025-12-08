import numpy as np
import heapq
import math
import cv2
import json
import os

class DStarLite:
    def __init__(self, map_pgm_path, landmarks_json_path, origin_x, origin_y, res):
        # Configuración del mapa
        self.map_res = res
        self.map_origin_x = origin_x
        self.map_origin_y = origin_y
        
        # Cargar mapas
        self.static_map = self.load_pgm_map(map_pgm_path)
        self.rows, self.cols = self.static_map.shape
        self.inject_landmarks(landmarks_json_path)
        
        # Mapa de costos actual
        self.cost_map = self.static_map.copy()
        
        # Estado D* Lite
        self.g = {}
        self.rhs = {}
        self.OPEN = []        # Heap (min-priority queue)
        self.open_set = {}    # Hash map para búsqueda rápida {nodo: clave}
        self.km = 0
        self.start = None
        self.goal = None
        self.last_start = None
        
        # Optimización: Set para rastrear obstáculos dinámicos activos
        self.active_dynamic_obstacles = set()
        
        # Pre-cache de movimientos (8-conectividad)
        # (dy, dx, costo)
        self.moves = [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), 
                      (1,1,1.414), (1,-1,1.414), (-1,1,1.414), (-1,-1,1.414)]

    def load_pgm_map(self, filepath):
        if not os.path.exists(filepath): return np.zeros((400, 400), dtype=int)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None: return np.zeros((400, 400), dtype=int)
        
        grid = np.zeros_like(img, dtype=int)
        grid[img < 250] = 100 
        grid[img >= 250] = 0  
        return np.flipud(grid)

    def inject_landmarks(self, filepath):
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

    # --- CORE D* LITE ---
    def heuristic(self, s1, s2):
        # Distancia Octile (más rápida y precisa para grillas que sqrt puro)
        dy = abs(s1[0] - s2[0])
        dx = abs(s1[1] - s2[1])
        return (dx + dy) + (1.414 - 2) * min(dx, dy)

    def calculate_key(self, s):
        # Usar .get es lento en bucles, acceso directo es preferible si se gestiona bien, 
        # pero mantenemos .get por seguridad de inicialización.
        g_val = self.g.get(s, float('inf'))
        rhs_val = self.rhs.get(s, float('inf'))
        min_val = min(g_val, rhs_val)
        return (min_val + self.heuristic(self.start, s) + self.km, min_val)

    def get_neighbors(self, s):
        y, x = s
        neighbors = []
        for dy, dx, cost in self.moves:
            ny, nx = y + dy, x + dx
            # Chequeo de límites
            if 0 <= ny < self.rows and 0 <= nx < self.cols:
                # Chequeo de transitabilidad
                if self.cost_map[ny, nx] < 100: 
                    neighbors.append(((ny, nx), cost))
        return neighbors

    def update_vertex(self, u):
        if u != self.goal:
            min_rhs = float('inf')
            for sprime, cost in self.get_neighbors(u):
                # Inlining simple de la suma para evitar llamadas a funciones
                g_sprime = self.g.get(sprime, float('inf'))
                if g_sprime < float('inf'):
                    temp = cost + g_sprime
                    if temp < min_rhs: min_rhs = temp
            self.rhs[u] = min_rhs
        
        # --- OPTIMIZACIÓN DE HEAP (LAZY REMOVAL) ---
        # Si el nodo está en la lista OPEN, lo marcamos como "sucio" eliminándolo del dict
        # No lo sacamos del heap físicamente (costoso), solo insertamos la nueva versión.
        if u in self.open_set:
            del self.open_set[u] 
        
        # Si es inconsistente, lo insertamos con su nueva clave
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            k = self.calculate_key(u)
            heapq.heappush(self.OPEN, (k, u))
            self.open_set[u] = k # Registramos la clave válida actual

    def compute_shortest_path(self):
        while self.OPEN:
            # --- POP PEREZOSO ---
            # Sacamos el elemento con menor clave
            k_old, u = heapq.heappop(self.OPEN)
            
            # Verificamos si este elemento es válido
            # Si u no está en open_set, o la clave no coincide, es una versión antigua (basura)
            if u not in self.open_set or self.open_set[u] != k_old:
                continue
            
            # Condición de parada
            k_new = self.calculate_key(u)
            start_key = self.calculate_key(self.start)
            
            if k_old < start_key or self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf')):
                pass # Continuar expandiendo
            else:
                # Ya terminamos (el start es consistente y tiene la menor prioridad)
                # Reinsertamos el nodo actual porque era válido y no lo procesamos
                heapq.heappush(self.OPEN, (k_old, u)) 
                break

            # Si la clave actual es vieja (k_old < k_new), actualizamos
            if k_old < k_new:
                heapq.heappush(self.OPEN, (k_new, u))
                self.open_set[u] = k_new
            
            elif self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                # Sacamos de open_set porque ahora es consistente (g == rhs)
                if u in self.open_set: del self.open_set[u]
                for s, _ in self.get_neighbors(u): self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s, _ in self.get_neighbors(u): self.update_vertex(s)

    # --- INTERFAZ PÚBLICA ---
    def set_goal(self, start_world, goal_world):
        self.start = self.world_to_grid(*start_world)
        self.goal = self.world_to_grid(*goal_world)
        self.last_start = self.start
        
        # Reset rápido
        self.g.clear()
        self.rhs.clear()
        self.OPEN.clear()
        self.open_set.clear() # Reset del hash map
        self.km = 0
        
        self.rhs[self.goal] = 0
        k = self.calculate_key(self.goal)
        heapq.heappush(self.OPEN, (k, self.goal))
        self.open_set[self.goal] = k
        
        self.compute_shortest_path()

    def update_obstacles(self, robot_pos_world, new_dynamic_set):
        """
        Recibe un SET de celdas ocupadas.
        Usa diferencia de conjuntos para O(1) en lugar de escanear la matriz.
        """
        if self.start is None or self.goal is None: return

        new_start = self.world_to_grid(*robot_pos_world)
        if self.start != new_start:
            self.km += self.heuristic(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start

        # --- OPTIMIZACIÓN DE CAMBIOS ---
        # 1. Obstáculos que desaparecieron (estaban activos, pero no en el nuevo set)
        removed_obstacles = self.active_dynamic_obstacles - new_dynamic_set
        
        # 2. Obstáculos que aparecieron (están en el nuevo set, pero no estaban activos)
        added_obstacles = new_dynamic_set - self.active_dynamic_obstacles
        
        # Si no hubo cambios reales y el robot no se movió mucho, salir
        if not removed_obstacles and not added_obstacles and self.km == 0:
            return

        # Actualizar mapa y vértices
        for (y, x) in removed_obstacles:
            # Restaurar valor estático (si era 0, vuelve a 0)
            if self.static_map[y, x] == 0:
                self.cost_map[y, x] = 0
                self.update_vertex((y, x))
                # Propagar a vecinos
                for dy, dx, _ in self.moves:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        self.update_vertex((ny, nx))

        for (y, x) in added_obstacles:
            if self.cost_map[y, x] == 0: # Solo si estaba libre
                self.cost_map[y, x] = 100
                self.g[(y,x)] = float('inf')
                self.rhs[(y,x)] = float('inf')
                self.update_vertex((y, x))
                # Propagar a vecinos
                for dy, dx, _ in self.moves:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.rows and 0 <= nx < self.cols:
                        self.update_vertex((ny, nx))

        # Actualizar el registro de obstáculos activos
        self.active_dynamic_obstacles = new_dynamic_set.copy()

        self.compute_shortest_path()

    def get_path_world(self):
        if self.start is None or self.goal is None: return None
        if self.g.get(self.start, float('inf')) == float('inf'): return None

        path_world = []
        curr = self.start
        path_world.append(self.grid_to_world(*curr))
        
        max_steps = 2000 # Evitar loops infinitos
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
        gx = int((x - self.map_origin_x) / self.map_res)
        gy = int((y - self.map_origin_y) / self.map_res)
        gx = max(0, min(self.cols - 1, gx))
        gy = max(0, min(self.rows - 1, gy))
        return (gy, gx)

    def grid_to_world(self, gy, gx):
        wx = (gx * self.map_res) + self.map_origin_x
        wy = (gy * self.map_res) + self.map_origin_y
        return wx, wy