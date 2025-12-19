"""
Entorno de simulación del robot con Pygame
Se ejecuta en la LAPTOP
"""
import pygame
import numpy as np
import random
from typing import Tuple, List

class RobotEnvironment:
    def __init__(self, width=400, height=400, grid_size=20, num_obstacles=5, max_steps=200):
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DQN Robot Navigation")
        
        # Colores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.GRAY = (128, 128, 128)
        
        # Estado del robot
        self.robot_pos = [0, 0]
        self.goal_pos = [0, 0]
        self.obstacles = []
        
        # Parámetros del entorno
        self.max_steps = max_steps
        self.current_step = 0
        
        # Métricas
        self.total_reward = 0
        self.episodes = 0
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
    def reset(self) -> np.ndarray:
        """Reinicia el entorno y retorna el estado inicial"""
        # Generar posición inicial del robot (en esquinas o bordes)
        positions = [
            [self.grid_size, self.grid_size],  # Esquina superior izquierda
            [self.width - 2*self.grid_size, self.grid_size],  # Superior derecha
            [self.grid_size, self.height - 2*self.grid_size],  # Inferior izquierda
            [self.width - 2*self.grid_size, self.height - 2*self.grid_size],  # Inferior derecha
        ]
        self.robot_pos = random.choice(positions).copy()
        
        # Generar posición del objetivo (en el lado opuesto)
        # Si robot está arriba, objetivo abajo. Si está izquierda, objetivo derecha
        grid_cols = self.width // self.grid_size
        grid_rows = self.height // self.grid_size
        
        robot_col = self.robot_pos[0] // self.grid_size
        robot_row = self.robot_pos[1] // self.grid_size
        
        # Objetivo en el cuadrante opuesto
        if robot_col < grid_cols // 2:
            goal_col = random.randint(grid_cols // 2 + 1, grid_cols - 2)
        else:
            goal_col = random.randint(1, grid_cols // 2 - 1)
            
        if robot_row < grid_rows // 2:
            goal_row = random.randint(grid_rows // 2 + 1, grid_rows - 2)
        else:
            goal_row = random.randint(1, grid_rows // 2 - 1)
            
        self.goal_pos = [goal_col * self.grid_size, goal_row * self.grid_size]
        
        # Generar obstáculos aleatorios (menos obstáculos)
        self.obstacles = self._generate_obstacles(num_obstacles=self.num_obstacles)
        
        self.current_step = 0
        self.total_reward = 0
        
        return self._get_state()
    
    def _generate_obstacles(self, num_obstacles: int) -> List[List[int]]:
        """Genera obstáculos que no bloqueen completamente el camino"""
        obstacles = []
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 50:
                obs_pos = [
                    random.randint(1, self.width // self.grid_size - 2) * self.grid_size,
                    random.randint(1, self.height // self.grid_size - 2) * self.grid_size
                ]
                # Evitar colocar obstáculos sobre el robot o el objetivo
                if (obs_pos != self.robot_pos and 
                    obs_pos != self.goal_pos and
                    obs_pos not in obstacles):
                    obstacles.append(obs_pos)
                    break
                attempts += 1
        return obstacles
    
    def _get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual del entorno
        Estado: [robot_x, robot_y, goal_x, goal_y, dist_to_goal, 
                 nearest_obstacle_dist, angle_to_goal, 8 sensores de proximidad]
        """
        # Posición normalizada
        robot_x = self.robot_pos[0] / self.width
        robot_y = self.robot_pos[1] / self.height
        goal_x = self.goal_pos[0] / self.width
        goal_y = self.goal_pos[1] / self.height
        
        # Distancia al objetivo (normalizada)
        dist_to_goal = self._calculate_distance(self.robot_pos, self.goal_pos) / np.sqrt(self.width**2 + self.height**2)
        
        # Distancia al obstáculo más cercano (normalizada)
        nearest_obstacle_dist = self._get_nearest_obstacle_distance() / np.sqrt(self.width**2 + self.height**2)
        
        # Ángulo hacia el objetivo
        angle_to_goal = self._calculate_angle_to_goal()
        
        # Sensores de proximidad en 8 direcciones
        proximity_sensors = self._get_proximity_sensors()
        
        state = np.array([
            robot_x, robot_y, goal_x, goal_y, 
            dist_to_goal, nearest_obstacle_dist, angle_to_goal
        ] + proximity_sensors, dtype=np.float32)
        
        return state
    
    def _get_proximity_sensors(self) -> List[float]:
        """8 sensores de proximidad alrededor del robot"""
        sensor_range = 100  # pixels
        directions = [
            (0, -1),   # Norte
            (1, -1),   # Noreste
            (1, 0),    # Este
            (1, 1),    # Sureste
            (0, 1),    # Sur
            (-1, 1),   # Suroeste
            (-1, 0),   # Oeste
            (-1, -1)   # Noroeste
        ]
        
        sensors = []
        for dx, dy in directions:
            # Verificar distancia a obstáculos en esta dirección
            min_dist = sensor_range
            for step in range(1, sensor_range, self.grid_size):
                check_x = self.robot_pos[0] + dx * step
                check_y = self.robot_pos[1] + dy * step
                
                # Verificar límites
                if check_x < 0 or check_x >= self.width or check_y < 0 or check_y >= self.height:
                    min_dist = step
                    break
                
                # Verificar obstáculos
                for obs in self.obstacles:
                    if abs(check_x - obs[0]) < self.grid_size and abs(check_y - obs[1]) < self.grid_size:
                        min_dist = step
                        break
                
                if min_dist < sensor_range:
                    break
            
            sensors.append(min_dist / sensor_range)  # Normalizado
        
        return sensors
    
    def _calculate_distance(self, pos1: List[int], pos2: List[int]) -> float:
        """Calcula la distancia euclidiana entre dos posiciones"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _calculate_angle_to_goal(self) -> float:
        """Calcula el ángulo hacia el objetivo (normalizado entre -1 y 1)"""
        dx = self.goal_pos[0] - self.robot_pos[0]
        dy = self.goal_pos[1] - self.robot_pos[1]
        angle = np.arctan2(dy, dx)
        return angle / np.pi  # Normalizar entre -1 y 1
    
    def _get_nearest_obstacle_distance(self) -> float:
        """Obtiene la distancia al obstáculo más cercano"""
        if not self.obstacles:
            return 1000  # Valor grande si no hay obstáculos
        
        distances = [self._calculate_distance(self.robot_pos, obs) for obs in self.obstacles]
        return min(distances)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Ejecuta una acción en el entorno
        Acciones: 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha, 
                 4=Arriba-Derecha, 5=Arriba-Izquierda, 
                 6=Abajo-Derecha, 7=Abajo-Izquierda
        """
        self.current_step += 1
        
        # Guardar posición anterior
        old_distance = self._calculate_distance(self.robot_pos, self.goal_pos)
        old_pos = self.robot_pos.copy()
        
        # Ejecutar acción
        if action == 0:  # Arriba
            self.robot_pos[1] -= self.grid_size
        elif action == 1:  # Abajo
            self.robot_pos[1] += self.grid_size
        elif action == 2:  # Izquierda
            self.robot_pos[0] -= self.grid_size
        elif action == 3:  # Derecha
            self.robot_pos[0] += self.grid_size
        elif action == 4:  # Arriba-Derecha
            self.robot_pos[0] += self.grid_size
            self.robot_pos[1] -= self.grid_size
        elif action == 5:  # Arriba-Izquierda
            self.robot_pos[0] -= self.grid_size
            self.robot_pos[1] -= self.grid_size
        elif action == 6:  # Abajo-Derecha
            self.robot_pos[0] += self.grid_size
            self.robot_pos[1] += self.grid_size
        elif action == 7:  # Abajo-Izquierda
            self.robot_pos[0] -= self.grid_size
            self.robot_pos[1] += self.grid_size
        
        # Verificar colisiones y límites
        collision = False
        out_of_bounds = False
        
        # Verificar límites
        if (self.robot_pos[0] < 0 or self.robot_pos[0] >= self.width or
            self.robot_pos[1] < 0 or self.robot_pos[1] >= self.height):
            self.robot_pos = old_pos
            out_of_bounds = True
        
        # Verificar colisión con obstáculos
        for obs in self.obstacles:
            if abs(self.robot_pos[0] - obs[0]) < self.grid_size and \
               abs(self.robot_pos[1] - obs[1]) < self.grid_size:
                self.robot_pos = old_pos
                collision = True
                break
        
        # Calcular recompensa
        reward = self._calculate_reward(old_distance, collision, out_of_bounds)
        self.total_reward += reward
        
        # Verificar si alcanzó el objetivo
        done = False
        new_distance = self._calculate_distance(self.robot_pos, self.goal_pos)
        if new_distance < self.grid_size * 1.5:
            reward += 100  # Bonificación por llegar al objetivo
            done = True
        
        # Verificar si excedió el límite de pasos
        if self.current_step >= self.max_steps:
            done = True
        
        # Obtener nuevo estado
        next_state = self._get_state()
        
        info = {
            'collision': collision,
            'out_of_bounds': out_of_bounds,
            'distance_to_goal': new_distance,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, old_distance: float, collision: bool, out_of_bounds: bool) -> float:
        """Calcula la recompensa basada en la acción"""
        new_distance = self._calculate_distance(self.robot_pos, self.goal_pos)
        
        # Recompensa GRANDE por acercarse al objetivo (multiplicado por 5)
        distance_reward = (old_distance - new_distance) / self.grid_size * 5.0
        reward = distance_reward
        
        # Penalización FUERTE por estar lejos del objetivo
        normalized_distance = new_distance / np.sqrt(self.width**2 + self.height**2)
        if normalized_distance > 0.5:  # Si está muy lejos
            reward -= 5.0 * normalized_distance
        
        # Recompensa por estar cerca del objetivo
        if new_distance < 100:
            reward += 2.0
        if new_distance < 50:
            reward += 5.0
        
        # Penalización por colisión
        if collision:
            reward -= 15.0
        
        # Penalización por salirse de los límites
        if out_of_bounds:
            reward -= 15.0
        
        # Pequeña penalización por cada paso
        reward -= 0.5
        
        return reward
    
    def render(self, mode='human'):
        """Renderiza el entorno"""
        # Limpiar pantalla
        self.screen.fill(self.WHITE)
        
        # Dibujar grid
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y), 1)
        
        # Dibujar obstáculos
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.BLACK, 
                           (obs[0], obs[1], self.grid_size, self.grid_size))
        
        # Dibujar objetivo
        pygame.draw.rect(self.screen, self.GREEN, 
                        (self.goal_pos[0], self.goal_pos[1], self.grid_size, self.grid_size))
        
        # Dibujar robot
        pygame.draw.rect(self.screen, self.BLUE, 
                        (self.robot_pos[0], self.robot_pos[1], self.grid_size, self.grid_size))
        
        # Dibujar información
        info_text = [
            f"Episode: {self.episodes}",
            f"Step: {self.current_step}/{self.max_steps}",
            f"Reward: {self.total_reward:.2f}",
            f"Distance: {self._calculate_distance(self.robot_pos, self.goal_pos):.1f}"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
    
    def close(self):
        """Cierra el entorno"""
        pygame.quit()
    
    def get_state_size(self) -> int:
        """Retorna el tamaño del estado"""
        return 15  # 7 características básicas + 8 sensores
    
    def get_action_size(self) -> int:
        """Retorna el número de acciones posibles"""
        return 8  # 8 direcciones posibles