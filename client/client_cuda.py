"""
Cliente Python simplificado para conectarse al servidor CUDA C
Se ejecuta en la LAPTOP
"""
import socket
import numpy as np
import pygame
from robot_environment import RobotEnvironment
import time
import configparser
import os

class CUDAClient:
    def __init__(self, host, port=5556):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Conecta al servidor CUDA C"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ Connected to CUDA server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def send_state(self, state):
        """Envía el estado al servidor y recibe acción"""
        try:
            # Formatear estado como string CSV
            state_str = "STATE:" + ",".join([f"{x:.6f}" for x in state])
            self.socket.sendall(state_str.encode())
            
            # Recibir respuesta
            response = self.socket.recv(1024).decode()
            parts = response.split(',')
            action = int(parts[0])
            epsilon = float(parts[1]) if len(parts) > 1 else 0.0
            
            return action, epsilon
        except Exception as e:
            print(f"Error in communication: {e}")
            self.connected = False
            return None, None
    
    def notify_episode_end(self):
        """Notifica al servidor que el episodio terminó"""
        try:
            self.socket.sendall(b"EPISODE_END")
            response = self.socket.recv(1024).decode()
            return response == "OK"
        except Exception as e:
            print(f"Error notifying episode end: {e}")
            return False

    def send_feedback(self, state, action, reward, next_state, done):
        """Envía feedback de entrenamiento al servidor"""
        try:
            # Formatear: FEEDBACK:state|action|reward|next_state|done
            state_str = ",".join([f"{x:.6f}" for x in state])
            next_state_str = ",".join([f"{x:.6f}" for x in next_state])
            
            msg = f"FEEDBACK:{state_str}|{action}|{reward:.2f}|{next_state_str}|{int(done)}"
            self.socket.sendall(msg.encode())
            
            # Esperar confirmación (breve)
            response = self.socket.recv(1024).decode()
            return response == "OK"
        except Exception as e:
            print(f"Error sending feedback: {e}")
            self.connected = False
            return False
    
    def save_model(self):
        """Solicita al servidor guardar el modelo"""
        try:
            self.socket.sendall(b"SAVE")
            response = self.socket.recv(1024).decode()
            return response == "SAVED"
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def disconnect(self):
        """Desconecta del servidor"""
        if self.connected:
            try:
                self.socket.sendall(b"QUIT")
                response = self.socket.recv(1024).decode()
                self.socket.close()
                self.connected = False
                print("✓ Disconnected from server")
            except:
                pass


def train_with_cuda_server(client, env, num_episodes=100, render=True, save_interval=50):
    """
    Entrenar usando servidor CUDA C
    """
    print("\n=== Training with CUDA Server ===")
    print(f"Episodes: {num_episodes}")
    print(f"Model will be saved every {save_interval} episodes")
    print("Press 'Q' to quit\n")
    
    best_reward = float('-inf')
    
    for episode in range(num_episodes):
        state = env.reset()
        env.episodes = episode + 1
        done = False
        episode_reward = 0
        step_count = 0
        epsilon = 0.0
        
        while not done:
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Quitting...")
                        return
            
            # Obtener acción del servidor CUDA
            action, epsilon = client.send_state(state)
            if action is None:
                print("Lost connection to CUDA server")
                return
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
        # Mostrar progreso
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Best reward: {best_reward:.2f}")

        # Enviar feedback de entrenamiento DESPUÉS de ejecutar la acción
        # Se envía: estado_actual, acción_tomada, recompensa_recibida, siguiente_estado, si_terminó
        if not client.send_feedback(state, action, reward, next_state, done):
             print("Warning: Failed to send feedback to server")

        state = next_state
        
        # Renderizar
        if render:
            env.render()
        
        # Notificar fin de episodio
        client.notify_episode_end()
        
        # Actualizar mejor recompensa
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Guardar modelo periódicamente
        if (episode + 1) % save_interval == 0:
            print(f"\n=== Saving model at episode {episode + 1} ===")
            if client.save_model():
                print("✓ Model saved successfully on Jetson")
            else:
                print("⚠ Failed to save model")
        
        # Mostrar progreso
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Epsilon: {epsilon:.4f}")
            print(f"  Best reward: {best_reward:.2f}")


def test_cuda_connection(host, port=5556):
    """Prueba rápida de conexión"""
    print("Testing CUDA server connection...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        
        # Enviar estado de prueba
        test_state = "STATE:" + ",".join(["0.5"] * 15)
        sock.sendall(test_state.encode())
        
        # Recibir respuesta
        response = sock.recv(1024).decode()
        print(f"✓ Server responded: {response}")
        
        # Cerrar
        sock.sendall(b"QUIT")
        sock.close()
        
        return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


def load_config():
    """Carga configuración desde archivo"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.ini')
    
    if os.path.exists(config_path):
        config.read(config_path)
        return config
    return None

def main():
    import argparse
    
    # Cargar configuración
    config = load_config()
    
    # Valores por defecto desde config
    default_host = config.get('network', 'jetson_ip') if config else '172.22.234.49'
    default_port = int(config.get('network', 'training_port')) if config else 5556
    default_episodes = int(config.get('training', 'episodes')) if config else 200
    default_save_interval = int(config.get('training', 'save_interval')) if config else 50
    default_width = int(config.get('environment', 'width')) if config else 400
    default_height = int(config.get('environment', 'height')) if config else 400
    default_obstacles = int(config.get('environment', 'num_obstacles')) if config else 5
    default_max_steps = int(config.get('environment', 'max_steps')) if config else 200
    
    parser = argparse.ArgumentParser(description='DQN Training Client')
    parser.add_argument('--host', type=str, default=default_host, help=f'Jetson IP (default: {default_host})')
    parser.add_argument('--port', type=int, default=default_port, help=f'Server port (default: {default_port})')
    parser.add_argument('--episodes', type=int, default=default_episodes, help=f'Number of episodes (default: {default_episodes})')
    parser.add_argument('--save-interval', type=int, default=default_save_interval, help=f'Save every N episodes (default: {default_save_interval})')
    parser.add_argument('--width', type=int, default=default_width, help=f'Environment width (default: {default_width})')
    parser.add_argument('--height', type=int, default=default_height, help=f'Environment height (default: {default_height})')
    parser.add_argument('--obstacles', type=int, default=default_obstacles, help=f'Number of obstacles (default: {default_obstacles})')
    parser.add_argument('--max-steps', type=int, default=default_max_steps, help=f'Max steps per episode (default: {default_max_steps})')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    # Test de conexión
    if args.test:
        test_cuda_connection(args.host, args.port)
        return
    
    # Crear entorno con configuración
    env = RobotEnvironment(
        width=args.width, 
        height=args.height,
        num_obstacles=args.obstacles,
        max_steps=args.max_steps
    )
    
    # Crear cliente
    client = CUDAClient(host=args.host, port=args.port)
    
    print("Connecting to CUDA server...")
    if not client.connect():
        print("Failed to connect. Make sure the CUDA server is running:")
        print(f"  On Jetson: ./bin/dqn_server_cuda")
        return
    
    try:
        # Entrenar
        train_with_cuda_server(
            client, 
            env, 
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            render=not args.no_render
        )
        
        # Guardar modelo final
        print("\n=== Saving final model ===")
        if client.save_model():
            print("✓ Final model saved on Jetson")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("Saving model before exit...")
        client.save_model()
    finally:
        client.disconnect()
        env.close()
        print("Client closed")


if __name__ == '__main__':
    main()
