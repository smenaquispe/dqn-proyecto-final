"""
Cliente de Inferencia Python para DQN CUDA
Se ejecuta en la LAPTOP
Usa modelo entrenado en el Jetson para controlar el robot
"""
import socket
import numpy as np
import pygame
from robot_environment import RobotEnvironment
import time
import argparse
import configparser
import os

class InferenceClient:
    def __init__(self, host, port=5557):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Conecta al servidor de inferencia"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✓ Connected to inference server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def get_action(self, state):
        """Envía el estado y recibe acción con Q-values"""
        try:
            # Formatear estado como string CSV
            state_str = "STATE:" + ",".join([f"{x:.6f}" for x in state])
            self.socket.sendall(state_str.encode())
            
            # Recibir respuesta: action,q0,q1,q2,...,q7
            response = self.socket.recv(1024).decode()
            parts = response.split(',')
            
            action = int(parts[0])
            q_values = [float(q) for q in parts[1:]] if len(parts) > 1 else [0.0] * 8
            
            return action, q_values
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


def run_inference(client, env, num_episodes=10, render=True, delay=0.05):
    """
    Ejecutar inferencia usando el modelo entrenado
    """
    print("\n=== Running Inference with Trained Model ===")
    print(f"Episodes: {num_episodes}")
    print(f"Server: {client.host}:{client.port}")
    print("Press 'Q' to quit\n")
    
    total_rewards = []
    total_steps = []
    successes = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        env.episodes = episode + 1
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        while not done:
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total_rewards, total_steps, successes
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        print("Quitting...")
                        return total_rewards, total_steps, successes
            
            # Obtener acción del servidor (inferencia)
            action, q_values = client.get_action(state)
            if action is None:
                print("Lost connection to inference server")
                return total_rewards, total_steps, successes
            
            # Ejecutar acción
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
            state = next_state
            
            # Renderizar
            if render:
                env.render()
                
                # Mostrar Q-values en consola cada 10 steps
                if step_count % 10 == 0:
                    action_names = ['↑', '↓', '←', '→', '↗', '↖', '↘', '↙']
                    print(f"  Step {step_count}: Action={action_names[action]}, Max Q={max(q_values):.3f}")
                
                # Pequeño delay para visualizar mejor
                time.sleep(delay)
        
        # Notificar fin de episodio
        client.notify_episode_end()
        
        # Guardar estadísticas
        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        
        # Verificar si llegó al objetivo
        distance_to_goal = env._calculate_distance(env.robot_pos, env.goal_pos)
        success = distance_to_goal < env.grid_size * 1.5
        if success:
            successes += 1
        
        # Mostrar resumen del episodio
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1} Summary:")
        print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Final distance to goal: {distance_to_goal:.1f}")
        print(f"  Success rate: {successes}/{episode+1} ({100*successes/(episode+1):.1f}%)")
        print(f"{'='*50}")
    
    return total_rewards, total_steps, successes


def print_statistics(rewards, steps, successes, num_episodes):
    """Imprime estadísticas finales"""
    print("\n" + "="*60)
    print("INFERENCE STATISTICS")
    print("="*60)
    print(f"Total episodes: {num_episodes}")
    print(f"Successful episodes: {successes} ({100*successes/num_episodes:.1f}%)")
    print(f"\nRewards:")
    print(f"  Average: {np.mean(rewards):.2f}")
    print(f"  Best: {np.max(rewards):.2f}")
    print(f"  Worst: {np.min(rewards):.2f}")
    print(f"  Std Dev: {np.std(rewards):.2f}")
    print(f"\nSteps:")
    print(f"  Average: {np.mean(steps):.1f}")
    print(f"  Best: {np.min(steps)}")
    print(f"  Worst: {np.max(steps)}")
    print("="*60)


def test_connection(host, port=5557):
    """Prueba rápida de conexión al servidor de inferencia"""
    print("Testing inference server connection...")
    
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
    # Cargar configuración
    config = load_config()
    
    # Valores por defecto desde config
    default_host = config.get('network', 'jetson_ip') if config else '172.22.234.49'
    default_port = int(config.get('network', 'inference_port')) if config else 5557
    default_episodes = int(config.get('inference', 'episodes')) if config else 10
    default_delay = float(config.get('inference', 'delay')) if config else 0.05
    default_width = int(config.get('environment', 'width')) if config else 400
    default_height = int(config.get('environment', 'height')) if config else 400
    default_obstacles = int(config.get('environment', 'num_obstacles')) if config else 5
    default_max_steps = int(config.get('environment', 'max_steps')) if config else 200
    
    parser = argparse.ArgumentParser(description='DQN Inference Client')
    parser.add_argument('--host', type=str, default=default_host, help=f'Jetson IP (default: {default_host})')
    parser.add_argument('--port', type=int, default=default_port, help=f'Port (default: {default_port})')
    parser.add_argument('--episodes', type=int, default=default_episodes, help=f'Episodes (default: {default_episodes})')
    parser.add_argument('--delay', type=float, default=default_delay, help=f'Delay (default: {default_delay})')
    parser.add_argument('--width', type=int, default=default_width, help=f'Width (default: {default_width})')
    parser.add_argument('--height', type=int, default=default_height, help=f'Height (default: {default_height})')
    parser.add_argument('--obstacles', type=int, default=default_obstacles, help=f'Obstacles (default: {default_obstacles})')
    parser.add_argument('--max-steps', type=int, default=default_max_steps, help=f'Max steps (default: {default_max_steps})')
    parser.add_argument('--test', action='store_true', help='Test connection only')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    # Test de conexión
    if args.test:
        test_connection(args.host, args.port)
        return
    
    # Crear entorno con configuración
    env = RobotEnvironment(
        width=args.width,
        height=args.height,
        num_obstacles=args.obstacles,
        max_steps=args.max_steps
    )
    
    # Crear cliente de inferencia
    client = InferenceClient(host=args.host, port=args.port)
    
    print("="*60)
    print("DQN INFERENCE MODE")
    print("="*60)
    print(f"Connecting to inference server at {args.host}:{args.port}...")
    
    if not client.connect():
        print("\n✗ Failed to connect to inference server!")
        print("\nMake sure the inference server is running on Jetson:")
        print(f"  ssh -i ~/.ssh/id_jetson jetson@{args.host}")
        print(f"  cd ~/dqn_cuda_project")
        print(f"  ./bin/dqn_inference_server dqn_model_final.bin")
        return
    
    print("✓ Connected! Starting inference...\n")
    
    try:
        # Ejecutar inferencia
        rewards, steps, successes = run_inference(
            client, 
            env, 
            num_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay
        )
        
        # Mostrar estadísticas
        if rewards:
            print_statistics(rewards, steps, successes, args.episodes)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        client.disconnect()
        env.close()
        print("\nClient closed")


if __name__ == '__main__':
    main()
