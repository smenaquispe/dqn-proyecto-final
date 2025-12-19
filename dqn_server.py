import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os
import argparse
import time

# --- Configuration ---
STATE_SIZE = 15
ACTION_SIZE = 8
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Neural Network ---
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# --- Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps = 0
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
                
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
        
        # Compute Huber loss
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
        
    def load(self, filename):
        if os.path.exists(filename):
            self.policy_net.load_state_dict(torch.load(filename))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            return True
        return False

# --- Server ---
def start_server(host='0.0.0.0', port=5556):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((host, port))
    except Exception as e:
        print(f"Error binding to port {port}: {e}")
        return

    server.listen(1)
    print(f"Python DQN Server listening on {host}:{port}")
    
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    episode_count = 0
    
    print("Waiting for client connection...")
    
    while True:
        client_socket, addr = server.accept()
        print(f"Connected by {addr}")
        
        current_episode_loss = 0
        loss_count = 0
        
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                    
                message = data.decode().strip()
                
                # Protocol: 
                # STEP:state_csv -> action
                # TRAIN:state_csv|action|reward|next_state_csv|done -> loss
                # EPISODE_END -> OK
                # SAVE -> SAVED
                
                if message.startswith("STEP:"):
                    # Just inference
                    parts = message.split(":")[1].split(",")
                    state = [float(x) for x in parts]
                    action = agent.select_action(state)
                    response = f"{action},{agent.epsilon:.4f}"
                    client_socket.sendall(response.encode())
                    
                elif message.startswith("TRAIN:"):
                    # Store experience and train
                    payload = message.split(":")[1]
                    parts = payload.split("|")
                    
                    state = [float(x) for x in parts[0].split(",")]
                    action = int(parts[1])
                    reward = float(parts[2])
                    next_state = [float(x) for x in parts[3].split(",")]
                    done = int(parts[4])
                    
                    agent.memory.push(state, action, reward, next_state, done)
                    
                    loss = agent.optimize_model()
                    if loss > 0:
                        current_episode_loss += loss
                        loss_count += 1
                        
                    client_socket.sendall(b"OK")
                    
                elif message == "EPISODE_END":
                    episode_count += 1
                    # Decay epsilon
                    if agent.epsilon > EPSILON_END:
                        agent.epsilon *= EPSILON_DECAY
                        
                    # Update target network
                    if episode_count % TARGET_UPDATE == 0:
                        agent.update_target_network()
                        print(f"Target network updated at episode {episode_count}")
                        
                    avg_loss = current_episode_loss / max(1, loss_count)
                    print(f"Episode {episode_count} ended. Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
                    
                    client_socket.sendall(b"OK")
                    current_episode_loss = 0
                    loss_count = 0
                    
                elif message == "SAVE":
                    agent.save(f"dqn_model_ep{episode_count}.pth")
                    print(f"Model saved at episode {episode_count}")
                    client_socket.sendall(b"SAVED")
                    
                elif message == "QUIT":
                    print("Client requested quit")
                    break
                    
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
            print("Client disconnected, waiting for new connection...")
            # Optional: break outer loop to stop server completely, or continue waiting

if __name__ == "__main__":
    start_server()
