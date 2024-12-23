import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import traci
import matplotlib.pyplot as plt
import pickle
from collections import deque
import logging

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
class DQNAgent:
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 memory_size=10000,
                 batch_size=64):
        
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural Network for DQN
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)        
        # For tracking metrics
        self.loss_history = []
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)
        # Normalize rewards
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        # Target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.loss_history.append(loss.item())
        return loss.item()

class RampMeteringEnv:
    def __init__(self, sumo_config):
        self.sumo_config = sumo_config
        self.ramp_tl_id = "J4"
        self.state_size = 5
        self.action_space_size = 3
        self.max_steps = 3600
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        try:
            traci.close()
        except:
            pass
            
        traci.start(self.sumo_config)
        return self._get_state()
    
    def _get_state(self):
        try:
            highway_vehicles = len(traci.vehicle.getIDList())
            ramp_vehicles = len([v for v in traci.vehicle.getIDList() 
                               if traci.vehicle.getRoadID(v).startswith('E0')])
            
            highway_speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
            avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
            
            ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                  for v in traci.vehicle.getIDList() 
                                  if traci.vehicle.getRoadID(v).startswith('E0'))
            
            current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
            
            # Normalize state values
            normalized_state = [
                highway_vehicles / 100,  
                ramp_vehicles / 20,      
                avg_highway_speed / 30,   
                min(ramp_waiting_time, 300) / 300,  
                current_phase / 2        # Assuming 3 phases (0,1,2)
            ]
            
            return normalized_state
            
        except Exception as e:
            logging.error(f"Error getting state: {e}")
            return [0] * self.state_size
    
    def step(self, action):
        self.current_step += 1
        
        # Map action to traffic light phase duration
        phase_durations = [20, 40, 60]
        selected_duration = phase_durations[action]
        
        try:
            traci.trafficlight.setPhaseDuration(self.ramp_tl_id, selected_duration)
            traci.simulationStep()
            
            reward = self._calculate_reward()
            next_state = self._get_state()
            
            # Check terminal conditions
            done = (self.current_step >= self.max_steps or 
                   traci.simulation.getMinExpectedNumber() <= 0)
            
            return next_state, reward, done
            
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return self._get_state(), -100, True
    
    def _calculate_reward(self):
        # Reward function considering:
        # 1. Highway flow (penalize congestion)
        # 2. Ramp waiting time
        # 3. Average vehicle speed
        
        highway_vehicles = len(traci.vehicle.getIDList())
        ramp_vehicles = len([v for v in traci.vehicle.getIDList() 
                              if traci.vehicle.getRoadID(v).startswith('E0')])
        
        highway_speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
        
        ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                 for v in traci.vehicle.getIDList() 
                                 if traci.vehicle.getRoadID(v).startswith('E0'))
        
        congestion_penalty = (highway_vehicles - 50) * 0.2 
        waiting_time_penalty = ramp_waiting_time / 50  
        speed_bonus = max(0, avg_highway_speed)
        
        reward = speed_bonus - congestion_penalty - waiting_time_penalty
        
        return reward
    

def train_dqn(episodes=50):

    sumo_binary = "sumo"
    agent = DQNAgent(state_size=5, action_size=3)
    best_reward = float('-inf')

    # Load best model if available
    if os.path.exists('best_dqn_model.pth'):
        agent.model.load_state_dict(torch.load('best_dqn_model.pth'))
        agent.target_model.load_state_dict(agent.model.state_dict())
        logging.info("Loaded best previous model")
    
    # Training metrics
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        logging.info(f"Episode {episode + 1}/{episodes}")
        
        # SUMO configuration
        sumo_cmd = [
            sumo_binary,
            "-c", "project.sumocfg",
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--random",
        ]
        
        # Initialize environment
        env = RampMeteringEnv(sumo_cmd)
        state = env.reset()
        total_reward = 0
        losses = []
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss:
                losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
       # threshold_reward=4800
        #if total_reward > threshold_reward:
         #    agent.epsilon = max(agent.epsilon * 0.99, 0.01)
        #else:
         #    agent.epsilon = max(agent.epsilon * 0.997, 0.01)

        # Update target network every 5 episodes
        if episode % 5 == 0:
            agent.update_target_model()
        
        episode_rewards.append(total_reward)
        avg_loss = np.mean(losses) if losses else 0
        
        logging.info(f"Episode {episode + 1} - "
                    f"Total Reward: {total_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}, "
                    f"Avg Loss: {avg_loss:.3f}")
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), 'best_dqn_model.pth')
            logging.info(f"New best model saved with reward {total_reward:.2f}")
        


    # Save training progress plot
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
            
    plt.subplot(1, 2, 2)
    plt.plot(agent.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
            
            
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    plt.close()
    
    # Save final model
    torch.save(agent.model.state_dict(), 'final_dqn_ramp_metering_model.pth')
    logging.info("Training completed")
    
    return episode_rewards, agent.loss_history


"""""
class Metrics:
    def __init__(self):
        self.avg_highway_speeds = []
        self.avg_waiting_times = []
        self.throughput = []
        self.avg_travel_times = []
        
    def update(self, env):
        # Collect metrics each step
        speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        self.avg_highway_speeds.append(np.mean(speeds) if speeds else 0)
        # Add other metrics...

def run_baseline():
    # Similar setup but without traffic light control
    sumo_cmd = [
        "sumo",
        "-c", "project.sumocfg",
        "--start",
        "--quit-on-end"
    ]
    env = RampMeteringEnv(sumo_cmd)
    metrics = Metrics()
    
    state = env.reset()
    done = False
    
    while not done:
        # Run simulation without control
        _, _, done = env.step(0)  # No action
        metrics.update(env)
    
    return metrics
"""
if __name__ == "__main__":
    # Ensure SUMO_HOME is set
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
    # Start training
    rewards, losses = train_dqn()