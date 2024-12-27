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
import logging
from collections import deque

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.min_memory_size = 5000  # Wait for more experiences before learning
        self.target_update_frequency = 500  # Update target network every 500 steps
        self.update_frequency = 4  # Update main network every 4 steps
        self.batch_size = 64  # Increased batch size
        self.steps = 0
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model
    
    def update_target_model(self):
        # Soft updates
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * local_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        self.steps += 1
        
        if len(self.memory) < self.min_memory_size:
            return
        
        if self.steps % self.update_frequency != 0:
            return
            
        if self.steps % self.target_update_frequency == 0:
            self.update_target_model()
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch])
        actions = torch.LongTensor([m[1] for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([m[4] for m in minibatch])
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # gradient clipping
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RampMeteringEnv:
    def __init__(self, sumo_config, min_phase_duration=30):
        self.sumo_config = sumo_config
        self.ramp_tl_id = "J4"
        self.state_size = 5  # Expanded state representation
        self.action_space_size = 2  # Switch to green or red
        self.min_phase_duration = min_phase_duration
        self.current_phase_duration = 0

    def reset(self):
        try:
            traci.close()
        except:
            pass
        
        traci.start(self.sumo_config)
        self.current_phase_duration = 0
        return self._get_state()
    
    def _get_state(self):
        highway_vehicles = len(traci.vehicle.getIDList())
        ramp_vehicles = len([v for v in traci.vehicle.getIDList() if traci.vehicle.getRoadID(v).startswith('E0')])
        highway_speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
        ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList() 
                                if traci.vehicle.getRoadID(v).startswith('E0'))
        current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
        return [highway_vehicles / 100, ramp_vehicles / 20, avg_highway_speed / 30, 
                ramp_waiting_time / 300, current_phase / 2]
    
    def step(self, action):
        current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
        
        if self.current_phase_duration >= self.min_phase_duration:
            if action == 0 and current_phase != 0:
                traci.trafficlight.setPhase(self.ramp_tl_id, 0)
                self.current_phase_duration = 0
            elif action == 1 and current_phase != 2:
                traci.trafficlight.setPhase(self.ramp_tl_id, 2)
                self.current_phase_duration = 0
        
        traci.simulationStep()
        self.current_phase_duration += 1
        
        reward = self._calculate_reward()
        next_state = self._get_state()
        done = traci.simulation.getTime() >= 1200 
        
        return next_state, reward, done
    
    def _calculate_reward(self):
        highway_vehicles = len(traci.vehicle.getIDList())
        ramp_vehicles = len([v for v in traci.vehicle.getIDList() if traci.vehicle.getRoadID(v).startswith('E0')])
        highway_speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
        ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList() 
                                if traci.vehicle.getRoadID(v).startswith('E0'))
        current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
        
        # Compute penalties and rewards
        congestion_penalty = max(0, (highway_vehicles - 50) * 0.1)
        waiting_time_penalty = ramp_waiting_time / 50
        efficiency_penalty = 10 if current_phase == 0 and ramp_vehicles > 0 and highway_vehicles > 50 else 0
        reward = avg_highway_speed - congestion_penalty - waiting_time_penalty - efficiency_penalty
        
        return np.clip(reward, -1, 1)

def train_dqn(episodes=150, simulation_steps=1200):  
    sumo_binary = "sumo-gui"
    sumo_cmd = [
        sumo_binary, "-c", "project.sumocfg", "--start", "--quit-on-end",
        "--no-warnings", "--no-step-log", "--random"
    ]
    env = RampMeteringEnv(sumo_cmd)
    agent = DQNAgent(state_size=5, action_size=2)
    
    episode_rewards = []
    avg_rewards = []  # Track moving average
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < simulation_steps:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(agent.batch_size)
            step += 1
        
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:])  # 10-episode moving average
        avg_rewards.append(avg_reward)
        
        logging.info(f"Episode {episode + 1} - "
                    f"Total Reward: {total_reward:.2f}, "
                    f"Average Reward (10 ep): {avg_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}")
    
    # Plot both actual and smoothed rewards
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, alpha=0.6, label='Episode Rewards')
    plt.plot(avg_rewards, label='10-Episode Moving Average')
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    train_dqn()