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
from collections import deque
import logging

# Logging configuration
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
                 epsilon=1,
                 epsilon_decay=0.97,
                 epsilon_min=0.05,
                 memory_size=5000,
                 batch_size=32):
        
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
            nn.Linear(self.state_size, 64),  # Reduced network size
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
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
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
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
            
            normalized_state = [
                highway_vehicles / 100,  
                ramp_vehicles / 20,      
                avg_highway_speed / 30,   
                min(ramp_waiting_time, 300) / 300,  
                current_phase / 2
            ]
            
            return normalized_state
            
        except Exception as e:
            logging.error(f"Error getting state: {e}")
            return [0] * self.state_size
    
    def step(self, action):
        self.current_step += 1
        
        phase_durations = [20, 40, 60]
        selected_duration = phase_durations[action]
        
        try:
            traci.trafficlight.setPhaseDuration(self.ramp_tl_id, selected_duration)
            traci.simulationStep()
            
            next_state = self._get_state()
            reward = self._calculate_reward(next_state)
            
            done = (self.current_step >= self.max_steps or 
                   traci.simulation.getMinExpectedNumber() <= 0)
            
            return next_state, reward, done
            
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return self._get_state(), -1, True
    
    def _calculate_reward(self, state):
        try:
            highway_density = state[0]
            ramp_density = state[1]
            avg_speed = state[2]
            waiting_time = state[3]
            
            # Reward components
            flow_reward = avg_speed * (1 - highway_density)
            waiting_penalty = -waiting_time * 2
            density_balance = -abs(highway_density - 0.5)
            
            # Combine rewards
            reward = (
                3.0 * flow_reward +
                2.0 * waiting_penalty +
                1.0 * density_balance
            )
            
            return np.clip(reward, -1, 1)
            
        except Exception as e:
            logging.error(f"Error calculating reward: {e}")
            return -1

def train_dqn(episodes=50):
    sumo_binary = "sumo"
    agent = DQNAgent(state_size=5, action_size=3)
    best_reward = float('-inf')
    
    if os.path.exists('best_dqn_ramp_metering_model.pth'):
        agent.model.load_state_dict(torch.load('best_dqn_ramp_metering_model.pth'))
        agent.target_model.load_state_dict(agent.model.state_dict())
        logging.info("Loaded best previous model")
    
    episode_rewards = []
    avg_rewards = []
    
    # Early stopping parameters
    patience = 5
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    for episode in range(episodes):
        logging.info(f"Episode {episode + 1}/{episodes}")
        
        sumo_cmd = [
            sumo_binary,
            "-c", "project.sumocfg",
            "--start",
            "--quit-on-end",
            "--no-warnings",
            "--no-step-log",
            "--random",
        ]
        
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
        
        if episode % 2 == 0:
            agent.update_target_model()
        
        episode_rewards.append(total_reward)
        
        # Calculate running average
        window_size = 3
        if len(episode_rewards) >= window_size:
            avg_reward = np.mean(episode_rewards[-window_size:])
            avg_rewards.append(avg_reward)
            
            # Early stopping check
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logging.info("Early stopping triggered!")
                break
            
            if len(avg_rewards) > 1:
                reward_change = abs(avg_rewards[-1] - avg_rewards[-2])
                logging.info(f"Reward change: {reward_change:.2f}")
        
        avg_loss = np.mean(losses) if losses else 0
        
        logging.info(f"Episode {episode + 1} - "
                    f"Total Reward: {total_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}, "
                    f"Avg Loss: {avg_loss:.3f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), 'best_dqn_ramp_metering_model.pth')
            logging.info(f"New best model saved with reward {total_reward:.2f}")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(agent.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(avg_rewards)
    plt.title('Average Rewards (Window=3)')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    plt.close()
    
    torch.save(agent.model.state_dict(), 'final_dqn_ramp_metering_model.pth')
    logging.info("Training completed")
    
    return episode_rewards, agent.loss_history

if __name__ == "__main__":
    if 'SUMO_HOME' not in os.environ:
        sys.exit("Please declare environment variable 'SUMO_HOME'")
    
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    
    rewards, losses = train_dqn()