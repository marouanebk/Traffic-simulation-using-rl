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

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.max_memory_size = 10000  # Limit memory size
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Neural Network for DQN
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),  # Increased network complexity
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Adaptive exploration: more exploration early, less later
        explore_threshold = max(self.epsilon_min, 
                                1.0 - (0.9 / (1 + np.exp(-len(self.memory)/1000))))
        
        if np.random.rand() <= explore_threshold:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act_values = self.model(state_tensor)
            return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch])
        actions = torch.LongTensor([m[1] for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor([m[3] for m in minibatch])
        dones = torch.FloatTensor([m[4] for m in minibatch])
        
        # Compute Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss with Huber loss for more stable training
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
        
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RampMeteringEnv:
    def __init__(self, sumo_config):
        self.sumo_config = sumo_config
        self.ramp_tl_id = "J4"  # Traffic light ID for the ramp
        self.state_size = 5  # Number of state variables
        self.action_space_size = 3  # Different traffic light timings
        self.steps_since_reset = 0
        
    def reset(self):
        # Close existing connection
        try:
            traci.close()
        except Exception as e:
            print(f"Error closing TraCI connection: {e}")
        
        # Start new simulation
        try:
            traci.start(self.sumo_config)
            print("SUMO simulation started.")
        except Exception as e:
            print(f"Error starting simulation: {e}")
            sys.exit("Failed to start SUMO simulation.")
        
        self.steps_since_reset = 0
        return self._get_state()
    
    def _get_state(self):
        # More robust state extraction
        try:
            all_vehicle_ids = traci.vehicle.getIDList()
            highway_vehicles = [v for v in all_vehicle_ids 
                                if not v.startswith('ramp')]
            ramp_vehicles = [v for v in all_vehicle_ids 
                             if v.startswith('ramp')]
            
            highway_speeds = [traci.vehicle.getSpeed(v) for v in highway_vehicles]
            avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
            
            ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                    for v in ramp_vehicles)
            
            current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
            
            return [
                len(highway_vehicles), 
                len(ramp_vehicles), 
                avg_highway_speed, 
                ramp_waiting_time, 
                current_phase
            ]
        except Exception as e:
            print(f"Error getting state: {e}")
            return [0, 0, 0, 0, 0]
    
    def step(self, action):
        # Map action to traffic light phase duration
        phase_durations = [20, 40, 60]  # Different ramp metering timings
        selected_duration = phase_durations[action]
        
        # Set traffic light phase
        traci.trafficlight.setPhaseDuration(self.ramp_tl_id, selected_duration)
        
        # Run simulation step
        traci.simulationStep()
        
        self.steps_since_reset += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        next_state = self._get_state()
        
        # Check if simulation is done
        done = (self.steps_since_reset >= 1000 or  # Limit episode length
                traci.simulation.getTime() >= 3600)  
        
        return next_state, reward, done
    
    def _calculate_reward(self):
        # More nuanced reward function
        try:
            all_vehicle_ids = traci.vehicle.getIDList()
            highway_vehicles = [v for v in all_vehicle_ids 
                                if not v.startswith('ramp')]
            ramp_vehicles = [v for v in all_vehicle_ids 
                             if v.startswith('ramp')]
            
            # Highway congestion metrics
            highway_speeds = [traci.vehicle.getSpeed(v) for v in highway_vehicles]
            avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
            
            # Calculate waiting times
            highway_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                       for v in highway_vehicles)
            ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                    for v in ramp_vehicles)
            
            # Reward components
            speed_reward = avg_highway_speed  # Reward for maintaining highway speed
            congestion_penalty = max(0, len(highway_vehicles) - 50) * 10  # Penalize overcrowding
            waiting_time_penalty = (highway_waiting_time + ramp_waiting_time) / 100  # Penalize waiting
            
            # Balance different reward components
            reward = speed_reward - congestion_penalty - waiting_time_penalty
            
            return reward
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0

def train_dqn(episodes=4, batch_size=64):  # Increased episodes and batch size
    # SUMO configuration
    sumo_binary = "sumo"  # Use non-gui for faster training
    sumo_cmd = [sumo_binary, 
                "-c", "project.sumocfg",
                "--start","--no-step-log", "--no-warnings"]  
    
    # Initialize environment and agent
    env = RampMeteringEnv(sumo_cmd)
    agent = DQNAgent(state_size=5, action_size=3)
    
    # Training loop with more logging and periodic model saving
    episode_rewards = []
    
    for episode in range(episodes):
        try:
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            while not done:
                # Choose action
                action = agent.act(state)
                
                # Take action and observe
                next_state, reward, done = env.step(action)
                
                # Remember the experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and reward tracking
                state = next_state
                total_reward += reward
                step += 1
                
                # Perform experience replay
                agent.replay(batch_size)
                
                # Early stopping if no progress
                if step >= 1000:
                    break
            
            # Periodically update target network
            if episode % 10 == 0:
                agent.update_target_model()
                # Save model periodically
                torch.save(agent.model.state_dict(), f'dqn_model_episode_{episode}.pth')
            
            # Track rewards
            episode_rewards.append(total_reward)
            
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}, Steps: {step}")
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            break
    
    # Plotting and saving results
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.tight_layout()
    plt.savefig('training_rewards.png')
    plt.close()  # Close the plot to prevent display
    
    # Save final model
    torch.save(agent.model.state_dict(), 'dqn_ramp_metering_model.pth')
    
    # Save episode rewards for later analysis
    np.save('episode_rewards.npy', np.array(episode_rewards))
    
    print("Training completed. Model and rewards saved.")

if __name__ == "__main__":
    train_dqn()