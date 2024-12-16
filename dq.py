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
    def __init__(self, state_size, action_size, learning_rate=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        
        # Neural Network for DQN
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
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
        next_q_values = self.target_model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
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
        
    def reset(self):
        # Attempt to close any existing connection
        try:
            traci.close()
        except Exception as e:
            print(f"Error closing TraCI connection: {e}")
        
        # Attempt to start a new simulation connection
        try:
            traci.start(self.sumo_config)
            print("SUMO simulation started.")
        except Exception as e:
            print(f"Error starting simulation: {e}")
            sys.exit("Failed to start SUMO simulation.")
        
        return self._get_state()

    
    def _get_state(self):
        # State representation:
        # 1. Number of vehicles on highway
        # 2. Number of vehicles on ramp
        # 3. Average speed on highway
        # 4. Waiting time on ramp
        # 5. Current traffic light phase
        
        highway_vehicles = len(traci.vehicle.getIDList())
        ramp_vehicles = len([v for v in traci.vehicle.getIDList() 
                              if traci.vehicle.getRoadID(v).startswith('E0')])
        
        highway_speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
        avg_highway_speed = np.mean(highway_speeds) if highway_speeds else 0
        
        ramp_waiting_time = sum(traci.vehicle.getWaitingTime(v) 
                                 for v in traci.vehicle.getIDList() 
                                 if traci.vehicle.getRoadID(v).startswith('E0'))
        
        current_phase = traci.trafficlight.getPhase(self.ramp_tl_id)
        
        return [highway_vehicles, ramp_vehicles, avg_highway_speed, 
                ramp_waiting_time, current_phase]
    
    def step(self, action):
        # Map action to traffic light phase duration
        phase_durations = [20, 40, 60]  # Different ramp metering timings
        selected_duration = phase_durations[action]
        
        # Set traffic light phase
        traci.trafficlight.setPhaseDuration(self.ramp_tl_id, selected_duration)
        
        # Run simulation step
        traci.simulationStep()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        next_state = self._get_state()
        
        # Check if simulation is done
        done = traci.simulation.getTime() >= 3600
        
        return next_state, reward, done
    
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
        
        # Debugging: print these intermediate values
       # print(f"Highway vehicles: {highway_vehicles}, Ramp vehicles: {ramp_vehicles}")
        #print(f"Avg Speed: {avg_highway_speed}, Waiting time: {ramp_waiting_time}")
        
        congestion_penalty = (highway_vehicles - 50) * 0.1  # Reduced penalty scaling
        waiting_time_penalty = ramp_waiting_time / 50  # Reduced waiting time penalty
        speed_bonus = max(0, avg_highway_speed)
        
        reward = speed_bonus - congestion_penalty - waiting_time_penalty
        print(f"Reward: {reward}")  # Print the final reward value
        
        return reward

def train_dqn(episodes=5, batch_size=32):
    # SUMO configuration
    sumo_binary = "sumo-gui"  
    sumo_cmd = [sumo_binary, 
                "-c", "project.sumocfg",
                "--start", "--delay", "50"]  
    
    # Initialize environment and agent
    env = RampMeteringEnv(sumo_cmd)
    agent = DQNAgent(state_size=5, action_size=3)
    
    # Load previous training data if available
    if os.path.exists('replay_buffer.pkl'):
        with open('replay_buffer.pkl', 'rb') as f:
            agent.memory = pickle.load(f)
            print("Replay buffer loaded successfully.")
    
    if os.path.exists('dqn_ramp_metering_model.pth'):
        agent.model.load_state_dict(torch.load('dqn_ramp_metering_model.pth'))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print("Model loaded successfully.")

    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Additional debugging prints
        print(f"\n--- Starting Episode {episode + 1} ---")
        print(f"Initial Epsilon: {agent.epsilon:.4f}")
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action and observe
            next_state, reward, done = env.step(action)
            
            # Print detailed debugging information
            print(f"Step Reward: {reward}")
            print(f"Action Taken: {action}")
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward tracking
            state = next_state
            total_reward += reward
            
            # Perform experience replay
            agent.replay(batch_size)
        
        # Print episode summary
        print(f"Episode {episode+1}/{episodes}")
        print(f"Total Reward: {total_reward}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        
        episode_rewards.append(total_reward)
    
    # Visualize rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.tight_layout()
    plt.savefig('training_rewards.png')
    plt.show()
    # Save final model
    torch.save(agent.model.state_dict(), 'dqn_ramp_metering_model.pth')
    print("Final model saved.")

if __name__ == "__main__":
    train_dqn()
