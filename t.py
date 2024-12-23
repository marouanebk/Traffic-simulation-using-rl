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

def modify_traffic_scale(episode):
    # Dynamically adjust traffic scale based on the episode number or any other metric
    scale_factor = 1.0 + 0.05 * random.random()  # Reduced scale factor for smoother variations
    return scale_factor

# Ensure SUMO_HOME is set
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = gamma    
        self.epsilon = epsilon  
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
        loss = F.smooth_l1_loss(current_q_values, target_q_values.detach())
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * local_param.data)

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
        # close any existing connection
        try:
            traci.close()
        except Exception as e:
            print(f"Error closing TraCI connection: {e}")
        
        # start a new simulation connection
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
        done = traci.simulation.getTime() >= 600
        
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
        
        congestion_penalty = (highway_vehicles - 50) * 0.1  # Reduced penalty scaling
        waiting_time_penalty = ramp_waiting_time / 50  # Reduced waiting time penalty
        speed_bonus = max(0, avg_highway_speed)
        
        reward = speed_bonus - congestion_penalty - waiting_time_penalty
        
        return reward

def train_dqn(episodes=50, batch_size=32):
    # SUMO configuration
    sumo_binary = "sumo-gui"
    agent = DQNAgent(state_size=5, action_size=3)
    # Initialize the log file
    log_file = "training_log.txt"
    with open(log_file, "a") as log:
        log.write("Training Start\n")
    
    # Initialize best performance tracking
    best_reward = float('-inf')  # Initialize with a very low value
    best_model_path = 'best_dqn_ramp_metering_model.pth'
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        # Modify traffic scale dynamically for each episode
        traffic_scale = modify_traffic_scale(episode)
        print(f"Episode {episode + 1}: Traffic Scale Factor: {traffic_scale:.2f}")
        
        # Adjust SUMO configuration for this episode based on the traffic scale
        sumo_cmd = [
            sumo_binary,
            "-c", "project.sumocfg",  # Path to your SUMO configuration
            "--start",
            "--quit-on-end",
            "--delay", "0",
            "--scale", str(traffic_scale)  # Adjust traffic scale dynamically here
        ]
        
        # Initialize environment and agent
        env = RampMeteringEnv(sumo_cmd)  # Re-initialize environment with updated SUMO configuration
        #agent = DQNAgent(state_size=5, action_size=3)
        
        # Load previous training data if available
        if os.path.exists('replay_buffer.pkl'):
            with open('replay_buffer.pkl', 'rb') as f:
                agent.memory = pickle.load(f)
                print("Replay buffer loaded successfully.")
        
        if os.path.exists('dqn_ramp_metering_model.pth'):
            agent.model.load_state_dict(torch.load('dqn_ramp_metering_model.pth'))
            agent.target_model.load_state_dict(agent.model.state_dict())
            print("Model loaded successfully.")

        # Reset environment with the new scale
        state = env.reset()
        total_reward = 0
        done = False
        
        # Training for this episode
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(batch_size)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1} completed with total reward {total_reward}")
        print(f"Final Epsilon: {agent.epsilon:.4f}")
        
        # Write the episode details to the log file
        with open(log_file, "a") as log:
            log.write(f"Episode {episode + 1} completed with total reward {total_reward}\n")
            log.write(f"Final Epsilon: {agent.epsilon:.4f}\n")
        
        # Save the model if this is the best reward so far
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.model.state_dict(), best_model_path)  # Save the best model
            print(f"New best model saved with reward {total_reward}.")
        traci.close()
    # Save the final model and replay buffer
    torch.save(agent.model.state_dict(), 'final_dqn_ramp_metering_model.pth')
    with open('replay_buffer.pkl', 'wb') as f:
        pickle.dump(agent.memory, f)
    
    plt.plot(episode_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('training_rewards.png')
    plt.show()
    
    return episode_rewards

# Start the training
train_dqn()
