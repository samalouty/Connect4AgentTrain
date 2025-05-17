import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim1=128, hidden_dim2=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, model_path=None, learning_rate=0.001, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.999,
                 buffer_size=20000, hidden_dim1=128, hidden_dim2=128):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = DQNNetwork(state_size, action_size, hidden_dim1, hidden_dim2).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size, hidden_dim1, hidden_dim2).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load(model_path) 
            print(f"Loaded model from {model_path}")
        else:
            print("Built a new model.")
            
        self.update_target_model() 
        self.target_model.eval()  

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state_np, valid_actions):
        if not valid_actions:
            return None 

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions) 
        self.model.eval() 
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).detach().cpu().numpy()[0]
        self.model.train() 

        masked_q_values = np.full(self.action_size, -np.inf)
        for action in valid_actions:
            masked_q_values[action] = q_values[action]
        
        best_action = np.argmax(masked_q_values)
        return best_action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0 

        minibatch = random.sample(self.memory, batch_size)
        states_np = np.array([transition[0] for transition in minibatch])
        actions_np = np.array([transition[1] for transition in minibatch])
        rewards_np = np.array([transition[2] for transition in minibatch])
        next_states_np = np.array([transition[3] for transition in minibatch])
        dones_np = np.array([transition[4] for transition in minibatch])
        states = torch.FloatTensor(states_np).to(self.device)
        actions = torch.LongTensor(actions_np).unsqueeze(1).to(self.device) 
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        dones = torch.BoolTensor(dones_np).to(self.device)
        current_q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values_target_net = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values_target_net * (~dones))
        loss = self.criterion(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)