
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random 

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
_model = None
_state_size = 6 * 7 
_action_size = 7    
MODEL_WEIGHTS_PATH = "dqn_connect4_model.pth" 
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model_for_inference():
    global _model
    if _model is None:
        _model = DQNNetwork(_state_size, _action_size).to(_device)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, '..', MODEL_WEIGHTS_PATH) 

        if os.path.exists(weights_path):
            try:
                _model.load_state_dict(torch.load(weights_path, map_location=_device))
                _model.eval() 
                print(f"DQN PyTorch Agent: Loaded model weights from {weights_path}")
            except Exception as e:
                print(f"DQN PyTorch Agent: Error loading model weights from {weights_path}: {e}")
                print("Using uninitialized model (will perform poorly).")
        else:
            print(f"DQN PyTorch Agent: Model weights not found at {weights_path}. Using uninitialized model (will perform poorly).")
    return _model

def my_agent(observation, configuration):
    """
    observation.board: 1D list, 0=empty, 1=player1, 2=player2
    observation.mark: Current player's mark (1 or 2)
    configuration.columns: 7
    """
    model = load_trained_model_for_inference()
    
    board_1d_backend_format = observation.board[:] 
    my_mark_backend = observation.mark 
    opponent_mark_backend = 2 if my_mark_backend == 1 else 1
    nn_input_board = np.zeros(_state_size, dtype=np.float32)
    for i in range(len(board_1d_backend_format)):
        if board_1d_backend_format[i] == my_mark_backend:
            nn_input_board[i] = 1.0  
        elif board_1d_backend_format[i] == opponent_mark_backend:
            nn_input_board[i] = -1.0 
    state_tensor = torch.FloatTensor(nn_input_board).unsqueeze(0).to(_device)
    
    with torch.no_grad(): 
        q_values_tensor = model(state_tensor)
    q_values = q_values_tensor.detach().cpu().numpy()[0]
    valid_actions = []
    for col in range(configuration.columns):
        if observation.board[col] == 0: 
            valid_actions.append(col)
            
    if not valid_actions: 
        print("DQN PyTorch Agent: No valid actions found, defaulting to 0.")
        return 0 

    
    best_action = -1
    max_q = -float('inf') 
    
    
    for action_idx in valid_actions:
        if q_values[action_idx] > max_q:
            max_q = q_values[action_idx]
            best_action = action_idx
            
    if best_action == -1: 
        print("DQN PyTorch Agent: All valid actions had -inf Q-value. Choosing random valid action.")
        return random.choice(valid_actions) if valid_actions else 0 

    return best_action