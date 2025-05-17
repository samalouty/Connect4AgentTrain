import numpy as np
import random

class Connect4Env:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player_env = 1 
        self.action_space_size = cols

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player_env = 1
        return self._get_obs()

    def _get_obs(self):
        obs_board = np.copy(self.board)
        if self.current_player_env == 1: 
            obs_board[obs_board == 2] = -1 
        else: 
            obs_board[obs_board == 1] = -1 
            obs_board[obs_board == 2] = 1  
        return obs_board.flatten()


    def get_valid_actions(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def _drop_piece(self, col, piece):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = piece
                return True
        return False

    def check_winner(self, piece):
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if self.board[r,c] == piece and self.board[r,c+1] == piece and self.board[r,c+2] == piece and self.board[r,c+3] == piece:
                    return True
        
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if self.board[r,c] == piece and self.board[r+1,c] == piece and self.board[r+2,c] == piece and self.board[r+3,c] == piece:
                    return True
        
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if self.board[r,c] == piece and self.board[r+1,c+1] == piece and self.board[r+2,c+2] == piece and self.board[r+3,c+3] == piece:
                    return True
        
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if self.board[r,c] == piece and self.board[r-1,c+1] == piece and self.board[r-2,c+2] == piece and self.board[r-3,c+3] == piece:
                    return True
        return False

    def is_board_full(self):
        return not (self.board[0] == 0).any()

    def step(self, action_col):
        
        agent_piece = 1
        opponent_piece = 2

        if action_col not in self.get_valid_actions():
            return self._get_obs(), -10, True, {"error": "Invalid action by agent"} 

        self._drop_piece(action_col, agent_piece)

        if self.check_winner(agent_piece):
            return self._get_obs(), 1, True, {} 

        if self.is_board_full():
            return self._get_obs(), 0, True, {} 

        
        self.current_player_env = opponent_piece 
        valid_opponent_actions = self.get_valid_actions()
        if not valid_opponent_actions: 
             return self._get_obs(), 0, True, {"error": "Board full before opponent move, but not draw?"}

        opponent_action = random.choice(valid_opponent_actions)
        self._drop_piece(opponent_action, opponent_piece)

        if self.check_winner(opponent_piece):
            return self._get_obs(), -1, True, {} 

        if self.is_board_full():
            return self._get_obs(), 0, True, {} 

        self.current_player_env = agent_piece 
        return self._get_obs(), -0.01, False, {} 