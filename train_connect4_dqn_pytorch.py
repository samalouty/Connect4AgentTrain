
from connect4_env import Connect4Env
from dqn_agent_pytorch import DQNAgent
import numpy as np
import os
import matplotlib.pyplot as plt

EPISODES = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10 
MODEL_SAVE_PATH = "dqn_connect4_model.pth" 

if __name__ == "__main__":
    env = Connect4Env()
    state_size = env.rows * env.cols
    action_size = env.cols
    agent = DQNAgent(state_size, action_size, model_path=None) 
    episode_rewards = []
    losses = []

    for e in range(EPISODES):
        state_flat = env.reset() 
        total_reward = 0
        for time_step in range(env.rows * env.cols): 
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break 
            
            action = agent.act(state_flat, valid_actions) 
            if action is None:
                 print("Error: No valid action returned by agent.")
                 break

            next_state_flat, reward, done, info = env.step(action)
            
            agent.remember(state_flat, action, reward, next_state_flat, done)
            state_flat = next_state_flat 
            total_reward += reward
            
            if done:
                break
        
        loss = agent.replay(BATCH_SIZE) 
        if loss is not None: 
            losses.append(loss)
        
        episode_rewards.append(total_reward)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (e + 1) % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
            print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}, Epsilon: {agent.epsilon:.3f}, Avg Loss (last 10): {np.mean(losses[-10:]):.4f}")
            if (e+1) % 100 == 0: 
                 agent.save(MODEL_SAVE_PATH)
        
        if (e + 1) % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            print(f"--- Episode {e+1} --- Avg Reward (last 100): {avg_reward_100:.2f} ---")


    print("Training finished.")
    agent.save(MODEL_SAVE_PATH)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss (per batch)')
    plt.xlabel('Training Step (Batch)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_plot_pytorch.png")
    plt.show()