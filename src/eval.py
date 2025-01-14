import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env import MortalKombatEnv
import time
import csv
from datetime import datetime
import os

def evaluate_model(model_path, num_episodes=10, render=True, record_video=False):
    env = MortalKombatEnv()
    model = PPO.load(model_path)

    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            
            if render:
                env.render()
                time.sleep(0.016)  
            
            total_reward += reward
            steps += 1

            if info.get('enemy_health', 1) == 0:
                wins += 1
                
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode reward: {total_reward}")
        print(f"Episode length: {steps} steps")
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    win_rate = wins / num_episodes
    
    print("\nEvaluation Results:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Win rate: {win_rate:.2%}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'win_rate': win_rate,
        'episode_rewards': episode_rewards
    }

