import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from stable_baselines3.common.results_plotter import plot_results
import numpy as np
from eval import evaluate_model

def plot_training_progress(log_dir="logs"):
    """Plot training metrics from CSV logs"""
    data = pd.read_csv(f"{log_dir}/training_metrics.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    sns.lineplot(data=data, x='timestep', y='reward', ax=ax1)
    ax1.set_title('Training Rewards over Time')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward')

    sns.lineplot(data=data, x='timestep', y='episode_length', ax=ax2)
    ax2.set_title('Episode Lengths over Time')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Steps per Episode')
    
    data['win_rate'] = data['wins'].rolling(window=100).mean()
    sns.lineplot(data=data, x='timestep', y='win_rate', ax=ax3)
    ax3.set_title('Win Rate over Time (Rolling Average)')
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Win Rate')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = evaluate_model(
        model_path="models/mortal_kombat_ppo",
        num_episodes=5,
        render=True
    )
    plot_training_progress()