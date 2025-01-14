import yaml
import os
from env import MortalKombatEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: MortalKombatEnv()])
    
    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        verbose=1
    )
    
    model.learn(
        total_timesteps=config['total_timesteps'],
        progress_bar=True
    )
    
    model.save('models/mortal_kombat_ppo')

if __name__ == '__main__':
    main()
