from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy  

def create_model(env, config):
    model = PPO(
        ActorCriticCnnPolicy,
        env,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        verbose=1
    )
    return model