# Mortal-KomBot
This project implements a reinforcement learning agent trained to play Mortal Kombat using PPO (Proximal Policy Optimization) from Stable Baselines 3. The agent learns to fight opponents by maximizing in-game rewards based on health differences and match outcomes.

## Prerequisites

- Python 3.6/3.7/3.8
- ROM file for Mortal Kombat (Genesis/Mega Drive version)
- OpenAI Gymnasium
- Stable Baselines 3
- PyGame
- OpenCV
- Retro

Note: The ROM for Mortal Kombat should be legally obtained. Import it to retro by running python -m retro.import /path/to/your/rom/directory.

## Environment 
The custom environment (env.py) implements:

- Observation space: 84x84 grayscale images
- Action space: Binary space for Genesis controller buttons
- Reward structure:

  +1.0 for gaining health
  -1.0 for losing health
  +2.0 for dealing damage
  +10.0 for winning
  -10.0 for losing

## Training

The hyperparameters for training are listed in config.yaml.

## Evaluation

Running visualization.py will:

1. Load the trained model
2. Run evaluation episodes
3. Display the game screen
4. Generate performance visualizations

## Contributions

Feel free to open issues or submit pull requests for improvements. Some areas for potential enhancement:

- Improved reward shaping
- Additional enemy difficulty levels
- Support for different characters
- Multi-environment parallel training

## Acknowledgements

- OpenAI Gymnasium
- Stable Baselines3
- Midway Games, creators of Mortal Kombat 1
- SEGA, creators of SEGA Genesis
