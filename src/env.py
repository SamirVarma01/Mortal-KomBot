import retro
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import cv2

class MortalKombatEnv(gym.Env):
    def __init__(self):
        self.env = retro.make('MortalKombat-Genesis')
        self.viewer = None
        pygame.init()
        
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        
        self.action_space = spaces.MultiBinary(len(buttons))
        
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        self.previous_health = 0
        self.previous_enemy_health = 0
        self.screen = None
    
    def preprocess_observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)
    
    def step(self, action):
        action = np.array(action, dtype=np.int8)
        
        obs, reward, done, info = self.env.step(action)
        processed_obs = self.preprocess_observation(obs)
        
        reward = 0
        current_health = info.get('health', 0)
        enemy_health = info.get('enemy_health', 0)
        
        if current_health > self.previous_health:
            reward += 1.0
        elif current_health < self.previous_health:
            reward -= 1.0
            
        if enemy_health < self.previous_enemy_health:
            reward += 2.0
            
        if enemy_health == 0:
            reward += 10.0
            
        if current_health == 0:
            reward -= 10.0
            
        self.previous_health = current_health
        self.previous_enemy_health = enemy_health
        
        return processed_obs, reward, done, False, info
    
    def render(self):
        if self.screen is None:
            screen_width = 320
            screen_height = 240
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('Mortal Kombat')

        game_screen = self.env.get_screen()
        
        game_surface = pygame.surfarray.make_surface(
            np.transpose(game_screen, (1, 0, 2))
        )
        
        self.screen.blit(game_surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        return True
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        self.previous_health = 0
        self.previous_enemy_health = 0
        processed_obs = self.preprocess_observation(obs)
        return processed_obs, {}
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        self.env.close()
