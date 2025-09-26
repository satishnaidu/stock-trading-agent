import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3 import PPO


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None

    def create_model(self, env):
        self.model = PPO("MlpPolicy",
                         env,
                         verbose=1,
                         tensorboard_log="./ppo_tensorboard/")

    def train(self, env, num_episodes=1000):
        # If model hasn't been created yet, create it
        if self.model is None:
            self.create_model(env)

        # Convert episodes to timesteps (approximate)
        # Assuming each episode has about 100 steps on average
        total_timesteps = num_episodes * 100

        # Train the model using SB3's learn method
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, state):
        if self.model is None:
            return np.random.randint(self.action_dim)

        action, _ = self.model.predict(state, deterministic=False)
        return action

    def save(self, path):
        if self.model is not None:
            self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)

