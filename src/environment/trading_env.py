# src/environment/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnvironment, self).__init__()

        self.df = df
        print("DataFrame columns:", self.df.columns.tolist())  # Debug print

        # Verify required columns exist
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {self.df.columns.tolist()}")

        self.initial_balance = initial_balance

        # Define action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        num_features = len(self.df.columns) + 2  # +2 for balance and holdings
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0

        # Safely get the current price
        try:
            self.current_price = float(self.df.iloc[self.current_step]['close'])
        except KeyError as e:
            print(f"Available columns: {self.df.columns.tolist()}")
            raise KeyError(f"Column 'close' not found in DataFrame. Available columns: {self.df.columns.tolist()}")

        return self._get_observation(), {}

    def _get_observation(self):
        # Get the current state of the environment
        features = self.df.iloc[self.current_step].values

        # Add balance and shares held to the observation
        obs = np.append(features, [
            self.balance,
            self.shares_held
        ])

        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1

        if self.current_step >= len(self.df):
            return self._get_observation(), 0, True, False, {}

        self.current_price = float(self.df.iloc[self.current_step]['close'])

        # Execute trading action
        reward = self._take_action(action)

        # Calculate if done
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

    def _take_action(self, action):
        reward = 0

        if action == 1:  # Buy
            if self.balance >= self.current_price:
                self.shares_held += 1
                self.balance -= self.current_price
                reward = 1
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += self.current_price
                reward = 1

        return reward
