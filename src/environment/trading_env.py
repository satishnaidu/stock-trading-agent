import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=100000, max_position=100):
        super().__init__()

        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold

        # Define observation space with specific low and high values
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -self.max_position, 0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, self.max_position, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_pnl = 0

        return self._get_observation(), {}

    def _get_observation(self):
        current_data = self.data[self.current_step]
        return np.array([
            float(current_data['price']),
            float(current_data['volume']),
            float(self.position),
            float(self.balance),
            float(self.total_pnl),
            float(current_data['market_feature'])
        ], dtype=np.float32)

    def step(self, action):
        # Execute trading action
        reward = self._execute_trade(action)

        # Move to next time step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Calculate metrics
        info = {
            'total_pnl': float(self.total_pnl),
            'position': float(self.position),
            'balance': float(self.balance)
        }

        return self._get_observation(), float(reward), done, False, info

    def _execute_trade(self, action):
        current_price = float(self.data[self.current_step]['price'])
        previous_value = self.balance + self.position * current_price

        if action == 0:  # Buy
            if self.position < self.max_position:
                shares_to_buy = min(
                    self.max_position - self.position,
                    self.balance // current_price
                )
                self.position += shares_to_buy
                self.balance -= shares_to_buy * current_price

        elif action == 1:  # Sell
            if self.position > -self.max_position:
                shares_to_sell = min(
                    self.max_position + self.position,
                    self.position
                )
                self.position -= shares_to_sell
                self.balance += shares_to_sell * current_price

        current_value = self.balance + self.position * current_price
        reward = current_value - previous_value
        self.total_pnl = current_value - self.initial_balance

        return float(reward)
