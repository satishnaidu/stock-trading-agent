# src/environment/trading_env.py

import numpy as np
import pandas as pd
# src/environment/trading_env.py

class TradingEnvironment:
    def __init__(self, data):
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        else:
            self.data = data

        self.initial_balance = 10000.0
        self.reset()

    def reset(self):
        """
        Reset the environment to initial state
        Returns:
            tuple: (observation, info)
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.shares = 0
        self.total_value = self.balance

        # Return both state and info dict to match gym interface
        observation = self._get_state()
        info = {
            'balance': self.balance,
            'total_value': self.total_value,
            'step': self.current_step
        }
        return observation, info

    def _get_state(self):
        current_data = self.data.iloc[self.current_step]

        # Calculate price normalization factor
        price_scale = self.data['Close'].max()
        volume_scale = self.data['Volume'].max()

        state = np.array([
            current_data['Open'] / price_scale,
            current_data['High'] / price_scale,
            current_data['Low'] / price_scale,
            current_data['Close'] / price_scale,
            current_data['Volume'] / volume_scale,
            self.position,
            self.balance / self.initial_balance
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Execute one step in the environment
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Store current price for reward calculation
        current_price = self.data.iloc[self.current_step]['Close']

        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                shares_to_buy = self.balance // current_price
                self.shares = shares_to_buy
                self.balance -= shares_to_buy * current_price
                self.position = 1
        elif action == 2:  # Sell
            if self.position == 1:
                self.balance += self.shares * current_price
                self.shares = 0
                self.position = 0

        # Move to next step
        self.current_step += 1

        # Calculate reward
        new_total_value = self.balance + (self.shares * current_price)
        reward = ((new_total_value - self.total_value) / self.total_value) * 100
        self.total_value = new_total_value

        # Check if done
        done = self.current_step >= len(self.data) - 1

        # Get next state
        next_state = self._get_state()

        info = {
            'balance': self.balance,
            'total_value': self.total_value,
            'shares': self.shares,
            'current_price': current_price
        }

        return next_state, reward, done, False, info

