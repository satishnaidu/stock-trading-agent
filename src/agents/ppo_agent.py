import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.batch_size = 64

        # Initialize networks
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        self.memory = []

    def select_action(self, state):
        # Convert state to tensor properly
        state = np.array(state, dtype=np.float32)  # Ensure numpy array
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

        self.memory.append({
            'state': state,
            'action': action,
            'action_log_prob': action_log_prob
        })

        return action.item()

    def train(self, env, num_episodes=1000):
        for episode in range(num_episodes):
            state, _ = env.reset()  # Properly unpack the reset return value
            episode_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)  # Updated to handle new step return

                # Store transition
                self.store_transition(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                if len(self.memory) >= self.batch_size:
                    self.update()

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}, Average Reward: {episode_reward:.2f}")

    def backtest(self, env):
        """
        Perform backtest of the trained agent
        """
        state, _ = env.reset()  # Properly unpack the reset return
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            if steps % 100 == 0:
                print(f"Step: {steps}")
                print(f"Average Reward: {total_reward / steps:.2f}")
                print(f"Total P&L: ${info['total_value'] - env.initial_balance:.2f}")
                print("-------------------")

        return total_reward

