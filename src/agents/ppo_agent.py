import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimplePPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_features = self.shared(x)
        return self.policy_head(shared_features), self.value_head(shared_features)

class TradingAgent:
    def __init__(self, env, learning_rate=3e-4):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = SimplePPONetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.gamma = 0.99
        self.epsilon = 0.2
        self.batch_size = 64

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy, _ = self.network(state)
            action = torch.multinomial(policy, 1).item()
        return action

    def train(self, total_timesteps):
        print(f"Training on device: {self.device}")

        episode_rewards = []
        total_episodes = 0
        timestep = 0

        while timestep < total_timesteps:
            # Collect trajectory
            states, actions, rewards, values, log_probs = [], [], [], [], []
            done = False
            episode_reward = 0
            state, _ = self.env.reset()

            while not done and timestep < total_timesteps:
                state_tensor = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    policy, value = self.network(state_tensor)

                action = torch.multinomial(policy, 1).item()
                log_prob = torch.log(policy[action])

                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value.item())
                log_probs.append(log_prob.item())

                state = next_state
                episode_reward += reward
                timestep += 1

                if len(states) >= self.batch_size:
                    self._update_policy(states, actions, rewards, values, log_probs)
                    states, actions, rewards, values, log_probs = [], [], [], [], []

            episode_rewards.append(episode_reward)
            total_episodes += 1

            if total_episodes % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {total_episodes}, Average Reward: {avg_reward:.2f}")

    def _update_policy(self, states, actions, rewards, values, old_log_probs):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Compute returns and advantages
        returns = self._compute_returns(rewards)
        advantages = returns - torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(5):
            policy, values_pred = self.network(states)
            values_pred = values_pred.squeeze()

            new_log_probs = torch.log(policy.gather(1, actions.unsqueeze(1))).squeeze()
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (returns - values_pred).pow(2).mean()

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _compute_returns(self, rewards):
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
        return torch.FloatTensor(returns).to(self.device)

    def predict(self, observation):
        """Method for compatibility with the trading environment"""
        return self.select_action(observation)
