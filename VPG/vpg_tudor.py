from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.nn import Linear, ReLU
import numpy as np
import gym
from gym.spaces import Box, Discrete


class MLP(nn.Module):
    def __init__(self, input_size, fc1_size, fc2_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
                Linear(input_size, fc1_size),
                ReLU(),
                Linear(fc1_size, fc2_size),
                ReLU(),
                Linear(fc2_size, output_size),
                ReLU())

    def forward(self, x):
        x = self.model(x)
        return x


class REINFORCE:

    def __init__(self, env, hidden_state=128, gamma=0.99, lr=0.003):
        if (isinstance(env.action_space, Discrete) and
                isinstance(env.observation_space, Box)):

            self.action_space = env.action_space.n
            self.observation_space = env.observation_space.shape[0]

            self.policy_network = MLP(self.observation_space,
                                      hidden_state, hidden_state,
                                      self.action_space)
            self.optimizer = Adam(self.policy_network.parameters(), lr=lr)
            self.env = env
            self.gamma = gamma

    @torch.no_grad()
    def policy(self, observation):
        observation = torch.from_numpy(observation).float()
        logits = self.policy_network(observation)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        return log_prob, action.numpy().item()

    def compute_loss(self, observation, action, discounted_reward):
        logits = self.policy_network(observation)
        probs = Categorical(logits=logits)
        log_prob = probs.log_prob(action)
        return -(log_prob*discounted_reward).sum()

    def learn(self, n_episodes=10):
        episode_reward = []
        for episode in range(n_episodes):
            observations = []
            actions = []
            log_probs = []
            rewards = []

            obs = self.env.reset()
            done = False
            while not done:
                log_prob, action = self.policy(obs)
                observations.append(obs)
                log_probs.append(log_prob)
                actions.append(action)
                obs, reward, done, info = self.env.step(action)
                rewards.append(reward)

            discounted_rewards = []
            for i in range(len(rewards)):
                R = 0
                for t in range(i, len(rewards)):
                    R += rewards[t] * self.gamma**(t-i)
                discounted_rewards.append(R)

            episode_reward.append(np.sum(rewards))
            print("Episode:", episode, "Mean Reward:", np.mean(episode_reward))

            observations = torch.Tensor(observations)
            actions = torch.Tensor(actions)
            discounted_rewards = torch.Tensor(discounted_rewards)
            log_probs = torch.Tensor(log_probs)

            self.optimizer.zero_grad()
            loss = self.compute_loss(observations, actions, discounted_rewards)
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env, gamma=0.98)
    agent.learn(n_episodes=20000)
