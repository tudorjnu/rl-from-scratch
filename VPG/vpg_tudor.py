from torch import nn
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Box, Discrete


class MLP(nn.Module):
    def __init__(self, input_size, fc1_size, fc2_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_size),
            nn.Softmax())

    def forward(self, x):
        x = self.model(x)
        return x


class REINFORCE:

    def __init__(self, env, hidden_state=128, gamma=0.99, lr=0.003):
        # check if action space is Discrete and if the observation space is contiinuous
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
        # transform observation to torch tensor
        observation = torch.from_numpy(observation).float()
        # get the probabilities of each action
        probs = self.policy_network(observation)
        # create a categorical distribution and sample an action
        probs = Categorical(probs=probs)
        action = probs.sample().numpy().item()
        return action

    def compute_loss(self, observation, action, discounted_reward):
        # get the probabilities of each action
        probs = self.policy_network(observation)
        # create a categorical distribution
        prob_distrib = Categorical(probs=probs)
        # get the log probability of the action
        log_prob = prob_distrib.log_prob(action)
        return -(log_prob*discounted_reward).sum()

    def learn(self, n_episodes=100):
        episode_reward = []
        for episode in range(n_episodes):
            observations = []
            actions = []
            rewards = []

            obs = self.env.reset()
            done = False
            while not done:
                action = self.policy(obs)
                observations.append(obs)
                actions.append(action)
                obs, reward, done, info = self.env.step(action)
                rewards.append(reward)

            # compute the discounted reward
            discounted_rewards = []
            # for each step in the episode
            for i in range(len(rewards)):
                # set the initial reward to 0
                R = 0
                # for each of the following steps
                for t in range(i, len(rewards)):
                    # add the discounted reward to R
                    # t - i is because the rewards are discounted from the
                    # current step to the end of the episode
                    R += rewards[t] * self.gamma**(t-i)
                # add the discounted reward to the list of discounted rewards
                discounted_rewards.append(R)

            # for displaying the progress
            episode_reward.append(np.sum(rewards))
            print("Episode:", episode, "Mean Rewards:",
                  np.mean(episode_reward[-100:]))

            observations = torch.Tensor(observations)
            actions = torch.Tensor(actions)
            discounted_rewards = torch.Tensor(discounted_rewards)

            self.optimizer.zero_grad()
            loss = self.compute_loss(observations, actions, discounted_rewards)
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = REINFORCE(env, gamma=0.98)
    agent.learn(n_episodes=20000)
