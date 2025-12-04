import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from src.Model import Model

class PGagent():
    def __init__(self, lr=0.1, gamma=0.9, model=Model(), n_episodes=100):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.n_episodes = n_episodes

        self.tetta_mu = np.zeros(4)
        self.tetta_sigma = np.ones(4) * 0.1

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            self.rollout()
            difference_mu = 0
            difference_sigma = 0

            for i in range(len(self.actions)):
                G = 0  # считаем доход
                for j in range(i, len(self.rewards)):
                    G += self.gamma ** (j-i) * self.rewards[j]

                action, state = self.actions[i], self.states[i]

                difference_mu += self.acceptance_mu(action, state) * G
                difference_sigma += self.acceptance_sigma(action, state) * G

            self.tetta_mu = self.tetta_mu + self.lr * difference_mu
            self.tetta_sigma = self.tetta_sigma + self.lr * difference_sigma

            print(f"Episode {episode_ind}, last_reward = {self.rewards[-1]:.2f}")

    def rollout(self):
        state = self.reset_state()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < self.model.episode_length:
            mu_val = self.mu(state)
            sigma_val = self.sigma(state)
            action = self.policy(mu_val, sigma_val)

            new_state, reward, done = self.model.step(state, action)

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

            state = new_state
            total_reward += reward
            steps += 1

        return total_reward

    def reset_state(self):
        self.states = []
        self.actions = []
        self.rewards = []
        state = np.array([
            np.random.uniform(-1.0, 1.0),         # x
            np.random.uniform(-0.1, 0.1),         # x_dot
            np.random.uniform(-np.pi/6, np.pi/6), # tetta
            np.random.uniform(-0.1, 0.1)          # tetta_dot
        ], dtype=float)
        return state

    def policy(self, mu, sigma):
        mu_tensor = torch.tensor(mu, dtype=torch.float32)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

        dist = torch.distributions.Normal(mu_tensor, sigma_tensor)
        a = dist.sample()
        return a.numpy()

    def sigma(self, s):
        z = np.dot(s, self.tetta_sigma)
        z = np.clip(z, -10, 10)  # ограничиваем диапазон скалярного произведения
        sigma = np.exp(z)
        return sigma

    def mu(self, s):
        mu = s @ self.tetta_mu
        return mu

    def acceptance_mu(self, action, state):
        acceptance = state * (action - self.mu(state)) / self.sigma(state)
        return acceptance

    def acceptance_sigma(self, action, state):
        acceptance = state * (action - self.mu(state)) ** 2 / self.sigma(state) ** 2
        return acceptance