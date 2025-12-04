import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.Model import Model

class HCagent():
    def __init__(self, model=Model(), sigma=0.5, sigma_decay=0.995, n_episodes=100):
        self.w = np.random.uniform(-1, 1, size=4)
        self.best_w = self.w.copy()
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.n_episodes = n_episodes
        self.model = model
        self.best_reward = -np.inf

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            delta = np.random.normal(0, self.sigma, size=4)
            new_w = self.w + delta

            reward = self.rollout(new_w)

            print(f"Episode {episode_ind}, best_reward = {self.best_reward:.2f}")

            if reward > self.best_reward:
                self.w = new_w
                self.best_reward = reward
                self.best_w = new_w.copy()

            self.sigma = max(self.sigma * self.sigma_decay, 0.01)

    def rollout(self, w):
        state = self.reset_state()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < self.model.episode_length:
            action = np.clip(np.dot(w, state), -self.model.Fmax, self.model.Fmax)

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
