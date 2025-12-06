import numpy as np

from src.Model import Model

class ACagent():
    def __init__(self, lr_actor=5e-4, lr_critic=1e-3, gamma=0.95, model=Model(), n_episodes=1000):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.model = model
        self.n_episodes = n_episodes

        self.w = np.random.uniform(-1, 1, size=4)
        self.tetta_mu = np.zeros(4)
        # np.log(0.5) для начального значения sigma, подобрано при выборе гиперпараметров
        self.tetta_sigma = np.ones(4) * np.log(0.5)

        self.sigma_clip_min = 0.05
        self.sigma_clip_max = 2.0
        self.tetta_sigma_clip = 5.0
        self.tetta_mu_clip = 10

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            self.rollout()
            for i in range(len(self.actions)):
                action, state, reward = self.actions[i], self.states[i], self.rewards[i]
                next_value = self.value_func(self.states[i+1]) if i+1 < len(self.states) else 0
                TD_error = reward + self.gamma * next_value - self.value_func(state)

                self.tetta_mu += self.lr_actor * TD_error * self.acceptance_mu(action, state)
                self.tetta_sigma += self.lr_actor * TD_error * self.acceptance_sigma(action, state)

                self.tetta_sigma = np.clip(self.tetta_sigma, -self.tetta_sigma_clip, self.tetta_sigma_clip)
                self.tetta_mu = np.clip(self.tetta_mu, -self.tetta_mu_clip, self.tetta_mu_clip)

                self.w += self.lr_critic * TD_error * state

            # print(f"Episode {episode_ind}, last_reward = {self.rewards[-1]:.2f}")

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

    def value_func(self, state):
        return np.dot(state, self.w)

    def policy(self, mu, sigma):
        a = np.random.normal(mu, sigma)
        return np.clip(a, -self.model.Fmax, self.model.Fmax)

    def sigma(self, s):
        z = np.dot(s, self.tetta_sigma)
        z = np.clip(z, -10, 10)
        sigma = np.clip(np.exp(z), self.sigma_clip_min, self.sigma_clip_max)
        return sigma

    def mu(self, s):
        mu = np.dot(s, self.tetta_mu)
        return mu

    def acceptance_mu(self, action, state):
        acceptance = state * (action - self.mu(state)) / (self.sigma(state) ** 2 + 1e-8)
        return acceptance

    def acceptance_sigma(self, action, state):
        acceptance = state * ((action - self.mu(state)) ** 2 / (self.sigma(state) ** 2 + 1e-8) - 1)
        return acceptance

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