import numpy as np

from src.Model import Model

class Qagent():
    def __init__(self, k_actions = 20, model=Model(), n_episodes=100):
        model = Model()
        n_episodes = 100

        # Дискретизируем пространство действий
        action_space = np.linspace(-model.Fmax, model.Fmax, k_actions)

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            self.rollout()

    def rollout(self):
        state = self.reset_state()
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < self.model.episode_length:
            # mu_val = self.mu(state)
            # sigma_val = self.sigma(state)
            # action = self.policy(mu_val, sigma_val)



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
            np.random.uniform(-1.0, 1.0),  # x
            np.random.uniform(-0.1, 0.1),  # x_dot
            np.random.uniform(-np.pi / 6, np.pi / 6),  # tetta
            np.random.uniform(-0.1, 0.1)  # tetta_dot
        ], dtype=float)
        return state