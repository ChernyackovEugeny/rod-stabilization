import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool

from src.Model import Model
from src.BRSagent import BRSagent
from src.HCagent import HCagent

# вынесен еще раз с добавлением best_reward, в исходный класс не добавлен из-за ненадобности
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

        self.best_reward = 0

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            total_reward = self.rollout()
            self.best_reward = max(self.best_reward, total_reward)
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

def u_run_single_experiment(args):
    w_range, n_episodes = args

    agent = BRSagent(n_episodes=n_episodes)
    agent.w = np.random.uniform(-w_range, w_range, size=4)
    agent.learn()
    return agent.best_reward

def n_run_single_experiment(args):
    mu, sigma, n_episodes = args

    agent = BRSagent(n_episodes=n_episodes)
    agent.w = np.random.normal(loc=mu, scale=sigma, size=4)
    agent.learn()
    return agent.best_reward

def parallel_sweep():
    n_episodes = 200
    n_repeats = 2
    w_ranges = [0.01, 0.1, 0.2, 0.4, 1]

    with Pool() as pool:
        print('равномерное распределение')
        for w_range in w_ranges:
            args = [(w_range, n_episodes) for _ in range(n_repeats)]
            rewards = pool.map(u_run_single_experiment, args)
            print(w_range, np.mean(rewards))

        print('нормальное распределение')
        for mu in [-1, 0, 1]:
            for sigma in [0.01, 0.1, 1]:
                args = [(mu, sigma, n_episodes) for _ in range(n_repeats)]
                rewards = pool.map(n_run_single_experiment, args)
                print(mu, sigma, np.mean(rewards))

        print('влияние количества эпизодов')
        for n_episodes in [100, 1000, 2000]:
            args = [(0.4, n_episodes) for _ in range(n_repeats)]
            rewards = pool.map(u_run_single_experiment, args)
            print(n_episodes, np.mean(rewards))

if __name__ == "__main__":
    parallel_sweep()

# равномерное распределение
# 0.01 998.9141257856691
# 0.1 998.7839960446985
# 0.2 999.1440254994335
# 0.4 999.3042626303999
# 1 999.0095708146024
# нормальное распределение
# -1 0.01 999.0169209999115
# -1 0.1 999.2394455450673
# -1 1 999.450635225698
# 0 0.01 999.3754556563393
# 0 0.1 999.2267466801623
# 0 1 999.1982944346506
# 1 0.01 998.9806637754264
# 1 0.1 999.2110103523194
# 1 1 999.4233751706248
# влияние количества эпизодов
# 100 999.0520245520588
# 1000 999.7307275025455
# 2000 999.8275370384988


# def u_run_single_experiment(args):
#     w_range, n_episodes = args
#
#     agent = HCagent(n_episodes=n_episodes)
#     agent.w = np.random.uniform(-w_range, w_range, size=4)
#     agent.learn()
#     return agent.best_reward
#
# def n_run_single_experiment(args):
#     mu, sigma, n_episodes = args
#
#     agent = HCagent(n_episodes=n_episodes)
#     agent.w = np.random.normal(loc=mu, scale=sigma, size=4)
#     agent.learn()
#     return agent.best_reward
#
# def d_run_single_experiment(args):
#     delta_sigma, n_episodes = args
#
#     agent = HCagent(sigma=delta_sigma, n_episodes=n_episodes)
#     agent.learn()
#     return agent.best_reward
#
# def decay_run_single_experiment(args):
#     decay, n_episodes = args
#
#     agent = HCagent(sigma_decay=decay, n_episodes=n_episodes)
#     agent.learn()
#     return agent.best_reward
#
# def parallel_sweep():
#     n_episodes = 200
#     n_repeats = 10
#     w_ranges = [0.01, 0.1, 0.2, 0.4, 1]
#
#     print('равномерное распределение')
#     for w_range in w_ranges:
#         args = [(w_range, n_episodes) for _ in range(n_repeats)]
#         with Pool() as pool:
#             rewards = pool.map(u_run_single_experiment, args)
#         print(w_range, np.mean(rewards))
#
#     print('нормальное распределение')
#     for mu in [-1, 0, 1]:
#         for sigma in [0.01, 0.1, 1]:
#             args = [(mu, sigma, n_episodes) for _ in range(n_repeats)]
#             with Pool() as pool:
#                 rewards = pool.map(n_run_single_experiment, args)
#             print(mu, sigma, np.mean(rewards))
#
#     print('влияние дисперсии шума')
#     for delta_sigma in [0.01, 0.1, 0.5, 1]:
#         args = [(delta_sigma, n_episodes) for _ in range(n_repeats)]
#         with Pool() as pool:
#             rewards = pool.map(d_run_single_experiment, args)
#         print(delta_sigma, np.mean(rewards))
#
#     print('влияние затухания')
#     for decay in [0.5, 0.8, 0.9, 0.99, 0.999, 1]:
#         args = [(decay, n_episodes) for _ in range(n_repeats)]
#         with Pool() as pool:
#             rewards = pool.map(decay_run_single_experiment, args)
#         print(decay, np.mean(rewards))
#
#
# if __name__ == "__main__":
#     parallel_sweep()

# равномерное распределение
# 0.01 999.8535681296365
# 0.1 999.8691029989268
# 0.2 999.8246132721122
# 0.4 999.9113256950643
# 1 999.8766941675256
# нормальное распределение
# -1 0.01 999.8803637058818
# -1 0.1 999.8795777244799
# -1 1 999.9065507176219
# 0 0.01 953.0370151651302
# 0 0.1 999.6478352662783
# 0 1 898.5822072795008
# 1 0.01 611.452835425768
# 1 0.1 807.4625890749583
# 1 1 581.2197210748665
# влияние дисперсии шума
# 0.01 936.265424430899
# 0.1 869.7256351179756
# 0.5 967.4021303839065
# 1 999.8717094814165
# влияние затухания
# 0.5 878.7217791795708
# 0.8 672.9117170457527
# 0.9 840.3621421344458
# 0.99 947.276035368378
# 0.999 999.880612613459
# 1 999.8521522370878


# def lr_experiment(args):
#     lr_actor, lr_critic, n_episodes = args
#     agent = ACagent(lr_actor=lr_actor, lr_critic=lr_critic, n_episodes=n_episodes)
#     agent.learn()
#     return agent.best_reward
#
# def sigma_experiment(args):
#     sigma_init, n_episodes = args
#     agent = ACagent(n_episodes=n_episodes)
#     agent.tetta_sigma = np.ones(4) * np.log(sigma_init)  # sigma = exp(tetta_sigma)
#     agent.learn()
#     return agent.best_reward
#
# def gamma_experiment(args):
#     gamma, n_episodes = args
#     agent = ACagent(gamma=gamma, n_episodes=n_episodes)
#     agent.learn()
#     return agent.best_reward
#
# def parallel_sweep():
#     n_episodes = 200
#     n_repeats = 5  # несколько повторов для усреднения
#
#     with Pool() as pool:
#         print("=== Sweep lr_actor, lr_critic ===")
#         for lr_actor in [1e-4, 3e-4, 5e-4]:
#             for lr_critic in [5e-4, 1e-3, 3e-3]:
#                 args = [(lr_actor, lr_critic, n_episodes) for _ in range(n_repeats)]
#                 rewards = pool.map(lr_experiment, args)
#                 print(f"lr_actor={lr_actor}, lr_critic={lr_critic}, avg_reward={np.mean(rewards):.2f}")
#
#         print("=== Sweep initial sigma ===")
#         for sigma_init in [0.2, 0.3, 0.5]:
#             args = [(sigma_init, n_episodes) for _ in range(n_repeats)]
#             rewards = pool.map(sigma_experiment, args)
#             print(f"sigma_init={sigma_init}, avg_reward={np.mean(rewards):.2f}")
#
#         print("=== Sweep gamma ===")
#         for gamma in [0.95, 0.98, 0.99]:
#             args = [(gamma, n_episodes) for _ in range(n_repeats)]
#             rewards = pool.map(gamma_experiment, args)
#             print(f"gamma={gamma}, avg_reward={np.mean(rewards):.2f}")
#
#
# if __name__ == "__main__":
#     parallel_sweep()

# === Sweep lr_actor, lr_critic ===
# lr_actor=0.0001, lr_critic=0.0005, avg_reward=997.22
# lr_actor=0.0001, lr_critic=0.001, avg_reward=997.76
# lr_actor=0.0001, lr_critic=0.003, avg_reward=997.81
# lr_actor=0.0003, lr_critic=0.0005, avg_reward=999.04
# lr_actor=0.0003, lr_critic=0.001, avg_reward=998.87
# lr_actor=0.0003, lr_critic=0.003, avg_reward=998.45
# lr_actor=0.0005, lr_critic=0.0005, avg_reward=998.80
# lr_actor=0.0005, lr_critic=0.001, avg_reward=999.52
# lr_actor=0.0005, lr_critic=0.003, avg_reward=999.11
# === Sweep initial sigma ===
# sigma_init=0.2, avg_reward=998.94
# sigma_init=0.3, avg_reward=999.50
# sigma_init=0.5, avg_reward=999.61
# === Sweep gamma ===
# gamma=0.95, avg_reward=999.68
# gamma=0.98, avg_reward=998.73
# gamma=0.99, avg_reward=999.64
