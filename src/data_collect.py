import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import pickle

from src.Model import Model

import numpy as np

from src.Model import Model

class BRSagent():
    def __init__(self, model=Model(), n_episodes=1000):
        self.w = np.random.normal(loc=-1, scale=1, size=4)
        self.n_episodes = n_episodes
        self.model = model
        self.best_w = self.w.copy()
        self.best_reward = -np.inf

        self.states = []
        self.actions = []
        self.rewards = []

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            self.w = np.random.normal(loc=-1, scale=1, size=4)
            r = self.rollout(self.w)

            # print(f"Episode {episode_ind}, best_reward = {self.best_reward:.2f}")

            if r > self.best_reward:
                self.best_reward = r
                self.best_w = self.w.copy()

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

        self.all_states.append(np.array(self.states, dtype=np.float32))
        self.all_actions.append(np.array(self.actions, dtype=np.float32))
        self.all_rewards.append(np.array(self.rewards, dtype=np.float32))

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

# веса подобраны усреднениями 10 лучших найденых векторов весов после прогонов BRS агента
class BestBRSagent():
    def __init__(self, model=Model(), n_episodes=1000):
        self.w = [-2.25190609, -0.92523376, -1.38602878,  0.28684791]
        self.n_episodes = n_episodes
        self.model = model
        self.best_w = self.w.copy()
        self.best_reward = -np.inf

        self.states = []
        self.actions = []
        self.rewards = []

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            r = self.rollout(self.w)

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

        self.all_states.append(np.array(self.states, dtype=np.float32))
        self.all_actions.append(np.array(self.actions, dtype=np.float32))
        self.all_rewards.append(np.array(self.rewards, dtype=np.float32))

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

class HCagent():
    def __init__(self, model=Model(), sigma=1, sigma_decay=0.99, n_episodes=1000):
        self.w = np.random.uniform(-0.4, 0.4, size=4)
        self.best_w = self.w.copy()
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.n_episodes = n_episodes
        self.model = model
        self.best_reward = -np.inf

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []

        self.states = []
        self.actions = []
        self.rewards = []

    def learn(self):
        for episode_ind in range(self.n_episodes):
            delta = np.random.normal(0, self.sigma, size=4)
            new_w = self.w + delta

            reward = self.rollout(new_w)

            # print(f"Episode {episode_ind}, best_reward = {self.best_reward:.2f}")

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

        self.all_states.append(np.array(self.states, dtype=np.float32))
        self.all_actions.append(np.array(self.actions, dtype=np.float32))
        self.all_rewards.append(np.array(self.rewards, dtype=np.float32))

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

class ACagent():
    def __init__(self, lr_actor=3e-4, lr_critic=1e-2, gamma=0.95, model=Model(), n_episodes=1000):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.model = model
        self.n_episodes = n_episodes

        self.w = np.random.uniform(-1, 1, size=4)
        self.tetta_mu = np.zeros(4)
        self.tetta_sigma = np.ones(4) * 0.1

        self.sigma_clip_min = 0.05
        self.sigma_clip_max = 2.0
        self.tetta_sigma_clip = 5.0
        self.tetta_mu_clip = 10

        self.states = []
        self.actions = []
        self.rewards = []

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []

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

        self.all_states.append(np.array(self.states, dtype=np.float32))
        self.all_actions.append(np.array(self.actions, dtype=np.float32))
        self.all_rewards.append(np.array(self.rewards, dtype=np.float32))

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






# --- 1. ОДИН ПРОГОН ОДНОГО АГЕНТА --- #

def BRS_experiment(_):
    agent = BRSagent()
    agent.learn()
    return {
        "states": agent.all_states,
        "actions": agent.all_actions,
        "rewards": agent.all_rewards
    }

def HC_experiment(_):
    agent = HCagent()
    agent.learn()
    return {
        "states": agent.all_states,
        "actions": agent.all_actions,
        "rewards": agent.all_rewards
    }

def AC_experiment(_):
    agent = ACagent()
    agent.learn()
    return {
        "states": agent.all_states,
        "actions": agent.all_actions,
        "rewards": agent.all_rewards
    }
def BestBRS_experiment(_):
    agent = BestBRSagent()
    agent.learn()
    return {
        "states": agent.all_states,
        "actions": agent.all_actions,
        "rewards": agent.all_rewards
    }


def pad_episode(states, actions, rewards, max_steps, state_dim=4):
    T = len(rewards)

    padded_states  = np.zeros((max_steps, state_dim), dtype=np.float32)
    padded_actions = np.zeros(max_steps, dtype=np.float32)
    padded_rewards = np.zeros(max_steps, dtype=np.float32)

    padded_states[:T]  = states
    padded_actions[:T] = actions
    padded_rewards[:T] = rewards

    return padded_states, padded_actions, padded_rewards


# ===========================
# 3. ПАДДИНГ ВСЕХ ЭПИЗОДОВ ОДНОГО ПРОГОНА
# ===========================

def pad_run_all_episodes(run, max_steps, max_episodes, state_dim=4):

    padded_states  = np.zeros((max_episodes, max_steps, state_dim), dtype=np.float32)
    padded_actions = np.zeros((max_episodes, max_steps), dtype=np.float32)
    padded_rewards = np.zeros((max_episodes, max_steps), dtype=np.float32)

    num_eps = len(run["states"])

    for ep in range(num_eps):
        s, a, r = pad_episode(
            run["states"][ep],
            run["actions"][ep],
            run["rewards"][ep],
            max_steps=max_steps,
            state_dim=state_dim
        )
        padded_states[ep]  = s
        padded_actions[ep] = a
        padded_rewards[ep] = r

    return padded_states, padded_actions, padded_rewards


# ===========================
# 4. ЗАПУСК ВСЕХ АГЕНТОВ + ПАДДИНГ ДО ЕДИНОЙ ФОРМЫ
# ===========================

def run_all(n_repeats=10, max_steps=1000):

    agent_list = ["HC", "BRS", "BestBRS", "AC"]
    funcs = {"HC": HC_experiment, "BRS": BRS_experiment, "BestBRS": BestBRS_experiment, "AC": AC_experiment}

    # запускаем и собираем сырые данные
    raw_results = {}

    with Pool() as pool:
        for name in agent_list:
            print(f"=== Запуск {name} ===")
            runs = pool.map(funcs[name], range(n_repeats))
            raw_results[name] = runs

    max_episodes = 1000

    # ПАДДИНГ ДО (repeats, episodes, max_steps, ...)
    final_results = {}

    for name in agent_list:
        states_all  = []
        actions_all = []
        rewards_all = []

        for run in raw_results[name]:
            s, a, r = pad_run_all_episodes(run,
                                           max_steps=max_steps,
                                           max_episodes=max_episodes)
            states_all.append(s)
            actions_all.append(a)
            rewards_all.append(r)

        final_results[name] = {
            "states": np.array(states_all, dtype=np.float32),   # (repeats, episodes, steps, 4)
            "actions": np.array(actions_all, dtype=np.float32), # (repeats, episodes, steps)
            "rewards": np.array(rewards_all, dtype=np.float32)
        }

    return final_results


# ===========================
# 5. MAIN
# ===========================

if __name__ == "__main__":
    N_REPEATS = 20
    MAX_STEPS = 1000

    results = run_all(n_repeats=N_REPEATS, max_steps=MAX_STEPS)

    with open("agent_results_ready.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Готовый тензор сохранён в 'agent_results_ready.pkl'")