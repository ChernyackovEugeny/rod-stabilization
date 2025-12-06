import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

with open("agent_results_ready.pkl", "rb") as f:
    agent_results = pickle.load(f)

def plot_and_save_agent_metrics(agent_results, upright_threshold=0.087, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    agents = list(agent_results.keys())

    # ТЕКСТОВЫЙ ВЫВОД МЕТРИК
    for agent_name in agents:
        print(f"\n=== {agent_name} ===")
        rewards = agent_results[agent_name]["rewards"]  # (repeats, episodes, steps)
        states = agent_results[agent_name]["states"]    # (repeats, episodes, steps, 4)

        # 1. Среднее вознаграждение
        avg_reward = np.mean(rewards)
        avg_reward_last_step = np.mean(rewards[:, :, -1])
        print(f"Среднее вознаграждение по всем шагам: {avg_reward:.3f}")
        print(f"Среднее вознаграждение на последнем шаге эпизодов: {avg_reward_last_step:.3f}")

        # 2. Кумулятивное вознаграждение
        cum_rewards = np.cumsum(rewards, axis=2)
        avg_cum_reward_last_step = np.mean(cum_rewards[:, :, -1])
        print(f"Среднее кумулятивное вознаграждение на последнем шаге: {avg_cum_reward_last_step:.3f}")

        # 3. Длина эпизодов
        episode_lengths = np.sum(rewards != 0, axis=2).flatten()
        print(f"Средняя длина эпизодов: {np.mean(episode_lengths):.1f} шагов")
        print(f"Максимальная длина эпизода: {np.max(episode_lengths)} шагов")
        print(f"Минимальная длина эпизода: {np.min(episode_lengths)} шагов")

        # 4. RMSE θ
        theta = states[:, :, :, 2]
        rmse_theta = np.sqrt(np.mean(theta ** 2, axis=(1, 2)))
        print(f"Средний RMSE θ: {np.mean(rmse_theta):.4f} rad")
        print(f"Минимальный RMSE θ: {np.min(rmse_theta):.4f} rad")
        print(f"Максимальный RMSE θ: {np.max(rmse_theta):.4f} rad")

        # 5. Время удержания вертикали
        upright_steps = np.sum(np.abs(theta) <= upright_threshold, axis=(1, 2))
        print(f"Среднее количество шагов в вертикали: {np.mean(upright_steps):.1f}")
        print(f"Минимальное количество шагов в вертикали: {np.min(upright_steps)}")
        print(f"Максимальное количество шагов в вертикали: {np.max(upright_steps)}")

    # ГРАФИКИ
    # 1. Усреднённое вознаграждение по шагам
    plt.figure(figsize=(10, 5))
    for agent_name in agents:
        rewards = agent_results[agent_name]["rewards"]
        avg_rewards = np.mean(rewards, axis=(0, 1))
        plt.plot(avg_rewards, label=agent_name)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    plt.title("Усреднённое вознаграждение по шагам")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_reward_per_step.png")
    plt.show()

    # 2. Кумулятивное вознаграждение
    plt.figure(figsize=(10, 5))
    for agent_name in agents:
        rewards = agent_results[agent_name]["rewards"]
        cum_rewards = np.cumsum(rewards, axis=2)
        avg_cum_rewards = np.mean(cum_rewards, axis=(0, 1))
        plt.plot(avg_cum_rewards, label=agent_name)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Усреднённое кумулятивное вознаграждение")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_cumulative_reward.png")
    plt.show()

    # 3. Длина эпизодов
    plt.figure(figsize=(8, 5))
    for i, agent_name in enumerate(agents):
        rewards = agent_results[agent_name]["rewards"]
        lengths = np.sum(rewards != 0, axis=2).flatten()
        x = np.repeat(i, len(lengths))
        plt.scatter(x, lengths, alpha=0.6)
        plt.plot(i, np.mean(lengths), 'ro')
    plt.xticks(range(len(agents)), agents)
    plt.ylabel("Episode Length (steps)")
    plt.title("Длина эпизодов (каждая реплика + среднее)")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/episode_lengths.png")
    plt.show()

    # 4. RMSE θ и шаги в вертикали
    for agent_name in agents:
        states = agent_results[agent_name]["states"]
        theta = states[:, :, :, 2]
        rmse_theta = np.sqrt(np.mean(theta**2, axis=(1, 2)))
        upright_steps = np.sum(np.abs(theta) <= upright_threshold, axis=(1, 2))

        plt.figure(figsize=(12, 4))
        # RMSE θ
        plt.subplot(1,2,1)
        plt.scatter(range(len(rmse_theta)), rmse_theta, alpha=0.6)
        plt.plot(range(len(rmse_theta)), [np.mean(rmse_theta)]*len(rmse_theta), 'r-', label='Mean')
        plt.xlabel("Run")
        plt.ylabel("RMSE θ (rad)")
        plt.title(f"{agent_name} — RMSE θ")
        plt.grid(True)
        plt.legend()

        # Steps Upright
        plt.subplot(1,2,2)
        plt.scatter(range(len(upright_steps)), upright_steps, alpha=0.6)
        plt.plot(range(len(upright_steps)), [np.mean(upright_steps)]*len(upright_steps), 'r-', label='Mean')
        plt.xlabel("Run")
        plt.ylabel(f"Steps Upright (θ ±{upright_threshold} rad)")
        plt.title(f"{agent_name} — Время в вертикали")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{agent_name}_theta_upright.png")
        plt.show()

plot_and_save_agent_metrics(agent_results)
