import numpy as np
from src.BRSagent import BRSagent
from src.HCagent import HCagent
from src.ACagent import ACagent

def BRS_experiment(_):
    agent = BRSagent()
    agent.learn()
    return {"states": agent.states, "actions": agent.actions, "rewards": agent.rewards}

def HC_experiment(_):
    agent = HCagent()
    agent.learn()
    return {"states": agent.states, "actions": agent.actions, "rewards": agent.rewards}

def AC_experiment(_):
    agent = ACagent()
    agent.learn()
    return {"states": agent.states, "actions": agent.actions, "rewards": agent.rewards}

def debug_sweep(n_repeats=5):
    all_results = {"HC": [], "BRS": [], "AC": []}
    agents = {"BRS": BRS_experiment, "AC": AC_experiment}

    for name, func in agents.items():
        print(f"\n=== {name} ===")
        for i in range(n_repeats):
            result = func(i)
            all_results[name].append(result)
            n_steps = len(result["rewards"])
            n_states = len(result["states"])
            n_actions = len(result["actions"])
            print(f"Run {i+1}: steps={n_steps}, states={n_states}, actions={n_actions}")
            # можно проверить первые 3 значения
            print(f"  last rewards: {result['rewards'][-10:]}")
            print(f"  last theta: {np.array(result['states'])[-10:,2]}")
    return all_results

if __name__ == "__main__":
    results = debug_sweep(n_repeats=2)
