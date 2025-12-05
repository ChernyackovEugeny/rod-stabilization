import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.Visualizer import Visualizer
from src.Model import Model
from src.BRSagent import BRSagent
from src.HCagent import HCagent
from src.ACagent import ACagent

w = []
for i in range(10):
    print(i)
    agent = BRSagent(n_episodes=1000)
    agent.learn()
    w.append(agent.best_w)
print(np.mean(w), np.max(w), w)