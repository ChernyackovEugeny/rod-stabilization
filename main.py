import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from src.Visualizer import Visualizer
from src.Model import Model
from src.BRSagent import BRSagent
from src.HCagent import HCagent
from src.PGagent import PGagent

agent = HCagent(n_episodes=100)
agent.learn()
visual = Visualizer()
visual.animate(agent.states)