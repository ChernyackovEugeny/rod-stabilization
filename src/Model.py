import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Model():
    def __init__(self, episode_length=1000, Fmax=10, dt=0.01):
        self.l = 1.0
        self.m = 0.5
        self.M = 1.2

        self.tetta_fail = np.pi/2
        self.x_fail = 10.0

        self.Fmax = Fmax
        self.dt = dt

        self.episode_length = episode_length

    def step(self, state, action):
        action = np.clip(action, -self.Fmax, self.Fmax)
        new_state = self.apply_action(state, action, self.dt)

        reward = self.reward(new_state)

        done = abs(new_state[0]) > self.x_fail or abs(new_state[2]) > self.tetta_fail

        return new_state, reward, done

    def g(self, state, action):
        x, x_der, tetta, tetta_der = state
        g = 9.8
        I = self.m * self.l**2 / 3
        a = self.l / 2
        delta = I * (self.M + self.m) - (self.m * a * np.cos(tetta))**2
        delta = max(delta, 1e-4)  # численная стабилизация для знаменателя

        x_2der = (I * (action + self.m * a * tetta_der**2 * np.sin(tetta)) +
                  self.m**2 * g * a**2 * np.cos(tetta) * np.sin(tetta)) / delta
        tetta_2der = (-self.m * a * np.cos(tetta) * (action + self.m * a * tetta_der**2 * np.sin(tetta)) -
                      (self.M + self.m) * self.m * g * a * np.sin(tetta)) / delta

        return np.array([x_der, x_2der, tetta_der, tetta_2der])

    def apply_action(self, state, action, dt):

        k1 = self.g(state, action)
        k2 = self.g(state + dt * 0.5 * k1, action)
        k3 = self.g(state + dt * 0.5 * k2, action)
        k4 = self.g(state + dt * k3, action)

        new_state = state + dt * (1/6 * (k1 + 2*k2 + 2*k3 + k4))
        return new_state

    def reward(self, state):
        x, x_der, tetta, tetta_der = state
        r = 1 - (0.1 * tetta ** 2 + 0.1 * x ** 2 + 0.01 * tetta_der ** 2 + 0.01 * x_der ** 2)
        if abs(x) > self.x_fail or abs(tetta) > self.tetta_fail:
            r -= 10
        return r
