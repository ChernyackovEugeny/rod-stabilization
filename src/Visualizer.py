import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Visualizer():
    def __init__(self, cart_width=0.4, cart_height=0.2, pole_length=1.0):
        self.cart_width = cart_width
        self.cart_height = cart_height
        self.pole_length = pole_length

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-0.5, 2.0)
        self.ax.set_aspect('equal')

        self.cart_patch = plt.Rectangle((0, -cart_height/2), cart_width, cart_height, color='blue')
        self.ax.add_patch(self.cart_patch)
        self.pole_line, = self.ax.plot([], [], color='red', linewidth=4)

    # функция для обновления тележки и стержня на каждом кадре
    def update(self, state):
        x, x_dot, theta, theta_dot = state
        cart_top = self.cart_height / 2

        # обновляем координаты тележки
        self.cart_patch.set_xy((x - self.cart_width / 2, -self.cart_height / 2))

        # обновляем координаты стержня
        pole_x = [x, x + self.pole_length * np.sin(theta)]
        pole_y = [cart_top, cart_top + self.pole_length * np.cos(theta)]
        self.pole_line.set_data(pole_x, pole_y)

        return self.cart_patch, self.pole_line  # возвращаем объекты для FuncAnimation

    # функция для создания анимации
    def animate(self, states, interval=20):
        # функция для FuncAnimation
        def anim_func(i):
            return self.update(states[i])

        # создаем анимацию
        ani = animation.FuncAnimation(self.fig, anim_func, frames=len(states), interval=interval, blit=True)

        plt.show()
        return ani