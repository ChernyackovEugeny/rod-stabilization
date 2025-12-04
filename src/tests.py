import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

a = np.random.uniform(-10, 10, size=4)
b = np.random.uniform(-10, 10, size=4)
q, w, e, r = a
print(a.dot(b), a @ b)







def visualize_episode(self, w, steps=200):
    self.state = np.array([
        np.random.uniform(-0.05, 0.05),  # x
        0.0,  # x_der
        np.random.uniform(-1, 1),  # tetta (угол небольшого отклонения)
        0.0  # tetta_der
    ], dtype=float)

    xs, thetas, actions = [], [], []

    for _ in range(steps):
        action = np.clip(w @ self.state, -self.Fmax, self.Fmax)
        self.state = self.apply_action(self.state, action, self.dt)
        x, _, theta, _ = self.state
        xs.append(x)
        thetas.append(theta)
        actions.append(action)

    fig, ax = plt.subplots()
    ax.set_ylim(-0.2, self.l * 1.2)
    ax.set_aspect('equal')
    ax.grid(True)

    cart_width = 0.4
    cart_height = 0.2

    # Земля
    ground = plt.Line2D([-100, 100], [0, 0], color='black', lw=2)
    ax.add_line(ground)

    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue')
    rod_line, = ax.plot([], [], lw=3, c='red')
    force_arrow = ax.arrow(0, cart_height * 0.5, 0, 0, head_width=0.05, head_length=0.05, fc='green', ec='green')
    ax.add_patch(cart_patch)

    def init():
        cart_patch.set_xy((-cart_width / 2, 0))
        rod_line.set_data([], [])
        return cart_patch, rod_line

    def animate(i):
        x = xs[i]
        theta = thetas[i]
        action = actions[i]

        # Сдвигаем камеру (ось X)
        ax.set_xlim(x - 5, x + 5)

        # Тележка
        cart_patch.set_xy((x - cart_width / 2, 0))

        # Стержень
        x0, y0 = x, cart_height
        x1 = x0 + self.l * np.sin(theta)
        y1 = y0 + self.l * np.cos(theta)
        rod_line.set_data([x0, x1], [y0, y1])

        # Цвет стержня при падении
        rod_line.set_color('orange' if abs(theta) > self.tetta_fail else 'red')

        # Сила как стрелка
        force_len = 0.2 * action / self.Fmax
        # Удаляем старую стрелку
        for patch in ax.patches[1:]:
            patch.remove()
        force_arrow = ax.arrow(x, cart_height * 0.5, force_len, 0, head_width=0.05, head_length=0.05,
                               fc='green', ec='green')
        ax.add_patch(force_arrow)

        return cart_patch, rod_line, force_arrow

    ani = animation.FuncAnimation(fig, animate, frames=steps,
                                  init_func=init, blit=False, interval=50)
    plt.show()
