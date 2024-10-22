import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import main


def generate_hist(pairs: int, experiments: int, ax: Axes) -> float:
    points = main.generate_points(pairs, experiments)
    pi: np.array = main.calculate_pi(points=points)
    pi_mean = pi.mean()
    pi_std = pi.std()

    ax.hist(pi)
    ax.axvline(pi_mean, color='r')
    ax.axvline(pi_mean + pi_std, color='r', linestyle="--")
    ax.axvline(pi_mean - pi_std, color='r', linestyle="--")
    ax.set_title(f"$P = {pairs}$ $X = {experiments}$")

    return pi_std


fig, ax = plt.subplots(4, 4, figsize=(20, 16))

x = [10, 100, 1_000, 10_000]
uncertainties = []  # uncertainties depending on pairs and experiments

for i, pairs in enumerate(x):
    for j, experiments in enumerate(x):
        std = generate_hist(pairs, experiments, ax[i, j])
        uncertainties.append([pairs, experiments, std])

# First row = pairs
# Second row = experiments
# Third row = uncertainties
uncertainties = np.array(uncertainties)

# uncertainties as a function of X and constant P
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for i in range(4):
    data = uncertainties[4 * i: 4 * (i + 1)]
    ax[i].scatter(data[:, 1], data[:, 2])
    ax[i].set_title(f"$P = {int(uncertainties[4 * i][0])}$")
    ax[i].set_yscale("log")
    ax[i].set_xscale("log")
fig.savefig(main.FIGS_DIR/"1.5_uncertainty_function_of_x.pdf")

# uncertainties as a function of P and constant X
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
for i in range(4):
    indices = [False, False, False, False]
    indices[i] = True
    indices = indices * 4
    data = uncertainties[indices]
    ax[i].scatter(data[:, 0], data[:, 2])
    ax[i].set_title(f"$X = {int(data[0, 1])}$")
    ax[i].set_yscale("log")
    ax[i].set_xscale("log")
fig.savefig(main.FIGS_DIR/"1.5_uncertainty_function_of_p.pdf")

plt.show()
