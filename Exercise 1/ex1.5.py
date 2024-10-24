# %% Generate Data
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import main


def generate_hist(pairs: int, experiments: int, ax: Axes) -> typing.Tuple[float, float]:
    points = main.generate_points(pairs, experiments)
    pi: np.array = main.calculate_pi(points=points)
    pi_mean = pi.mean()
    pi_std = pi.std()

    ax.hist(pi, label=r"$\pi_x$ distribution")
    ax.axvline(pi_mean, color='blue', linestyle=":", label=r"$\pi_\text{f}$")
    ax.axvline(np.pi, color="black", linestyle="--", label=r"true $\pi$ value")
    ax.axvspan(pi_mean - pi_std, pi_mean + pi_std, alpha=0.3, color="blue", zorder=0, label="uncertainty")
    ax.set_title(f"$P = {pairs}$ $X = {experiments}$")

    return pi_mean, pi_std


# %% Plot different P and X combinations
fig, ax = plt.subplots(4, 4, figsize=(20, 16))

x = [10, 100, 1_000, 10_000]
pi_values: dict[str, (float, float)] = dict()  # uncertainties depending on pairs and experiments

for i, pairs in enumerate(x):
    for j, experiments in enumerate(x):
        pi, std = generate_hist(pairs, experiments, ax[i, j])
        pi_values[f"{pairs}_{experiments}"] = (pi, std)

# get the extremal limits
x_min = np.inf
x_max = -np.inf
for i in range(4):
    for j in range(4):
        limits = ax[i, j].get_xlim()
        x_min = min(x_min, limits[0])
        x_max = max(x_max, limits[1])

# set for comparison the same x_limits for all hists
for i in range(4):
    for j in range(4):
        ax[i, j].set_xlim(x_min, x_max)

handles, labels = ax[0, 0].get_legend_handles_labels()
plt.legend(handles, labels, loc="upper right", frameon=True, bbox_to_anchor=(1.55, 4.5))
plt.tight_layout()
plt.savefig(main.FIGS_DIR / "ex1.5_px_combinations.pdf")
plt.show()

# %% Create a latex table for pi values
keys = [x.split("_") for x in pi_values.keys()]
index1 = [int(x[0]) for x in keys]
index2 = [int(x[1]) for x in keys]
multi_index = pd.MultiIndex.from_arrays([index1, index2], names=["P", "X"])

pi_df = pd.DataFrame({
    "Pi"   : [f"\\num{{{round(x[0], 3)} \\pm {round(x[1], 3)}}}" for x in pi_values.values()],
    "value": [x[0] for x in pi_values.values()],
    "std"  : [x[1] for x in pi_values.values()]
}, index=multi_index)
data_matrix = pi_df["Pi"].unstack(level="P")
styler = data_matrix.style
with open(main.TABLES_DIR / "ex1.5_table.tex", "w") as f:
    f.write(r"%$X \text\textbackslash P$ & 10 & 100 & 1000 & 10000 \\" + "\n")
    caption = r"""
    Calcuated $\pi$ values for different $P$ and $X$.
    """
    label = r"tab:ex1.5_pi_values"
    styler.to_latex(buf=f, hrules=True, position_float="centering", label=label,
                    caption=caption, column_format=r"c|cccc")

# %% uncertainties as a function of X and constant P
import matplotlib.ticker as mticker
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].set_ylabel("Uncertainty")
for ax_index, i in enumerate(x):
    data = pi_df.query(f"P == {i}")["std"]
    x_axis = data.index.get_level_values("X")
    y_axis = data.to_numpy()
    ax[ax_index].scatter(x_axis, y_axis, s=60)
    ax[ax_index].set_title(f"$P = {i}$")
    ax[ax_index].set_yscale("log")
    ax[ax_index].set_xscale("log")
    ax[ax_index].yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax[ax_index].set_xlabel(r"$X$")
fig.savefig(main.FIGS_DIR / "1.5_uncertainty_function_of_x.pdf")
plt.show()

# %% uncertainties as a function of P and constant X
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].set_ylabel("Uncertainty")
for ax_index, i in enumerate(x):
    data = pi_df.query(f"X == {i}")["std"]
    x_axis = data.index.get_level_values("P")
    y_axis = data.to_numpy()
    ax[ax_index].set_xlabel(r"$P$")
    ax[ax_index].scatter(x_axis, y_axis, s=60)
    ax[ax_index].set_title(f"$X = {i}$")
    ax[ax_index].set_yscale("log")
    ax[ax_index].set_xscale("log")
fig.savefig(main.FIGS_DIR / "1.5_uncertainty_function_of_p.pdf")
plt.show()
