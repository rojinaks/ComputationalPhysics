import matplotlib.pyplot as plt
import numpy as np
import typing
import matplotlib.pyplot as plt

import main


def generate_hist_for_pi(pi_array: np.array, pairs: int, experiments: int, bins: int = 10, show: bool = True,
                         save_path: typing.Optional[str] = None
                         ) -> None:
    pi_mean = pi_array.mean(axis=0)
    pi_std = pi_array.std(axis=0)
    plt.hist(pi_array, bins=10)
    plt.title(f"$P = {pairs} X = {experiments}$")
    plt.axvline(pi_mean, color='blue', linestyle=":")
    plt.axvline(np.pi, color="black", linestyle="--")
    plt.axvspan(pi_mean - pi_std, pi_mean + pi_std, alpha=0.3, color="blue", zorder=0)
    plt.xlabel(r"$\pi_x$")
    plt.ylabel(r"$\pi_x$ count")
    # plt.axvline(pi_mean + pi_std, color='r', linestyle='--')
    # plt.axvline(pi_mean - pi_std, color='r', linestyle='--')

    if save_path:
        plt.savefig(save_path)
    plt.show()


for pairs, experiments in [(100, 100), (1, 10_000)]:
    points = main.generate_points(pairs, experiments)
    pi = main.calculate_pi(points=points)

    pi_mean = pi.mean()
    pi_std = pi.std()

    print(f"Calculated π = {pi_mean:.3f} +- {pi_std:.3f}")
    print(f"Real π: {np.pi:.3f}\n")

    generate_hist_for_pi(pi, pairs, experiments, save_path=main.FIGS_DIR / f"ex1.2_pi_hist_{pairs}_{experiments}.pdf")
