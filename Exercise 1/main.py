import numpy as np
import matplotlib.pyplot as plt
import typing


def generate_points(pairs: int, experiments: int = 1) -> np.array:
    return np.random.uniform(low=-1.0, high=1.0, size=(2, pairs, experiments,))


def calculate_radii(points: np.array) -> np.array:
    radius = np.sqrt((points ** 2).sum(axis=0))

    return radius


def calculate_pi(**kwargs) -> np.array:
    """

    Either input the radii or the points that were generated before as
    keyword arguments.

    """
    if (radii := kwargs.get("radii")) is None:
        radii = (kwargs.get("points") ** 2).sum(axis=0)
    inside_the_circle = (radii <= 1)
    return 4 * inside_the_circle.sum(axis=0) / len(radii)


def generate_hist_for_pi(pi_array: np.array, pairs: int, experiments: int, bins: int = 10, show: bool = True,
                         save_path: typing.Optional[str] = None
                         ) -> None:
    pi_mean = pi_array.mean(axis=0)
    pi_std = pi_array.std(axis=0)
    plt.hist(pi_array, bins=10)
    plt.title(f"$P = {pairs} X = {experiments}$")
    plt.axvline(pi_mean, color='r')
    plt.axvline(pi_mean + pi_std, color='r', linestyle='--')
    plt.axvline(pi_mean - pi_std, color='r', linestyle='--')
    plt.show()
    # TODO Save
