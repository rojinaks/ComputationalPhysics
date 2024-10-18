import numpy as np


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
