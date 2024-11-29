import numpy as np


class Bootstrap:

    def __init__(self, data, a, B):
        self.data = data
        # Indices into the data
        self.bootstrap_draws = np.random.randint(low=0, high=len(data), size=(B, a))

    def expectation(self, observable):
        # Gives the expectation value of the observable for every bootstrap␣sample B
        return np.fromiter((observable(self.data[draw]).mean() for draw in self.bootstrap_draws), dtype=float)

    def estimate(self, observable):
        o = self.expectation(observable)

        mean = o.mean()
        uncertainty = o.std()
        return mean, uncertainty


def self_bootstrap(data: np.ndarray, a: int, B: int, axis: tuple | None = None) -> tuple[float, float]:
    """

    Measure the mean and uncertainty of a data set using the bootstrapping method.

    :param int a: number of elements per bootstrapping step.
    :param int B: number of bootstrapping steps.

    :return: mean and std of data
    :rtype tuple[float, float]:

    """

    expect = np.fromiter(
        (data[draw].mean() for draw in np.random.randint(low=0, high=len(data), size=(B, a))), dtype=float
    )

    return expect.mean(), expect.std()
