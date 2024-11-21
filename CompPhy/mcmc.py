import typing
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

SpinConfig = NDArray[np.int8]


class Ising2D:
    """

    Markov Chain Monte Carlo for the 2D Ising Model

    :param float J: coupling strength between neighbouring spins
    :param float h: magnetic field strength
    :param int lattice_sites: number of spins in one dimension

    """

    def __init__(self, J: float, h: float, lattice_sites: int):
        self.J = J
        self.h = h
        self.lattice_sites = lattice_sites

        # Not yet implemented for Exercise 5
        if self.h != 0:
            raise NotImplementedError

    def generate_spin_config(self) -> SpinConfig:
        """Generate one 2d Array of spin values."""
        return np.random.choice(
            np.array([-1, 1], dtype=np.int8),
            size=(self.lattice_sites, self.lattice_sites),
        )

    def _calc_h(self, config: SpinConfig) -> NDArray[np.float64]:
        """

        Calculate the Term
        H = - J Σ σ_x σ_y

        :param NDArray[np.int8] config: Spin Configuration

        """

        # Make Case for one config or multiple configs
        if config.ndim == 2:
            term1 = (config * np.roll(config, shift=1, axis=0)).sum()
            term2 = (config * np.roll(config, shift=1, axis=1)).sum()

            return -self.J * (term1 + term2)

        else:
            term1 = (config * np.roll(config, shift=1, axis=1)).sum(axis=(1, 2))
            term2 = (config * np.roll(config, shift=1, axis=2)).sum(axis=(1, 2))

            return -self.J * (term1 + term2)

    def _calc_m(self, config: SpinConfig) -> np.float64:
        """

        Calculate the Term
        M =  - h * Σ σ_x

        """

        return -self.h * config.sum()

    def _calc_action_change(self, config: SpinConfig, x: int, y: int):
        N = self.lattice_sites

        neighbours = (
            config[(x + 1) % N, y]
            + config[(x - 1) % N, y]
            + config[x, (y + 1) % N]
            + config[x, (y - 1) % N]
        )

        return 2 * self.J * config[x, y] * neighbours

    def _make_proposal(self, config: SpinConfig) -> SpinConfig:

        x, y = np.random.randint(low=0, high=self.lattice_sites, size=2)

        a = min(1, np.exp(-self._calc_action_change(config, x, y)))
        u = np.random.uniform(0, 1)

        new_config = deepcopy(config)
        if a >= u:
            new_config[x, y] *= -1

        return new_config

    def __call__(
        self,
        number_of_configs: int,
        starting_config: typing.Optional[SpinConfig] = None,
    ):
        # Generate a random starting position if not specified
        if starting_config is None:
            starting_config = self.generate_spin_config()

        configurations = [starting_config]
        for i in range(number_of_configs - 1):
            configurations.append(self._make_proposal(configurations[-1]))

        return np.array(configurations)
