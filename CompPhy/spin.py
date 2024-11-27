import abc
import numpy as np
from numpy.typing import NDArray
import typing


Spin = typing.Literal[-1, 1]


class SpinSystem(abc.ABC):
    """

    Defines an abstract class for Spin Systems with any number of dimensions and
    lattice sites

    """

    DIM: int
    N: int

    def __init__(self):
        self.indexing = self.generate_indexing()

    @property
    def shape(self) -> tuple:
        """Shape of a spin configuration"""
        return tuple([self.N] * self.DIM)

    @property
    def lattice_site_count(self):
        return pow(self.N, self.DIM)

    @abc.abstractmethod
    def action_change(self, cfg: NDArray[np.int8], x: typing.Any) -> float:
        """

        Calculate the Action Change if the configuration `cfg` flips its spin
        on position `x`

        """
        pass

    def generate_cfg(self) -> NDArray[np.int8]:
        """Generate a randomized spin configuration."""
        x = np.random.choice(np.array([-1, 1], dtype=np.int8), size=self.shape)
        return x

    def generate_spinup_cfg(self):
        """Generate a spin up configuration."""
        return np.ones(self.shape, dtype=np.int8)

    def generate_spindown_cfg(self):
        """Generate a spin down configuration."""
        return np.ones(self.shape, dtype=np.int8) * (-1)

    def generate_indexing(self) -> NDArray[np.int64]:
        """Generate an array of the indices of all lattice sites."""

        DIM = self.DIM
        N = self.N
        x = np.zeros(shape=(*[N] * DIM, DIM), dtype=np.int64)
        X = np.arange(0, N, 1)

        def fill(x, dim, index=[]):
            for i in X:
                temp_index = [*index, i]
                if dim == 1:
                    x[*temp_index] = temp_index
                else:
                    fill(x, dim - 1, temp_index)

        fill(x, DIM)

        return x.reshape(pow(N, DIM), DIM)

    def magnetization(self, cfg: NDArray[np.int8]):
        return cfg.mean()

    @abc.abstractmethod
    def action(self, cfg: NDArray[np.int8]) -> float:
        """Calculate the hamiltonian for a specific spin configuration."""
        pass

    def generate_sample(
        self, sample_count: int, start_cfg: typing.Optional[NDArray[np.int8]] = None
    ) -> NDArray[np.int8]:
        """

        Generate a sample of Spin Configurations according to the Markov-Chain Monte Carlo method.
        For each sweep visit all lattice sites at random order and propose flipping the
        spin there, accepting or rejecting as it goes

        :param int sample_count: number of samples to generate.

        :return NDArray[np.int8]: Sample of randomized spin configurations

        """

        if start_cfg is None:
            start_cfg = self.generate_cfg()

        indexing = self.generate_indexing()
        sample = np.zeros(shape=(sample_count, *([self.N] * self.DIM)), dtype=np.int8)

        current_cfg = start_cfg

        for step in range(sample_count):
            for X in np.random.permutation(indexing):
                A = min(1, np.exp(-self.action_change(current_cfg, X)))
                if A >= np.random.uniform(0, 1):
                    # Accept the flip
                    current_cfg[*X] *= -1
            sample[step] = current_cfg

        return sample


class Ising1D(SpinSystem):
    def __init__(self, lattice_sites: int, j: float, h: float):
        self.lattice_sites = lattice_sites
        self.j = j
        self.h = h
        self.DIM = 1
        self.N = self.lattice_sites
        super().__init__()

    def action_change(self, cfg: NDArray[np.int8], x: int) -> float:
        spin = cfg[x]
        left = cfg[(x - 1) % self.N]
        right = cfg[(x + 1) % self.N]

        return 2 * spin * (self.h + self.j * (left + right))

    def action(self, cfg):
        # return super().hamiltonian(cfg)
        raise NotImplementedError()


class Ising2D(SpinSystem):
    def __init__(self, lattice_sites: int, j: float, h: float):
        self.lattice_sites = lattice_sites
        self.j = j
        self.h = h
        if self.h != 0:
            raise NotImplementedError()
        self.DIM = 2
        self.N = self.lattice_sites
        super().__init__()

    def action_change(self, cfg: NDArray[np.int8], X: typing.Sequence[int]) -> float:
        N = self.N
        x, y = X

        neighbours = cfg[(x + 1) % N, y] + cfg[(x - 1) % N, y] + cfg[x, (y + 1) % N] + cfg[x, (y - 1) % N]

        return 2 * self.j * cfg[x, y] * neighbours

    def calc_h(self, cfg: NDArray[np.int8]) -> NDArray[np.float64] | np.float64:
        """

        Calculate the Term
        H = - J Σ σ_x σ_y

        :param NDArray[np.int8] config: Spin Configuration

        """

        # Make Case for one config or multiple configs
        if cfg.ndim == 2:
            term1 = (cfg * np.roll(cfg, shift=1, axis=0)).sum()
            term2 = (cfg * np.roll(cfg, shift=1, axis=1)).sum()

            return -self.j * (term1 + term2)

        elif cfg.ndim == 3:
            term1 = (cfg * np.roll(cfg, shift=1, axis=1)).sum(axis=(1, 2))
            term2 = (cfg * np.roll(cfg, shift=1, axis=2)).sum(axis=(1, 2))

            return -self.j * (term1 + term2)
        else:
            raise TypeError("Wrong dimension for cfg")

    def action(self, cfg):
        raise NotImplementedError
