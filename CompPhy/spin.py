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

    shape: tuple
    """Tuple of dimension count of the system and lattice sites per dimension"""

    @abc.abstractmethod
    def action_change(self, cfg: NDArray[np.int8], x: typing.Any) -> float:
        """

        Calculate the Action Change if the configuration `cfg` flips its spin
        on position `x`

        """
        pass

    def generate_cfg(self) -> NDArray[np.int8]:
        return np.random.choice(np.array([-1, 1], dtype=np.int8), size=self.shape).flatten()

    def generate_spinup_cfg(self):
        return np.ones(self.shape, dtype=np.int8).flatten()

    def generate_spindown_cfg(self):
        return np.ones(self.shape, dtype=np.int8).flatten() * (-1)


class Ising1D(SpinSystem):
    def __init__(self, lattice_sites: int, j: float, h: float):
        self.lattice_sites = lattice_sites
        self.j = j
        self.h = h
        self.shape = (1, lattice_sites)

    def action_change(self, cfg: NDArray[np.int8], x: int) -> float:
        return 1


model = Ising1D(5, 1, 1)
print(model.generate_cfg())
