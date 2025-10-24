import numpy as np
from typing import Dict, Tuple
from ..units import Units
from .basic_force import Force


class HardSphere(Force):
    def __init__(
        self,
        r1: float = 1.0,
        r2: float = 1.0,
        exclude_intermol: bool = False,
        units: Units = None,
    ):
        super().__init__(units)
        self.r1 = r1
        self.r2 = r2
        self.exclude_intermol = exclude_intermol

        self.big_number = float(np.inf)

    def __call__(
        self,
        pos: np.ndarray,
        r_ij: np.ndarray,
        charge1: float,
        charge2: float,
        system=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = r_ij.shape[0]
        energies = np.zeros(N)
        forces = np.zeros((N, 3))

        # Calculate distances
        r = np.linalg.norm(r_ij, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            mask = r < (self.r1 + self.r2)

            # Energy calculation
            energies[mask] = self.big_number

            # Force calculation
            forces[mask] = self.big_number

        return np.sum(energies), forces
