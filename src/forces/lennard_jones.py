import numpy as np
from typing import Dict, Tuple
from ..units import Units
from .basic_force import Force


class LennardJones(Force):
    def __init__(
        self,
        sigma: float = 1.0,
        epsilon: float = 1.0,
        cutoff: float = 2.5,
        units: Units = None,
    ):
        super().__init__(units)
        self.sigma = sigma
        self.epsilon = epsilon
        self.cutoff = cutoff

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
            mask = (r > 1e-12) & (r < self.cutoff)
            # inv_r = np.where(mask, 1.0 / r, 0.0)
            inv_r = 1.0 / r[mask]
            inv_r2 = inv_r**2
            s_over_r = self.sigma * inv_r
            s6 = s_over_r**6
            s12 = s6**2

            # Energy calculation
            energies[mask] = 4.0 * self.epsilon * (s12 - s6)

            # Force calculation
            force_mag = 24.0 * self.epsilon * (2.0 * s12 - s6) * inv_r2
            force_vec = force_mag[:, np.newaxis] * r_ij[mask]
            forces[mask] = force_vec

        return np.sum(energies), forces
