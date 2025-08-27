import numpy as np
from typing import Dict, Tuple
from units import Units
from .basic_force import Force


class Coulomb(Force):
    def __init__(self, units: Units = None, cutoff: float = None):
        super().__init__(units)
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
            if self.cutoff is not None:
                mask = (r > 1e-12) & (r < self.cutoff)
            else:
                mask = r > 1e-12

            inv_r = 1.0 / r[mask]
            inv_r3 = inv_r**3

            # Energy calculation
            energies[mask] = charge1 * charge2 * inv_r

            # Force calculation
            force_mag = charge1 * charge2 * inv_r3
            force_vec = force_mag[:, np.newaxis] * r_ij[mask]
            forces[mask] = force_vec

        # Apply reduced Bjerrum length scaling for LJ units
        if system.ewald_handler is not None:
            energies *= system.ewald_handler.lB_star
            forces *= system.ewald_handler.lB_star

        return np.sum(energies), forces
