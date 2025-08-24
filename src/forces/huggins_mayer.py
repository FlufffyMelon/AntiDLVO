import numpy as np
from typing import Dict, Tuple
from units import Units
from .basic_force import Force


class HugginsMayer(Force):
    def __init__(
        self,
        sigma: float = 1.0,
        b: float = 1.0,
        B: float = 1.0,
        c: float = 1.0,
        d: float = 1.0,
        cutoff: float = 2.5,
        units: Units = None,
    ):
        super().__init__(units)
        self.sigma = sigma
        self.b = b
        self.B = B
        self.c = c
        self.d = d
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
            r_m = r[mask]
            inv_r = 1 / r_m
            inv_r6 = inv_r**6
            inv_r8 = inv_r**8

            # Energy calculation
            energies[mask] = (
                self.b * np.exp(self.B * (self.sigma - r_m))
                + self.c * inv_r6
                + self.d * inv_r8
            )

            # Force calculation
            force_mag = (
                self.b * self.B * np.exp(self.B * (self.sigma - r_m))
                + (6 * self.c * inv_r6 + 8 * self.d * inv_r8) * inv_r
            ) * inv_r
            # print("inv_r", inv_r.shape)
            # print("r_ij", r_ij[mask].shape)
            forces[mask] = force_mag[:, np.newaxis] * r_ij[mask]

        return np.sum(energies), forces
