import numpy as np
from typing import Tuple
from ..units import Units
from ..system import System
from .basic_force import Force
from scipy.special import erfc


class Ewald(Force):
    """
    Real-space part of Ewald summation for electrostatic interactions.
    This class handles only the real-space pair interactions.
    The reciprocal-space and self-energy terms are managed by EwaldHandler.
    """

    def __init__(self, exclude_intermol: bool = False, units: Units = None):
        super().__init__(units)
        self.exclude_intermol = exclude_intermol

    def __call__(
        self,
        pos: np.ndarray,
        r_ij: np.ndarray,
        charge1: float,
        charge2: float,
        system: System,
        **kwargs,
    ) -> Tuple[float, np.ndarray]:
        if not system.ewald_handler:
            raise ValueError("Ewald handler not found in system")

        N = r_ij.shape[0]

        forces = np.zeros((N, 3), dtype=np.float64)
        energy = charge1 * charge2 * self._nu_e3dtf(r_ij, system)

        # if system.ewald_handler.dipole_correction:
        #     energy -= 2 * np.pi * pos[0, 2] ** 2 / system.get_volume()

        # Apply reduced Bjerrum length scaling for proper units
        energy *= system.ewald_handler.lB_star
        forces *= system.ewald_handler.lB_star

        return energy, forces

    def _nu_e3dtf(
        self,
        r_ij: np.ndarray,
        system: System,
    ) -> float:
        N = r_ij.shape[0]
        E = np.zeros(N, dtype=np.float64)

        # 1. First constant term is `tau_3D`
        E += system.ewald_handler.tau_3D

        # 2. Second term is sum in real space
        alpha = system.ewald_handler.alpha
        n = system.ewald_handler.n_real
        for nx in range(-n, n + 1):
            for ny in range(-n, n + 1):
                for nz in range(-n, n + 1):
                    r_ij_image = r_ij + np.array([nx, ny, nz]) * system.box
                    r_ij_image_norm = np.linalg.norm(r_ij_image)

                    E += erfc(np.sqrt(alpha) * r_ij_image_norm) / r_ij_image_norm

        # 3. Third term is sum in reciprocal space
        k_dot_r_ij = system.ewald_handler.kvecs @ r_ij.T
        E += 2 * np.sum(system.ewald_handler.Ak[:, np.newaxis] * np.cos(k_dot_r_ij))

        # 4. Fourth constant term which is canceling
        E -= np.pi / (alpha * system.get_volume())

        return np.sum(E)
