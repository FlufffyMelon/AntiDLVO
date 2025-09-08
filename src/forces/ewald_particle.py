import numpy as np
from typing import Tuple
from units import Units
from .basic_force import Force
from scipy.special import erfc


class EwaldReal(Force):
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
        system=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if system is None or not system.ewald_handler:
            N = r_ij.shape[0]
            return np.zeros(N), np.zeros((N, 3))

        # Get Ewald parameters from handler
        alpha = system.ewald_handler.alpha
        r_cut = system.ewald_handler.real_cut

        # Real-space pair term (screened Coulomb)
        energy, forces = self._real_space_part(
            r_ij, float(charge1), float(charge2), float(alpha), float(r_cut)
        )

        # Apply reduced Bjerrum length scaling for LJ units
        if system.ewald_handler is not None:
            energy *= system.ewald_handler.lB_star
            forces *= system.ewald_handler.lB_star

        return energy, forces

    @staticmethod
    def _real_space_part(
        r_ij: np.ndarray,
        qi: float,
        qj: float,
        alpha: float,
        r_cut: float,
    ) -> Tuple[float, np.ndarray]:
        N = r_ij.shape[0]
        energies = np.zeros(N)
        forces = np.zeros((N, 3))

        r = np.linalg.norm(r_ij, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = (r > 1e-12) & (r <= r_cut)
            r_t = r[mask]
            dr = r_ij[mask, :]

            # Energy
            erfc_term = erfc(np.sqrt(alpha) * r_t)
            e = qi * qj * erfc_term / r_t
            energies[mask] = e

            # Force magnitude on j from i (negative gradient wrt r_j)
            # F = - d/dr (qi qj erfc(sqrt(alpha) r)/r) * r_hat
            inv_r = 1.0 / r_t
            inv_r2 = inv_r * inv_r
            exp_term = np.exp(-alpha * r_t**2)
            bracket = (
                erfc_term * inv_r2 + 2.0 * np.sqrt(alpha / np.pi) * exp_term * inv_r
            )

            force_mag = qi * qj * bracket
            forces_vec = force_mag[:, np.newaxis] * dr * inv_r[:, np.newaxis]
            forces[mask] = forces_vec

        return np.sum(energies), forces
