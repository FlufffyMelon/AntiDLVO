import numpy as np
from typing import Tuple
from ..units import Units
from .basic_force import Force
from scipy.special import erfc

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_MIN_R_SQ = 1e-24  # ~1e-12 nm tolerance squared


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
        if N == 0:
            return 0.0, np.zeros((0, 3), dtype=r_ij.dtype)

        # Compute squared distances first (avoids extra sqrt work for filtered-out entries)
        r_sq = np.einsum("ij,ij->i", r_ij, r_ij)
        r_cut_sq = r_cut * r_cut

        mask = (r_sq > _MIN_R_SQ) & (r_sq <= r_cut_sq)
        if not np.any(mask):
            return 0.0, np.zeros_like(r_ij)

        r_sq_sel = r_sq[mask]
        r_sel = np.sqrt(r_sq_sel)
        inv_r = 1.0 / r_sel
        inv_r2 = 1.0 / r_sq_sel

        sqrt_alpha = np.sqrt(alpha)
        scaled_r = sqrt_alpha * r_sel

        erfc_term = erfc(scaled_r)
        exp_term = np.exp(-alpha * r_sq_sel)

        prefactor = qi * qj
        bracket = erfc_term * inv_r + (2.0 * sqrt_alpha * _INV_SQRT_PI) * exp_term

        forces = np.zeros_like(r_ij)
        forces_sel = (
            prefactor * bracket[:, np.newaxis] * r_ij[mask, :] * inv_r2[:, np.newaxis]
        )
        forces[mask, :] = forces_sel

        energy = prefactor * np.sum(erfc_term * inv_r)
        return float(energy), forces
