import numpy as np
from typing import Tuple
from units import Units
from .basic_force import Force


class ExternalUniformField(Force):
    def __init__(self, gradient: np.ndarray, units: Units = None):
        super().__init__(units)
        self.g = np.array(gradient, dtype=np.float64)

    def __call__(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_arr = np.asarray(pos, dtype=np.float64)
        if pos_arr.ndim == 1:
            # Single position vector
            energy = np.array([float(np.dot(self.g, pos_arr))])
            forces = -self.g
            return energy, forces
        # Multiple positions (N,3)
        energies = pos_arr @ self.g
        forces = -np.tile(self.g, (pos_arr.shape[0], 1))
        return np.sum(energies), forces

    def get_energy(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
    ) -> np.ndarray:
        return np.array([float(np.dot(self.g, pos))])

    def get_forces(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
    ) -> np.ndarray:
        return -self.g  # Force is negative gradient of potential
