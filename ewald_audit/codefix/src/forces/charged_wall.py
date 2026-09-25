import numpy as np
from typing import Tuple
from ..units import Units
from ..system import System
from .basic_force import OneBodyForce


class ChargedWall(OneBodyForce):
    """ """

    def __init__(
        self,
        sigma_1: float,
        sigma_2: float,
        units: Units = None,
    ):
        super().__init__(units)

        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        self.C1 = None

    def __call__(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float,
        system: System,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energy and force for a charged particle in an electric field.

        Args:
            pos: Particle positions (N, 3)
            atom_type: Type ID of particles
            charge: Charge of particles (in elementary charge units)
            system: System object (for box dimensions)

        Returns:
            Tuple of (energy, forces)
        """
        if not system.ewald_handler:
            raise ValueError("Ewald handler not found in system")

        if self.C1 is None:
            self.C1 = self.calculate_C1(system)

        # Ensure pos is 2D array of shape (N, 3)
        pos_arr = np.asarray(pos).reshape(-1, 3)
        N = pos_arr.shape[0]
        forces = np.zeros((N, 3), dtype=np.float64)

        energy = 2 * np.pi * (self.sigma_2 - self.sigma_1) * pos_arr[:, 2] + self.C1

        # Apply reduced Bjerrum length scaling for proper units
        energy *= system.ewald_handler.lB_star
        forces *= system.ewald_handler.lB_star

        return np.sum(energy), forces

    def calculate_C1(self, system: System) -> float:
        return (
            self.sigma_1 + self.sigma_2
        ) * system.ewald_handler.C1_factor - 2 * np.pi * self.sigma_2 * self.system.box[
            2
        ]
