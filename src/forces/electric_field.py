import numpy as np
from typing import Tuple
from ..units import Units
from .basic_force import OneBodyForce


class ElectricField(OneBodyForce):
    """
    External electric field potential acting on charged particles.

    Energy: U(r) = q * E * r  (dot product)
    Force:  F(r) = q * E      (vector)

    When configured with voltage between walls, field is computed as E = U/D
    and automatically directed along z-axis.
    """

    def __init__(
        self,
        voltage: float = None,  # Voltage between walls (V)
        direction: str = "up",
        units: Units = None,
    ):
        super().__init__(units)

        self.voltage = voltage
        self.direction = direction
        assert direction in ["up", "down"], "Direction must be 'up' or 'down'"

        # Conversion factor from V/nm to kJ/(mol nm)
        # 1 V/nm * 1 e = 96.485 kJ/mol/nm
        # This is because 1 eV = 96.485 kJ/mol
        conversion_factor = 96.485
        self.voltage = self.voltage * conversion_factor

    def __call__(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
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
        # Ensure pos is 2D array of shape (N, 3)
        pos_arr = np.asarray(pos).reshape(-1, 3)

        # Get electric field vector
        if self.direction == "up":
            E = self.voltage / system.box[2]
        elif self.direction == "down":
            E = -self.voltage / system.box[2]
        else:
            raise ValueError(f"Invalid direction: {self.direction}")

        # Energy calculation: U = q * E * z
        # energy is in kJ/mol
        energy = -charge * E * pos_arr[:, 2]

        # Force calculation: F = q * E
        forces = np.zeros((pos_arr.shape[0], 3), dtype=np.float64)
        forces[:, 2] = -charge * E

        return np.sum(energy), forces
