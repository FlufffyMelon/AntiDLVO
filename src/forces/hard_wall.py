import numpy as np
from typing import Tuple
from units import Units
from .basic_force import OneBodyForce


class HardWall(OneBodyForce):
    """
    Hard wall potential acting along z from planar wall.
    U(z) = inf for z < 0, 0 for z > 0.
    """

    def __init__(self, r_wall: float, style: str, units: Units = None):
        super().__init__(units)
        assert style in ["bottom", "top"], f"Unknown wall style: {style}"
        self.r_wall = r_wall
        self.style = style

        self.big_number = float(np.inf)

    def __call__(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos_arr = pos.reshape(-1, 3)

        if system is None:
            H = np.inf
        else:
            H = float(system.box[2])

        if self.style == "bottom":
            z = pos_arr[:, 2]
        elif self.style == "top":
            z = H - pos_arr[:, 2]

        z_eff = np.maximum(z, 1e-12)

        # Energy calculation: U(z)
        energy = np.where(z_eff < self.r_wall, self.big_number, 0.0)

        # Force calculation: Fz = -dU/dz = -U'(z)
        Fz = np.where(z_eff < self.r_wall, self.big_number, 0.0)

        if self.style == "top":
            Fz = -Fz

        forces = np.zeros_like(pos_arr)
        forces[:, 2] = Fz

        return np.sum(energy), forces
