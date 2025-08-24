import numpy as np
from typing import Tuple
from units import Units
from .basic_force import OneBodyForce


class ExternalWallPotential(OneBodyForce):
    """
    External wall potential acting along z from planar wall.

    U(z) = 2*pi*[ 2/(5*z^10) - 1/z^4 - 1/(3*Delta* (z + 0.61*Delta)^3 ) ]
    with Delta = 1/sqrt(2).
    """

    def __init__(self, style: str, units: Units = None):
        super().__init__(units)
        assert style in ["bottom", "top"], f"Unknown wall style: {style}"
        self.style = style
        self._two_pi = 2.0 * np.pi
        self._delta = 1.0 / np.sqrt(2.0)
        self._a = 0.61 * self._delta

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
        term1_z = (2.0 / 5.0) * (z_eff**-10)
        term2_z = -(z_eff**-4)
        term3_z = -1.0 / (3.0 * self._delta * ((z_eff + self._a) ** 3))
        energy = self._two_pi * (term1_z + term2_z + term3_z)

        # Force calculation: Fz = -dU/dz = -U'(z)
        d1_z = -4.0 * (z_eff**-11)
        d2_z = 4.0 * (z_eff**-5)
        d3_z = (1.0 / self._delta) * ((z_eff + self._a) ** -4)
        Fz = -self._two_pi * (d1_z + d2_z + d3_z)

        if self.style == "top":
            Fz = -Fz

        forces = np.zeros((pos_arr.shape[0], 3), dtype=np.float64)
        forces[:, 2] = Fz

        return np.sum(energy), forces
