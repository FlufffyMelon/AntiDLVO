"""
Ewald summation handler for electrostatic interactions in periodic systems.
This module contains the EwaldHandler class that manages reciprocal-space calculations
and cached data for Ewald summation.
"""

import numpy as np
from typing import Optional
from .units import Units


class EwaldHandler:
    """
    Handles Ewald summation calculations and caching for electrostatic interactions.

    This class manages the reciprocal-space part of the Ewald summation, including:
    - Storing and updating k-vectors
    - Computing structure factors S(k)
    - Calculating reciprocal-space energies and forces
    - Managing the reduced Bjerrum length for proper LJ unit scaling
    - Calculating dipole correction for slab geometry
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        real_cut: Optional[float] = None,
        n_c: Optional[int] = None,
        dielectric: Optional[float] = None,
        z_scale_factor: float = 1.0,
        dipole_correction: bool = False,
        units: Units = None,
    ):
        """
        Initialize the Ewald handler with parameters.

        Args:
            alpha: Ewald splitting parameter
            real_cut: Real-space cutoff distance
            n_c: Integer parameter for reciprocal space cutoff (k_cut = 2π/(L*n_c))
            dielectric: Relative permittivity of the medium
            z_scale_factor: Scale factor for z dimension to avoid periodic interactions
            dipole_correction: Whether to use dipole correction
            units: Units system
        """
        self.alpha = alpha
        self.real_cut = real_cut
        self.n_c = n_c
        self.dielectric = dielectric
        self.z_scale_factor = z_scale_factor
        self.dipole_correction = dipole_correction
        self.units = units

        # Cached data for k-space calculations
        self.kvecs: Optional[np.ndarray] = None  # (K, 3)
        self.k_sq: Optional[np.ndarray] = None  # (K,)
        self.Ak: Optional[np.ndarray] = None  # (K,)
        self.Sc: Optional[np.ndarray] = None  # (K,)
        self.Ss: Optional[np.ndarray] = None  # (K,)

        # Reduced Bjerrum length (for LJ units)
        self.lB_star: Optional[float] = None

        # Dipole correction for slab geometry
        self.dipole_Mz: float = 0.0  # Total z-component of dipole moment
        self.dipole_Q_Gz: float = 0.0  # Total z-component of dipole moment squared

        self.update_bjerrum_length(1.0)  # Default temperature

    def update_bjerrum_length(self, temperature: float) -> None:
        """
        Update the reduced Bjerrum length based on temperature.

        In LJ units, lB* = 1/(εr * T*) where T* is reduced temperature.

        In standard units, lB* = kc / εr where kc is the Coulomb constant.

        Args:
            temperature: System temperature in appropriate units
        """
        if self.units and getattr(self.units, "system_name", "") == "lj":
            self.lB_star = 1.0 / max(self.dielectric * temperature, 1e-12)
        elif self.units and getattr(self.units, "system_name", "") == "standard":
            # Fallback for non-LJ units
            kc = 138.935456  # kJ nm / mol
            self.lB_star = kc / max(self.dielectric, 1e-12)

        return self.lB_star

    def initialize_k_vectors(self, box: np.ndarray) -> None:
        """
        Initialize reciprocal vectors and Ak coefficients for the current box.

        Args:
            box: Simulation box dimensions [Lx, Ly, Lz]
        """
        if self.n_c is None or self.alpha is None:
            return

        Lx, Ly, Lz = map(float, box)
        # Apply z_scale_factor to Lz for k-vector calculation
        Lz_scaled = Lz * self.z_scale_factor

        b1 = 2.0 * np.pi / Lx
        b2 = 2.0 * np.pi / Ly
        b3 = 2.0 * np.pi / Lz_scaled

        # Build integer grid from -n_c..n_c and exclude zero vector
        rng = np.arange(-self.n_c, self.n_c + 1, dtype=int)
        ms = np.array(np.meshgrid(rng, rng, rng, indexing="ij"))
        ms = ms.reshape(3, -1).T  # (M,3)
        mask_nonzero = ~np.all(ms == 0, axis=1)
        ms = ms[mask_nonzero]

        # k vectors
        kvecs = np.zeros_like(ms, dtype=np.float64)
        kvecs[:, 0] = ms[:, 0] * b1
        kvecs[:, 1] = ms[:, 1] * b2
        kvecs[:, 2] = ms[:, 2] * b3
        k_sq = np.sum(kvecs * kvecs, axis=1)

        # Volume and A(k) coefficients - use scaled volume for k-space
        V = float(Lx * Ly * Lz_scaled)
        with np.errstate(divide="ignore", invalid="ignore"):
            Ak = (
                (2.0 * np.pi / V)
                * np.exp(-k_sq / (4.0 * self.alpha))
                / np.where(k_sq > 0, k_sq, np.inf)
            )

        self.kvecs = kvecs
        self.k_sq = k_sq
        self.Ak = Ak

        # Initialize structure factors to zeros
        self.Sc = np.zeros(len(self.Ak))
        self.Ss = np.zeros(len(self.Ak))

    def update_structure_factors(
        self, positions: np.ndarray, charges: np.ndarray
    ) -> None:
        """
        Update structure factors S_c and S_s for current positions and charges.

        Args:
            positions: Particle positions (N, 3)
            charges: Particle charges (N,)
        """
        if self.kvecs is None or self.Ak is None:
            return

        if len(positions) == 0:
            self.Sc = np.zeros(len(self.Ak))
            self.Ss = np.zeros(len(self.Ak))
            return

        # Dot products k·r_i for all k and all i: (K,3) @ (3,N) = (K,N)
        kr = self.kvecs @ positions.T  # shape (K, N)
        cos_kr = np.cos(kr)
        sin_kr = np.sin(kr)
        q = charges[np.newaxis, :]  # (1, N)

        # S_c(k) = sum_i q_i * cos(k·r_i)
        # S_s(k) = sum_i q_i * sin(k·r_i)
        self.Sc = q * cos_kr
        self.Ss = q * sin_kr

    def update_dipole_moment(self, positions: np.ndarray, charges: np.ndarray) -> None:
        """
        Update the dipole moment z-component.

        Args:
            positions: Particle positions (N, 3)
            charges: Particle charges (N,)
        """
        self.dipole_Mz = np.sum(charges * positions[:, 2])
        self.dipole_Q_Gz = np.sum(charges) * np.sum(charges * positions[:, 2]**2)

    def update_structure_factors_from_delta(
        self, indices: np.ndarray, delta_S_c: np.ndarray, delta_S_s: np.ndarray
    ) -> None:
        """
        Update structure factors S_c and S_s for current positions and charges.
        """
        self.Sc[:, indices] += delta_S_c
        self.Ss[:, indices] += delta_S_s

    def compute_total_kspace_energy(self) -> float:
        """
        Compute the total k-space energy for the entire system.

        U_k = (2π/V) ∑_{k≠0} [exp(-k²/(4α²))/k²] * |S(k)|²

        Returns:
            Total k-space energy
        """
        if self.Ak is None or self.Sc is None or self.Ss is None:
            return 0.0

        # |S(k)|² = S_c(k)² + S_s(k)²
        S_squared = np.sum(self.Sc, axis=1) ** 2 + np.sum(self.Ss, axis=1) ** 2
        energy = np.sum(self.Ak * S_squared)

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return energy

    def compute_total_self_energy(self, charges: np.ndarray) -> float:
        """
        Compute the total self-energy correction for the entire system.

        U_self = -√(α/π) ∑_i q_i²

        Args:
            charges: Particle charges (N,)

        Returns:
            Total self-energy correction
        """
        if self.alpha is None:
            return 0.0

        if len(charges) == 0:
            return 0.0

        q_squared_sum = np.sum(charges**2)
        energy = -np.sqrt(self.alpha / np.pi) * q_squared_sum

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return energy

    def compute_dipole_correction(self, volume: float) -> float:
        """
        Compute the dipole correction energy for slab geometry.

        U_d = 2π/V * M_z^2, where M_z = sum_i q_i * z_i
        U_sq = 2π/V * Q_t * G_z^2, where Q_t = sum_i q_i and G_z = sum_i q_i * z_i^2

        U_c = U_d - U_sq

        Returns:
            Dipole correction energy
        """
        if not self.dipole_correction:
            return 0.0

        if volume <= 0.0:
            return 0.0

        # Apply z_scale_factor to volume for dipole correction
        scaled_volume = volume * self.z_scale_factor

        # Calculate dipole correction
        energy = 2.0 * np.pi * (self.dipole_Mz**2 - self.dipole_Q_Gz) / scaled_volume

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return energy

    def delta_kspace_energy(
        self,
        new_positions: np.ndarray = None,
        old_positions: np.ndarray = None,
        charges: np.ndarray = None,
    ):
        """
        Compute the change in k-space energy due to translation, addition, or deletion of a M particles.

        ΔS_c(k) = q_i [cos(k·r_new) - cos(k·r_old)]
        ΔS_s(k) = q_i [sin(k·r_new) - sin(k·r_old)]

        ΔU_k = (2π / V) Σ_{k≠0} A(k) [ 2( S_c ΔS_c + S_s ΔS_s ) + (ΔS_c^2 + ΔS_s^2) ].

        Args:
            new_position: New particle position (M, 3)
            old_position: Old particle position (M, 3)
            charge: Particle charge (M,)
        """
        if charges is None:
            raise ValueError("Charge must be provided")

        if new_positions is None and old_positions is None:
            raise ValueError("Either new_position or old_position must be provided")

        new_c, new_s, old_c, old_s = 0, 0, 0, 0

        if new_positions is not None:
            new_c = np.cos(self.kvecs @ new_positions.T)
            new_s = np.sin(self.kvecs @ new_positions.T)
        if old_positions is not None:
            old_c = np.cos(self.kvecs @ old_positions.T)
            old_s = np.sin(self.kvecs @ old_positions.T)

        delta_S_c = charges * (new_c - old_c)
        delta_S_s = charges * (new_s - old_s)

        dot_product = np.sum(self.Sc, axis=1) * np.sum(delta_S_c, axis=1) + np.sum(
            self.Ss, axis=1
        ) * np.sum(delta_S_s, axis=1)

        delta_S_squared = (
            np.sum(delta_S_c, axis=1) ** 2 + np.sum(delta_S_s, axis=1) ** 2
        )

        delta_energy = np.sum(self.Ak * (2 * dot_product + delta_S_squared))

        if self.lB_star is not None:
            delta_energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return delta_energy, delta_S_c, delta_S_s

    def delta_self_energy(self, charge: float) -> float:
        """
        Compute the change in self-energy correction due to translation, addition, or deletion of a particle.
        """
        if self.alpha is None:
            return 0.0

        delta_self_energy = -np.sqrt(self.alpha / np.pi) * charge**2

        if self.lB_star is not None:
            delta_self_energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return delta_self_energy

    def delta_dipole_correction(
        self,
        new_positions: np.ndarray = None,
        old_positions: np.ndarray = None,
        charges: np.ndarray = None,
        volume: float = None,
        total_charge: float = None,
    ) -> float:
        """
        M_new = M_old - q * z_old + q * z_new
        ΔM_z^2 = M_new^2 - M_old^2 = (2 * M_old + q * z_new - q * z_old) * (q * z_new - q * z_old)
        ΔU_c = -2π/V * ΔM_z^2

        Args:
            new_position: New particle position (M, 3)
            old_position: Old particle position (M, 3)
            charge: Particle charge (M,)
            volume: System volume

        Returns:
            Change in dipole correction energy
        """
        if not self.dipole_correction:
            return 0.0

        # Apply z_scale_factor to volume for dipole correction
        scaled_volume = volume * self.z_scale_factor

        if new_positions is not None and old_positions is not None:
            delta_Mz_squared = (
                2 * self.dipole_Mz
                + np.sum(charges * (new_positions[:, 2] - old_positions[:, 2]))
            ) * np.sum(charges * (new_positions[:, 2] - old_positions[:, 2]))

            delta_Gz = np.sum(charges * (new_positions[:, 2] ** 2 - old_positions[:, 2] ** 2))
        elif new_positions is not None:
            delta_Mz_squared = (
                2 * self.dipole_Mz + np.sum(charges * new_positions[:, 2])
            ) * np.sum(charges * new_positions[:, 2])

            delta_Gz = np.sum(charges * new_positions[:, 2] ** 2)
        elif old_positions is not None:
            delta_Mz_squared = -(
                2 * self.dipole_Mz - np.sum(charges * old_positions[:, 2])
            ) * np.sum(charges * old_positions[:, 2])

            delta_Gz = -np.sum(charges * old_positions[:, 2] ** 2)
        else:
            raise ValueError("Either new_positions or old_positions must be provided")

        delta_dipole_energy = 2.0 * np.pi * (delta_Mz_squared - total_charge * delta_Gz) / scaled_volume

        if self.lB_star is not None:
            delta_dipole_energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return delta_dipole_energy

    def compute_pressure_virial(self, volume: float) -> float:
        """
        Compute the pressure due to virial interactions in the Ewald sum.

        Args:
            volume: System volume
        """
        if self.kvecs is None or self.Ak is None or self.Sc is None or self.Ss is None:
            return 0.0

        if volume <= 0.0:
            return 0.0

        # k_sq = np.sum(self.kvecs * self.kvecs, axis=1)
        # |S(k)|² = S_c(k)² + S_s(k)²
        S_squared = np.sum(self.Sc, axis=1) ** 2 + np.sum(self.Ss, axis=1) ** 2

        virial_pressure = (
            np.sum(S_squared * self.Ak * (1 / 3 - self.k_sq / (6 * self.alpha)))
            / volume
        )

        return virial_pressure
