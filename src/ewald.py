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
        dipole_correction: bool = True,
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
        self.dipole_z: float = 0.0  # Total z-component of dipole moment

        self.update_bjerrum_length(1.0)  # Default temperature

    def update_bjerrum_length(self, temperature: float) -> None:
        """
        Update the reduced Bjerrum length based on temperature.

        In LJ units, lB* = 1/(εr * T*) where T* is reduced temperature.

        Args:
            temperature: System temperature in appropriate units
        """
        if self.units and getattr(self.units, "system_name", "") == "lj":
            self.lB_star = 1.0 / max(self.dielectric * temperature, 1e-12)
        else:
            # Fallback for non-LJ units
            self.lB_star = 1.0

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
                * np.exp(-k_sq / (4.0 * (self.alpha**2)))
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
        Sc = np.sum(q * cos_kr, axis=1)
        Ss = np.sum(q * sin_kr, axis=1)

        self.Sc = Sc
        self.Ss = Ss

    def update_dipole_moment(self, positions: np.ndarray, charges: np.ndarray) -> None:
        """
        Update the dipole moment z-component.

        Args:
            positions: Particle positions (N, 3)
            charges: Particle charges (N,)
        """
        self.dipole_z = np.sum(charges * positions[:, 2])

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
        S_squared = self.Sc**2 + self.Ss**2
        energy = 0.5 * np.sum(self.Ak * S_squared)

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star

        return energy

    def compute_total_self_energy(self, charges: np.ndarray) -> float:
        """
        Compute the total self-energy correction for the entire system.

        U_self = -(α/√π) ∑_i q_i²

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
        energy = -(self.alpha / np.sqrt(np.pi)) * q_squared_sum

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star

        return energy

    def compute_dipole_correction(self, volume: float) -> float:
        """
        Compute the dipole correction energy for slab geometry.

        U_c = -2π/V * M_z^2, where M_z = sum_i q_i * z_i

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
        energy = -2.0 * np.pi * self.dipole_z**2 / scaled_volume

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star

        return energy

    def delta_kspace_energy(
        self,
        new_position: np.ndarray = None,
        old_position: np.ndarray = None,
        charge: float = None,
    ) -> float:
        """
        Compute the change in k-space energy due to translation, addition, or deletion of a particle.

        ΔS_c(k) = q_i [cos(k·r_new) - cos(k·r_old)]
        ΔS_s(k) = q_i [sin(k·r_new) - sin(k·r_old)]

        ΔU_k = (2π / V) Σ_{k≠0} A(k) [ 2( S_c ΔS_c + S_s ΔS_s ) + (ΔS_c^2 + ΔS_s^2) ].

        Args:
            new_position: New particle position (3,)
            old_position: Old particle position (3,)
            charge: Particle charge
        """
        if charge is None:
            raise ValueError("Charge must be provided")

        if new_position is None and old_position is None:
            raise ValueError("Either new_position or old_position must be provided")

        new_c, new_s, old_c, old_s = 0, 0, 0, 0
        # Check if we have valid k-vectors
        # if self.kvecs is None or len(self.kvecs) == 0:
        #     return 0.0

        # # Initialize with zeros of the correct shape
        # new_c, new_s = np.zeros_like(self.Sc), np.zeros_like(self.Ss)
        # old_c, old_s = np.zeros_like(self.Sc), np.zeros_like(self.Ss)

        if new_position is not None:
            new_c = np.cos(self.kvecs @ new_position)
            new_s = np.sin(self.kvecs @ new_position)
        if old_position is not None:
            old_c = np.cos(self.kvecs @ old_position)
            old_s = np.sin(self.kvecs @ old_position)

        delta_S_c = charge * (new_c - old_c)
        delta_S_s = charge * (new_s - old_s)

        # For reciprocal space energy, we need to account for the 0.5 factor
        # The change in energy is:
        # ΔU_k = 0.5 * q * (phi_new - phi_old)
        # Where phi = (4π/V) * sum_k [A(k) * (S_c*cos(k·r) - S_s*sin(k·r))]

        # For translation: phi_new - phi_old involves only the change in cos/sin terms
        # For insertion/deletion: we need to consider the full contribution

        dot_product = self.Sc * delta_S_c + self.Ss * delta_S_s
        delta_S_squared = delta_S_c**2 + delta_S_s**2

        # The factor of 0.5 is because U_recip(i) = 0.5 * q_i * phi(r_i)
        delta_energy = 0.5 * np.sum(self.Ak * (2 * dot_product + delta_S_squared))

        if self.lB_star is not None:
            delta_energy *= self.lB_star

        return delta_energy

    def delta_self_energy(self, charge: float) -> float:
        """
        Compute the change in self-energy correction due to translation, addition, or deletion of a particle.
        """
        if self.alpha is None:
            return 0.0

        delta_self_energy = -(self.alpha / np.sqrt(np.pi)) * charge**2

        if self.lB_star is not None:
            delta_self_energy *= self.lB_star

        return delta_self_energy

    # def delta_dipole_correction(
    #     self,
    #     new_positions: np.ndarray = None,
    #     old_positions: np.ndarray = None,
    #     charges: np.ndarray = None,
    #     volume: float = None,
    # ) -> float:
    #     # Calculate change in dipole correction
    #     # energy = -2.0 * np.pi * (self.dipole_z**2) / volume
    #     new_dipole_z, old_dipole_z = 0.0, 0.0

    #     if new_positions is not None:
    #         new_dipole_z = np.sum(charges * new_positions[:, 2])
    #     if old_positions is not None:
    #         old_dipole_z = np.sum(charges * old_positions[:, 2])

    #     delta_dipole_z_squared = new_dipole_z**2 - old_dipole_z**2
    #     delta_dipole_energy = -2.0 * np.pi * delta_dipole_z_squared / volume

    #     if self.lB_star is not None:
    #         delta_dipole_energy *= self.lB_star

    #     return delta_dipole_energy

    def delta_dipole_correction(
        self,
        new_position: np.ndarray = None,
        old_position: np.ndarray = None,
        charge: float = None,
        volume: float = None,
    ) -> float:
        """
        M_new = M_old - q * z_old + q * z_new
        ΔM_z^2 = M_new^2 - M_old^2 = (2 * M_old + q * z_new - q * z_old) * (q * z_new - q * z_old)
        ΔU_c = -2π/V * ΔM_z^2

        Args:
            new_position: New particle position (3,)
            old_position: Old particle position (3,)
            charge: Particle charge
            volume: System volume

        Returns:
            Change in dipole correction energy
        """
        # Apply z_scale_factor to volume for dipole correction
        scaled_volume = volume * self.z_scale_factor

        if new_position is not None and old_position is not None:
            delta_dipole_z_squared = (
                (2 * self.dipole_z + charge * (new_position[2] - old_position[2]))
                * charge
                * (new_position[2] - old_position[2])
            )
        elif new_position is not None:
            delta_dipole_z_squared = (
                (2 * self.dipole_z + charge * new_position[2])
                * charge
                * new_position[2]
            )
        elif old_position is not None:
            delta_dipole_z_squared = (
                -(2 * self.dipole_z - charge * old_position[2])
                * charge
                * old_position[2]
            )

        delta_dipole_energy = -2.0 * np.pi * delta_dipole_z_squared / scaled_volume

        if self.lB_star is not None:
            delta_dipole_energy *= self.lB_star

        return delta_dipole_energy
