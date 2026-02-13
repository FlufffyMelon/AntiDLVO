"""
Ewald summation handler for electrostatic interactions in periodic systems.
This module contains the EwaldHandler class that manages reciprocal-space calculations
and cached data for Ewald summation.
"""

import numpy as np
from typing import Optional

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - GPU optional
    cp = None
    CUPY_AVAILABLE = False

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
        use_gpu: bool = False,
        gpu_device: int = 0,
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
            use_gpu: Enable CuPy acceleration (requires CuPy)
            gpu_device: GPU device ID to use (0, 1, 2, etc.)
        """
        self.alpha = alpha
        self.real_cut = real_cut
        self.n_c = n_c
        self.dielectric = dielectric
        self.z_scale_factor = z_scale_factor
        self.dipole_correction = dipole_correction
        self.units = units
        self.use_gpu = bool(use_gpu and CUPY_AVAILABLE)
        if use_gpu and not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not installed but GPU acceleration was requested."
            )

        # Set GPU device if GPU is enabled
        if self.use_gpu:
            if gpu_device < 0:
                raise ValueError(
                    f"GPU device ID must be non-negative, got {gpu_device}"
                )
            if gpu_device >= cp.cuda.runtime.getDeviceCount():
                raise ValueError(
                    f"GPU device {gpu_device} is not available. "
                    f"Only {cp.cuda.runtime.getDeviceCount()} device(s) available."
                )
            cp.cuda.Device(gpu_device).use()
            self.gpu_device = gpu_device
        else:
            self.gpu_device = None

        self.xp = cp if self.use_gpu else np

        # Cached data for k-space calculations
        self.kvecs: Optional[np.ndarray] = None  # (K, 3)
        self.k_sq: Optional[np.ndarray] = None  # (K,)
        self.Ak: Optional[np.ndarray] = None  # (K,)
        self.Sc: Optional[np.ndarray] = None  # (K,)
        self.Ss: Optional[np.ndarray] = None  # (K,)
        self.Ak_kvecs: Optional[np.ndarray] = None  # (K, 3)

        # Reduced Bjerrum length (for LJ units)
        self.lB_star: Optional[float] = None

        # Dipole correction for slab geometry
        self.dipole_Mz: float = 0.0  # Total z-component of dipole moment
        self.dipole_Q_Gz: float = 0.0  # Total z-component of dipole moment squared

        self.update_bjerrum_length(1.0)  # Default temperature

    def _to_backend(self, array, dtype=None):
        if array is None:
            return None
        xp = self.xp
        return xp.asarray(array, dtype=dtype)

    def _to_cpu(self, array):
        if array is None or not self.use_gpu:
            return array
        return cp.asnumpy(array)

    def _as_index_array(self, indices: np.ndarray):
        if not self.use_gpu:
            return indices
        return cp.asarray(indices, dtype=cp.int64)

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

        kvecs_host = np.ascontiguousarray(kvecs)
        k_sq_host = np.ascontiguousarray(k_sq)
        Ak_host = np.ascontiguousarray(Ak)

        self.kvecs = self._to_backend(kvecs_host)
        self.k_sq = self._to_backend(k_sq_host)
        self.Ak = self._to_backend(Ak_host)
        self.Ak_kvecs = self._to_backend(
            np.ascontiguousarray(Ak_host[:, np.newaxis] * kvecs_host)
        )

        # Initialize structure factors to zeros
        xp = self.xp
        self.Sc = xp.zeros((len(self.Ak), 0), dtype=self.Ak.dtype)
        self.Ss = xp.zeros((len(self.Ak), 0), dtype=self.Ak.dtype)

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

        xp = self.xp
        if len(positions) == 0:
            self.Sc = xp.zeros((len(self.Ak), 0), dtype=self.Ak.dtype)
            self.Ss = xp.zeros((len(self.Ak), 0), dtype=self.Ak.dtype)
            return

        positions_backend = self._to_backend(positions)
        charges_backend = self._to_backend(charges)

        # Dot products k·r_i for all k and all i: (K,3) @ (3,N) = (K,N)
        kr = self.kvecs @ positions_backend.T  # (K, N)
        cos_kr = xp.cos(kr)  # (K, N)
        sin_kr = xp.sin(kr)  # (K, N)
        q = charges_backend[xp.newaxis, :]  # (1, N)

        # S_c(k) = sum_i q_i * cos(k·r_i)
        # S_s(k) = sum_i q_i * sin(k·r_i)
        self.Sc = q * cos_kr  # (K, N)
        self.Ss = q * sin_kr  # (K, N)

    # def update_dipole_moment(self, positions: np.ndarray, charges: np.ndarray) -> None:
    #     """
    #     Update the dipole moment z-component.

    #     Args:
    #         positions: Particle positions (N, 3)
    #         charges: Particle charges (N,)
    #     """
    #     self.dipole_Mz = np.sum(charges * positions[:, 2])
    #     self.dipole_Q_Gz = np.sum(charges) * np.sum(charges * positions[:, 2] ** 2)

    def update_structure_factors_from_delta(
        self, indices: np.ndarray, delta_S_c: np.ndarray, delta_S_s: np.ndarray
    ) -> None:
        """
        Update structure factors S_c and S_s for current positions and charges.
        """
        idx = self._as_index_array(np.asarray(indices, dtype=int))
        self.Sc[:, idx] += self._to_backend(delta_S_c)
        self.Ss[:, idx] += self._to_backend(delta_S_s)

    def compute_total_kspace_energy_forces(self) -> float:
        """
        Compute the total k-space energy for the entire system.

        U_k = (2π/V) ∑_{k≠0} [exp(-k²/(4α²))/k²] * |S(k)|²

        F_k = -(4π/V) ∑_{k≠0} [exp(-k²/(4α²))/k²] * q_i * [sin(k·r_i) * S_c - cos(k·r_i) * S_s]

        Returns:
            Total k-space energy
        """
        if self.Ak is None or self.Sc is None or self.Ss is None:
            return 0.0

        xp = self.xp
        # |S(k)|² = S_c(k)² + S_s(k)²
        Sc_sum = xp.sum(self.Sc, axis=1)  # (K,)
        Ss_sum = xp.sum(self.Ss, axis=1)  # (K,)
        energy = xp.sum(self.Ak * (Sc_sum**2 + Ss_sum**2))

        force_struct = Sc_sum[:, xp.newaxis] * self.Ss - Ss_sum[:, xp.newaxis] * self.Sc
        forces = 2 * force_struct.T @ self.Ak_kvecs  # (N, 3)

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star
            forces *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return float(self._to_cpu(energy)), self._to_cpu(forces)

    # def compute_total_self_energy(self, charges: np.ndarray) -> float:
    #     """
    #     Compute the total self-energy correction for the entire system.

    #     U_self = -√(α/π) ∑_i q_i²

    #     Args:
    #         charges: Particle charges (N,)

    #     Returns:
    #         Total self-energy correction
    #     """
    #     if self.alpha is None:
    #         return 0.0

    #     if len(charges) == 0:
    #         return 0.0

    #     q_squared_sum = np.sum(charges**2)
    #     energy = -np.sqrt(self.alpha / np.pi) * q_squared_sum

    #     # Scale by reduced Bjerrum length for LJ units
    #     if self.lB_star is not None:
    #         energy *= self.lB_star
    #     else:
    #         raise ValueError("Reduced Bjerrum length is not set")

    #     return energy

    def compute_dipole_correction_energy_forces(
        self, positions: np.ndarray, charges: np.ndarray, volume: float
    ) -> (float, np.ndarray):
        """
        Compute the dipole correction energy for slab geometry.

        U_d = 2π/V * M_z^2, where M_z = sum_i q_i * z_i
        U_sq = 2π/V * Q_t * G_z^2, where Q_t = sum_i q_i and G_z = sum_i q_i * z_i^2

        U_c = U_d - U_sq

        Returns:
            Dipole correction energy
        """
        forces = np.zeros_like(positions)
        if not self.dipole_correction:
            return 0.0, forces

        # Apply z_scale_factor to volume for dipole correction
        scaled_volume = volume * self.z_scale_factor

        dipole_Mz = np.sum(charges * positions[:, 2])
        dipole_Q_Gz = np.sum(charges) * np.sum(charges * positions[:, 2] ** 2)

        # Calculate dipole correction
        energy = 2.0 * np.pi * (dipole_Mz**2 - dipole_Q_Gz) / scaled_volume
        forces[:, 2] = (
            -4.0
            * np.pi
            * (dipole_Mz * charges - np.sum(charges) * charges * positions[:, 2])
            / scaled_volume
        )

        # Scale by reduced Bjerrum length for LJ units
        if self.lB_star is not None:
            energy *= self.lB_star
            forces *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return energy, forces

    # def delta_kspace_energy(
    #     self,
    #     indices: np.ndarray = None,
    #     positions: np.ndarray = None,
    #     new_positions: np.ndarray = None,
    #     # old_positions: np.ndarray = None,
    #     charges: np.ndarray = None,
    # ):
    #     """
    #     Compute the change in k-space energy due to translation, addition, or deletion of a M particles.

    #     ΔS_c(k) = q_i [cos(k·r_new) - cos(k·r_old)]
    #     ΔS_s(k) = q_i [sin(k·r_new) - sin(k·r_old)]

    #     ΔU_k = (2π / V) Σ_{k≠0} A(k) [ 2( S_c ΔS_c + S_s ΔS_s ) + (ΔS_c^2 + ΔS_s^2) ].

    #     Args:
    #         new_position: New particle position (M, 3)
    #         old_position: Old particle position (M, 3)
    #         charge: Particle charge (M,)
    #     """
    #     if charges is None:
    #         raise ValueError("Charge must be provided")

    #     # if new_positions is None and old_positions is None:
    #     #     raise ValueError("Either new_position or old_position must be provided")

    #     additional_indices = np.setdiff1d(np.arange(len(positions)), indices)

    #     new_c, new_s, old_c, old_s = 0, 0, 0, 0

    #     old_positions = positions[indices].copy()

    #     if new_positions is not None:
    #         new_c = np.cos(self.kvecs @ new_positions.T)
    #         new_s = np.sin(self.kvecs @ new_positions.T)
    #     if old_positions is not None:
    #         old_c = np.cos(self.kvecs @ old_positions.T)
    #         old_s = np.sin(self.kvecs @ old_positions.T)

    #     delta_S_c = charges * (new_c - old_c)
    #     delta_S_s = charges * (new_s - old_s)

    #     # |S(k)|² = S_c(k)² + S_s(k)²
    #     Sc_sum = np.sum(self.Sc, axis=1)  # (K,)
    #     Ss_sum = np.sum(self.Ss, axis=1)  # (K,)

    #     dot_product = Sc_sum * np.sum(delta_S_c, axis=1) + Ss_sum * np.sum(
    #         delta_S_s, axis=1
    #     )

    #     delta_S_squared = (
    #         np.sum(delta_S_c, axis=1) ** 2 + np.sum(delta_S_s, axis=1) ** 2
    #     )

    #     delta_energy = np.sum(self.Ak * (2 * dot_product + delta_S_squared))

    #     # ----------------------------------------------------------------

    #     delta_forces = np.zeros_like(positions)

    #     # Changes in forces of old new atoms
    #     force_coef = self.Ak * ((self.Sc + delta_S_c) * -Sc_sum * self.Ss)  # (K, N)

    #     delta_forces = -2 * np.sum(
    #         force_coef[:, :, np.newaxis] * self.kvecs[np.newaxis, :, :], axis=0
    #     )  # (N, 3)

    #     # Changes in forces of all other atoms due to the structure factor changing
    #     force_coef = self.Ak * (Ss_sum * delta_S_c - Sc_sum * delta_S_s)  # (K, N)
    #     forces = 2 * np.sum(
    #         force_coef[:, :, np.newaxis] * self.kvecs[np.newaxis, :, :], axis=0
    #     )  # (N, 3)
    #     # ------------------

    #     # ----------------
    #     # |S(k)|² = S_c(k)² + S_s(k)²
    #     Sc_sum = np.sum(self.Sc, axis=1)  # (K,)
    #     Ss_sum = np.sum(self.Ss, axis=1)  # (K,)
    #     energy = np.sum(self.Ak * (Sc_sum**2 + Ss_sum**2))

    #     force_coef = self.Ak * (Ss_sum * self.Sc - Sc_sum * self.Ss)  # (K, N)
    #     forces = -2 * np.sum(
    #         force_coef[:, :, np.newaxis] * self.kvecs[np.newaxis, :, :], axis=0
    #     )  # (N, 3)
    #     # ------------------

    #     if self.lB_star is not None:
    #         delta_energy *= self.lB_star
    #     else:
    #         raise ValueError("Reduced Bjerrum length is not set")

    #     return delta_energy, delta_S_c, delta_S_s

    def delta_kspace_energy_forces(
        self,
        indices: np.ndarray = None,
        positions: np.ndarray = None,
        new_positions: np.ndarray = None,
        # old_positions: np.ndarray = None,
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

        xp = self.xp
        positions_np = np.asarray(positions)
        indices_np = np.asarray(indices, dtype=int)
        charges_backend = self._to_backend(charges)

        new_c, new_s, old_c, old_s = 0, 0, 0, 0

        old_positions = positions_np[indices_np].copy()

        if new_positions is not None:
            new_positions_backend = self._to_backend(new_positions)
            new_c = xp.cos(self.kvecs @ new_positions_backend.T)
            new_s = xp.sin(self.kvecs @ new_positions_backend.T)
        if old_positions is not None:
            old_positions_backend = self._to_backend(old_positions)
            old_c = xp.cos(self.kvecs @ old_positions_backend.T)
            old_s = xp.sin(self.kvecs @ old_positions_backend.T)

        delta_S_c = charges_backend * (new_c - old_c)
        delta_S_s = charges_backend * (new_s - old_s)

        # |S(k)|² = S_c(k)² + S_s(k)²
        Sc_sum = xp.sum(self.Sc, axis=1)  # (K,)
        Ss_sum = xp.sum(self.Ss, axis=1)  # (K,)

        dot_product = Sc_sum * xp.sum(delta_S_c, axis=1) + Ss_sum * xp.sum(
            delta_S_s, axis=1
        )

        delta_S_squared = (
            xp.sum(delta_S_c, axis=1) ** 2 + xp.sum(delta_S_s, axis=1) ** 2
        )

        delta_energy = xp.sum(self.Ak * (2 * dot_product + delta_S_squared))

        # ----------------------------------------------------------------

        delta_Sc_sum = xp.sum(delta_S_c, axis=1)
        delta_Ss_sum = xp.sum(delta_S_s, axis=1)

        Sc_sum_new = Sc_sum + delta_Sc_sum
        Ss_sum_new = Ss_sum + delta_Ss_sum

        positions_backend = self._to_backend(positions_np)
        delta_forces = xp.zeros_like(positions_backend)

        total_particles = positions_np.shape[0]
        all_indices = np.arange(total_particles)
        mask = np.ones(total_particles, dtype=bool)
        mask[indices_np] = False
        unaffected = all_indices[mask]

        scaled_delta_sc = delta_Sc_sum[:, xp.newaxis] * self.Ak_kvecs
        scaled_delta_ss = delta_Ss_sum[:, xp.newaxis] * self.Ak_kvecs

        if len(unaffected) > 0:
            unaffected_idx = self._as_index_array(unaffected)
            Ss_unaff_T = xp.transpose(self.Ss[:, unaffected_idx])
            Sc_unaff_T = xp.transpose(self.Sc[:, unaffected_idx])
            delta_forces[unaffected_idx] = 2.0 * (
                Ss_unaff_T @ scaled_delta_sc - Sc_unaff_T @ scaled_delta_ss
            )

        if len(indices_np) > 0:
            indices_backend = self._as_index_array(indices_np)
            Sc_old = self.Sc[:, indices_backend]
            Ss_old = self.Ss[:, indices_backend]
            Sc_new = Sc_old + delta_S_c
            Ss_new = Ss_old + delta_S_s
            new_mix = (
                Sc_sum_new[:, xp.newaxis] * Ss_new - Ss_sum_new[:, xp.newaxis] * Sc_new
            )
            old_mix = Sc_sum[:, xp.newaxis] * Ss_old - Ss_sum[:, xp.newaxis] * Sc_old
            delta_forces[indices_backend] = 2.0 * (new_mix - old_mix).T @ self.Ak_kvecs

        if self.lB_star is not None:
            delta_energy *= self.lB_star
            delta_forces *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return (
            float(self._to_cpu(delta_energy)),
            self._to_cpu(delta_forces),
            self._to_cpu(delta_S_c),
            self._to_cpu(delta_S_s),
        )

    # def delta_self_energy(self, charge: float) -> float:
    #     """
    #     Compute the change in self-energy correction due to translation, addition, or deletion of a particle.
    #     """
    #     if self.alpha is None:
    #         return 0.0

    #     delta_self_energy = -np.sqrt(self.alpha / np.pi) * charge**2

    #     if self.lB_star is not None:
    #         delta_self_energy *= self.lB_star
    #     else:
    #         raise ValueError("Reduced Bjerrum length is not set")

    #     return delta_self_energy

    # def delta_dipole_correction(
    #     self,
    #     positions: np.ndarray = None,
    #     charges_full: np.ndarray = None,
    #     new_positions: np.ndarray = None,
    #     old_positions: np.ndarray = None,
    #     charges: np.ndarray = None,
    #     volume: float = None,
    # ) -> float:
    #     """
    #     M_new = M_old - q * z_old + q * z_new
    #     ΔM_z^2 = M_new^2 - M_old^2 = (2 * M_old + q * z_new - q * z_old) * (q * z_new - q * z_old)
    #     ΔG_z = - Q * q * (z_new^2 - z_old^2)
    #     ΔU_c = -2π/V * ΔM_z^2

    #     Args:
    #         new_position: New particle position (M, 3)
    #         old_position: Old particle position (M, 3)
    #         charge: Particle charge (M,)
    #         volume: System volume

    #     Returns:
    #         Change in dipole correction energy
    #     """
    #     if not self.dipole_correction:
    #         return 0.0

    #     # Apply z_scale_factor to volume for dipole correction
    #     scaled_volume = volume * self.z_scale_factor

    #     dipole_Mz = np.sum(charges_full * positions[:, 2])
    #     total_charge = np.sum(charges_full)
    #     # dipole_Q_Gz = np.sum(charges_full) * np.sum(charges_full * positions[:, 2] ** 2)

    #     if new_positions is not None and old_positions is not None:
    #         delta_Mz_squared = (
    #             2 * dipole_Mz
    #             + np.sum(charges * (new_positions[:, 2] - old_positions[:, 2]))
    #         ) * np.sum(charges * (new_positions[:, 2] - old_positions[:, 2]))

    #         delta_Gz = np.sum(
    #             charges * (new_positions[:, 2] ** 2 - old_positions[:, 2] ** 2)
    #         )
    #     elif new_positions is not None:
    #         delta_Mz_squared = (
    #             2 * dipole_Mz + np.sum(charges * new_positions[:, 2])
    #         ) * np.sum(charges * new_positions[:, 2])

    #         delta_Gz = np.sum(charges * new_positions[:, 2] ** 2)
    #     elif old_positions is not None:
    #         delta_Mz_squared = -(
    #             2 * dipole_Mz - np.sum(charges * old_positions[:, 2])
    #         ) * np.sum(charges * old_positions[:, 2])

    #         delta_Gz = -np.sum(charges * old_positions[:, 2] ** 2)
    #     else:
    #         raise ValueError("Either new_positions or old_positions must be provided")

    #     delta_dipole_energy = (
    #         2.0 * np.pi * (delta_Mz_squared - total_charge * delta_Gz) / scaled_volume
    #     )

    #     if self.lB_star is not None:
    #         delta_dipole_energy *= self.lB_star
    #     else:
    #         raise ValueError("Reduced Bjerrum length is not set")

    #     return delta_dipole_energy

    def delta_dipole_correction_energy_forces(
        self,
        positions: np.ndarray = None,
        charges: np.ndarray = None,
        indices: np.ndarray = None,
        new_positions: np.ndarray = None,
        volume: float = None,
    ) -> float:
        """

        Args:
            new_position: New particle position (M, 3)
            old_position: Old particle position (M, 3)
            charge: Particle charge (M,)
            volume: System volume

        Returns:
            Change in dipole correction energy
        """
        if not self.dipole_correction:
            return 0.0, np.zeros_like(positions)

        old_energy, old_forces = self.compute_dipole_correction_energy_forces(
            positions, charges, volume
        )

        updated_positions = positions.copy()
        updated_positions[indices] = new_positions

        new_energy, new_forces = self.compute_dipole_correction_energy_forces(
            positions, charges, volume
        )

        return new_energy - old_energy, new_forces - old_forces

    # def compute_pressure_virial(self, volume: float) -> float:
    #     """
    #     Compute the pressure due to virial interactions in the Ewald sum.

    #     Args:
    #         volume: System volume
    #     """
    #     if self.kvecs is None or self.Ak is None or self.Sc is None or self.Ss is None:
    #         return 0.0

    #     if volume <= 0.0:
    #         return 0.0

    #     # k_sq = np.sum(self.kvecs * self.kvecs, axis=1)
    #     # |S(k)|² = S_c(k)² + S_s(k)²
    #     S_squared = np.sum(self.Sc, axis=1) ** 2 + np.sum(self.Ss, axis=1) ** 2

    #     virial_pressure = (
    #         np.sum(S_squared * self.Ak * (1 / 3 - self.k_sq / (6 * self.alpha)))
    #         / volume
    #     )

    #     return virial_pressure
