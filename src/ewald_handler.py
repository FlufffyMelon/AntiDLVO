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


def _get_gpu_memory_usage_mb():
    """Get current GPU memory usage in MB. Returns (used, total_allocated)."""
    if not CUPY_AVAILABLE:
        return None, None
    try:
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        total_bytes = mempool.total_bytes()
        used_mb = used_bytes / (1024 * 1024)  # Convert to MB
        total_mb = total_bytes / (1024 * 1024)  # Convert to MB
        return used_mb, total_mb
    except:
        return None, None


def _print_memory(label: str):
    """Print GPU memory usage with a label."""
    used_mb, total_mb = _get_gpu_memory_usage_mb()
    if used_mb is not None and total_mb is not None:
        print(f"[GPU_MEMORY] {label}: used={used_mb:.2f} MB, total_allocated={total_mb:.2f} MB")
    else:
        # Fallback to CPU memory info if GPU not available
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)
            print(f"[CPU_MEMORY] {label}: {mem_mb:.2f} MB")
        except:
            pass


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
        self.Sc: Optional[np.ndarray] = None  # (K, capacity) - structure factor cosine
        self.Ss: Optional[np.ndarray] = None  # (K, capacity) - structure factor sine
        self.Ak_kvecs: Optional[np.ndarray] = None  # (K, 3)

        # Internal capacity tracking (synchronized with System.capacity)
        self.capacity: int = 0

        # Track memory cleanup frequency to avoid calling free_all_blocks too often
        self._memory_cleanup_counter: int = 0

        # Track memory usage to decide when to free blocks
        self._last_total_allocated: float = 0.0

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
        # Check if array is already on the correct backend
        if self.use_gpu and isinstance(array, cp.ndarray):
            # Already on GPU, only convert dtype if needed
            if dtype is None or array.dtype == dtype:
                return array
            return array.astype(dtype)
        elif not self.use_gpu and isinstance(array, np.ndarray):
            # Already on CPU, only convert dtype if needed
            if dtype is None or array.dtype == dtype:
                return array
            return array.astype(dtype)
        # Convert to appropriate backend
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

        # Structure factors will be initialized when system capacity is known
        # They will be resized to match system.capacity when needed

    def ensure_capacity(self, capacity: int) -> None:
        """
        Ensure structure factor arrays have sufficient capacity.
        Called when system capacity changes.

        Args:
            capacity: Required capacity for structure factor arrays
        """
        if self.Ak is None:
            return

        _print_memory(f"ensure_capacity START (capacity={capacity})")

        # Update internal capacity
        self.capacity = capacity

        xp = self.xp
        current_capacity = self.Sc.shape[1] if self.Sc is not None else 0

        if capacity > current_capacity:
            # Resize arrays
            if self.Sc is None:
                self.Sc = xp.zeros((len(self.Ak), capacity), dtype=self.Ak.dtype)
                self.Ss = xp.zeros((len(self.Ak), capacity), dtype=self.Ak.dtype)
                _print_memory(f"ensure_capacity INIT (Sc shape={self.Sc.shape}, Ss shape={self.Ss.shape})")
            else:
                # Create new arrays with larger capacity
                new_Sc = xp.zeros((len(self.Ak), capacity), dtype=self.Ak.dtype)
                new_Ss = xp.zeros((len(self.Ak), capacity), dtype=self.Ak.dtype)
                _print_memory(f"ensure_capacity ALLOC (new_Sc shape={new_Sc.shape}, new_Ss shape={new_Ss.shape})")

                # Copy existing data
                new_Sc[:, :current_capacity] = self.Sc
                new_Ss[:, :current_capacity] = self.Ss
                _print_memory(f"ensure_capacity COPY (copied {current_capacity} columns)")

                # Free old GPU memory if using CuPy
                if self.use_gpu:
                    del self.Sc
                    del self.Ss
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except:
                        pass
                    _print_memory("ensure_capacity GPU_CLEANUP")

                self.Sc = new_Sc
                self.Ss = new_Ss
                _print_memory(f"ensure_capacity END (Sc shape={self.Sc.shape}, Ss shape={self.Ss.shape})")

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
        n_atoms = len(positions)

        # Ensure capacity is sufficient (use internal capacity)
        # if n_atoms > self.capacity:
        #     self.ensure_capacity(n_atoms)

        if n_atoms == 0:
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
        # Update only the active columns
        self.Sc[:, :n_atoms] = q * cos_kr  # (K, N)
        self.Ss[:, :n_atoms] = q * sin_kr  # (K, N)

        # Clean up temporary arrays if using GPU to prevent memory accumulation
        if self.use_gpu:
            del kr, cos_kr, sin_kr, q
            # Periodically check memory and free blocks only if fragmentation is high
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except:
                pass


    def update_structure_factors_from_delta(
        self, indices: np.ndarray, delta_S_c: np.ndarray, delta_S_s: np.ndarray
    ) -> None:
        """
        Update structure factors S_c and S_s for current positions and charges.
        """
        idx = self._as_index_array(np.asarray(indices, dtype=int))
        self.Sc[:, idx] += self._to_backend(delta_S_c)
        self.Ss[:, idx] += self._to_backend(delta_S_s)

    def delta_kspace_energy_insertion(
        self, new_positions: np.ndarray, charges: np.ndarray, n_atoms: int
    ):
        """
        Compute k-space energy change when inserting atoms.

        dU_k = lB* * sum_k A(k) * [2*(Sc_sum*dSc + Ss_sum*dSs) + dSc^2 + dSs^2]

        where dSc = sum_i q_i * cos(k · r_i), dSs = sum_i q_i * sin(k · r_i)

        Args:
            new_positions: Positions of new atoms (M, 3)
            charges: Charges of new atoms (M,)
            n_atoms: Current number of atoms in system (before insertion)

        Returns:
            (delta_energy, new_Sc_cols, new_Ss_cols) where the columns are (K, M)
                arrays to be appended after acceptance.
        """
        if self.kvecs is None or self.Ak is None:
            return 0.0, None, None

        xp = self.xp
        pos_backend = self._to_backend(new_positions)
        q_backend = self._to_backend(charges)

        # k · r for each k-vector and each new atom: (K, M)
        kr = self.kvecs @ pos_backend.T
        cos_kr = xp.cos(kr)
        sin_kr = xp.sin(kr)

        # New structure factor columns: (K, M)
        new_Sc = q_backend[xp.newaxis, :] * cos_kr
        new_Ss = q_backend[xp.newaxis, :] * sin_kr

        # Sum over new atoms to get total change: (K,)
        dSc = xp.sum(new_Sc, axis=1)
        dSs = xp.sum(new_Ss, axis=1)

        # Current structure factor sums (only over active atoms)
        Sc_sum = xp.sum(self.Sc[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))
        Ss_sum = xp.sum(self.Ss[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))

        # Delta energy
        delta_energy = xp.sum(
            self.Ak * (2.0 * (Sc_sum * dSc + Ss_sum * dSs) + dSc**2 + dSs**2)
        )

        if self.lB_star is not None:
            delta_energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        # Return arrays on their current backend (GPU if using GPU, CPU if not)
        # This avoids unnecessary GPU->CPU->GPU conversions that cause memory leaks
        # The caller (append_structure_factors) will handle the backend conversion if needed
        return (
            float(self._to_cpu(delta_energy)),
            new_Sc,  # Keep on current backend (GPU if use_gpu=True)
            new_Ss,  # Keep on current backend (GPU if use_gpu=True)
        )

    def delta_kspace_energy_deletion(self, atom_indices: np.ndarray, n_atoms: int):
        """
        Compute k-space energy change when deleting atoms.

        Removing atoms is equivalent to dSc = -Sc_del, dSs = -Ss_del where
        Sc_del/Ss_del are the structure factor contributions of the removed atoms.

        dU_k = lB* * sum_k A(k) * [-2*(Sc_sum*Sc_del + Ss_sum*Ss_del) + Sc_del^2 + Ss_del^2]

        Args:
            atom_indices: Global indices of atoms to delete
            n_atoms: Current number of atoms in system (before deletion)

        Returns:
            delta_energy (float)
        """
        if self.kvecs is None or self.Ak is None:
            return 0.0

        xp = self.xp
        idx = self._as_index_array(np.asarray(atom_indices, dtype=int))

        # Structure factor contributions of atoms being deleted: (K,)
        Sc_del = xp.sum(self.Sc[:, idx], axis=1)
        Ss_del = xp.sum(self.Ss[:, idx], axis=1)

        # Current total sums (only over active atoms)
        Sc_sum = xp.sum(self.Sc[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))
        Ss_sum = xp.sum(self.Ss[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))

        # Delta energy (note the sign: dSc = -Sc_del)
        delta_energy = xp.sum(
            self.Ak
            * (
                -2.0 * (Sc_sum * Sc_del + Ss_sum * Ss_del)
                + Sc_del**2
                + Ss_del**2
            )
        )

        if self.lB_star is not None:
            delta_energy *= self.lB_star
        else:
            raise ValueError("Reduced Bjerrum length is not set")

        return float(self._to_cpu(delta_energy))

    def append_structure_factors(
        self, new_Sc: np.ndarray, new_Ss: np.ndarray, start_idx: int
    ) -> None:
        """
        Append new columns to structure factor arrays after accepted insertion.

        Args:
            new_Sc: New structure factor cosine columns (K, M)
            new_Ss: New structure factor sine columns (K, M)
            start_idx: Starting index where to append (current N_atoms)
        """
        if new_Sc is None or new_Ss is None:
            return
        _print_memory(f"append_structure_factors START (start_idx={start_idx}, new_Sc shape={new_Sc.shape})")
        xp = self.xp
        M = new_Sc.shape[1]  # Number of new columns
        end_idx = start_idx + M

        # Ensure capacity is sufficient (use internal capacity)
        # if end_idx > self.capacity:
        #     self.ensure_capacity(end_idx)

        # Convert to backend only if needed - reuse GPU arrays if already on GPU
        # This avoids creating unnecessary copies that fragment the memory pool
        if self.use_gpu:
            # If arrays are already on GPU and contiguous, reuse them directly
            if isinstance(new_Sc, cp.ndarray) and new_Sc.flags.c_contiguous:
                new_Sc_backend = new_Sc
            else:
                new_Sc_backend = self._to_backend(new_Sc)

            if isinstance(new_Ss, cp.ndarray) and new_Ss.flags.c_contiguous:
                new_Ss_backend = new_Ss
            else:
                new_Ss_backend = self._to_backend(new_Ss)
        else:
            # CPU mode - simple conversion
            new_Sc_backend = self._to_backend(new_Sc)
            new_Ss_backend = self._to_backend(new_Ss)

        _print_memory(f"append_structure_factors CONVERTED (Sc shape={self.Sc.shape if self.Sc is not None else None})")

        # Append new columns using in-place assignment
        # Use direct assignment to avoid creating temporary views
        self.Sc[:, start_idx:end_idx] = new_Sc_backend
        self.Ss[:, start_idx:end_idx] = new_Ss_backend

        # Clean up temporary arrays if we created new ones
        if self.use_gpu:
            # Only delete if we created new arrays (not if we reused existing GPU arrays)
            if new_Sc_backend is not new_Sc:
                del new_Sc_backend
            if new_Ss_backend is not new_Ss:
                del new_Ss_backend

            # Memory cleanup is now handled at the MC step level in sampler.move()
            # No need to free here to avoid re-allocation spikes during operations

        _print_memory(f"append_structure_factors END (appended {M} columns)")

    def remove_structure_factors(self, atom_indices: np.ndarray, n_atoms: int) -> None:
        """
        Remove columns from structure factor arrays after accepted deletion.

        This method removes the specified columns and compacts the remaining columns
        to maintain the mapping: column index = atom index (after system compaction).

        Args:
            atom_indices: Global indices of atoms to remove (before system compaction)
            n_atoms: Current number of atoms in system (before deletion)
        """
        if len(atom_indices) == 0:
            return

        _print_memory(f"remove_structure_factors START (n_atoms={n_atoms}, removing {len(atom_indices)} atoms)")

        xp = self.xp
        atom_indices_arr = np.asarray(atom_indices, dtype=int)

        # Filter to only valid indices within active range
        valid_indices = atom_indices_arr[(atom_indices_arr >= 0) & (atom_indices_arr < n_atoms)]
        if len(valid_indices) == 0:
            return

        # Create keep mask (more efficient than removing one by one)
        keep_mask = np.ones(n_atoms, dtype=bool)
        keep_mask[valid_indices] = False
        n_keep = int(np.sum(keep_mask))

        if n_keep < n_atoms:
            # Get indices of kept columns
            keep_indices = np.where(keep_mask)[0]
            keep_indices_backend = self._as_index_array(keep_indices)

            # Use advanced indexing to copy columns - this creates a temporary but it's necessary
            # We'll free it immediately after use
            if self.use_gpu:
                # Create temporary copy using advanced indexing
                # This is the most efficient way, but creates a temporary array
                temp_Sc = self.Sc[:, keep_indices_backend].copy()
                temp_Ss = self.Ss[:, keep_indices_backend].copy()

                # Copy to destination
                self.Sc[:, :n_keep] = temp_Sc
                self.Ss[:, :n_keep] = temp_Ss

                # Explicitly delete temporaries
                del temp_Sc, temp_Ss
                # Memory cleanup is now handled at the MC step level in sampler.move()
            else:
                # For CPU, advanced indexing is fine
                self.Sc[:, :n_keep] = self.Sc[:, keep_indices_backend]
                self.Ss[:, :n_keep] = self.Ss[:, keep_indices_backend]

            # Zero out the tail
            self.Sc[:, n_keep:n_atoms] = 0.0
            self.Ss[:, n_keep:n_atoms] = 0.0

            # Memory cleanup is now handled at the MC step level in sampler.move()
            # This prevents re-allocation spikes during operations

            _print_memory(f"remove_structure_factors END (kept {n_keep}/{n_atoms} atoms)")

    def compute_total_kspace_energy_forces(self, n_atoms: int) -> float:
        """
        Compute the total k-space energy for the entire system.

        U_k = (2π/V) ∑_{k≠0} [exp(-k²/(4α²))/k²] * |S(k)|²

        F_k = -(4π/V) ∑_{k≠0} [exp(-k²/(4α²))/k²] * q_i * [sin(k·r_i) * S_c - cos(k·r_i) * S_s]

        Args:
            n_atoms: Current number of atoms in system

        Returns:
            (energy, forces) tuple
        """
        if self.Ak is None or self.Sc is None or self.Ss is None:
            return 0.0, np.zeros((n_atoms, 3))

        xp = self.xp
        # |S(k)|² = S_c(k)² + S_s(k)²
        # Sum only over active atoms
        Sc_sum = xp.sum(self.Sc[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))  # (K,)
        Ss_sum = xp.sum(self.Ss[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))  # (K,)
        energy = xp.sum(self.Ak * (Sc_sum**2 + Ss_sum**2))

        # Forces only for active atoms
        if n_atoms > 0:
            force_struct = Sc_sum[:, xp.newaxis] * self.Ss[:, :n_atoms] - Ss_sum[:, xp.newaxis] * self.Sc[:, :n_atoms]
            forces = 2 * force_struct.T @ self.Ak_kvecs  # (N, 3)
        else:
            forces = np.zeros((0, 3))

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

        # |S(k)|² = S_c(k)² + S_s(k)² (sum only over active atoms)
        # Get n_atoms from positions array
        n_atoms = positions_np.shape[0]
        Sc_sum = xp.sum(self.Sc[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))  # (K,)
        Ss_sum = xp.sum(self.Ss[:, :n_atoms], axis=1) if n_atoms > 0 else xp.zeros(len(self.Ak))  # (K,)

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
