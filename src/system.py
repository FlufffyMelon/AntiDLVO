"""
System class for managing atoms, geometry, and ensemble parameters in Monte Carlo simulations.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .units import Units
from .ewald import EwaldHandler


class System:
    """
    System class containing information about all atoms, geometry, and ensemble parameters.
    """

    def __init__(
        self,
        n_atoms: int = 0,
        initial_capacity: int = 128,
        box: Union[List[float], np.ndarray] = None,
        pbc: Union[List[bool], np.ndarray] = None,
        ensemble: str = "NVT",
        temp: Optional[float] = None,
        pressure: Optional[float] = None,
        mu: Optional[float] = None,
        ewald_handler: Optional[EwaldHandler] = None,
        units: Units = None,
    ):
        """
        Initialize the system.

        Args:
            n_atoms: Number of atoms in the system
            initial_capacity: Initial capacity for arrays
            box: Simulation box dimensions [x, y, z]
            pbc: Periodic boundary conditions [x, y, z]
            ensemble: Type of ensemble ("NVT", "NPT", "muVT")
            temp: Temperature (required for all ensembles)
            pressure: Pressure (required for NPT)
            mu: Chemical potential (required for muVT)
            ewald_handler: Ewald handler object
            units: Units system to use
        """
        self.N = n_atoms
        self.capacity = max(initial_capacity, n_atoms)
        self.ensemble = ensemble
        self.temp = temp
        self.pressure = pressure
        self.mu = mu
        self.units = units or Units()

        # Cached properties updated during MC moves
        self.potential_energy: Optional[float] = None
        self.virial: Optional[float] = None
        self.mu_id: Optional[float] = None

        # Initialize Ewald handler if parameters are provided
        self.ewald_handler = ewald_handler

        # Set box dimensions
        self.box = np.array(
            box if box is not None else [10.0, 10.0, 10.0], dtype=np.float64
        )

        # Update Bjerrum length with current temperature
        if self.ewald_handler and self.temp is not None:
            self.ewald_handler.update_bjerrum_length(self.temp)

        # Validate ensemble parameters
        self._validate_ensemble_parameters()

        # Initialize arrays
        self._initialize_arrays()

        # Set PBC
        self.pbc = np.array(pbc if pbc is not None else [True, True, True], dtype=bool)

        # Pre-compute inverse box for optimization
        self.inv_box = np.where(self.pbc, 1.0 / self.box, 0.0)

        # Initialize Ewald k-vectors if enabled
        if self.ewald_handler:
            self.ewald_handler.initialize_k_vectors(self.box)
            self.ewald_handler.update_structure_factors(np.zeros((0, 3)), np.zeros(0))

    def _validate_ensemble_parameters(self) -> None:
        """Validate required parameters for the chosen ensemble."""
        if self.ensemble.lower() not in ["nvt", "npt", "muvt"]:
            raise ValueError(f"Unsupported ensemble: {self.ensemble}")

        if self.temp is None:
            raise ValueError("Temperature is required for all ensembles")

        if self.ensemble == "NPT" and self.pressure is None:
            raise ValueError("Pressure is required for NPT ensemble")

        if self.ensemble == "muVT" and self.mu is None:
            raise ValueError("Chemical potential is required for muVT ensemble")

    def _initialize_arrays(self) -> None:
        """Initialize system arrays with given capacity."""
        self.charges = np.zeros(self.capacity, dtype=np.float64)
        self.masses = np.zeros(self.capacity, dtype=np.float64)
        self.positions = np.zeros((self.capacity, 3), dtype=np.float64)
        self.types = np.zeros(self.capacity, dtype=np.int32)
        self.names = np.empty(self.capacity, dtype="U10")

    def _resize_arrays(self, new_capacity: int) -> None:
        """Resize all arrays to new capacity."""
        old_capacity = self.capacity
        self.capacity = new_capacity

        # Resize arrays while preserving existing data
        new_charges = np.zeros(new_capacity, dtype=np.float64)
        new_masses = np.zeros(new_capacity, dtype=np.float64)
        new_positions = np.zeros((new_capacity, 3), dtype=np.float64)
        new_types = np.zeros(new_capacity, dtype=np.int32)
        new_names = np.empty(new_capacity, dtype="U10")

        # Copy existing data
        copy_size = min(old_capacity, new_capacity)
        new_charges[:copy_size] = self.charges[:copy_size]
        new_masses[:copy_size] = self.masses[:copy_size]
        new_positions[:copy_size] = self.positions[:copy_size]
        new_types[:copy_size] = self.types[:copy_size]
        new_names[:copy_size] = self.names[:copy_size]

        # Update arrays
        self.charges = new_charges
        self.masses = new_masses
        self.positions = new_positions
        self.types = new_types
        self.names = new_names

    def add_atom(
        self,
        position: Union[List[float], np.ndarray],
        atom_type: int,
        name: str,
        charge: float = 0.0,
        mass: float = 1.0,
    ) -> int:
        """
        Add an atom to the system.

        Args:
            position: Atom position [x, y, z]
            atom_type: Atom type identifier
            name: Atom name
            charge: Atom charge
            mass: Atom mass

        Returns:
            Index of the added atom
        """
        # Check if we need to resize
        if self.N >= self.capacity:
            self._resize_arrays(self.capacity * 2)

        # Add atom data
        atom_id = self.N
        self.positions[atom_id] = np.array(position, dtype=np.float64)
        self.types[atom_id] = atom_type
        self.names[atom_id] = name
        self.charges[atom_id] = charge
        self.masses[atom_id] = mass

        self.N += 1

        # Update Ewald structure factors if enabled
        if self.ewald_handler:
            self.ewald_handler.update_structure_factors(
                self.positions[: self.N], self.charges[: self.N]
            )
            self.ewald_handler.update_dipole_moment(
                self.positions[: self.N], self.charges[: self.N]
            )

        return atom_id

    def remove_atom(self, atom_id: int) -> None:
        """
        Remove an atom from the system.

        Args:
            atom_id: Index of atom to remove
        """
        if atom_id >= self.N or atom_id < 0:
            raise IndexError(f"Atom index {atom_id} out of range")

        # Shift all atoms after the removed one
        if atom_id < self.N - 1:
            self.positions[atom_id : self.N - 1] = self.positions[atom_id + 1 : self.N]
            self.types[atom_id : self.N - 1] = self.types[atom_id + 1 : self.N]
            self.names[atom_id : self.N - 1] = self.names[atom_id + 1 : self.N]
            self.charges[atom_id : self.N - 1] = self.charges[atom_id + 1 : self.N]
            self.masses[atom_id : self.N - 1] = self.masses[atom_id + 1 : self.N]

        # Zero out the last position
        self.positions[self.N - 1] = 0.0
        self.types[self.N - 1] = 0
        self.names[self.N - 1] = ""
        self.charges[self.N - 1] = 0.0
        self.masses[self.N - 1] = 0.0

        self.N -= 1

        # Update Ewald structure factors if enabled
        if self.ewald_handler:
            self.ewald_handler.update_structure_factors(
                self.positions[: self.N], self.charges[: self.N]
            )
            self.ewald_handler.update_dipole_moment(
                self.positions[: self.N], self.charges[: self.N]
            )

    def get_active_atoms(self) -> Tuple[np.ndarray, ...]:
        """
        Get data for all active atoms (first N atoms).

        Returns:
            Tuple of (positions, types, names, charges, masses)
        """
        return (
            self.positions[: self.N],
            self.types[: self.N],
            self.names[: self.N],
            self.charges[: self.N],
            self.masses[: self.N],
        )

    def apply_pbc(self, position: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions to a position.

        Args:
            position: Position to wrap

        Returns:
            Wrapped position
        """
        wrapped = position.copy()
        wrapped = np.where(
            self.pbc, wrapped - self.box * np.round(wrapped * self.inv_box), wrapped
        )
        return wrapped

    def get_minimum_image_distance(
        self, pos1: np.ndarray, pos2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate minimum image distance between two positions.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            Minimum image distance vector
        """
        dr = pos2 - pos1
        dr = np.where(self.pbc, dr - self.box * np.round(dr * self.inv_box), dr)
        return dr

    def get_all_distances(self, pos1: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Calculate distances from one position to all others.

        Args:
            pos1: Single position (3,)
            positions: Array of positions (N, 3)

        Returns:
            Array of distances (N,)
        """
        dr = positions - pos1[np.newaxis, :]
        dr = np.where(
            self.pbc[np.newaxis, :], dr - self.box * np.round(dr * self.inv_box), dr
        )
        return np.linalg.norm(dr, axis=1)

    def update_box(self, scale_factor: float) -> None:
        """
        Update box dimensions and derived properties.

        Args:
            scale_factor: Scale factor to apply to box dimensions
        """
        # Update box dimensions
        self.box *= scale_factor
        self.inv_box = np.where(self.pbc, 1.0 / self.box, 0.0)

        # Update Ewald k-vectors if enabled
        if self.ewald_handler:
            self.ewald_handler.initialize_k_vectors(self.box)
            self.ewald_handler.update_structure_factors(
                self.positions[: self.N], self.charges[: self.N]
            )
            self.ewald_handler.update_dipole_moment(
                self.positions[: self.N], self.charges[: self.N]
            )

    def get_volume(self) -> float:
        """
        Get the system volume.
        """
        return float(np.prod(self.box))

    def get_density(self) -> float:
        """Get the number density (atoms per unit volume)."""
        return self.N / self.get_volume() if self.N > 0 else 0.0

    def __str__(self) -> str:
        """String representation of the system."""
        return (
            f"System: {self.N} atoms, {self.ensemble} ensemble, "
            f"T={self.temp}, box={self.box}, capacity={self.capacity}"
        )
