"""
System class for managing atoms, geometry, and ensemble parameters in Monte Carlo simulations.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .units import Units
from .ewald_handler import EwaldHandler
from .molecule import Molecule


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
        self.N_atoms = n_atoms
        self.N_molecules = 0
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
        self.lB_star = None
        if self.ewald_handler and self.temp is not None:
            self.lB_star = self.ewald_handler.update_bjerrum_length(self.temp)

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
        self.forces = np.zeros((self.capacity, 3), dtype=np.float64)
        self.types = np.zeros(self.capacity, dtype=np.int32)
        self.names = np.empty(self.capacity, dtype="U10")
        self.molecule_ids = np.zeros(self.capacity, dtype=np.int32)
        self.statics = np.zeros(self.capacity, dtype=bool)

    def _resize_arrays(self, new_capacity: int) -> None:
        """Resize all arrays to new capacity."""
        old_capacity = self.capacity
        self.capacity = new_capacity

        # Resize arrays while preserving existing data
        new_charges = np.zeros(new_capacity, dtype=np.float64)
        new_masses = np.zeros(new_capacity, dtype=np.float64)
        new_positions = np.zeros((new_capacity, 3), dtype=np.float64)
        new_forces = np.zeros((new_capacity, 3), dtype=np.float64)
        new_types = np.zeros(new_capacity, dtype=np.int32)
        new_names = np.empty(new_capacity, dtype="U10")
        new_molecule_ids = np.zeros(new_capacity, dtype=np.int32)
        new_statics = np.zeros(new_capacity, dtype=bool)

        # Copy existing data
        copy_size = min(old_capacity, new_capacity)
        new_charges[:copy_size] = self.charges[:copy_size]
        new_masses[:copy_size] = self.masses[:copy_size]
        new_positions[:copy_size] = self.positions[:copy_size]
        new_forces[:copy_size] = self.forces[:copy_size]
        new_types[:copy_size] = self.types[:copy_size]
        new_names[:copy_size] = self.names[:copy_size]
        new_molecule_ids[:copy_size] = self.molecule_ids[:copy_size]
        new_statics[:copy_size] = self.statics[:copy_size]

        # Update arrays
        self.charges = new_charges
        self.masses = new_masses
        self.positions = new_positions
        self.forces = new_forces
        self.types = new_types
        self.names = new_names
        self.molecule_ids = new_molecule_ids
        self.statics = new_statics

    def add_molecule(self, molecule: Molecule) -> int:
        """
        Add a molecule to the system.

        Args:
            molecule: Molecule to add

        Returns:
            Index of the molecule in the molecules list
        """
        # Assign a unique ID to the molecule
        molecule_id = self.N_molecules
        self.N_molecules += 1

        # Add all particles from the molecule
        particles = molecule.get_particles()
        for position, type_id, name, charge, mass, static in particles:
            self._add_atom(position, type_id, name, charge, mass, molecule_id, static)

        return molecule_id

    def _add_atom(
        self,
        position: Union[List[float], np.ndarray],
        atom_type: int,
        name: str,
        charge: float = 0.0,
        mass: float = 1.0,
        molecule_id: int = -1,
        static: bool = False,
    ) -> int:
        """
        Internal method to add an atom to the system.

        Args:
            position: Atom position [x, y, z]
            atom_type: Atom type identifier
            name: Atom name
            charge: Atom charge
            mass: Atom mass
            molecule_id: ID of the molecule this atom belongs to
            static: Whether the atom is static

        Returns:
            Index of the added atom
        """
        # Check if we need to resize
        if self.N_atoms >= self.capacity:
            self._resize_arrays(self.capacity * 2)

        # Add atom data
        atom_id = self.N_atoms
        self.positions[atom_id] = np.array(position, dtype=np.float64)
        self.types[atom_id] = atom_type
        self.names[atom_id] = name
        self.charges[atom_id] = charge
        self.masses[atom_id] = mass
        self.molecule_ids[atom_id] = molecule_id
        self.statics[atom_id] = static

        self.N_atoms += 1

        return atom_id

    def remove_atom(self, atom_id: int) -> None:
        """
        Remove an atom from the system.

        Args:
            atom_id: Index of atom to remove
        """
        if atom_id >= self.N_atoms or atom_id < 0:
            raise IndexError(f"Atom index {atom_id} out of range")

        # Shift all atoms after the removed one
        if atom_id < self.N_atoms - 1:
            self.positions[atom_id : self.N_atoms - 1] = self.positions[
                atom_id + 1 : self.N_atoms
            ]
            self.forces[atom_id : self.N_atoms - 1] = self.positions[
                atom_id + 1 : self.N_atoms
            ]
            self.types[atom_id : self.N_atoms - 1] = self.types[
                atom_id + 1 : self.N_atoms
            ]
            self.names[atom_id : self.N_atoms - 1] = self.names[
                atom_id + 1 : self.N_atoms
            ]
            self.charges[atom_id : self.N_atoms - 1] = self.charges[
                atom_id + 1 : self.N_atoms
            ]
            self.masses[atom_id : self.N_atoms - 1] = self.masses[
                atom_id + 1 : self.N_atoms
            ]
            self.molecule_ids[atom_id : self.N_atoms - 1] = self.molecule_ids[
                atom_id + 1 : self.N_atoms
            ]
            self.statics[atom_id : self.N_atoms - 1] = self.statics[
                atom_id + 1 : self.N_atoms
            ]

        # Zero out the last position
        self.positions[self.N_atoms - 1] = 0.0
        self.forces[self.N_atoms - 1] = 0.0
        self.types[self.N_atoms - 1] = 0
        self.names[self.N_atoms - 1] = ""
        self.charges[self.N_atoms - 1] = 0.0
        self.masses[self.N_atoms - 1] = 0.0
        self.molecule_ids[self.N_atoms - 1] = 0
        self.statics[self.N_atoms - 1] = False

        self.N_atoms -= 1

        # Update Ewald structure factors if enabled
        if self.ewald_handler:
            self.ewald_handler.update_structure_factors(
                self.positions[: self.N_atoms], self.charges[: self.N_atoms]
            )
            # self.ewald_handler.update_dipole_moment(
            #     self.positions[: self.N_atoms], self.charges[: self.N_atoms]
            # )

    def get_active_atoms(self) -> Tuple[np.ndarray, ...]:
        """
        Get data for all active atoms (first N atoms).

        Returns:
            Tuple of (positions, types, names, charges, masses)
        """
        return (
            self.positions[: self.N_atoms],
            self.forces[: self.N_atoms],
            self.molecule_ids[: self.N_atoms],
            self.types[: self.N_atoms],
            self.names[: self.N_atoms],
            self.charges[: self.N_atoms],
            self.masses[: self.N_atoms],
            self.statics[: self.N_atoms],
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
        dr = np.where(
            self.pbc[np.newaxis, :],
            dr - self.box[np.newaxis, :] * np.round(dr * self.inv_box[np.newaxis, :]),
            dr,
        )
        return dr

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
                self.positions[: self.N_atoms], self.charges[: self.N_atoms]
            )
            # self.ewald_handler.update_dipole_moment(
            #     self.positions[: self.N_atoms], self.charges[: self.N_atoms]
            # )

    def get_volume(self) -> float:
        """
        Get the system volume.
        """
        return float(np.prod(self.box))

    def get_density(self) -> float:
        """Get the number density (atoms per unit volume)."""
        return self.N_molecules / self.get_volume() if self.N_molecules > 0 else 0.0

    def get_molecule_atoms(self, molecule_id: int) -> List[int]:
        """
        Get the atom indices that belong to a specific molecule.

        Args:
            molecule_id: ID of the molecule

        Returns:
            List of atom indices belonging to the molecule
        """
        return np.argwhere(self.molecule_ids[: self.N_atoms] == molecule_id).flatten()
        # return np.arange(self.N_atoms)[self.molecule_ids[: self.N_atoms] == molecule_id]

    def get_molecule_by_atom(self, atom_id: int) -> Optional[Molecule]:
        """
        Get the molecule that an atom belongs to.

        Args:
            atom_id: Index of the atom

        Returns:
            Molecule object or None if the atom doesn't belong to a molecule
        """
        if atom_id >= self.N_atoms or atom_id < 0:
            raise IndexError(f"Atom index {atom_id} out of range")

        molecule_id = self.molecule_ids[atom_id]
        if molecule_id < 0 or molecule_id >= len(self.molecules):
            return None

        return self.molecules[molecule_id]

    def unwrap_molecule(self, positions: np.ndarray) -> np.ndarray:
        """
        Unwrap a molecule that might be split across periodic boundaries.

        Args:
            positions: Array of positions (N, 3) of atoms in the molecule

        Returns:
            Unwrapped positions (N, 3) where the molecule is continuous
        """
        if len(positions) <= 1:
            return positions.copy()

        # Make a copy to avoid modifying the original
        unwrapped = positions.copy()

        # Use first atom as reference, adjust others to be close to it
        for i in range(1, len(unwrapped)):
            for dim in range(3):
                # Apply minimum image convention to ensure atoms are close to reference
                while unwrapped[i, dim] - unwrapped[0, dim] > self.box[dim] / 2:
                    unwrapped[i, dim] -= self.box[dim]
                while unwrapped[i, dim] - unwrapped[0, dim] < -self.box[dim] / 2:
                    unwrapped[i, dim] += self.box[dim]

        return unwrapped

    def __str__(self) -> str:
        """String representation of the system."""
        return (
            f"System: {self.N_atoms} atoms, {self.N_molecules} molecules, {self.ensemble} ensemble, "
            f"T={self.temp}, box={self.box}, capacity={self.capacity}"
        )
