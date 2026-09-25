"""
Molecule classes for Monte Carlo simulations.
Provides abstractions for different types of molecular entities.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Molecule(ABC):
    """
    Abstract base class for all molecular entities.
    Used as a convenient wrapper for creating particles in the system.
    """

    def __init__(self, name: str):
        """
        Initialize the molecule.

        Args:
            name: Name of the molecule
        """
        self.name = name

    @abstractmethod
    def get_particles(self) -> List[Tuple[np.ndarray, int, str, float, float]]:
        """
        Get list of particles that make up this molecule.

        Returns:
            List of tuples with (position, type_id, name, charge, mass) for each particle
        """
        pass


class Particle(Molecule):
    """
    Single particle implementation of Molecule.
    """

    def __init__(
        self,
        position: np.ndarray,
        type_id: int,
        name: str,
        charge: float,
        mass: float,
        static: bool = False,
    ):
        """
        Initialize a single particle.

        Args:
            position: Position vector [x, y, z]
            type_id: Particle type identifier
            name: Particle name
            charge: Particle charge
            mass: Particle mass
        """
        super().__init__(name)
        self.position = np.array(position, dtype=np.float64)
        self.type_id = type_id
        self.charge = charge
        self.mass = mass
        self.static = static

    def get_particles(self) -> List[Tuple[np.ndarray, int, str, float, float]]:
        """Return this particle's data as a single-element list."""
        return [
            (
                self.position,
                self.type_id,
                self.name,
                self.charge,
                self.mass,
                self.static,
            )
        ]


class Dipole(Molecule):
    """
    Dipole molecule consisting of two charged particles.
    """

    def __init__(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        length: float,
        type_plus: int,
        type_plus_name: str,
        type_minus: int,
        type_minus_name: str,
        type_ghost: Optional[int] = None,
        type_ghost_name: Optional[str] = None,
        charge: float = 1.0,
        mass: float = 1.0,
        ghost_count: int = 0,
        static: bool = False,
    ):
        """
        Initialize a dipole molecule.

        Args:
            position: Position of the center of the dipole [x, y, z]
            orientation: Unit vector defining dipole orientation
            length: Length of the dipole
            type_plus: Particle type identifier for the positive end
            type_minus: Particle type identifier for the negative end
            type_ghost: Particle type identifier for ghost particles (optional)
            name: Base name for the dipole particles
            charge: Magnitude of charge for dipole ends (positive/negative)
            mass: Mass of each particle in the dipole
            ghost_count: Number of ghost particles to place along the dipole
        """
        super().__init__("")
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)
        # Ensure orientation is a unit vector
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm
        self.length = float(length)
        self.type_plus = type_plus
        self.type_plus_name = type_plus_name
        self.type_minus = type_minus
        self.type_minus_name = type_minus_name
        self.type_ghost = type_ghost if type_ghost is not None else type_plus
        self.type_ghost_name = (
            type_ghost_name if type_ghost_name is not None else type_plus_name
        )
        self.charge = float(charge)
        self.mass = float(mass)
        self.ghost_count = int(ghost_count)
        self.static = static

    def get_particles(self) -> List[Tuple[np.ndarray, int, str, float, float]]:
        """Return the particles that make up this dipole."""
        half_length = self.length / 2.0
        direction = self.orientation * half_length

        # Create the two charged particles
        positive_pos = self.position + direction
        negative_pos = self.position - direction

        particles = [
            (
                positive_pos,
                self.type_plus,
                self.type_plus_name,
                self.charge,
                self.mass,
                self.static,
            ),
            (
                negative_pos,
                self.type_minus,
                self.type_minus_name,
                -self.charge,
                self.mass,
                self.static,
            ),
        ]

        # Add ghost particles if requested
        if self.ghost_count > 0:
            step = self.length / (self.ghost_count + 1)
            for i in range(self.ghost_count):
                # Position along the dipole, starting from the negative end
                pos = negative_pos + self.orientation * step * (i + 1)
                particles.append(
                    (pos, self.type_ghost, self.type_ghost_name, 0.0, 0.0, self.static)
                )

        return particles
