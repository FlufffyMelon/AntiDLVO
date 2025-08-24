import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from units import Units


class Force(ABC):
    """
    Abstract base class for all pair force components.
    Implement __call__ with a vectorized API returning (energy_vector, force_matrix):
      __call__(r_ij: ndarray, type1: int, types2: ndarray, charge1: float, charges2: ndarray)
    where r_ij is the distance matrix from a single particle of type1 to neighbors of types2.
    """

    def __init__(self, units: Units = None):
        self.units = units or Units()
        # self.parameters = {}

    @abstractmethod
    def __call__(
        self,
        pos: np.ndarray,
        r_ij: np.ndarray,
        charge1: float,
        charge2: float,
        system=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate energies and forces for pairwise interactions.

        Args:
            pos: Position of particle i (3,)
            r_ij: Distance matrix (N, 3) from particle i to all particles j
            charge1: Charge of particle i
            charge2: Charges of particles j
            system: System object

        Returns:
            Tuple of (energy_vector, force_matrix) where:
            - energy_vector: (N,) array of pairwise energies
            - force_matrix: (N, 3) array of forces on particles j from particle i
        """
        pass

    # def set_parameters(self, params: Dict[Any, Any]) -> None:
    #     self.parameters = params


class OneBodyForce(ABC):
    def __init__(self, units: Units = None):
        self.units = units or Units()
        # self.parameters: Dict[Any, Any] = {}

    @abstractmethod
    def __call__(
        self,
        pos: np.ndarray,
        atom_type: int,
        charge: float = 0.0,
        system=None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate one-body energy and force.

        Args:
            pos: Position vector (3,) of the particle
            atom_type: Type of the particle
            charge: Charge of the particle
            system: System object for box information

        Returns:
            Tuple of (energy_vector, force_vector) where:
            - energy_vector: (1,) array containing the energy
            - force_vector: (3,) array containing the force
        """
        pass

    # def set_parameters(self, params: Dict[Any, Any]) -> None:
    #     self.parameters = params
