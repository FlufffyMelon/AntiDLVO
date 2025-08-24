"""
Units management system for Monte Carlo simulations.
Supports both Lennard-Jones and standard (Gromacs-compatible) unit systems.
"""

from pint import UnitRegistry
import numpy as np
from typing import Union, Any


class UnitsSystem:
    """Base units system class using pint for unit management."""

    def __init__(self):
        self.ureg = UnitRegistry()
        self.Q_ = self.ureg.Quantity
        self._setup_units()

    def _setup_units(self):
        """Setup unit definitions. Override in subclasses."""
        pass

    def quantity(self, value: Union[float, np.ndarray], unit: str) -> Any:
        """Create a quantity with units."""
        return self.Q_(value, unit)

    def strip_units(self, quantity: Any) -> Union[float, np.ndarray]:
        """Remove units and return magnitude."""
        return quantity.magnitude

    def add_units(self, value: Union[float, np.ndarray], unit: str) -> Any:
        """Add units to a value."""
        return self.Q_(value, unit)

    def convert_to(self, quantity: Any, target_unit: str) -> Any:
        """Convert quantity to target unit."""
        return quantity.to(target_unit)

    # Human-readable labels for logging/output
    @property
    def energy_label(self) -> str:
        return ""

    @property
    def length_label(self) -> str:
        return ""

    @property
    def temperature_label(self) -> str:
        return ""

    @property
    def pressure_label(self) -> str:
        return ""


class LJUnits(UnitsSystem):
    """Lennard-Jones units system."""

    def _setup_units(self):
        """Setup LJ unit definitions."""
        # Define LJ base units
        self.ureg.define("lj_energy = [energy]")
        self.ureg.define("lj_length = [length]")
        self.ureg.define(
            "lj_time = sqrt(lj_mass * lj_length ** 2 / lj_energy) = [time]"
        )
        self.ureg.define("lj_force = lj_energy / lj_length = [force]")
        self.ureg.define("lj_pressure = lj_energy / lj_length ** 3 = [pressure]")
        self.ureg.define("lj_temperature = lj_energy = [temperature]")
        self.ureg.define("lj_mass = [mass]")

    @property
    def energy_label(self) -> str:
        return "LJ"

    @property
    def length_label(self) -> str:
        return "LJ"

    @property
    def temperature_label(self) -> str:
        return "LJ"

    @property
    def pressure_label(self) -> str:
        return "LJ"


class StandardUnits(UnitsSystem):
    """Standard units system (Gromacs-compatible)."""

    def _setup_units(self):
        """Setup standard unit definitions."""
        # Standard units are already defined in pint
        # Length: nm, Energy: kJ/mol, Mass: amu, Time: ps
        # Force: kJ/(mol*nm), Pressure: bar, Temperature: K
        pass

    @property
    def energy_label(self) -> str:
        return "kJ/mol"

    @property
    def length_label(self) -> str:
        return "nm"

    @property
    def temperature_label(self) -> str:
        return "K"

    @property
    def pressure_label(self) -> str:
        return "bar"


class Units:
    """Main units interface that can switch between unit systems."""

    def __init__(self, system: str = "standard"):
        """
        Initialize units system.

        Args:
            system: "standard" for Gromacs units or "lj" for Lennard-Jones units
        """
        self.system_name = system

        if system == "lj":
            self.system = LJUnits()
        elif system == "standard":
            self.system = StandardUnits()
        else:
            raise ValueError(f"Unknown unit system: {system}")

    def quantity(self, value: Union[float, np.ndarray], unit: str) -> Any:
        """Create a quantity with units."""
        return self.system.quantity(value, unit)

    def strip_units(self, quantity: Any) -> Union[float, np.ndarray]:
        """Remove units and return magnitude."""
        return self.system.strip_units(quantity)

    def add_units(self, value: Union[float, np.ndarray], unit: str) -> Any:
        """Add units to a value."""
        return self.system.add_units(value, unit)

    def convert_to(self, quantity: Any, target_unit: str) -> Any:
        """Convert quantity to target unit."""
        return self.system.convert_to(quantity, target_unit)

    @property
    def ureg(self):
        """Access to underlying unit registry."""
        return self.system.ureg

    # Expose labels for convenience
    @property
    def energy_label(self) -> str:
        return self.system.energy_label

    @property
    def length_label(self) -> str:
        return self.system.length_label

    @property
    def temperature_label(self) -> str:
        return self.system.temperature_label

    @property
    def pressure_label(self) -> str:
        return self.system.pressure_label
