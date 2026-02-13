#!/usr/bin/env python3
"""
Base test class for Ewald method validation tests.
Contains common functionality shared across all test implementations.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    load_config,
    create_units,
    create_system,
    create_topology,
)
from src.molecule import Particle, Dipole


class BaseEwaldTest:
    """Base class for Ewald method validation tests."""

    def __init__(
        self, config_file: str, results_dir: str = "results", verbose: bool = False
    ):
        """Initialize the test.

        Args:
            config_file: Path to test configuration file
            results_dir: Directory to save results
            verbose: Whether to show detailed output
        """
        self.config_file = Path(config_file)
        self.results_dir = Path(results_dir)
        self.verbose = verbose

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True)

        # Load configuration
        self.cfg = load_config(str(self.config_file))

        # Create system components
        self.units = create_units(self.cfg)
        self.system = create_system(self.cfg, self.units)
        self.topology = create_topology(self.cfg, self.units)

        # Get test parameters
        self.test_params = self.cfg.get("test_params", {})

        # Results storage
        self.results = {}

    def run_test(self) -> Dict[str, Any]:
        """Run the test and return results.

        Returns:
            Dictionary containing test results
        """
        raise NotImplementedError("Subclasses must implement run_test method")

    def _create_energy_plot(
        self,
        x_values: List[float],
        computed_energies: List[float],
        real_energies: List[float],
        title: str,
        filename: str,
    ):
        """Create energy comparison plot."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            x_values, computed_energies, "o-", label="Ewald", linewidth=2, markersize=6
        )
        plt.plot(
            x_values, real_energies, "s-", label="Coulomb", linewidth=1, markersize=4
        )
        plt.xlabel("Distance (nm)")
        plt.ylabel("Energy (kJ/mol)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        errors = np.abs(1 - np.array(computed_energies) / np.array(real_energies))
        plt.plot(x_values, errors, "ro-", linewidth=2, markersize=6)
        plt.xlabel("Distance (nm)")
        plt.ylabel("Relative Error")
        plt.title("Ewald Error")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _create_crystal_plot(
        self,
        lattice_sizes: List[int],
        computed_energies: List[float],
        theoretical_energies: List[float],
        title: str,
        filename: str,
    ):
        """Create crystal energy comparison plot."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            lattice_sizes,
            computed_energies,
            "o-",
            label="Ewald",
            linewidth=2,
            markersize=6,
        )
        plt.plot(
            lattice_sizes,
            theoretical_energies,
            "s-",
            label="Madelung",
            linewidth=2,
            markersize=4,
        )
        plt.xlabel("Lattice Size (unit cells)")
        plt.ylabel("Energy per Ion Pair (kJ/mol)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        errors = np.abs(
            1 - np.array(computed_energies) / np.array(theoretical_energies)
        )
        plt.plot(lattice_sizes, errors, "ro-", linewidth=2, markersize=6)
        plt.xlabel("Lattice Size (unit cells)")
        plt.ylabel("Relative Error")
        plt.title("Ewald Error")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def _extract_ewald_params(self) -> Dict[str, Any]:
        """Extract Ewald parameters from configuration."""
        ewald_cfg = self.cfg.get("ewald", {})
        return {
            "alpha": ewald_cfg.get("alpha"),
            "real_cut": ewald_cfg.get("real_cut"),
            "n_c": ewald_cfg.get("n_c"),
            "eps": ewald_cfg.get("eps"),
            "dielectric": ewald_cfg.get("dielectric"),
        }

    def _calculate_errors(
        self, computed: List[float], theoretical: List[float]
    ) -> Dict[str, float]:
        """Calculate error metrics."""
        computed_arr = np.array(computed)
        theoretical_arr = np.array(theoretical)

        l2_error = np.sqrt(np.sum((computed_arr - theoretical_arr) ** 2))
        max_error = np.max(np.abs(computed_arr - theoretical_arr))
        mean_error = np.mean(np.abs(computed_arr - theoretical_arr))
        relative_error = np.abs(computed_arr - theoretical_arr) / theoretical_arr

        return {
            "l2_error": l2_error,
            "max_error": max_error,
            "mean_error": mean_error,
            "relative_error": relative_error,
        }

    def _clear_system(self):
        """Clear all atoms from the system."""
        self.system.N_atoms = 0
        self.system.N_molecules = 0

    def _update_ewald_structure_factors(self):
        """Update Ewald structure factors after adding atoms."""
        if self.system.ewald_handler:
            self.system.ewald_handler.update_structure_factors(
                self.system.positions[: self.system.N_atoms],
                self.system.charges[: self.system.N_atoms],
            )
            # self.system.ewald_handler.update_dipole_moment(
            #     self.system.positions[: self.system.N_atoms],
            #     self.system.charges[: self.system.N_atoms],
            # )

    def _get_center_position(self) -> np.ndarray:
        """Get center position of the simulation box."""
        Lx, Ly, Lz = self.system.box
        return np.array([Lx / 2, Ly / 2, Lz / 2])

    def _get_type_ids(self) -> Dict[str, int]:
        """Get type IDs from topology."""
        return self.topology.type_name_to_id.copy()

    def _compute_energy(self) -> float:
        """Compute energy using the topology."""
        computed_energy, _ = self.topology.compute_energy_virial(self.system)
        return computed_energy
