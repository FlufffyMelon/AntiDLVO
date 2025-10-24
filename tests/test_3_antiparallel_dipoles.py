#!/usr/bin/env python3
"""
Test 3: Two antiparallel dipoles at varying distances.
Validates Ewald method against dipole-dipole interaction.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_test import BaseEwaldTest
from src.molecule import Dipole


class TestAntiparallelDipoles(BaseEwaldTest):
    """Test for two antiparallel dipoles at varying distances."""

    def __init__(
        self, config_file: str = "configs/test_3_antiparallel_dipoles.yaml", **kwargs
    ):
        """Initialize the antiparallel dipoles test."""
        super().__init__(config_file, **kwargs)
        self.test_name = "Antiparallel Dipoles"

    def run_test(self) -> Dict[str, Any]:
        """Run the antiparallel dipoles test."""
        distances = self.test_params.get("distances", np.linspace(2, 20, 20).tolist())
        dipole_length = self.test_params.get("dipole_length", 1.0)

        computed_energies = []
        real_energies = []

        for distance in distances:
            comp_energy, real_energy = self._setup_antiparallel_dipoles(
                distance, dipole_length
            )
            computed_energies.append(comp_energy)
            real_energies.append(real_energy)

            if self.verbose:
                print(
                    f"  Distance: {distance:.2f} nm, Computed: {comp_energy:.6f}, Real: {real_energy:.6f}"
                )

        # Calculate errors
        errors = self._calculate_errors(computed_energies, real_energies)

        # Create plot
        self._create_energy_plot(
            distances,
            computed_energies,
            real_energies,
            "Antiparallel Dipoles: Ewald vs Coulomb",
            "test_3_antiparallel_dipoles.png",
        )

        return {
            "test_name": self.test_name,
            "distances": distances,
            "computed_energies": computed_energies,
            "real_energies": real_energies,
            "dipole_length": dipole_length,
            **errors,
            "ewald_params": self._extract_ewald_params(),
        }

    def _setup_antiparallel_dipoles(
        self, distance: float, dipole_length: float
    ) -> tuple[float, float]:
        """Setup two antiparallel dipoles at specified distance."""
        # Clear existing atoms
        self._clear_system()

        # Get type IDs from topology
        type_ids = self._get_type_ids()
        ip_type_id = type_ids.get("Ip", 0)
        im_type_id = type_ids.get("Im", 1)

        # Add two dipoles at specified distance
        center = self._get_center_position()

        # Add first dipole (oriented along +z)
        self.system.add_molecule(
            Dipole(
                position=center + np.array([-distance / 2, 0, 0]),
                orientation=np.array([0, 0, 1]),
                length=dipole_length,
                type_plus=ip_type_id,
                type_plus_name="Ip",
                type_minus=im_type_id,
                type_minus_name="Im",
                type_ghost=None,
                type_ghost_name=None,
                charge=1.0,
                mass=18.01528,
                ghost_count=0,
            )
        )

        # Add second dipole (oriented along -z)
        self.system.add_molecule(
            Dipole(
                position=center + np.array([distance / 2, 0, 0]),
                orientation=np.array([0, 0, -1]),
                length=dipole_length,
                type_plus=ip_type_id,
                type_plus_name="Ip",
                type_minus=im_type_id,
                type_minus_name="Im",
                type_ghost=None,
                type_ghost_name=None,
                charge=1.0,
                mass=18.01528,
                ghost_count=0,
            )
        )

        # Update Ewald structure factors
        self._update_ewald_structure_factors()

        # Calculate energies
        computed_energy = self._compute_energy()
        # Real energy: Coulomb interaction between two antiparallel dipoles
        # For antiparallel dipoles: E = k*q²*(1/(r^2 + l^2)^0.5 - 1/r - 1/l)
        # where r is distance between centers, l is half dipole length
        real_energy = (
            2
            * 138.935456
            * (
                1 / (distance**2 + dipole_length**2) ** 0.5
                - 1 / distance
                - 1 / dipole_length
            )
        )

        return computed_energy, real_energy


def main():
    """Run the antiparallel dipoles test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test 3: Antiparallel Dipoles")
    parser.add_argument(
        "--config",
        default="configs/test_3_antiparallel_dipoles.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("TEST 3: ANTIPARALLEL DIPOLES")
    print("=" * 60)

    try:
        test = TestAntiparallelDipoles(
            config_file=args.config, results_dir=args.results_dir, verbose=args.verbose
        )

        result = test.run_test()

        print("\nEwald parameters:")
        for key, value in result["ewald_params"].items():
            print(f"{key}: {value}")

        print(f"\nTest completed successfully!")
        print(f"L2 error: {result['l2_error']:.6e}")
        print(f"Max error: {result['max_error']:.6e}")
        print(f"Mean error: {result['mean_error']:.6e}")
        print(f"Dipole length: {result['dipole_length']:.3f} nm")
        print(f"Plot saved to: {test.results_dir / 'test_3_antiparallel_dipoles.png'}")

    except Exception as e:
        print(f"Test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
