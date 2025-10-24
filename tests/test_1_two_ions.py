#!/usr/bin/env python3
"""
Test 1: Two Na+ and Cl- ions at varying distances.
Validates Ewald method against Coulomb interaction.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_test import BaseEwaldTest
from src.molecule import Particle


class TestTwoIons(BaseEwaldTest):
    """Test for two ions at varying distances."""

    def __init__(self, config_file: str = "configs/test_1_two_ions.yaml", **kwargs):
        """Initialize the two ions test."""
        super().__init__(config_file, **kwargs)
        self.test_name = "Two Ions"

    def run_test(self) -> Dict[str, Any]:
        """Run the two ions test."""
        distances = self.test_params.get("distances", np.linspace(2, 20, 20).tolist())

        computed_energies = []
        real_energies = []

        for distance in distances:
            comp_energy, real_energy = self._setup_two_ions(distance)
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
            "Two Ions: Ewald vs Coulomb",
            "test_1_two_ions.png",
        )

        return {
            "test_name": self.test_name,
            "distances": distances,
            "computed_energies": computed_energies,
            "real_energies": real_energies,
            **errors,
            "ewald_params": self._extract_ewald_params(),
        }

    def _setup_two_ions(self, distance: float) -> tuple[float, float]:
        """Setup two ions at specified distance."""
        # Clear existing atoms
        self._clear_system()

        # Get type IDs from topology
        type_ids = self._get_type_ids()
        na_type_id = type_ids.get("Na", 0)
        cl_type_id = type_ids.get("Cl", 1)

        # Add two ions at specified distance
        center = self._get_center_position()

        # Add first ion (Na+)
        self.system.add_molecule(
            Particle(
                position=center + np.array([-distance / 2, 0, 0]),
                type_id=na_type_id,
                name="Na",
                charge=1.0,
                mass=39.0983,
            )
        )

        # Add second ion (Cl-)
        self.system.add_molecule(
            Particle(
                position=center + np.array([distance / 2, 0, 0]),
                type_id=cl_type_id,
                name="Cl",
                charge=-1.0,
                mass=35.453,
            )
        )

        # Update Ewald structure factors
        self._update_ewald_structure_factors()

        # Calculate energies
        computed_energy = self._compute_energy()
        # Real energy: Coulomb interaction between two point charges
        # E = k * q1 * q2 / r, where k = 138.935456 kJ⋅nm/mol⋅e²
        real_energy = -138.935456 / distance

        return computed_energy, real_energy


def main():
    """Run the two ions test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test 1: Two Ions")
    parser.add_argument(
        "--config",
        default="configs/test_1_two_ions.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("TEST 1: TWO IONS")
    print("=" * 60)

    try:
        test = TestTwoIons(
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
        print(f"Plot saved to: {test.results_dir / 'test_1_two_ions.png'}")

    except Exception as e:
        print(f"Test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
