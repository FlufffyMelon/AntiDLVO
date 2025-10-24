#!/usr/bin/env python3
"""
Test 1: Two Na+ and Cl- ions at varying distances.
Validates Ewald method against Coulomb interaction.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_test import BaseEwaldTest
from src.molecule import Particle
from src.utils import create_topology


class TestTwoIonsColumb(BaseEwaldTest):
    """Test for two ions at varying distances."""

    def __init__(self, config_file: str = "configs/test_1_two_ions.yaml", **kwargs):
        """Initialize the two ions test."""
        super().__init__(config_file, **kwargs)
        self.test_name = "Two Ions"

    def run_test(self) -> Dict[str, Any]:
        """Run the two ions test."""
        distances = self.test_params.get("distances", np.linspace(2, 20, 20).tolist())
        n_images = self.test_params.get("n_images", [1, 2, 3, 4, 5, 6])

        ewald_energies = []
        columb_energies = []

        for i, n_image in enumerate(n_images):
            ewald_energies.append([])
            columb_energies.append([])
            for distance in distances:
                ewald_energy = self._setup_two_ions(distance)
                columb_energy = self._compute_columb_energy(n_image)
                columb_energies[i].append(columb_energy)
                ewald_energies[i].append(ewald_energy)

            if self.verbose:
                print(
                    f"  Distance: {distance:.2f} nm, Ewald: {ewald_energy:.6f}, Columb: {columb_energies[i][-1]:.6f}"
                )

        # Calculate errors
        errors = []
        for i in range(len(n_images)):
            errors.append(self._calculate_errors(ewald_energies[i], columb_energies[i]))

        # Create plot
        self._create_energy_plot(
            distances,
            ewald_energies,
            columb_energies,
            "Two Ions: Ewald vs Coulomb (n-images)",
            "test_9_two_ions_columb.png",
        )

        return {
            "test_name": self.test_name,
            "distances": distances,
            "ewald_energies": ewald_energies,
            "columb_energies": columb_energies,
            **errors[0],
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
        ewald_energy = self._compute_energy()

        return ewald_energy

    def _compute_columb_energy(self, n_image: int) -> tuple[float, float]:
        """Calculate energy using direct Coulomb summation with periodic images through topology."""
        # Create a modified copy of the configuration with Coulomb interactions
        coulomb_cfg = copy.deepcopy(self.cfg)
        # print(coulomb_cfg.pairs_coulomb)
        for pair in coulomb_cfg.pairs_coulomb:
            for force in pair.forces:
                force["nx_images"] = n_image
                force["ny_images"] = n_image
                force["nz_images"] = n_image
        coulomb_cfg.pairs = coulomb_cfg.pairs_coulomb

        # Create a new system and topology using the modified config
        coulomb_topology = create_topology(coulomb_cfg, self.units)
        coulomb_system = copy.deepcopy(self.system)
        coulomb_system.ewald_handler = None

        # Calculate the energy using the Coulomb topology
        columb_energy, _ = coulomb_topology.compute_energy_virial(coulomb_system)

        return columb_energy

    def _create_energy_plot(
        self,
        x_values: List[float],
        ewald_energies: List[List[float]],
        columb_energies: List[List[float]],
        title: str,
        filename: str,
    ):
        """Create energy comparison plot."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            x_values, ewald_energies[0], "o-", label="Ewald", linewidth=2, markersize=6
        )
        for i in range(len(columb_energies)):
            plt.plot(
                x_values,
                columb_energies[i],
                "s-",
                label=f"Coulomb {i + 1}",
                linewidth=1,
                markersize=4,
            )

        plt.xlabel("Distance (nm)")
        plt.ylabel("Energy (kJ/mol)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        for i in range(len(columb_energies)):
            errors = np.abs(
                1 - np.array(ewald_energies[i]) / np.array(columb_energies[i])
            )
            plt.plot(x_values, errors, "ro-", linewidth=2, markersize=6)

        plt.xlabel("Distance (nm)")
        plt.ylabel("Relative Error")
        plt.title("Ewald Error")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


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
        test = TestTwoIonsColumb(
            config_file=args.config, results_dir=args.results_dir, verbose=args.verbose
        )

        result = test.run_test()

        print("\nEwald parameters:")
        for key, value in result["ewald_params"].items():
            print(f"{key}: {value}")

        print("\nTest completed successfully!")
        print(f"L2 error: {result['l2_error']:.6e}")
        print(f"Max error: {result['max_error']:.6e}")
        print(f"Mean error: {result['mean_error']:.6e}")
        print(f"Plot saved to: {test.results_dir / 'test_9_two_ions_columb.png'}")

    except Exception as e:
        print(f"Test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
