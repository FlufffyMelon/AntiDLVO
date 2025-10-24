#!/usr/bin/env python3
"""
Test 8: System with ions and dipoles in 2D periodic arrangement.
Compares Ewald method against direct Coulomb summation with varying number of image cells
in the X and Y dimensions only.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import copy

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_test import BaseEwaldTest
from src.molecule import Particle, Dipole
from src.forces.coulomb import Coulomb
from src.forces import EwaldReal
from src.utils import create_topology, create_system, create_units, _init_random_configuration


class Test2DPeriodicSystem(BaseEwaldTest):
    """Test for 2D periodic system with ions and dipoles."""

    def __init__(
        self, config_file: str = "configs/test_8_2d_periodic_system.yaml", **kwargs
    ):
        """Initialize the 2D periodic system test."""
        super().__init__(config_file, **kwargs)
        self.test_name = "2D Periodic System"

    def run_test(self) -> Dict[str, Any]:
        """Run the 2D periodic system test."""
        # Get test parameters
        image_counts = self.test_params.get("image_counts", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Setup the 2D periodic system once
        self._setup_periodic_system()

        # Calculate Ewald energy as reference
        ewald_energy = self._compute_energy()
        if self.verbose:
            print(f"Ewald energy: {ewald_energy:.6f}")

        # Calculate direct Coulomb energy with varying image counts (2D only)
        coulomb_energies = []
        for n_images in image_counts:
            coulomb_energy = self._calculate_coulomb_energy(n_images)
            coulomb_energies.append(coulomb_energy)

            if self.verbose:
                print(f"  Images: {n_images}, Coulomb energy: {coulomb_energy:.6f}, Diff: {coulomb_energy - ewald_energy:.6f}")

        # Calculate errors
        errors = self._calculate_errors(coulomb_energies, [ewald_energy] * len(coulomb_energies))

        # Create plot
        self._create_image_count_plot(
            image_counts,
            coulomb_energies,
            ewald_energy,
            "2D Periodic System: Coulomb vs Ewald",
            "test_8_2d_periodic_system.png",
        )

        return {
            "test_name": self.test_name,
            "image_counts": image_counts,
            "coulomb_energies": coulomb_energies,
            "ewald_energy": ewald_energy,
            **errors,
            "ewald_params": self._extract_ewald_params(),
        }

    def _setup_periodic_system(self):
        """Setup the 2D periodic system with ions and dipoles randomly distributed."""
        # Clear existing atoms
        self._clear_system()

        # Set random seed for reproducible placement
        random_seed = self.test_params.get("random_seed", 42)
        np.random.seed(random_seed)

        # Get number of particles from config
        n_ions = self.test_params.get("n_ions", 8)
        n_na_ions = n_ions // 2  # Half Na+ ions
        n_cl_ions = n_ions - n_na_ions  # Half Cl- ions
        n_dipoles = self.test_params.get("n_dipoles", 8)
        dipole_length = self.test_params.get("dipole_length", 1.0)

        # Prepare type specifications for random configuration
        processed_specs = []

        # Add Na+ ions
        processed_specs.append(
            ("Particle", {
                "type": "Na",
                "name": "Na",
                "mass": 39.0983,
                "charge": 1.0
            }, n_na_ions)
        )

        # Add Cl- ions
        processed_specs.append(
            ("Particle", {
                "type": "Cl",
                "name": "Cl",
                "mass": 35.453,
                "charge": -1.0
            }, n_cl_ions)
        )

        # For 2D test, we'll use custom dipoles with orientations in the xy-plane
        # First, we need to create a custom dipole class that will orient in xy-plane
        class XYPlaneDipole(Dipole):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Override orientation to be in xy-plane
                theta = np.random.uniform(0, 2 * np.pi)
                self.orientation = np.array([np.cos(theta), np.sin(theta), 0])
                self.orientation = self.orientation / np.linalg.norm(self.orientation)

        # Store the original Dipole class to restore it later
        original_dipole = sys.modules["src.molecule"].Dipole

        # Replace with our custom class
        sys.modules["src.molecule"].Dipole = XYPlaneDipole

        # Add dipoles with xy-plane orientation
        processed_specs.append(
            ("Dipole", {
                "type_plus": "Ip",
                "type_minus": "Im",
                "type_ghost": "G",
                "length": dipole_length,
                "mass": 18.01528,
                "charge": 1.0,
                "ghost_count": 0
            }, n_dipoles)
        )

        # Calculate safe padding and min_distance to avoid overlaps
        min_distance = 0.5  # Minimum distance between particles
        padding = np.array([0.5, 0.5, 0.5])  # Padding from box edges

        # Use the existing _init_random_configuration function to place molecules
        _init_random_configuration(
            self.system,
            processed_specs,
            self.topology.type_name_to_id,
            min_distance,
            padding
        )

        # Restore original Dipole class
        sys.modules["src.molecule"].Dipole = original_dipole

        if self.verbose:
            print(f"System initialized with {n_ions} ions and {n_dipoles} dipoles")
            print(f"Total atom count: {self.system.N_atoms}")

        # Update Ewald structure factors
        self._update_ewald_structure_factors()

    def _calculate_coulomb_energy(self, n_images: int) -> float:
        """Calculate energy using direct Coulomb summation with periodic images in 2D through topology."""
        # Create a modified copy of the configuration with Coulomb interactions
        coulomb_cfg = copy.deepcopy(self.cfg)

        # Modify the configuration to use Coulomb forces with the specified image count
        if "pairs" in coulomb_cfg:
            for pair in coulomb_cfg.pairs:
                # Replace forces list entirely
                new_forces = []
                for force in pair.forces:
                    if force.get("kind", "").lower() in ("ewald", "ewald_real"):
                        new_forces.append({
                            "kind": "coulomb",
                            "nx_images": n_images,
                            "ny_images": n_images,
                            "nz_images": 0,  # No images in z-direction for 2D periodicity
                        })

                # Only if we found forces to replace, update the pair's forces
                if new_forces:
                    pair.forces = new_forces

        # Create a new system and topology using the modified config
        coulomb_system = copy.deepcopy(self.system)
        coulomb_topology = create_topology(coulomb_cfg, self.units)

        # Calculate the energy using the Coulomb topology
        energy, _ = coulomb_topology.compute_energy_virial(coulomb_system)

        return energy

    def _create_image_count_plot(
        self,
        image_counts: List[int],
        coulomb_energies: List[float],
        ewald_energy: float,
        title: str,
        filename: str,
    ):
        """Create plot comparing Coulomb energies with different image counts to Ewald energy."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            image_counts, coulomb_energies, "o-", label="Coulomb (2D)", linewidth=2, markersize=6
        )
        plt.axhline(
            y=ewald_energy, color="r", linestyle="--", label="Ewald (2D)", linewidth=2
        )
        plt.xlabel("Number of Images in X and Y")
        plt.ylabel("Energy (kJ/mol)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        relative_errors = np.abs((np.array(coulomb_energies) - ewald_energy) / ewald_energy)
        plt.plot(image_counts, relative_errors, "bo-", linewidth=2, markersize=6)
        plt.xlabel("Number of Images in X and Y")
        plt.ylabel("Relative Error")
        plt.title("Convergence to Ewald")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Run the 2D periodic system test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test 8: 2D Periodic System")
    parser.add_argument(
        "--config",
        default="configs/test_8_2d_periodic_system.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("TEST 8: 2D PERIODIC SYSTEM")
    print("=" * 60)

    try:
        test = Test2DPeriodicSystem(
            config_file=args.config, results_dir=args.results_dir, verbose=args.verbose
        )

        result = test.run_test()

        print("\nEwald parameters:")
        for key, value in result["ewald_params"].items():
            print(f"{key}: {value}")

        print(f"\nTest completed successfully!")
        print(f"Ewald energy: {result['ewald_energy']:.6f}")
        print(f"L2 error: {result['l2_error']:.6e}")
        print(f"Max error: {result['max_error']:.6e}")
        print(f"Mean error: {result['mean_error']:.6e}")
        print(f"Plot saved to: {test.results_dir / 'test_8_2d_periodic_system.png'}")

    except Exception as e:
        print(f"Test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
