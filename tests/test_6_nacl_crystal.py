#!/usr/bin/env python3
"""
Test 6: NaCl crystal Madelung constant validation.
Validates Ewald method against Madelung constant.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
from scipy.special import lambertw

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from base_test import BaseEwaldTest


class TestNaClCrystal(BaseEwaldTest):
    """Test for NaCl crystal Madelung constant validation."""

    def __init__(self, config_file: str = "configs/test_6_nacl_crystal.yaml", **kwargs):
        """Initialize the NaCl crystal test."""
        super().__init__(config_file, **kwargs)
        self.test_name = "NaCl Crystal Madelung"

    def run_test(self) -> Dict[str, Any]:
        """Run the NaCl crystal test."""
        lattice_sizes = self.test_params.get("lattice_sizes", [2, 3, 4, 5, 6, 7, 8])
        lattice_constant = self.test_params.get("lattice_constant", 0.564)
        madelung_constant = self.test_params.get("madelung_constant", 1.74756)

        computed_energies = []
        theoretical_energies = []
        n_ion_pairs = []

        for size in lattice_sizes:
            comp_energy, theo_energy, n_pairs = self._setup_nacl_crystal(
                size, lattice_constant, madelung_constant
            )
            computed_energies.append(comp_energy)
            theoretical_energies.append(theo_energy)
            n_ion_pairs.append(n_pairs)

            if self.verbose:
                print(
                    f"  Size: {size}x{size}x{size}, Computed: {comp_energy:.6f}, Theoretical: {theo_energy:.6f}"
                )

        # Calculate errors
        errors = self._calculate_errors(computed_energies, theoretical_energies)

        # Create plot
        self._create_crystal_plot(
            lattice_sizes,
            computed_energies,
            theoretical_energies,
            "NaCl Crystal: Ewald vs Madelung Constant",
            "test_6_nacl_crystal.png",
        )

        return {
            "test_name": self.test_name,
            "lattice_sizes": lattice_sizes,
            "computed_energies": computed_energies,
            "theoretical_energies": theoretical_energies,
            "n_ion_pairs": n_ion_pairs,
            "lattice_constant": lattice_constant,
            "madelung_constant": madelung_constant,
            **errors,
            "ewald_params": self._extract_ewald_params(),
        }

    def _setup_nacl_crystal(
        self, size: int, lattice_constant: float, madelung_constant: float
    ) -> tuple[float, float, int]:
        """Setup NaCl crystal and calculate energy."""
        # Clear existing atoms
        self._clear_system()

        # Get type IDs from topology
        type_ids = self._get_type_ids()
        na_type_id = type_ids.get("Na", 0)
        cl_type_id = type_ids.get("Cl", 1)

        # Create NaCl crystal structure
        L = size * lattice_constant
        self.system.box = np.array([L, L, L])

        # Add atoms to the system in proper NaCl crystal structure
        positions = []
        types = []
        charges = []

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Na atom at (0,0,0) of each unit cell
                    pos_na = np.array([i, j, k]) * lattice_constant
                    positions.append(pos_na)
                    types.append(na_type_id)
                    charges.append(1.0)

                    # Cl atom at (0.5,0.5,0.5) of each unit cell
                    pos_cl = np.array([i + 0.5, j + 0.5, k + 0.5]) * lattice_constant
                    positions.append(pos_cl)
                    types.append(cl_type_id)
                    charges.append(-1.0)

        positions = np.array(positions)
        types = np.array(types)
        charges = np.array(charges)

        # Add atoms to system
        for i in range(len(positions)):
            self.system._add_atom(positions[i], types[i], f"Atom{i}", charges[i], 1.0)

        # Update Ewald structure factors
        self._update_ewald_structure_factors()

        # Calculate and apply optimal Ewald parameters if requested
        if self.cfg.ewald.get("use_optimal_params", False):
            if self.verbose:
                print(
                    f"  Calculating optimal Ewald parameters for {size}x{size}x{size} crystal..."
                )

            optimal_params = self._calculate_optimal_ewald_params(
                size, lattice_constant
            )

            # Update Ewald handler with optimal parameters
            if (
                hasattr(self.system, "ewald_handler")
                and self.system.ewald_handler is not None
            ):
                self.system.ewald_handler.alpha = optimal_params["alpha"]
                self.system.ewald_handler.real_cut = optimal_params["real_cut"]
                self.system.ewald_handler.n_c = optimal_params["n_c"]

                if self.verbose:
                    print(
                        f"  Applied optimal parameters: alpha={optimal_params['alpha']:.6f}, "
                        f"real_cut={optimal_params['real_cut']:.6f}, n_c={optimal_params['n_c']}"
                    )

                # Recalculate structure factors with new parameters
                self._update_ewald_structure_factors()
            else:
                if self.verbose:
                    print(
                        "  Warning: Ewald handler not initialized, using default parameters"
                    )

        # Calculate energy using Ewald
        # self.topology.recompute_caches(self.system)

        # Get individual energy components
        # pair_energy = self.system.potential_energy
        # kspace_energy = self.system.ewald_handler.compute_total_kspace_energy()
        # self_energy = self.system.ewald_handler.compute_total_self_energy(
        #     self.system.charges
        # )

        # # Total Ewald energy
        # total_ewald_energy = pair_energy + kspace_energy + self_energy

        # Calculate energies - get individual Ewald components
        pair_energy, _ = self.topology.compute_energy_virial(self.system)
        kspace_energy = self.system.ewald_handler.compute_total_kspace_energy()
        self_energy = self.system.ewald_handler.compute_total_self_energy(
            self.system.charges
        )

        # Total Ewald energy
        computed_energy = pair_energy

        if self.verbose:
            print(
                f"    Ewald components: pair={pair_energy:.6f}, kspace={kspace_energy:.6f}, self={self_energy:.6f}"
            )
            print(f"    Total Ewald energy: {computed_energy:.6f}")

        # Energy per ion pair (NaCl has 2 ions per unit cell)
        n_ion_pairs = self.system.N_atoms // 2  # Number of ion pairs (NaCl units)
        ewald_energy_per_pair = computed_energy / n_ion_pairs

        # Calculate theoretical energy from Madelung constant
        # The Madelung energy per ion pair is -M * e^2 / (4πε₀ * a)
        # In your units: -M * lB_star / lattice_const
        theoretical_energy = -138.935456 * madelung_constant / lattice_constant

        return ewald_energy_per_pair, theoretical_energy, n_ion_pairs

    def _calculate_optimal_ewald_params(
        self, size: int, lattice_constant: float
    ) -> Dict[str, float]:
        """Calculate optimal Ewald parameters using Lambert W function.

        Args:
            size: Crystal size (number of unit cells in each dimension)
            lattice_constant: Lattice constant in nm

        Returns:
            Dictionary with optimal alpha, real_cut, and n_c
        """
        # Get system properties
        L = size * lattice_constant
        V = L**3  # Volume

        # Calculate charge sum (for NaCl: 2 ions per unit cell, charges +1 and -1)
        n_unit_cells = size**3
        n_ions = n_unit_cells * 2  # 2 ions per unit cell
        Q = n_ions * 1.0**2  # Sum of squared charges (each ion has charge ±1)

        # Get parameters from config
        eps = self.cfg.ewald.get("optimal_eps", 1e-4)
        real_cut_ratio = self.cfg.ewald.get("optimal_real_cut_ratio", 0.25)
        fallback = self.cfg.ewald.get("fallback_to_standard", True)

        # Coulomb constant in your units
        kc = 138.93546  # kJ⋅nm/mol

        # Calculate real-space cutoff
        real_cut = real_cut_ratio * L

        if self.verbose:
            print(
                f"    System properties: L={L:.6f} nm, V={V:.6f} nm³, Q={Q:.6f}, eps={eps:.2e}"
            )
            print(f"    Real cutoff: {real_cut:.6f} nm")

        # Method 1: Lambert W approach
        try:
            # Calculate alpha using Lambert W function
            # alpha = W(Q * kc / eps * (r_cut / (2*V))^(1/2)) / r_cut^2
            arg1 = Q * kc / eps * (real_cut / (2 * V)) ** (1 / 2)

            if arg1 > 0:
                alpha_optimal = lambertw(arg1).real / real_cut**2

                # Calculate n_c using Lambert W function
                # n_c = sqrt(3 * alpha) * W(4/(3*V) * (Q^2 * kc^2 / (pi * alpha^(1/2) * eps^2))^(2/3))^(1/2)
                arg2 = (
                    4
                    / (3 * V)
                    * (Q**2 * kc**2 / (np.pi * alpha_optimal ** (1 / 2) * eps**2))
                    ** (2 / 3)
                )

                if arg2 > 0:
                    nc_optimal = np.sqrt(3 * alpha_optimal) * lambertw(arg2).real ** (
                        1 / 2
                    )

                    return {
                        "alpha": alpha_optimal,
                        "real_cut": real_cut,
                        "n_c": int(np.ceil(nc_optimal)),
                    }

                    # Check if results are reasonable
                    # if 0.01 <= alpha_optimal <= 10.0 and 1 <= nc_optimal <= 100:
                    #     if self.verbose:
                    #         print(
                    #             f"    Lambert W method: alpha={alpha_optimal:.6f}, n_c={nc_optimal:.6f}"
                    #         )
                    #     return {
                    #         "alpha": alpha_optimal,
                    #         "real_cut": real_cut,
                    #         "n_c": int(np.ceil(nc_optimal)),
                    #     }
                    # else:
                    #     if self.verbose:
                    #         print(
                    #             f"    Lambert W results out of range: alpha={alpha_optimal:.6f}, n_c={nc_optimal:.6f}"
                    #         )
        except Exception as e:
            if self.verbose:
                print(f"    Lambert W method failed: {e}")

        # Method 2: Standard approach (fallback)
        if fallback:
            if self.verbose:
                print("    Using standard method as fallback")

            # Standard formula: alpha = sqrt(-log(eps)) / r_cut
            alpha_standard = np.sqrt(-np.log(eps)) / real_cut

            # Calculate n_c based on alpha and desired accuracy
            k_cut = 2 * np.sqrt(alpha_standard * (-np.log(eps)))
            nc_standard = int(np.ceil(k_cut * L / (2 * np.pi)))

            if self.verbose:
                print(
                    f"    Standard method: alpha={alpha_standard:.6f}, n_c={nc_standard}"
                )

            return {"alpha": alpha_standard, "real_cut": real_cut, "n_c": nc_standard}

        # If all methods fail, return default values
        if self.verbose:
            print("    All methods failed, using default values")

        return {"alpha": 0.3, "real_cut": real_cut, "n_c": 10}


def main():
    """Run the NaCl crystal test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test 6: NaCl Crystal")
    parser.add_argument(
        "--config",
        default="configs/test_6_nacl_crystal.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("TEST 6: NACL CRYSTAL")
    print("=" * 60)

    try:
        test = TestNaClCrystal(
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
        print(f"Lattice constant: {result['lattice_constant']:.3f} nm")
        print(f"Madelung constant: {result['madelung_constant']:.6f}")
        print(f"Plot saved to: {test.results_dir / 'test_6_nacl_crystal.png'}")

    except Exception as e:
        print(f"Test failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
