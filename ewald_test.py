#!/usr/bin/env python3
"""
Test script to validate Ewald summation implementation by:
1. Comparing with analytical solution for NaCl crystal (Madelung constant)
2. Comparing with direct Coulomb summation with many periodic images
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.units import Units
from src.system import System
from src.topology import Topology
from src.ewald import EwaldHandler
from src.forces import Coulomb, EwaldReal
from scipy.special import erfcinv
from scipy.optimize import root_scalar


def madelung_constant_nacl():
    """
    Return the Madelung constant for NaCl structure.
    This is a well-established value used as reference.
    """
    return 1.74756  # Precise value for NaCl


def setup_nacl_system(size=4, lattice_const=1.0):
    """
    Create a simple cubic NaCl-like crystal system.

    Args:
        size: Number of unit cells in each dimension
        lattice_const: Lattice constant

    Returns:
        System object with NaCl crystal structure
    """
    units = Units(system="standard")

    # Create system with NaCl crystal structure
    L = size * lattice_const
    box = np.array([L, L, L])
    # Initialize with capacity for all atoms
    system = System(
        box=box,
        pbc=[True, True, True],
        units=units,
        temp=1.0,
        initial_capacity=size**3 * 2,
    )  # 2 atoms per unit cell

    # Set up Ewald handler
    # alpha = 1.2  # Following the recommendation in config file
    # # real_cut = 0.45 * L  # Similar to the config file
    # real_cut = 1.4
    # ewald_handler = EwaldHandler(
    #     alpha=alpha,
    #     real_cut=real_cut,
    #     n_c=7,  # More k-vectors for accuracy
    #     dielectric=1.0,
    #     units=units,
    # )
    # system.ewald_handler = ewald_handler

    # Create topology
    topology = Topology(units=units)

    # Register atom types
    topology.register_type(0, "Na")
    topology.register_type(1, "Cl")

    # Set type properties
    topology.set_type_properties(0, {"mass": 1.0, "charge": 1.0})
    topology.set_type_properties(1, {"mass": 1.0, "charge": -1.0})

    # Add Ewald force to topology
    # ewald_force = EwaldReal(units)
    # topology.add_interaction(0, 0, ewald_force)
    # topology.add_interaction(0, 1, ewald_force)
    # topology.add_interaction(1, 1, ewald_force)

    # Add atoms to the system in NaCl crystal structure
    positions = []
    types = []
    charges = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                # Na atom
                pos_na = np.array([i, j, k]) * lattice_const
                positions.append(pos_na)
                types.append(0)
                charges.append(1.0)

                # Cl atom
                pos_cl = np.array([i + 0.5, j + 0.5, k + 0.5]) * lattice_const
                positions.append(pos_cl)
                types.append(1)
                charges.append(-1.0)

    positions = np.array(positions)
    types = np.array(types)
    charges = np.array(charges)

    # Add atoms to system
    for i in range(len(positions)):
        system.add_atom(positions[i], types[i], f"Atom{i}", charges[i], 1.0)

    # Initialize k-vectors and structure factors
    # ewald_handler.initialize_k_vectors(box)
    # ewald_handler.update_structure_factors(positions, charges)
    # ewald_handler.update_dipole_moment(positions, charges)

    return system, topology


def test_nacl_madelung():
    """
    Test Ewald summation against the analytical Madelung constant for NaCl.
    """
    print("Testing Ewald summation with NaCl Madelung constant...")

    # eps = 1e-8
    # inv_eps = erfcinv(eps / 2)
    # for nc in range(1, 10):
    for nc in [1]:
        for alpha in [0.5]:
            # for alpha in [1]:
            # Setup a NaCl crystal
            size = 8
            lattice_const = 0.564
            system, topology = setup_nacl_system(size, lattice_const)
            print("System size:", system.N_atoms)

            # Set up Ewald handler
            # real_cut = 0.45 * system.box[0]  # Similar to the config file
            # real_cut = 1.4
            # alpha = inv_eps / real_cut  # Following the recommendation in config file
            # nc = system.box[0] * (2 * alpha * np.sqrt(np.log(1 / eps))) / (2 * np.pi)

            # real_cut = 0.45 * system.box[0]
            # alpha = 3.5 / real_cut
            # k_max = 7.5 * alpha
            # nc = k_max * system.box[0] / (2 * np.pi)

            def solve_for_s(eps):
                # Define the function whose root we want: f(s) = exp(-s^2)/s^2 - eps
                def f(s):
                    if s <= 0:
                        return np.inf
                    return np.exp(-(s**2)) / s**2 - eps

                # Initial guess: for small eps, s is large; try s in [1, 20]
                sol = root_scalar(f, bracket=[1e-6, 20], method="brentq")
                return sol.root

            eps = 1e-2
            alpha = (np.pi**3 * system.N_atoms / system.box[0] ** 6) ** (1 / 6)
            s = solve_for_s(eps)
            real_cut = s / alpha
            nc = s * system.box[0] * alpha / np.pi
            print(f"alpha: {alpha}, real_cut: {real_cut}, nc: {nc}")

            ewald_handler = EwaldHandler(
                alpha=alpha,
                real_cut=real_cut,
                n_c=nc,  # More k-vectors for accuracy
                dielectric=1.0,
                units=system.units,
            )
            system.ewald_handler = ewald_handler

            # Initialize k-vectors and structure factors
            ewald_handler.initialize_k_vectors(system.box)
            ewald_handler.update_structure_factors(system.positions, system.charges)
            ewald_handler.update_dipole_moment(system.positions, system.charges)

            ewald_force = EwaldReal(system.units)
            topology.add_interaction(0, 0, ewald_force)
            topology.add_interaction(0, 1, ewald_force)
            topology.add_interaction(1, 1, ewald_force)

            # Calculate energy using Ewald
            topology.recompute_caches(system)
            # print("pair_energy", system.potential_energy)
            # print("kspace_energy", ewald_handler.compute_total_kspace_energy())
            # print(
            #     "self_energy", ewald_handler.compute_total_self_energy(system.charges)
            # )
            n_ion_pairs = 2 * size**3
            ewald_energy = (
                system.potential_energy
                + ewald_handler.compute_total_kspace_energy()
                + ewald_handler.compute_total_self_energy(system.charges)
            ) / n_ion_pairs

            # Calculate theoretical energy from Madelung constant
            madelung = madelung_constant_nacl()
            theoretical_energy = -ewald_handler.lB_star * madelung / lattice_const

            print(f"nc: {nc}, alpha: {alpha}")
            print(f"Ewald energy: {ewald_energy:.6f}")
            print(f"Theoretical energy (Madelung): {theoretical_energy:.6f}")
            print(
                f"Relative error: {abs(ewald_energy - theoretical_energy) / abs(theoretical_energy) * 100:.6f}%"
            )
            print(f"--------------------------------")
            print()

    return ewald_energy, theoretical_energy


def test_direct_coulomb_convergence(max_size=11):
    """
    Test convergence of direct Coulomb summation with increasing number of periodic images.
    Compare with Ewald summation result.
    """
    print("\nTesting direct Coulomb summation convergence...")

    lattice_const = 0.564
    # Setup a smaller system for this test
    for size in range(1, max_size):
        system, topology, ewald_handler = setup_nacl_system(size, lattice_const)

        # Get positions and charges
        positions, types, names, charges, masses = system.get_active_atoms()

        coulomb_force = Coulomb(system.units, cutoff=0.45 * system.box[0])
        topology.add_interaction(0, 0, coulomb_force)
        topology.add_interaction(0, 1, coulomb_force)
        topology.add_interaction(1, 1, coulomb_force)

        # Calculate direct Coulomb energy with increasing lattice size
        madelung = madelung_constant_nacl()
        theoretical_energy = -ewald_handler.lB_star * madelung / lattice_const

        print(f"Theoretical energy (Madelung): {theoretical_energy:.6f}")

        energy, force, virial = topology._single_energy_forces_virial(
            positions[0, :], 0, charges[0], system, exclude_index=0
        )
        print(
            f"Direct Coulomb with lattice size {size}: {energy:.6f}, error: {abs(energy - ewald_energy) / abs(ewald_energy) * 100:.2f}%"
        )

    # Plot convergence
    # plt.figure(figsize=(10, 6))
    # plt.plot(image_counts, direct_energies, "o-", label="Direct Coulomb")
    # plt.axhline(y=ewald_energy, color="r", linestyle="--", label="Ewald energy")
    # plt.xlabel("Number of images in each direction")
    # plt.ylabel("Energy")
    # plt.title("Convergence of direct Coulomb summation")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("ewald_convergence.png")


if __name__ == "__main__":
    # Test Madelung constant comparison
    ewald_energy, theoretical_energy = test_nacl_madelung()

    # Test direct Coulomb convergence
    # test_direct_coulomb_convergence(max_size=17)

    print("\nTests completed! Check the generated plots for visualization.")
