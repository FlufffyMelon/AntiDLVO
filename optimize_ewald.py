#!/usr/bin/env python3
"""
Ewald parameter optimization script.
Optimizes alpha, real_cut, and n_c parameters to minimize L2 norm error
between computed and real Coulomb energies.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution, root_scalar
from scipy.special import lambertw
import matplotlib.pyplot as plt
from contextlib import redirect_stdout, redirect_stderr
import io
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
from src.units import Units
from src.utils import (
    load_config,
    create_units,
    create_system,
    create_topology,
)
from src.molecule import Particle


def calculate_theoretical_ewald_params(system):
    """
    Calculate theoretical Ewald parameters based on the system properties.

    Args:
        system: System object with box dimensions and number of atoms

    Returns:
        dict with theoretical alpha, real_cut, and n_c
    """
    # Get actual charges from the system
    positions, molecule_ids, types, names, charges, masses = system.get_active_atoms()
    Q = np.sum(charges**2)  # Sum of squared charges
    V = system.get_volume()

    # For two-ion system, we need to be more careful with the formulas
    # Let's use a more standard approach for Ewald parameter optimization

    # Method 1: Standard Ewald parameter estimation
    # For a two-ion system, we can use simpler formulas

    # Estimate alpha based on box size and desired accuracy
    eps = 1e-4  # Desired accuracy
    L = min(system.box)  # Use smallest box dimension

    # Standard formula: alpha = sqrt(-log(eps)) / r_cut
    # But we need to choose r_cut first
    real_cut = L / 4  # Use 1/4 of box size as cutoff

    # Calculate alpha using standard formula
    alpha = np.sqrt(-np.log(eps)) / real_cut

    # Calculate n_c based on alpha and desired accuracy
    # k_cut = 2*pi*n_c/L should be large enough to capture the Gaussian
    k_cut = 2 * np.sqrt(alpha * (-np.log(eps)))
    nc = int(np.ceil(k_cut * L / (2 * np.pi)))

    print(f"Standard method: alpha={alpha:.6f}, real_cut={real_cut:.6f}, n_c={nc}")

    # Alternative: Try the Lambert W approach with proper units
    try:
        # Use proper charge sum and units
        kc = 138.93546  # kJ⋅nm/mol (Coulomb constant in your units)
        eps_alt = 1e-4

        # For two-ion system: Q = q1^2 + q2^2 = 1^2 + (-1)^2 = 2
        Q_proper = Q  # This is already the sum of squared charges

        # Try the Lambert W approach with corrected units
        # Note: This might not be directly applicable to two-ion systems
        arg1_alt = Q_proper * kc / eps_alt * (real_cut / (2 * V)) ** (1/2)

        if arg1_alt > 0:  # Lambert W requires positive argument
            alpha_alt = lambertw(arg1_alt).real / real_cut**2

            arg2_alt = 4 / (3 * V) * (Q_proper**2 * kc**2 / (np.pi * alpha_alt**(1/2) * eps_alt**2)) ** (2/3)
            if arg2_alt > 0:
                nc_alt = np.sqrt(3 * alpha_alt) * lambertw(arg2_alt).real ** (1/2)

                print(f"Lambert W method: alpha={alpha_alt:.6f}, real_cut={real_cut:.6f}, n_c={nc_alt}")

                # Use Lambert W results if they seem reasonable
                if 0.01 <= alpha_alt <= 1.0 and 1 <= nc_alt <= 50:
                    alpha, nc = alpha_alt, nc_alt
                    print("Using Lambert W results")
    except Exception as e:
        print(f"Lambert W method failed: {e}")
        print("Using standard method results")

    return {
        'alpha': alpha,
        'real_cut': real_cut,
        'n_c': nc
    }


def compute_energy_with_params(a, alpha, real_cut, n_c, config_file, overrides=None):
    """
    Compute energy for given distance 'a' and Ewald parameters.
    Suppresses all output to avoid console spam.
    """
    # Suppress all output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            # Create overrides for Ewald parameters
            ewald_overrides = [
                f"ewald.alpha={alpha}",
                f"ewald.real_cut={real_cut}",
                f"ewald.n_c={n_c}"
            ]

            if overrides:
                ewald_overrides.extend(overrides)

            cfg = load_config(config_file, overrides=ewald_overrides)

            # Create components
            units = create_units(cfg)
            system = create_system(cfg, units)
            topology = create_topology(cfg, units)

            Lx, Ly, Lz = system.box
            system.add_molecule(
                Particle(
                    position=np.array([Lx/2, Ly/2, Lz/2]) + np.array([-a/2, 0, 0]),
                    type_id=0,
                    name="Na",
                    charge=1.0,
                    mass=1.0
                )
            )

            system.add_molecule(
                Particle(
                    position=np.array([Lx/2, Ly/2, Lz/2]) + np.array([a/2, 0, 0]),
                    type_id=1,
                    name="Cl",
                    charge=-1.0,
                    mass=1.0
                )
            )

            system.ewald_handler.update_structure_factors(
                system.positions[: system.N_atoms],
                system.charges[: system.N_atoms],
            )
            system.ewald_handler.update_dipole_moment(
                system.positions[: system.N_atoms],
                system.charges[: system.N_atoms],
            )

            computed_energy = topology.get_energy(system)
            real_energy = -138.935456 / a

            return computed_energy, real_energy

        except Exception as e:
            # Return large error values if computation fails
            return float('inf'), float('inf')


def compute_energy_for_distance(args):
    """
    Wrapper function for parallel computation of energy at a single distance.
    This function is used by multiprocessing.Pool.
    """
    a, alpha, real_cut, n_c, config_file, overrides = args
    return compute_energy_with_params(a, alpha, real_cut, n_c, config_file, overrides)


def objective_function(params, a_values, config_file, overrides=None, progress_bar=None, n_workers=1):
    """
    Objective function to minimize: L2 norm of error between computed and real energies.

    Args:
        params: [alpha, real_cut, n_c] - Ewald parameters
        a_values: List of distances to test
        config_file: Path to configuration file
        overrides: Additional configuration overrides
        progress_bar: Optional progress bar to update
        n_workers: Number of parallel workers (1 = sequential)

    Returns:
        L2 norm of the error
    """
    alpha, real_cut, n_c = params

    # Ensure n_c is an integer
    n_c = int(round(n_c))

    # Ensure parameters are within reasonable bounds
    if alpha <= 0 or real_cut <= 0 or n_c <= 0:
        return float('inf')

    if n_workers > 1 and len(a_values) > 1:
        # Parallel computation
        try:
            # Prepare arguments for parallel processing
            args_list = [(a, alpha, real_cut, n_c, config_file, overrides) for a in a_values]

            with Pool(processes=n_workers) as pool:
                results = pool.map(compute_energy_for_distance, args_list)

            computed_energies = []
            real_energies = []

            for computed_energy, real_energy in results:
                if computed_energy == float('inf') or real_energy == float('inf'):
                    return float('inf')
                computed_energies.append(computed_energy)
                real_energies.append(real_energy)

        except Exception as e:
            # Fallback to sequential computation if parallel fails
            print(f"Parallel computation failed, falling back to sequential: {e}")
            n_workers = 1

    if n_workers == 1:
        # Sequential computation
        computed_energies = []
        real_energies = []

        for a in a_values:
            computed_energy, real_energy = compute_energy_with_params(
                a, alpha, real_cut, n_c, config_file, overrides
            )

            if computed_energy == float('inf') or real_energy == float('inf'):
                return float('inf')

            computed_energies.append(computed_energy)
            real_energies.append(real_energy)

            # Update progress bar if provided
            if progress_bar:
                progress_bar.update(1)

    # Calculate L2 norm of error
    computed_energies = np.array(computed_energies)
    real_energies = np.array(real_energies)
    error = np.linalg.norm(computed_energies - real_energies)

    return error


def optimize_ewald_parameters(config_file, overrides=None, method='differential_evolution',
                             max_iterations=50, population_size=10, tolerance=1e-4, quick_mode=False, n_workers=1):
    """
    Optimize Ewald parameters to minimize L2 norm error.

    Args:
        config_file: Path to configuration file
        overrides: Additional configuration overrides
        method: Optimization method ('differential_evolution' or 'minimize')
        max_iterations: Maximum number of iterations
        population_size: Population size for differential evolution
        tolerance: Convergence tolerance
        quick_mode: Use reduced parameters for faster testing
        n_workers: Number of parallel workers for computation

    Returns:
        Optimized parameters and results
    """
    # Distance values to test (reduced set for faster computation)
    if quick_mode:
        a_values = [1, 3, 7]  # Very minimal set for quick testing
        max_iterations = min(max_iterations, 20)  # Limit iterations in quick mode
        population_size = min(population_size, 5)  # Smaller population
        tolerance = 1e-3
    else:
        # a_values = [0.5, 1, 2, 3, 5, 7, 10]
        a_values = np.linspace(0.5, 10, 20)

    # Parameter bounds: [alpha, real_cut, n_c]
    # alpha: typically 0.01 to 1.0
    # real_cut: typically 1.0 to 10.0
    # n_c: typically 5 to 50 (integer)
    bounds = [
        (0.01, 1.0),    # alpha
        (0.1, 5.0),    # real_cut
        (1, 20)         # n_c (will be rounded to integer)
    ]

    print("Starting Ewald parameter optimization...")
    print(f"Testing distances: {a_values}")
    print(f"Parameter bounds: alpha={bounds[0]}, real_cut={bounds[1]}, n_c={bounds[2]}")
    print(f"Optimization method: {method}")
    print(f"Max iterations: {max_iterations}, Population size: {population_size}")
    print(f"Tolerance: {tolerance}")
    print(f"Parallel workers: {n_workers}")

    # Calculate and display theoretical parameters
    print("\n" + "="*50)
    print("THEORETICAL EWALD PARAMETER PREDICTION")
    print("="*50)

    # Create a temporary system to calculate theoretical parameters
    # We need to create a system with the same box size but with 2 atoms (Na and Cl)
    temp_cfg = load_config(config_file, overrides=overrides)
    temp_units = create_units(temp_cfg)
    temp_system = create_system(temp_cfg, temp_units)

    # Add two atoms to the system for theoretical calculation
    Lx, Ly, Lz = temp_system.box
    temp_system.add_molecule(
        Particle(
            position=np.array([Lx/2, Ly/2, Lz/2]) + np.array([-1, 0, 0]),
            type_id=0,
            name="Na",
            charge=1.0,
            mass=1.0
        )
    )
    temp_system.add_molecule(
        Particle(
            position=np.array([Lx/2, Ly/2, Lz/2]) + np.array([1, 0, 0]),
            type_id=1,
            name="Cl",
            charge=-1.0,
            mass=1.0
        )
    )

    theoretical_params = calculate_theoretical_ewald_params(temp_system)
    print(f"Theoretical alpha: {theoretical_params['alpha']:.6f}")
    print(f"Theoretical real_cut: {theoretical_params['real_cut']:.6f}")
    print(f"Theoretical n_c: {theoretical_params['n_c']:.2f}")
    print("="*50)
    print()

    # Create progress bar for total evaluations
    total_evaluations = max_iterations * population_size if method == 'differential_evolution' else max_iterations * 5
    pbar = tqdm(total=total_evaluations, desc="Optimizing parameters", unit="eval")

    def objective_with_progress(params):
        return objective_function(params, a_values, config_file, overrides, pbar, n_workers)

    start_time = time.time()

    if method == 'differential_evolution':
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective_with_progress,
            bounds,
            seed=42,
            maxiter=max_iterations,
            popsize=population_size,
            atol=tolerance,
            tol=tolerance,
            workers=1,  # Keep scipy workers=1, we handle parallelism in objective function
            updating='deferred',
            callback=lambda xk, convergence: pbar.set_postfix({
                'best_error': f"{convergence:.2e}" if convergence < float('inf') else "inf"
            })
        )
    else:
        # Use local optimization with multiple starting points
        best_result = None
        best_error = float('inf')

        # Try multiple starting points
        initial_guesses = [
            [0.1, 2.0, 10],   # Default-like values
            [0.2, 3.0, 15],   # Higher alpha
            [0.05, 1.5, 8],   # Lower alpha
            [0.15, 4.0, 20],  # Higher cutoffs
            [0.3, 2.5, 12],   # Medium values
        ]

        for i, x0 in enumerate(initial_guesses):
            print(f"Trying initial guess {i+1}/{len(initial_guesses)}: {x0}")

            result = minimize(
                objective_with_progress,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': max_iterations, 'ftol': tolerance}
            )

            if result.fun < best_error:
                best_error = result.fun
                best_result = result

        result = best_result

    pbar.close()
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.2f} seconds")

    # Extract best parameters regardless of success status
    if hasattr(result, 'x') and result.x is not None:
        optimal_alpha, optimal_real_cut, optimal_n_c = result.x
        optimal_n_c = int(round(optimal_n_c))
        best_error = result.fun if hasattr(result, 'fun') and result.fun is not None else float('inf')

        if result.success:
            print("Optimization completed successfully!")
        else:
            print("Optimization did not converge, but best result found:")
            print(f"Message: {result.message}")

        print(f"Best parameters found:")
        print(f"  alpha = {optimal_alpha:.6f}")
        print(f"  real_cut = {optimal_real_cut:.6f}")
        print(f"  n_c = {optimal_n_c}")
        print(f"  L2 error = {best_error:.6e}")

        return {
            'alpha': optimal_alpha,
            'real_cut': optimal_real_cut,
            'n_c': optimal_n_c,
            'error': best_error,
            'success': result.success,
            'message': result.message if hasattr(result, 'message') else "Unknown",
            'theoretical_alpha': theoretical_params['alpha'],
            'theoretical_real_cut': theoretical_params['real_cut'],
            'theoretical_n_c': theoretical_params['n_c']
        }
    else:
        print("Optimization failed completely - no valid result found!")
        print(f"Message: {result.message if hasattr(result, 'message') else 'Unknown error'}")
        return {
            'alpha': None,
            'real_cut': None,
            'n_c': None,
            'error': float('inf'),
            'success': False,
            'message': result.message if hasattr(result, 'message') else 'Unknown error',
            'theoretical_alpha': theoretical_params['alpha'],
            'theoretical_real_cut': theoretical_params['real_cut'],
            'theoretical_n_c': theoretical_params['n_c']
        }


def plot_results(optimal_params, config_file, overrides=None, n_workers=1):
    """
    Plot comparison between computed and real energies using optimal parameters.
    """
    if optimal_params['alpha'] is None:
        print("Cannot plot results - no valid parameters found")
        return

    a_values = [0.5, 1, 1.5, 2, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    computed_energies = []
    real_energies = []

    print("\nComputing energies with optimal parameters for plotting...")

    if n_workers > 1 and len(a_values) > 1:
        # Parallel computation for plotting
        try:
            args_list = [(a, optimal_params['alpha'], optimal_params['real_cut'],
                         optimal_params['n_c'], config_file, overrides) for a in a_values]

            with Pool(processes=n_workers) as pool:
                results = pool.map(compute_energy_for_distance, args_list)

            for computed_energy, real_energy in results:
                computed_energies.append(computed_energy)
                real_energies.append(real_energy)

        except Exception as e:
            print(f"Parallel plotting failed, falling back to sequential: {e}")
            n_workers = 1

    if n_workers == 1:
        # Sequential computation for plotting
        with tqdm(a_values, desc="Computing energies for plot") as pbar:
            for a in pbar:
                computed_energy, real_energy = compute_energy_with_params(
                    a,
                    optimal_params['alpha'],
                    optimal_params['real_cut'],
                    optimal_params['n_c'],
                    config_file,
                    overrides
                )
                computed_energies.append(computed_energy)
                real_energies.append(real_energy)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, computed_energies, 'o-', label='Ewald (optimized)', linewidth=2, markersize=6)
    plt.plot(a_values, real_energies, 's-', label='Coulomb (exact)', linewidth=2, markersize=6)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Energy (kJ/mol)')
    plt.title(f'Ewald vs Coulomb Energy Comparison\n'
              f'α={optimal_params["alpha"]:.4f}, '
              f'rcut={optimal_params["real_cut"]:.2f}, '
              f'nc={optimal_params["n_c"]}, '
              f'L2 error={optimal_params["error"]:.2e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ewald_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved as 'ewald_optimization_results.png'")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize Ewald parameters to minimize L2 norm error",
        add_help=True
    )
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument(
        "--method",
        choices=['differential_evolution', 'minimize'],
        default='differential_evolution',
        help="Optimization method to use"
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        help="Generate comparison plot after optimization"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum number of optimization iterations (default: 50)"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Population size for differential evolution (default: 10)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Convergence tolerance (default: 1e-4)"
    )
    parser.add_argument(
        "--quick",
        action='store_true',
        help="Quick test mode with reduced iterations and fewer distance points"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=f"Number of parallel workers (default: 1, max: {cpu_count()})"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional overrides as dotlist, e.g. system.size=20 or --system.size 20",
    )

    args = parser.parse_args()

    # Validate workers argument
    max_workers = cpu_count()
    if args.workers < 1:
        print(f"Warning: Number of workers must be >= 1, setting to 1")
        args.workers = 1
    elif args.workers > max_workers:
        print(f"Warning: Number of workers ({args.workers}) exceeds available CPUs ({max_workers}), setting to {max_workers}")
        args.workers = max_workers

    # Normalize overrides into dotlist format
    overrides = []
    if args.overrides:
        pending_key = None
        for token in args.overrides:
            if token.startswith("--"):
                pending_key = token[2:]
            elif "=" in token and pending_key is None:
                overrides.append(token)
            elif pending_key is not None:
                overrides.append(f"{pending_key}={token}")
                pending_key = None
    try:
        # Run optimization
        optimal_params = optimize_ewald_parameters(
            args.config,
            overrides=overrides,
            method=args.method,
            max_iterations=args.max_iterations,
            population_size=args.population_size,
            tolerance=args.tolerance,
            quick_mode=args.quick,
            n_workers=args.workers
        )

        # Save results to file
        with open('ewald_optimization_results.txt', 'w') as f:
            f.write("Ewald Parameter Optimization Results\n")
            f.write("=" * 40 + "\n\n")

            # Add theoretical parameters for comparison
            f.write("THEORETICAL PREDICTIONS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Theoretical alpha: {optimal_params.get('theoretical_alpha', 'N/A'):.6f}\n")
            f.write(f"Theoretical real_cut: {optimal_params.get('theoretical_real_cut', 'N/A'):.6f}\n")
            f.write(f"Theoretical n_c: {optimal_params.get('theoretical_n_c', 'N/A'):.2f}\n\n")

            f.write("OPTIMIZATION RESULTS:\n")
            f.write("-" * 20 + "\n")
            if optimal_params['alpha'] is not None:
                f.write(f"Best alpha found: {optimal_params['alpha']:.6f}\n")
                f.write(f"Best real_cut found: {optimal_params['real_cut']:.6f}\n")
                f.write(f"Best n_c found: {optimal_params['n_c']}\n")
                f.write(f"L2 error: {optimal_params['error']:.6e}\n")
                f.write(f"Convergence: {'Success' if optimal_params['success'] else 'Failed'}\n")
                if not optimal_params['success']:
                    f.write(f"Message: {optimal_params.get('message', 'Unknown')}\n")
            else:
                f.write("Optimization failed completely - no valid parameters found!\n")
                f.write(f"Message: {optimal_params.get('message', 'Unknown error')}\n")

        print(f"\nResults saved to 'ewald_optimization_results.txt'")

        # Generate plot if requested (even if optimization didn't converge)
        if args.plot and optimal_params['alpha'] is not None:
            plot_results(optimal_params, args.config, overrides, args.workers)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
