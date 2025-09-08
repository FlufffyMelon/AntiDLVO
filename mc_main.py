#!/usr/bin/env python3
"""
Monte Carlo simulation runner with OmegaConf configuration management.
"""

import argparse
import sys
import traceback
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omegaconf import OmegaConf
from src.units import Units
from src.logger import Logger
from src.utils import (
    load_config,
    create_units,
    create_system,
    create_topology,
    create_logger,
    create_sampler,
    setup_initial_configuration,
)

from tqdm import tqdm


def run_simulation(config_file: str, overrides=None):
    """Run Monte Carlo simulation from configuration file."""
    print(f"Loading configuration from {config_file}")
    cfg = load_config(config_file, overrides=overrides)

    # Create components
    units = create_units(cfg)
    system = create_system(cfg, units)
    topology = create_topology(cfg, units)
    setup_initial_configuration(
        system,
        cfg.get("types", []),
        topology,
        cfg=cfg,
    )
    logger = create_logger(cfg)
    sampler = create_sampler(system, topology, cfg, logger, units)

    # Log simulation start
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger.log_simulation_start(system, config_dict)

    # Run simulation
    n_steps = cfg.simulation.get("n_steps", 1000)

    # Check initial configuration for problems
    # from src.utils import check_initial_configuration_energy

    # check_initial_configuration_energy(system, topology)

    print(f"Running {n_steps} Monte Carlo steps...")

    initial_energy = topology.get_energy(system)
    print(f"Initial energy: {initial_energy:.6f} {units.energy_label}")

    t0 = time.perf_counter()
    try:
        # Progress bar updated 10 times
        steps_remaining = n_steps
        chunks = 10
        chunk_size = max(1, n_steps // chunks)
        bar = None
        if tqdm is not None:
            bar = tqdm(total=n_steps, desc="MC", miniters=1, leave=False)
        for i in range(chunks):
            if steps_remaining <= 0:
                break
            take = chunk_size if i < chunks - 1 else steps_remaining
            sampler.move(take)
            steps_remaining -= take
            if bar is not None:
                bar.update(take)
        if bar is not None:
            bar.close()

        total_runtime = time.perf_counter() - t0

        final_energy = topology.get_energy(system)
        final_virial = system.virial
        print(f"Final energy: {final_energy:.6f} {units.energy_label}")
        print(f"Final number of atoms: {system.N_atoms}")

        # Recompute full energy from scratch for verification
        topology.recompute_caches(system)
        recomputed_energy = system.potential_energy
        recomputed_virial = system.virial

        print(
            f"Final energy (recomputed): {recomputed_energy:.6f} {units.energy_label}"
        )
        print(f"Final virial: {final_virial:.6f}")
        print(
            f"Energy delta (cached -> recomputed): {(recomputed_energy - final_energy):.6f} {units.energy_label}"
        )
        print(
            f"Virial delta (cached -> recomputed): {(recomputed_virial - final_virial):.6f} {units.pressure_label}"
        )
        logger.log_info(
            f"Final energy (recomputed): {recomputed_energy:.6f} {units.energy_label}; "
            f"delta vs cached: {(recomputed_energy - final_energy):.6f} {units.energy_label}"
        )

        ratios = sampler.get_acceptance_ratios()
        print("Acceptance ratios:")
        for action, ratio in ratios.items():
            print(f"  {action}: {ratio:.4f}")

        logger.log_simulation_end(system, sampler)
        print(f"Results saved to: {logger.get_output_directory()}")

        # Profiling summary if enabled
        if getattr(sampler, "debug", False):
            from src.profiler import Profiler  # local import

            summary = sampler.profiler.summary_text(sampler.n_moves, total_runtime)
            print("")
            print(summary)
            for line in summary.splitlines():
                logger.log_info(line)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        logger.log_warning("Simulation interrupted by user")
        logger.log_simulation_end(system, sampler)

    except Exception as e:
        print(f"\nSimulation failed with error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        logger.log_error(f"Simulation failed: {e}")
        raise

    finally:
        logger.finalize()

    print("Simulation completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation runner", add_help=True
    )
    parser.add_argument("config", help="Configuration file path")
    # Capture any extra args as dotlist overrides, e.g. --system.size 20 or system.size=20
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional overrides as dotlist, e.g. system.size=20 or --system.size 20",
    )

    args = parser.parse_args()

    # Normalize overrides into dotlist format
    overrides = []
    if args.overrides:
        # Accept both --key value and key=value styles
        pending_key = None
        for token in args.overrides:
            if token.startswith("--"):
                pending_key = token[2:]
            elif "=" in token and pending_key is None:
                overrides.append(token)
            elif pending_key is not None:
                overrides.append(f"{pending_key}={token}")
                pending_key = None
        # If odd number and a dangling key remains, ignore

    try:
        run_simulation(args.config, overrides=overrides)
    except Exception as e:
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
