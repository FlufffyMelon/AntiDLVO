#!/usr/bin/env python3
"""
Script to calculate local pressure distribution along the Oz axis from trajectory data.
The pressure is calculated using the formula P(z) = mean(F_z) / S, where
F_z = sum (z_j - z_i) / r_ij F_ij

This script reads trajectory data, calculates forces, and applies block averaging.
"""

import os.path as osp
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import shlex

# Constants
K_COULOMB = 138.935456  # kJ/mol * nm/e^2 (Coulomb constant in MD units)
K_VOLT = 96.485  # kJ/mol/V


def parse_key_value_string(s):
    """Parse key-value string from EXTXYZ header."""
    parts = shlex.split(s)
    result = {}

    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            value = value.strip('"')
            # Try to convert to number if possible
            if value.lower() == "nan":
                result[key] = float("nan")
            elif value.replace(".", "", 1).isdigit():
                result[key] = float(value)
            else:
                result[key] = value
    return result


def read_xyz_trajectory(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an EXTXYZ trajectory and return arrays:
      - positions: np.ndarray shape (M, N, 3) float64
      - species:   np.ndarray shape (M, N) (unicode strings)
      - charges:   np.ndarray shape (M, N) float64
      - boxes:     np.ndarray shape (M, 9) float64
      - energies:  np.ndarray shape (M,) float64
    """
    positions_frames: List[np.ndarray] = []
    species_frames: List[np.ndarray] = []
    charges_frames: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    energies: List[float] = []

    with open(path, "r") as f:
        frame_index = 0
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                break  # EOF
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue

            try:
                num_atoms = int(natoms_line)
            except ValueError:
                raise ValueError(
                    f"Invalid XYZ at frame {frame_index}: first line is not integer: '{natoms_line}'"
                )

            header_line = f.readline()
            if not header_line:
                raise ValueError(f"Unexpected EOF after natoms at frame {frame_index}")
            header_line = header_line.strip()

            properties = parse_key_value_string(header_line)
            boxes.append(
                np.array(
                    properties.get("Lattice", "0 0 0 0 0 0 0 0 0").split(),
                    dtype=np.float64,
                )
            )
            energies.append(float(properties.get("energy", 0.0)))

            frame_positions = np.empty((num_atoms, 3), dtype=np.float64)
            frame_species = np.empty((num_atoms,), dtype=object)
            frame_charges = np.empty((num_atoms,), dtype=np.float64)

            for ai in range(num_atoms):
                line = f.readline()
                if not line:
                    raise ValueError(
                        f"Unexpected EOF inside atom block at frame {frame_index}, atom {ai}"
                    )
                tokens = line.strip().split()
                if len(tokens) < 7:
                    raise ValueError(
                        f"Atom line too short at frame {frame_index}, atom {ai}: expected >= 7 tokens, got {len(tokens)}"
                    )

                # Species, Position, Charge, Mass, Type
                frame_species[ai] = tokens[0]
                frame_positions[ai, 0] = float(tokens[1])
                frame_positions[ai, 1] = float(tokens[2])
                frame_positions[ai, 2] = float(tokens[3])
                frame_charges[ai] = float(tokens[4])

            positions_frames.append(frame_positions)
            species_frames.append(frame_species)
            charges_frames.append(frame_charges)

            frame_index += 1

    positions = np.stack(positions_frames, axis=0)  # (M, N, 3)
    # Convert species to uniform dtype string array
    maxlen = max(max(len(s) for s in fr) for fr in species_frames)
    species = np.empty(
        (len(species_frames), species_frames[0].shape[0]), dtype=f"<U{maxlen}"
    )
    for i, fr in enumerate(species_frames):
        species[i, :] = fr

    charges = np.stack(charges_frames, axis=0)  # (M, N)
    boxes = np.stack(boxes, axis=0)  # (M, 9)
    energies = np.array(energies)  # (M,)

    return positions, species, charges, boxes, energies


def calculate_coulomb_forces(
    positions: np.ndarray,
    charges: np.ndarray,
    box: np.ndarray,
    dielectric: float = 78.0,
    n_images: int = 0,
) -> np.ndarray:
    """
    Calculate Coulomb forces between all particles.

    Args:
        positions: (N, 3) array of particle positions
        charges: (N,) array of particle charges
        box: (9,) array of box vectors
        dielectric: dielectric constant
        n_images: number of image cells to include

    Returns:
        forces: (N, 3) array of forces on each particle
    """
    N = positions.shape[0]
    forces = np.zeros((N, 3))

    # Extract box dimensions
    box_dims = np.array([box[0], box[4], box[8]])  # Assuming orthogonal box

    for i in range(N):
        # for j in range(i + 1, N):
        #     # Calculate distance vector
        #     r_ij = positions[j] - positions[i]

        #     # Apply minimum image convention
        #     for dim in range(3):
        #         if box_dims[dim] > 0:
        #             r_ij[dim] = r_ij[dim] - box_dims[dim] * np.round(
        #                 r_ij[dim] / box_dims[dim]
        #             )

        #     # Include periodic images
        #     for nx in range(-n_images, n_images + 1):
        #         for ny in range(-n_images, n_images + 1):
        #             for nz in range(-n_images, n_images + 1):
        #                 if nx == 0 and ny == 0 and nz == 0 and i == j:
        #                     continue

        #                 r_image = r_ij + np.array([nx, ny, nz]) * box_dims
        #                 r_norm = np.linalg.norm(r_image)

        #                 if r_norm > 1e-9:  # Avoid division by zero
        #                     # Coulomb force magnitude
        #                     force_mag = (
        #                         K_COULOMB
        #                         * charges[i]
        #                         * charges[j]
        #                         / (r_norm**2 * dielectric)
        #                     )

        #                     # Force vector
        #                     force_vec = force_mag * r_image / r_norm

        #                     # Add to forces (Newton's third law)
        #                     forces[i] += force_vec
        #                     forces[j] -= force_vec

        # for j in range(i, N):
        indices = np.arange(i, N)
        r_ij = positions[indices, :] - positions[i, :]
        for nx in range(-n_images, n_images + 1):
            for ny in range(-n_images, n_images + 1):
                r_image = r_ij + np.array([nx, ny, 0]) * box_dims
                r_norm = np.linalg.norm(r_image, axis=1)
                mask = r_norm > 1e-9

                # Coulomb force magnitude
                force_mag = (
                    K_COULOMB
                    * charges[i]
                    * charges[indices][mask]
                    / (r_norm[mask] ** 2 * dielectric)
                )

                # Force vector
                force_vec = (
                    force_mag.reshape(-1, 1)
                    * r_image[mask]
                    / r_norm[mask].reshape(-1, 1)
                )

                # Add to forces (Newton's third law)
                forces[i] += np.sum(force_vec, axis=0)
                forces[indices][mask] -= force_vec

    return forces


def calculate_external_field_forces(
    positions: np.ndarray, charges: np.ndarray, voltage: float, box: np.ndarray
) -> np.ndarray:
    """
    Calculate forces due to external electric field.

    Args:
        positions: (N, 3) array of particle positions
        charges: (N,) array of particle charges
        voltage: applied voltage
        box: (9,) array of box vectors

    Returns:
        forces: (N, 3) array of forces on each particle
    """
    N = positions.shape[0]
    forces = np.zeros((N, 3))

    # Electric field strength (assuming field is in z-direction)
    E_field = voltage / box[8]  # V/nm

    # Force due to electric field: F = q * E
    forces[:, 2] = -charges * E_field

    return forces


def calculate_local_pressure(
    positions: np.ndarray,
    forces: np.ndarray,
    species: np.ndarray,
    box: np.ndarray,
    n_bins: int = 100,
    selected_species: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate local pressure distribution along z-axis.

    Args:
        positions: (N, 3) array of particle positions
        forces: (N, 3) array of forces on each particle
        species: (N,) array of particle species
        box: (9,) array of box vectors
        n_bins: number of bins for z-coordinate
        selected_species: list of species to include in pressure calculation

    Returns:
        z_centers: (n_bins,) array of z-coordinate bin centers
        pressure: (n_bins,) array of pressure values
    """
    N = positions.shape[0]
    z_min, z_max = 0.0, box[8]  # Assuming box starts at origin

    # Create bins
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Calculate pressure contribution from each particle
    pressure = np.zeros(n_bins)

    # Create mask for selected species
    if selected_species is not None:
        species_mask = np.isin(species, selected_species)
        print(f"Selected species: {selected_species}")
        print(f"Particles included: {np.sum(species_mask)} out of {N}")
    else:
        species_mask = np.ones(N, dtype=bool)
        print("Including all species")

    for i in range(N):
        if not species_mask[i]:
            continue  # Skip particles not in selected species

        z_i = positions[i, 2]
        bin_idx = int((z_i - z_min) / (z_max - z_min) * n_bins)
        bin_idx = max(0, min(n_bins - 1, bin_idx))  # Clamp to valid range

        # Pressure contribution: F_z / S
        # S is the cross-sectional area (assuming square box in x-y)
        S = box[0] * box[4]  # Area in x-y plane
        pressure[bin_idx] += forces[i, 2] / S

    return z_centers, pressure


def calculate_average_pressure(pressure_data: np.ndarray) -> np.ndarray:
    """
    Calculate average pressure across all frames.

    Args:
        pressure_data: (n_frames, n_bins) array of pressure data

    Returns:
        average_pressure: (n_bins,) array of average pressure values
    """
    return np.mean(pressure_data, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate local pressure distribution from trajectory"
    )
    parser.add_argument(
        "folder", help="Path to results folder containing trajectory.xyz"
    )
    parser.add_argument(
        "--config", help="Path to config file (default: config_backup.yaml in folder)"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Starting frame index"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending frame index (default: all frames from start)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Number of frames to sample from the range (default: all frames in range)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=100, help="Number of bins for z-coordinate"
    )
    parser.add_argument(
        "--n-images", type=int, default=0, help="Number of periodic images to include"
    )
    parser.add_argument(
        "--dielectric", type=float, default=None, help="Dielectric constant"
    )
    parser.add_argument(
        "--voltage", type=float, default=None, help="Applied voltage (overrides config)"
    )
    parser.add_argument(
        "--output", help="Output filename (default: pressure_distribution.png)"
    )
    parser.add_argument(
        "--species",
        nargs="+",
        help="Species to include in pressure calculation (e.g., --species Ip Im)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = args.config or osp.join(args.folder, "config_backup.yaml")
    if osp.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract voltage from config if not provided
        if args.voltage is None:
            one_body_forces = config.get("one_body", [])
            for force_config in one_body_forces:
                forces = force_config.get("forces", [])
                for force in forces:
                    if force.get("kind") == "electric_field":
                        args.voltage = force.get("voltage", 0.0)
                        break
                if args.voltage is not None:
                    break

        # Extract dielectric from config
        ewald_config = config.get("ewald", {})
        if "dielectric" in ewald_config:
            args.dielectric = ewald_config["dielectric"]

    if args.voltage is None:
        args.voltage = 0.0
        print("Warning: No voltage found in config, using 0.0")

    if args.voltage is None:
        args.dielectric = 1.0
        print("Warning: No dielectric found in config, using 1.0")

    print(f"Using voltage: {args.voltage} V")
    print(f"Using dielectric: {args.dielectric}")
    if args.species:
        print(f"Analyzing species: {', '.join(args.species)}")
    else:
        print("Analyzing all species")

    # Read trajectory
    trajectory_path = osp.join(args.folder, "trajectory.xyz")
    if not osp.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print("Reading trajectory...")
    positions, species, charges, boxes, energies = read_xyz_trajectory(trajectory_path)

    n_frames, n_particles = positions.shape[:2]
    print(f"Loaded {n_frames} frames with {n_particles} particles")

    # Determine frame range and sampling
    start_frame = args.start_frame
    if args.end_frame is not None:
        end_frame = min(args.end_frame, n_frames)
    else:
        end_frame = n_frames

    # Calculate which frames to sample
    if args.n_frames is not None:
        # Sample n_frames evenly distributed within the range
        if args.n_frames > (end_frame - start_frame):
            print(
                f"Warning: Requested {args.n_frames} frames but only {end_frame - start_frame} available in range"
            )
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Create evenly spaced indices within the range
            frame_indices = np.linspace(
                start_frame, end_frame - 1, args.n_frames, dtype=int
            )
    else:
        # Use all frames in the range
        frame_indices = list(range(start_frame, end_frame))

    print(f"Frame range: {start_frame} to {end_frame - 1}")
    print(f"Sampling {len(frame_indices)} frames: {frame_indices}")

    # Calculate pressure for each sampled frame
    all_pressures = []
    z_centers = None

    for i, frame_idx in enumerate(frame_indices):
        if i % 10 == 0:
            print(f"Processing frame {frame_idx} ({i + 1}/{len(frame_indices)})")

        pos = positions[frame_idx]
        charge = charges[frame_idx]
        box = boxes[frame_idx]

        # Calculate forces
        coulomb_forces = calculate_coulomb_forces(
            pos, charge, box, dielectric=args.dielectric, n_images=args.n_images
        )
        external_forces = calculate_external_field_forces(
            pos, charge, args.voltage, box
        )
        total_forces = coulomb_forces + external_forces

        # Calculate local pressure
        z_centers, pressure = calculate_local_pressure(
            pos, total_forces, species[frame_idx], box, args.n_bins, args.species
        )
        all_pressures.append(pressure)

    # Convert to numpy array
    pressure_data = np.array(all_pressures)  # (n_frames, n_bins)

    # Calculate average pressure
    print(f"Calculating average pressure from {len(all_pressures)} frames")
    average_pressure = calculate_average_pressure(pressure_data)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(z_centers, average_pressure, linewidth=2, marker="o", markersize=4)
    plt.xlabel("z-coordinate (nm)")
    plt.ylabel("Pressure (kJ/mol/nm³)")
    species_str = ", ".join(args.species) if args.species else "All species"
    n_frames_used = len(all_pressures)
    plt.title(
        f"Average Pressure Distribution ({species_str})\n"
        f"Voltage: {args.voltage} V, Dielectric: {args.dielectric}, "
        f"Frames: {n_frames_used}"
    )
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = args.output or osp.join(args.folder, "pressure_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Save data
    data_path = osp.join(args.folder, "pressure_data.npz")
    np.savez(
        data_path,
        z_centers=z_centers,
        pressure=average_pressure,
        pressure_data=pressure_data,
        frame_indices=frame_indices,
        voltage=args.voltage,
        dielectric=args.dielectric,
        n_frames=len(all_pressures),
        start_frame=start_frame,
        end_frame=end_frame,
        selected_species=args.species if args.species else None,
    )
    print(f"Data saved to: {data_path}")

    plt.show()


if __name__ == "__main__":
    main()
