#!/usr/bin/env python3
"""
Build number density profile along Oz for selected species.

Reads EXTXYZ trajectory, selects species, samples frames within a range,
computes per-bin counts divided by bin volume, averages over sampled frames,
and outputs a simple line plot and .npz data in the results folder.
"""

import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import shlex
import yaml


def parse_key_value_string(s: str):
    parts = shlex.split(s)
    result = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            value = value.strip('"')
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
    positions_frames: List[np.ndarray] = []
    species_frames: List[np.ndarray] = []
    boxes: List[np.ndarray] = []
    energies: List[float] = []

    with open(path, "r") as f:
        frame_index = 0
        while True:
            natoms_line = f.readline()
            if not natoms_line:
                break
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

                frame_species[ai] = tokens[0]
                frame_positions[ai, 0] = float(tokens[1])
                frame_positions[ai, 1] = float(tokens[2])
                frame_positions[ai, 2] = float(tokens[3])

            positions_frames.append(frame_positions)
            species_frames.append(frame_species)
            frame_index += 1

    positions = np.stack(positions_frames, axis=0)
    maxlen = max(max(len(s) for s in fr) for fr in species_frames)
    species = np.empty(
        (len(species_frames), species_frames[0].shape[0]), dtype=f"<U{maxlen}"
    )
    for i, fr in enumerate(species_frames):
        species[i, :] = fr

    boxes = np.stack(boxes, axis=0)
    energies = np.array(energies)
    charges = np.empty((positions.shape[0], positions.shape[1]))  # placeholder
    return positions, species, charges, boxes, energies


def main():
    parser = argparse.ArgumentParser(
        description="Build number density profile along z for selected species"
    )
    parser.add_argument(
        "folder", help="Path to results folder containing trajectory.xyz"
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
        "--species",
        nargs="+",
        required=True,
        help="Species to include in density calculation (e.g., --species Ip Im)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=100, help="Number of bins along z"
    )
    parser.add_argument(
        "--output",
        help="Output plot filename (placed in calculation directory, default: density_profile.png)",
    )

    args = parser.parse_args()

    trajectory_path = osp.join(args.folder, "trajectory.xyz")
    if not osp.exists(trajectory_path):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print("Reading trajectory...")
    positions, species, _charges, boxes, _energies = read_xyz_trajectory(
        trajectory_path
    )

    n_frames_total, n_particles = positions.shape[:2]
    print(f"Loaded {n_frames_total} frames with {n_particles} particles")

    # Determine frame range and sampling
    start_frame = args.start_frame
    end_frame = args.end_frame if args.end_frame is not None else n_frames_total
    end_frame = min(end_frame, n_frames_total)

    if args.n_frames is not None:
        if args.n_frames > (end_frame - start_frame):
            print(
                f"Warning: Requested {args.n_frames} frames but only {end_frame - start_frame} available in range"
            )
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(
                start_frame, end_frame - 1, args.n_frames, dtype=int
            )
    else:
        frame_indices = list(range(start_frame, end_frame))

    print(f"Frame range: {start_frame} to {end_frame - 1}")
    print(f"Sampling {len(frame_indices)}")

    # Z binning setup
    # Assuming orthorhombic box and z from 0..Lz
    Lx = boxes[0, 0]
    Ly = boxes[0, 4]
    Lz = boxes[0, 8]
    area_xy = Lx * Ly
    z_edges = np.linspace(0.0, Lz, args.n_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    bin_height = z_edges[1] - z_edges[0]
    bin_volume = area_xy * bin_height

    # Accumulate counts per bin over sampled frames for each species separately
    species_data = {}
    for species_name in args.species:
        print(f"Processing species: {species_name}")
        all_counts = []
        for i, frame_idx in enumerate(frame_indices):
            pos = positions[frame_idx]
            sp = species[frame_idx]
            mask = sp == species_name
            z = pos[mask, 2]
            counts, _ = np.histogram(z, bins=z_edges)
            all_counts.append(counts)

        all_counts = np.array(all_counts)  # (F, n_bins)
        mean_counts = np.mean(all_counts, axis=0)
        number_density = mean_counts / bin_volume

        species_data[species_name] = {
            "counts": all_counts,
            "mean_counts": mean_counts,
            "number_density": number_density,
        }

    # Read config file
    config_path = osp.join(args.folder, "config_backup.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create single combined plot with all species and total
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    plt.figure(figsize=(10, 6))
    total_number_density = np.zeros_like(z_centers)
    for i, species_name in enumerate(args.species):
        data = species_data[species_name]
        color = colors[i % len(colors)]
        plt.plot(
            z_centers,
            data["number_density"] * 1.66053907 / len(args.species),
            linewidth=2,
            marker="o",
            markersize=3,
            color=color,
            label=species_name,
        )
        total_number_density += data["number_density"]

    # Total curve
    plt.plot(
        z_centers,
        total_number_density * 1.66053907 / len(args.species),
        linewidth=2.5,
        color="black",
        label="Total",
    )

    plt.xlabel("z-coordinate (nm)")
    plt.ylabel("Number density (mol/L)")
    # plt.title(
    #     f"Number density: {', '.join(args.species)}\n"
    #     f"Frames: {len(frame_indices)} ({start_frame}-{end_frame - 1}), "
    #     f"Bins: {args.n_bins}"
    # )
    plt.title(
        f"Number density: {', '.join(args.species)}\n"
        f"C: {config['C']}, Cz: {config['Cz']}, H: {config['H']}, sigma: {config['sigma']}, pz: {config['pz']}"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save single plot file in calculation directory
    if args.output:
        base_name = osp.splitext(args.output)[0]
        ext = osp.splitext(args.output)[1] or ".png"
        output_plot = osp.join(args.folder, f"{base_name}{ext}")
    else:
        output_plot = osp.join(args.folder, "density_profile.png")
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_plot}")
    plt.close()

    # Save data
    data_path = osp.join(args.folder, f"{args.output.split('.')[0]}.npz")
    save_data = {
        "z_edges": z_edges,
        "z_centers": z_centers,
        "bin_volume": bin_volume,
        "frame_indices": np.array(frame_indices),
        "species": np.array(args.species),
        "n_bins": args.n_bins,
    }

    # Add species-specific data
    for species_name in args.species:
        data = species_data[species_name]
        save_data[f"{species_name}_number_density"] = data["number_density"]
        save_data[f"{species_name}_mean_counts"] = data["mean_counts"]

    save_data["total_number_density"] = total_number_density
    np.savez(data_path, **save_data)
    print(f"Data saved to: {data_path}")
    # No interactive show; only saved output as a single file


if __name__ == "__main__":
    main()
