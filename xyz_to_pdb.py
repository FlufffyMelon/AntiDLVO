#!/usr/bin/env python3
"""
Script to extract the last frame from an XYZ trajectory file and save it as a PDB file.
"""

import argparse
import os
from pathlib import Path


def read_xyz_trajectory(filepath):
    """
    Read XYZ trajectory file and return all frames.

    Returns:
        list: List of frames, where each frame is a tuple (n_atoms, comment, atoms)
              atoms is a list of tuples (element, x, y, z, ...)
    """
    frames = []
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            # First line: number of atoms
            try:
                n_atoms = int(line.strip())
            except ValueError:
                break

            # Second line: comment/metadata
            comment = f.readline().strip()

            # Read atom lines
            atoms = []
            for i in range(n_atoms):
                atom_line = f.readline().strip()
                if not atom_line:
                    break
                parts = atom_line.split()
                if len(parts) >= 4:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    atoms.append((element, x, y, z))

            if len(atoms) == n_atoms:
                frames.append((n_atoms, comment, atoms))

    return frames


def write_pdb(atoms, output_path, comment="", convert_nm_to_angstrom=True):
    """
    Write atoms to a PDB file using strict column-based format.

    Args:
        atoms: List of tuples (element, x, y, z)
        output_path: Path to output PDB file
        comment: Optional comment string
        convert_nm_to_angstrom: If True, convert coordinates from nm to Angstrom (multiply by 10)
    """
    with open(output_path, 'w') as f:
        # Write header with proper REMARK format
        # REMARK format: columns 1-6 "REMARK", 7 space, 8-10 remark number, 11-79 text
        f.write("REMARK   1 Generated from XYZ trajectory\n")
        if comment:
            # Truncate comment if too long (max 69 chars for columns 11-79)
            comment_text = comment[:69]
            f.write(f"REMARK   2 {comment_text}\n")

        f.write("MODEL        1\n")

        # Write atoms using strict column positioning
        for i, (element, x, y, z) in enumerate(atoms, start=1):
            # PDB ATOM record format (80 columns total):
            # 1-6: "ATOM  "
            # 7-11: Atom serial number (right-justified)
            # 12: Space
            # 13-16: Atom name (left-justified for 1-2 chars, right-justified for 3-4 chars)
            # 17: Alternate location indicator (space)
            # 18-20: Residue name (right-justified)
            # 21: Space
            # 22: Chain identifier
            # 23-26: Residue sequence number (right-justified)
            # 27: Insertion code (space)
            # 28-30: Space
            # 31-38: X coordinate (8.3f, right-justified)
            # 39-46: Y coordinate (8.3f, right-justified)
            # 47-54: Z coordinate (8.3f, right-justified)
            # 55-60: Occupancy (6.2f, right-justified)
            # 61-66: Temperature factor (6.2f, right-justified)
            # 67-76: Space
            # 77-78: Element symbol (right-justified)
            # 79-80: Space + newline

            # Use element name in uppercase as residue name (max 3 chars for PDB format)
            residue_name = element.upper()[:3].rjust(3)  # Capitalize and right-justify to 3 chars
            chain = "A"
            res_seq = 1  # All atoms in one residue for simplicity

            # Format atom name according to PDB conventions:
            # 1 char: " C " (space, element, space, space)
            # 2 chars: " CA " (space, element, space)
            # 3-4 chars: right-justified in 4-char field
            if len(element) == 1:
                atom_name = f" {element}  "  # " C "
            elif len(element) == 2:
                atom_name = f" {element} "  # " CA "
            else:
                atom_name = element[:4].rjust(4)  # Right-justify if 3-4 chars

            # PDB coordinates are in Angstroms
            # Convert from nm to Angstrom if needed (common in MD simulations)
            if convert_nm_to_angstrom:
                x_coord = x * 10.0
                y_coord = y * 10.0
                z_coord = z * 10.0
            else:
                x_coord = x
                y_coord = y
                z_coord = z

            # Build line using strict column positioning
            line = "ATOM  "  # Columns 1-6
            line += f"{i:5d}"  # Columns 7-11
            line += " "  # Column 12
            line += atom_name  # Columns 13-16
            line += " "  # Column 17
            line += residue_name.rjust(3)  # Columns 18-20
            line += " "  # Column 21
            line += chain  # Column 22
            line += f"{res_seq:4d}"  # Columns 23-26
            line += " "  # Column 27 (insertion code)
            line += "   "  # Columns 28-30 (space)
            line += f"{x_coord:8.3f}"  # Columns 31-38
            line += f"{y_coord:8.3f}"  # Columns 39-46
            line += f"{z_coord:8.3f}"  # Columns 47-54
            line += f"{1.00:6.2f}"  # Columns 55-60 (occupancy)
            line += f"{0.00:6.2f}"  # Columns 61-66 (temperature factor)
            line += "          "  # Columns 67-76 (space)
            line += element[:2].rjust(2)  # Columns 77-78 (element symbol)
            line += " \n"  # Columns 79-80

            f.write(line)

        f.write("ENDMDL\n")
        f.write("END\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract the last frame from an XYZ trajectory and save as PDB"
    )
    parser.add_argument(
        "xyz_file",
        type=str,
        help="Path to the XYZ trajectory file"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Don't convert coordinates from nm to Angstrom (assume already in Angstrom)"
    )

    args = parser.parse_args()

    # Validate input file
    xyz_path = Path(args.xyz_file)
    if not xyz_path.exists():
        print(f"Error: File '{xyz_path}' does not exist.")
        return 1

    # Read trajectory
    print(f"Reading XYZ trajectory from: {xyz_path}")
    frames = read_xyz_trajectory(xyz_path)

    if not frames:
        print("Error: No frames found in the trajectory file.")
        return 1

    print(f"Found {len(frames)} frames in trajectory.")

    # Get last frame
    n_atoms, comment, atoms = frames[-1]
    print(f"Extracting last frame with {n_atoms} atoms.")

    # Generate output filename
    output_path = xyz_path.parent / f"{xyz_path.stem}_last_frame.pdb"

    # Convert coordinates if needed
    if not args.no_convert:
        # Check if units are in nm (common in MD simulations)
        # The comment line might indicate units, but we'll convert by default
        # User can use --no-convert if coordinates are already in Angstroms
        pass  # Conversion happens in write_pdb function

    # Write PDB file
    print(f"Writing PDB file to: {output_path}")
    convert_coords = not args.no_convert
    write_pdb(atoms, output_path, comment, convert_nm_to_angstrom=convert_coords)

    print(f"Successfully saved PDB file: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

