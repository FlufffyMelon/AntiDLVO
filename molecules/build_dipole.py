#!/usr/bin/env python3
"""
Build a dipole molecule template for LAMMPS (native format).
Generates correct Special Bond Counts and Special Bonds (1-2, 1-3, 1-4).
Usage: python build_dipole.py dipole_length n_virtual
"""
import sys
from collections import defaultdict, deque

def bfs_neighbors(graph, start, max_depth=3):
    """Return dict levels: levels[1] = set(nodes at distance 1), etc."""
    dist = {start: 0}
    q = deque([start])
    levels = {1:set(), 2:set(), 3:set()}
    while q:
        node = q.popleft()
        d = dist[node]
        if d == max_depth:
            continue
        for neigh in sorted(graph[node]):
            if neigh not in dist:
                nd = d + 1
                dist[neigh] = nd
                if 1 <= nd <= max_depth:
                    levels[nd].add(neigh)
                q.append(neigh)
    return levels

def main():
    if len(sys.argv) != 3:
        print("Usage: python build_dipole.py dipole_length n_virtual")
        sys.exit(1)

    dipole_length = float(sys.argv[1])
    n_virtual = int(sys.argv[2])
    if n_virtual < 0:
        raise SystemExit("n_virtual must be non-negative")

    n_atoms = 2 + n_virtual

    lines = []
    lines.append("# Dipole molecule with virtual sites")
    lines.append("")
    # header counts
    n_bonds = (n_atoms if n_virtual > 0 else 1)
    n_angles = n_virtual
    lines.append(f"{n_atoms} atoms")
    lines.append(f"{n_bonds} bonds")
    lines.append(f"{n_angles} angles")
    lines.append("0 dihedrals")
    lines.append("0 impropers")
    lines.append("# This molecule template is for atom_style full")
    lines.append("")

    # Coords
    lines.append("Coords")
    lines.append("")
    lines.append(f"1 0 0 0")
    lines.append(f"2 {dipole_length} 0 0")
    if n_virtual > 0:
        spacing = dipole_length / (n_virtual + 1)
        for i in range(n_virtual):
            pos = spacing * (i + 1)
            lines.append(f"{i + 3} {pos} 0 0")
    lines.append("")

    # Types
    lines.append("Types")
    lines.append("")
    lines.append("1 3")
    lines.append("2 4")
    for i in range(n_virtual):
        lines.append(f"{i + 3} 5")
    lines.append("")

    # Charges
    lines.append("Charges")
    lines.append("")
    lines.append("1 1.0")
    lines.append("2 -1.0")
    for i in range(n_virtual):
        lines.append(f"{i + 3} 0.0")
    lines.append("")

    # Bonds (IDs, types, atom1 atom2)
    lines.append("Bonds")
    lines.append("")
    bond_id = 1
    bonds = []

    # direct Ip-Im
    bonds.append((bond_id, 1, 1, 2)); bond_id += 1

    if n_virtual > 0:
        # Ip - first virtual
        bonds.append((bond_id, 1, 1, 3)); bond_id += 1
        # consecutive virtuals
        for i in range(n_virtual - 1):
            vs1 = i + 3
            vs2 = i + 4
            bonds.append((bond_id, 1, vs1, vs2)); bond_id += 1
        # last virtual - Im
        bonds.append((bond_id, 1, n_virtual + 2, 2)); bond_id += 1

    for b in bonds:
        lines.append(f"{b[0]} {b[1]} {b[2]} {b[3]}")
    lines.append("")

    # Angles
    if n_virtual > 0:
        lines.append("Angles")
        lines.append("")
        angle_id = 1
        if n_virtual == 1:
            lines.append(f"{angle_id} 1 1 3 2"); angle_id += 1
        else:
            lines.append(f"{angle_id} 1 1 3 4"); angle_id += 1
            for i in range(n_virtual - 2):
                vs1 = i + 3
                vs2 = i + 4
                vs3 = i + 5
                lines.append(f"{angle_id} 1 {vs1} {vs2} {vs3}")
                angle_id += 1
            if n_virtual > 1:
                lines.append(f"{angle_id} 1 {n_virtual+1} {n_virtual+2} 2")
        lines.append("")

    # Build adjacency graph from bonds (undirected)
    graph = defaultdict(set)
    for _, _, a1, a2 in bonds:
        graph[a1].add(a2)
        graph[a2].add(a1)

    # Special Bond Counts
    lines.append("Special Bond Counts")
    lines.append("")
    sb = {}
    for i in range(1, n_atoms + 1):
        levels = bfs_neighbors(graph, i, max_depth=3)
        neigh12 = levels[1]
        neigh13 = levels[2]
        neigh14 = levels[3]
        sb[i] = (neigh12, neigh13, neigh14)
        lines.append(f"{i} {len(neigh12)} {len(neigh13)} {len(neigh14)}")
    lines.append("")

    # Special Bonds: **no counts here** — list neighbors in order: 1-2, 1-3, 1-4
    lines.append("Special Bonds")
    lines.append("")
    for i in range(1, n_atoms + 1):
        neigh12, neigh13, neigh14 = sb[i]
        ordered = list(map(str, sorted(neigh12))) + list(map(str, sorted(neigh13))) + list(map(str, sorted(neigh14)))
        if ordered:
            lines.append(f"{i} " + " ".join(ordered))
        else:
            lines.append(f"{i}")
    lines.append("")

    # Shake sections (simple placeholders) — adjust if you use SHAKE clusters
    lines.append("Shake Flags")
    lines.append("")
    for i in range(1, n_atoms + 1):
        lines.append(f"{i} 0")
    lines.append("")

    lines.append("Shake Atoms")
    lines.append("")
    for i in range(1, n_atoms + 1):
        lines.append(f"{i}")
    lines.append("")

    lines.append("Shake Bond Types")
    lines.append("")
    for i in range(1, n_atoms + 1):
        lines.append(f"{i}")
    lines.append("")

    # Molecules
    lines.append("Molecules")
    lines.append("")
    for i in range(1, n_atoms + 1):
        lines.append(f"{i} 1")

    # Write output file
    output_file = sys.argv[0].replace(".py", ".txt")
    if output_file == sys.argv[0]:
        output_file = "dipole_molecule.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Created molecule template at {output_file}")

if __name__ == "__main__":
    main()
