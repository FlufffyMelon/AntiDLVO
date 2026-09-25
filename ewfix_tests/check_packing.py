#!/usr/bin/env python3
"""Stage 3.1: does random_hs actually build the systems we are about to run?

Builds the starting configuration for one parameter point and checks it against
the config rather than against itself:

  * how long the placement took;
  * particle counts recomputed from C, C_z, sigma and the box, with the same
    truncation the loader uses -- a placement that silently drops a particle it
    could not fit would otherwise go unnoticed;
  * no hard-sphere overlap, at zero margin (the placement keeps a margin; the
    requirement is only that spheres do not interpenetrate);
  * every atom inside the z slab its hard_wall allows;
  * net charge zero and initial energy finite.

    python ewfix_tests/check_packing.py H=3 Cz=0.7 pz=27.6

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (  # noqa: E402
    _build_hs_contact_matrix,
    _build_z_limits,
    create_system,
    create_topology,
    create_units,
    load_config,
    setup_initial_configuration,
)

CONFIG = "configs_prod/ions_dipole_ewfix.yaml"
NA_NM3_PER_MOL_L = 0.602214076  # particles per nm^3 per mol/l


def expected_counts(cfg):
    """What the config asks for, recomputed from the scalars it is built on."""
    C, Cz = float(cfg.C), float(cfg.Cz)
    L, H = float(cfg.L), float(cfg.H)
    n_wall = int(cfg.N_wall)
    v = L * L * H
    n_salt = int(NA_NM3_PER_MOL_L * C * v)
    return {
        "Na": int(NA_NM3_PER_MOL_L * C * v + 2 * n_wall),
        "Cl": n_salt,
        "Ip": int(NA_NM3_PER_MOL_L * Cz * v),
        "Im": int(NA_NM3_PER_MOL_L * Cz * v),
        "G": int(NA_NM3_PER_MOL_L * Cz * v),
        "Wb": n_wall,
        "Wt": n_wall,
    }


def min_pair_gap(pos, types, mol_ids, box, contact, chunk=512):
    """Worst (distance - contact) over all constrained pairs, and who it was.

    x and y are periodic, z is not. Atoms of the same molecule are skipped: a
    dipole's own sites sit at a fixed separation set by p_z, which the placement
    never tests and cannot change. Done in row chunks so the pair matrix never
    has to exist in full.
    """
    n = len(pos)
    worst = np.inf
    where = None
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = pos[start:stop, None, :] - pos[None, :, :]
        d[:, :, 0] -= box[0] * np.round(d[:, :, 0] / box[0])
        d[:, :, 1] -= box[1] * np.round(d[:, :, 1] / box[1])
        r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))

        c = contact[types[start:stop][:, None], types[None, :]].copy()
        same_mol = mol_ids[start:stop][:, None] == mol_ids[None, :]
        c[same_mol] = 0.0
        gap = np.where(c > 0.0, r - c, np.inf)
        k = int(np.argmin(gap))
        g = float(gap.flat[k])
        if g < worst:
            worst = g
            where = (start + k // n, k % n)
    return worst, where


def check_point(overrides, verbose=True):
    cfg = load_config(CONFIG, overrides)
    H = float(cfg.H)
    tag = " ".join(overrides) if overrides else "base"
    print(f"\n===== {tag} =====")
    print(f"  H={H} C={float(cfg.C)} Cz={float(cfg.Cz)} sigma={float(cfg.sigma)} "
          f"pz={float(cfg.pz)} N_wall={int(cfg.N_wall)} "
          f"z_scale={float(cfg.ewald.z_scale_factor):.4f} n_c={int(cfg.ewald.n_c)}")

    units = create_units(cfg)
    system = create_system(cfg, units)
    topology = create_topology(cfg, units)

    t0 = time.time()
    setup_initial_configuration(system, cfg.get("types", []), topology, cfg=cfg)
    t_place = time.time() - t0
    print(f"  placement: {t_place:.1f} s ({t_place / 60.0:.2f} min)")

    n = system.N_atoms
    pos = np.asarray(system.positions[:n], dtype=float)
    types = np.asarray(system.types[:n], dtype=int)
    charges = np.asarray(system.charges[:n], dtype=float)
    names = list(system.names[:n])
    box = np.asarray(system.box, dtype=float)

    failures = []

    # ---- counts
    got = {}
    for name in names:
        got[name] = got.get(name, 0) + 1
    want = expected_counts(cfg)
    print(f"  {'type':<6} {'placed':>8} {'expected':>9}")
    for name in ("Na", "Cl", "Ip", "Im", "G", "Wb", "Wt"):
        g, w = got.get(name, 0), want[name]
        flag = "" if g == w else "   <-- MISMATCH"
        print(f"  {name:<6} {g:>8} {w:>9}{flag}")
        if g != w:
            failures.append(f"{name}: placed {g}, config asks {w}")
    print(f"  total atoms: {n}")

    # ---- hard spheres
    contact = _build_hs_contact_matrix(cfg, topology.type_name_to_id, margin=0.0)
    mol_ids = np.asarray(system.molecule_ids[:n], dtype=int)
    _, sizes = np.unique(mol_ids, return_counts=True)
    print(f"  molecules: {len(sizes)}, largest {sizes.max()} atoms")
    if sizes.max() > 3:
        # The overlap test skips same-molecule pairs. If unrelated particles
        # shared a molecule id, that skip would quietly hide real overlaps.
        failures.append(f"a molecule has {sizes.max()} atoms; the intramolecular "
                        f"skip in the overlap test would not be trustworthy")
    if np.any(contact > 0):
        gap, where = min_pair_gap(pos, types, mol_ids, box, contact)
        i, j = where
        print(f"  worst hard-sphere gap: {gap:+.6f} nm "
              f"(atoms {i} {names[i]} / {j} {names[j]})")
        if gap < -1e-9:
            failures.append(f"hard-sphere overlap by {-gap:.6f} nm "
                            f"({names[i]}/{names[j]})")
    else:
        print("  no hard-sphere pairs declared")

    # ---- z slabs
    z_lo, z_hi = _build_z_limits(cfg, topology.type_name_to_id, H)
    lo = z_lo[types]
    hi = z_hi[types]
    below = pos[:, 2] - lo
    above = hi - pos[:, 2]
    print(f"  z margin: min below-limit {below.min():+.6f} nm, "
          f"min above-limit {above.min():+.6f} nm")
    if below.min() < -1e-9 or above.min() < -1e-9:
        k = int(np.argmin(np.minimum(below, above)))
        failures.append(f"atom {k} ({names[k]}) outside its z slab: "
                        f"z={pos[k, 2]:.6f}, allowed [{lo[k]:.6f}, {hi[k]:.6f}]")

    # ---- charge and energy
    q = float(charges.sum())
    print(f"  net charge: {q:+.3e} e")
    if abs(q) > 1e-9:
        failures.append(f"net charge {q:+.3e} e is not zero")

    t0 = time.time()
    e = float(topology.get_energy(system))
    print(f"  initial energy: {e:.6f} kJ/mol  ({time.time() - t0:.1f} s)")
    if not np.isfinite(e):
        failures.append(f"initial energy is {e}")

    if failures:
        print("  FAILED:")
        for f in failures:
            print("    -", f)
    else:
        print("  OK")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*",
                    help="dotlist overrides, e.g. H=11 Cz=0.7")
    args = ap.parse_args(argv)
    return 1 if check_point(list(args.overrides)) else 0


if __name__ == "__main__":
    sys.exit(main())
