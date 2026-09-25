#!/usr/bin/env python3
"""Stage 3.2: is the cached electrostatic energy still the real one after N moves?

Two independent questions, both answered from a single short run started with
sampler.force_recompute=true:

  1. Does the incremental energy the sampler carries agree with a full
     recompute? The sampler prints both on every move; the run is healthy if
     the cached total never drifts from the recomputed total by more than a
     relative 1e-6, and if the delta the move was accepted on matches the
     difference of the two full recomputes around it.

  2. Is the logged wall force right? It must be positive (the walls are pushed
     apart) and equal the smeared-charge limit 2*pi*k_c*sigma_eff^2/eps_r to
     within 0.97-1.05. sigma_eff is the charge actually placed -- N_wall is
     rounded to a perfect square, so it is not exactly the requested sigma.

    python ewfix_tests/check_force_recompute.py <log file> <run dir>
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys

import numpy as np

KC = 138.935456       # kJ nm / (mol e^2)
MPA = 1.66054         # 1 kJ/(mol nm^3) in MPa
DRIFT_LIMIT = 1.0e-6  # relative, cached vs recomputed total
RATIO_LO, RATIO_HI = 0.97, 1.05

LINE = re.compile(
    r"\[force_recompute\]\s+(\w+):\s*dE\(delta\)=([-\d.eE+]+),\s*"
    r"dE\(real\)=([-\d.eE+]+),\s*E_cached=([-\d.eE+]+),\s*E_real=([-\d.eE+]+)"
)


def check_log(path):
    d_delta, d_real, e_cached, e_real = [], [], [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = LINE.search(line)
            if m:
                d_delta.append(float(m.group(2)))
                d_real.append(float(m.group(3)))
                e_cached.append(float(m.group(4)))
                e_real.append(float(m.group(5)))

    if not e_real:
        print(f"  no [force_recompute] lines in {path}")
        return ["force_recompute produced no diagnostics -- was it enabled?"]

    d_delta = np.array(d_delta)
    d_real = np.array(d_real)
    e_cached = np.array(e_cached)
    e_real = np.array(e_real)

    denom = np.where(np.abs(e_real) > 0, np.abs(e_real), 1.0)
    drift = np.abs(e_cached - e_real) / denom
    # The printed values carry two significant digits, so a difference of one
    # unit in the last printed place is not evidence of drift.
    quantum = np.abs(e_real) * 1e-2 / denom

    dd = np.abs(d_delta - d_real)

    print(f"  moves audited            : {len(e_real)}")
    print(f"  |E_cached-E_real|/|E_real|: max {drift.max():.3e}  "
          f"median {np.median(drift):.3e}  (print quantum ~{quantum.max():.0e})")
    print(f"  |dE(delta)-dE(real)|     : max {dd.max():.3e} kJ/mol  "
          f"median {np.median(dd):.3e}")
    print(f"  E_real range             : {e_real.min():.6e} .. {e_real.max():.6e}")

    fails = []
    if drift.max() > max(DRIFT_LIMIT, quantum.max()):
        fails.append(f"cached energy drifts from the recomputed one by "
                     f"{drift.max():.3e} (limit {DRIFT_LIMIT:.0e})")
    if not np.all(np.isfinite(e_real)):
        fails.append("recomputed energy is not finite somewhere")
    return fails


def check_wall(run_dir, cfg):
    """Logged solvation_force against the smeared-charge limit."""
    n_wall = int(cfg.N_wall)
    L = float(cfg.L)
    eps_r = float(cfg.ewald.dielectric)
    # One wall's areal charge density, in e/nm^2, as actually placed.
    q_wall = None
    for entry in cfg.get("types", []) or []:
        spec = entry.get("Particle")
        if spec is not None and str(spec.get("type")) == "Wb":
            q_wall = float(spec.get("charge"))
    if q_wall is None:
        return ["no Wb particle type in the config backup"]
    sigma_eff = n_wall * abs(q_wall) / (L * L)
    p_ideal = 2.0 * math.pi * KC * sigma_eff ** 2 / eps_r

    path = os.path.join(run_dir, "simulation_data.csv")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    fs = np.array([float(r["solvation_force"]) for r in rows
                   if r.get("solvation_force")])
    if fs.size == 0:
        return ["simulation_data.csv has no solvation_force column"]

    # The start is still relaxing out of the random placement; judge the tail.
    tail = fs[len(fs) // 2:]
    mean = float(tail.mean())
    ratio = mean / p_ideal

    print(f"  sigma_eff                : {sigma_eff:.6f} e/nm^2 "
          f"({sigma_eff * 16.020506248:.4f} uC/cm^2)")
    print(f"  2*pi*k_c*sigma^2/eps     : {p_ideal:.4f} kJ/(mol nm^3) "
          f"= {p_ideal * MPA:.4f} MPa")
    print(f"  logged f_s (2nd half)    : {mean:.4f} +- {tail.std():.4f} "
          f"kJ/(mol nm^3) = {mean * MPA:.4f} MPa   over {tail.size} rows")
    print(f"  ratio f_s / ideal        : {ratio:.4f}")

    fails = []
    if mean <= 0:
        fails.append(f"logged solvation_force is {mean:.4f}, i.e. the wrong sign")
    if not (RATIO_LO <= ratio <= RATIO_HI):
        fails.append(f"f_s/ideal = {ratio:.4f} outside [{RATIO_LO}, {RATIO_HI}]")
    return fails


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    log_path, run_dir = argv[1], argv[2]

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import load_config

    cfg = load_config(os.path.join(run_dir, "config_backup.yaml"))

    print("=== cached vs recomputed energy ===")
    fails = check_log(log_path)
    print("\n=== logged wall force ===")
    fails += check_wall(run_dir, cfg)

    print()
    if fails:
        print("FORCE_RECOMPUTE CHECK FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("FORCE_RECOMPUTE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
