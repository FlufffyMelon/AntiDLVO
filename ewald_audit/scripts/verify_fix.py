#!/usr/bin/env python3
"""Does the *proposed* parameter set actually reach the target accuracy?

Re-runs the step-4 comparison, but with the patched code copy
(ewald_audit/codefix/src) and the corrected Ewald parameters instead of the
production ones.  Same frames, same reference, same metrics -- so the numbers
are directly comparable with ewald_audit/results/frames/*.json.

    python ewald_audit/scripts/verify_fix.py RUNDIR [RUNDIR ...] \
        --frames 2000 --n-moves 150 --gpu 0

Output: ewald_audit/results/frames_fixed/<tag>.json
"""
from __future__ import annotations

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # ~/AntiDLVO
CODEFIX = os.path.join(ROOT, "ewald_audit", "codefix")

# Resolve `src.*` to the PATCHED copy, and lock it into sys.modules *before*
# frame_audit is imported -- frame_audit rewrites sys.path on import, but an
# already-imported module wins over any path order.
sys.path.insert(0, ROOT)
sys.path.insert(0, CODEFIX)
import src  # noqa: E402
import src.utils  # noqa: E402,F401
import src.system  # noqa: E402,F401
import src.topology  # noqa: E402,F401
import src.ewald_handler  # noqa: E402,F401
import src.forces.ewald_particle  # noqa: E402,F401

assert os.path.abspath(src.__file__).startswith(CODEFIX), \
    f"src resolved to {src.__file__}, expected the patched copy under {CODEFIX}"

sys.path.insert(0, HERE)
from omegaconf import OmegaConf  # noqa: E402
import frame_audit as fa  # noqa: E402

# ---------------------------------------------------------------- parameters
# Proposed production setting (see EWALD_AUDIT_REPORT.md section 9).
EPS = 1.0e-6
R_CUT = 9.0          # nm; < L/2 = 10 nm, and cheap because the production
                     # real-space kernel computes all pair distances anyway


def z_scale_for(H):
    """Vacuum gap of at least 3 max(Lx,Ly); floor of 4 as in production."""
    return 8.0 if H <= 4.0 else 4.0


def corrected_build(run_dir, use_gpu=None):
    cfg = OmegaConf.load(os.path.join(run_dir, "config_backup.yaml"))
    if use_gpu is not None:
        cfg.ewald.use_gpu = bool(use_gpu)

    Lx, Ly, H = [float(b) for b in cfg.system.box]
    a = math.sqrt(-math.log(EPS)) / R_CUT
    z_scale = z_scale_for(H)
    k_max = 2.0 * math.sqrt(a * a * (-math.log(EPS)))
    n_c = int(math.ceil(k_max * max(Lx, Ly, H * z_scale) / (2.0 * math.pi)))

    cfg.ewald.eps = EPS
    cfg.ewald.real_cut = R_CUT
    cfg.ewald.alpha = a * a          # the code uses alpha as a^2
    cfg.ewald.z_scale_factor = z_scale
    cfg.ewald.n_c = n_c
    cfg.ewald.dipole_correction = True
    fa.relax_initial_placement(cfg)
    print(f"  corrected Ewald: alpha={a*a:.5f} (a={a:.4f}) r_cut={R_CUT} "
          f"z_scale={z_scale} k_max={k_max:.4f} n_c={n_c}", flush=True)

    units = fa.create_units(cfg)
    system = fa.create_system(cfg, units)
    topology = fa.create_topology(cfg, units)
    fa.setup_initial_configuration(system, cfg.get("types", []), topology, cfg=cfg)
    return cfg, units, system, topology


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", type=int, nargs="+", default=[2000])
    ap.add_argument("--n-moves", type=int, default=150)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "frames_fixed"))
    args = ap.parse_args()

    fa.build_production = corrected_build   # audit_run looks this up globally
    os.makedirs(args.out, exist_ok=True)

    for run in args.runs:
        tag = run.rstrip("/").replace("/", "__")
        fa.audit_run(run, args.frames, args.n_moves, bool(args.gpu), args.out,
                     tag=tag, convergence_check=False)


if __name__ == "__main__":
    main()
