#!/usr/bin/env python3
"""Step 1: inventory of Ewald parameters over all production runs.

Reads config_backup.yaml from every run directory under the given roots and
builds a table with the box, Ewald parameters, the actual k_max per axis and
the k_max required by the code's own accuracy formula.

Usage:  python inventory.py results_prod_neutral results_prod
Output: ewald_audit/results/inventory_<root>.csv (+ printed table)
"""
import sys
import os
import math
import csv
import glob
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(OUT, exist_ok=True)


def scan(root):
    rows = []
    for cfgpath in sorted(glob.glob(os.path.join(root, "**", "config_backup.yaml"), recursive=True)):
        rundir = os.path.dirname(cfgpath)
        with open(cfgpath) as f:
            cfg = yaml.safe_load(f)

        ew = cfg.get("ewald", {}) or {}
        box = cfg.get("system", {}).get("box", [None, None, None])
        pbc = cfg.get("system", {}).get("pbc", None)
        Lx, Ly, Lz = [float(b) for b in box]
        zs = float(ew.get("z_scale_factor", 1.0))
        Lz_eff = Lz * zs
        alpha = float(ew.get("alpha", float("nan")))
        rc = float(ew.get("real_cut", float("nan")))
        n_c = int(ew.get("n_c", 0))
        eps = float(ew.get("eps", float("nan")))

        # actual k_max per axis (largest |k_i| on the integer grid)
        kmax_x = n_c * 2 * math.pi / Lx
        kmax_y = n_c * 2 * math.pi / Ly
        kmax_z = n_c * 2 * math.pi / Lz_eff

        # required k_max from the code's own convention: exp(-k^2/(4 alpha)) = eps
        kmax_req = 2.0 * math.sqrt(alpha * (-math.log(eps)))
        # required n_c per axis if the grid were anisotropic
        nx_req = math.ceil(kmax_req * Lx / (2 * math.pi))
        ny_req = math.ceil(kmax_req * Ly / (2 * math.pi))
        nz_req = math.ceil(kmax_req * Lz_eff / (2 * math.pi))

        # real-space truncation error in the code's convention: erfc(sqrt(alpha)*rc)
        try:
            from math import erfc
            real_err = erfc(math.sqrt(alpha) * rc)
        except Exception:
            real_err = float("nan")
        # reciprocal-space Gaussian weight left at the actual cut (per axis)
        gx = math.exp(-kmax_x ** 2 / (4 * alpha))
        gz = math.exp(-kmax_z ** 2 / (4 * alpha))

        # counts of atoms by type from the config
        counts = {}
        for entry in cfg.get("types", []) or []:
            for kind, spec in entry.items():
                name = spec.get("type") or spec.get("type_plus")
                n = int(round(float(spec.get("count", 0))))
                counts[name] = n
                if kind == "Dipole":
                    counts["_ndip"] = n
        n_wall = counts.get("Wb", 0)
        n_atoms = (counts.get("Na", 0) + counts.get("Cl", 0)
                   + 3 * counts.get("_ndip", 0) + 2 * n_wall)

        rows.append(dict(
            run=os.path.relpath(rundir, root),
            pz=cfg.get("pz"), H=cfg.get("H"), L=cfg.get("L"),
            sigma=cfg.get("sigma"), C=cfg.get("C"), Cz=cfg.get("Cz"),
            n_steps=cfg.get("simulation", {}).get("n_steps"),
            Lx=Lx, Ly=Ly, Lz=Lz, z_scale=zs, Lz_eff=Lz_eff,
            pbc=str(pbc),
            N_wall=n_wall, N_atoms=n_atoms,
            eps=eps, real_cut=rc, alpha=alpha, n_c=n_c,
            dipole_correction=ew.get("dipole_correction"),
            dielectric=ew.get("dielectric"),
            kmax_x=kmax_x, kmax_y=kmax_y, kmax_z=kmax_z,
            kmax_req=kmax_req,
            nx_req=nx_req, ny_req=ny_req, nz_req=nz_req,
            real_err_erfc=real_err,
            gauss_left_xy=gx, gauss_left_z=gz,
            n_kvec=(2 * n_c + 1) ** 3 - 1,
            n_kvec_req=(2 * nx_req + 1) * (2 * ny_req + 1) * (2 * nz_req + 1) - 1,
        ))
    return rows


def main():
    roots = sys.argv[1:] or ["results_prod_neutral"]
    for root in roots:
        rows = scan(root)
        if not rows:
            print(f"[{root}] no config_backup.yaml found")
            continue
        tag = root.rstrip("/").replace("/", "_")
        path = os.path.join(OUT, f"inventory_{tag}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n### {root}: {len(rows)} runs -> {path}")
        hdr = ("pz", "H", "sigma", "C", "Cz", "N_atoms", "alpha", "real_cut",
               "n_c", "kmax_x", "kmax_z", "kmax_req", "nz_req", "n_steps")
        print(" | ".join(f"{h:>9}" for h in hdr))
        for r in rows:
            print(" | ".join(
                f"{r[h]:>9.3f}" if isinstance(r[h], float) else f"{str(r[h]):>9}"
                for h in hdr))
        # uniformity check
        keys = ("eps", "real_cut", "alpha", "n_c", "z_scale",
                "dipole_correction", "dielectric", "Lx", "Ly", "pbc")
        print("--- uniformity across runs ---")
        for k in keys:
            vals = sorted({repr(r[k]) for r in rows})
            print(f"  {k:20s}: {'SAME ' if len(vals) == 1 else 'DIFFER '} {vals}")


if __name__ == "__main__":
    main()
