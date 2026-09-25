#!/usr/bin/env python3
"""Step 5: observable-level comparison between the old production runs and the
runs made with corrected Ewald parameters.

The analysis pipeline (contact-plane histograms, linear extrapolation to
contact, mid-pore densities, P_contact / P_b / Pi) is a faithful port of
disjoining_pressure_contact.ipynb, so the numbers are directly comparable with
what went into the paper.  Two deliberate differences:

  * caches are written to ewald_audit/results/cache/, never into results*/
    (the audit must not touch the production output);
  * the disjoining pressure is reported both with the logged solvation_force
    and with an externally supplied one, because the logged value is affected
    by the force-sign bug (see EWALD_AUDIT_REPORT.md, defect D5).

    python ewald_audit/scripts/observables.py DIR [DIR ...] [--label L ...]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import os.path as osp
import re
import time
from collections import deque
from itertools import islice

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
OUT = osp.join(osp.dirname(HERE), "results")
FIG = osp.join(osp.dirname(HERE), "figures")
CACHE_DIR = osp.join(OUT, "cache")

# ---- identical to the notebook -------------------------------------------
START_FRAME = 2000
STRIDE = 5
N_BLOCKS = 5
BIN = 0.005
FIT_BINS = 8
MID_HALF_WIDTH = 0.25
SITES = ["Na", "Cl", "Ip", "Im", "G"]

T = 298.0
KT = 8.314462618e-3 * T
TO_MPA = 1.66053907
EPS0 = 8.8541878128e-12
EPS_W = 78.0
E_CHARGE = 1.602176634e-19
R_DEFAULT = {"Na": 0.331, "Cl": 0.383}


def parse_folder(name):
    out = {}
    for key, pat in [("pz", r"_pz_([0-9p.]+)_"), ("H", r"_H_([-0-9.]+)_"),
                     ("sigma", r"_sigma_([-0-9.]+)_"), ("C", r"_C_([-0-9.]+)_"),
                     ("Cz", r"_Cz_([-0-9.]+)_")]:
        m = re.search(pat, name)
        if m:
            out[key] = float(m.group(1).replace("p", "."))
    return out


def site_radii(pz):
    r = dict(R_DEFAULT)
    r["Ip"] = r["Na"]
    r["Im"] = r["Cl"]
    r["G"] = pz / 48.032047 / 2.0
    return r


def radii_from_config(calc_dir, pz):
    r = site_radii(pz)
    try:
        import yaml
        with open(osp.join(calc_dir, "config_backup.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        for s in SITES:
            v = cfg.get(f"r_{s}")
            if isinstance(v, (int, float)):
                r[s] = float(v)
    except Exception:
        pass
    return r


def count_rows(p):
    with open(p) as f:
        return sum(1 for _ in f) - 1


def _skip(f, n):
    deque(islice(f, n), maxlen=0)


def cache_path_for(calc_dir):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.md5(osp.realpath(calc_dir).encode()).hexdigest()[:10]
    return osp.join(CACHE_DIR, f"{osp.basename(calc_dir.rstrip('/'))}_{key}.npz")


def build_cache(calc_dir, pz, start_frame=START_FRAME, rebuild=False):
    traj = osp.join(calc_dir, "trajectory.xyz")
    cache = cache_path_for(calc_dir)
    if (not rebuild and osp.isfile(cache)
            and osp.getmtime(cache) >= osp.getmtime(traj)):
        with np.load(cache, allow_pickle=True) as d:
            if int(d["start_frame"]) == start_frame and int(d["stride"]) == STRIDE:
                return cache

    radii = radii_from_config(calc_dir, pz)
    t0 = time.time()
    n_rows = count_rows(osp.join(calc_dir, "simulation_data.csv"))

    with open(traj) as f:
        n_atoms = int(f.readline())
        header = f.readline()
        lat = re.search(r'Lattice="([^"]+)"', header).group(1).split()
        Lx, Ly, H = float(lat[0]), float(lat[4]), float(lat[8])
        first = list(islice(f, n_atoms))
        species = np.array([ln.split(None, 1)[0] for ln in first])
        charges = np.array([float(ln.split()[4]) for ln in first])
        n_wb = int(np.sum(species == "Wb"))
        q_wb = float(np.sum(charges[species == "Wb"]))

        masks = {s: species == s for s in SITES if np.any(species == s)}
        nbins = {s: int(np.ceil((H / 2 - radii[s]) / BIN)) for s in masks}
        hist = {s: np.zeros((N_BLOCKS, nbins[s])) for s in masks}
        frames_in_block = np.zeros(N_BLOCKS, dtype=int)

        n_total = max(n_rows, start_frame + 1)
        block_len = max((n_total - start_frame) / N_BLOCKS, 1)

        k = 1
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            n = int(line)
            f.readline()
            if n != n_atoms:
                raise RuntimeError(f"{calc_dir}: atom count changes at frame {k}")
            if k >= start_frame and (k - start_frame) % STRIDE == 0:
                block = list(islice(f, n))
                if len(block) < n:
                    break
                z = pd.read_csv(io.StringIO("".join(block)), sep=r"\s+", header=None,
                                usecols=[3], engine="c").values[:, 0]
                b = min(int((k - start_frame) / block_len), N_BLOCKS - 1)
                for s, m in masks.items():
                    zs = z[m]
                    dd = np.minimum(zs, H - zs) - radii[s]
                    h, _ = np.histogram(dd, bins=nbins[s],
                                        range=(0.0, nbins[s] * BIN))
                    hist[s][b] += h
                frames_in_block[b] += 1
            else:
                _skip(f, n)
            k += 1

    np.savez(cache, start_frame=start_frame, stride=STRIDE, bin=BIN,
             n_blocks=N_BLOCKS, Lx=Lx, Ly=Ly, H=H, n_wb=n_wb, q_wb=q_wb,
             n_frames_total=k, frames_in_block=frames_in_block,
             sites=np.array(list(masks)),
             radii=np.array([radii[s] for s in masks]),
             **{f"hist_{s}": hist[s] for s in masks})
    print(f"  cache {osp.basename(calc_dir)}: {k} frames, {time.time()-t0:.0f}s",
          flush=True)
    return cache


def contact_value(dens, fit_bins=FIT_BINS):
    x = (np.arange(fit_bins) + 0.5) * BIN
    a, b = np.polyfit(x, dens[:fit_bins], 1)
    return b


def analyze(calc_dir, pz, start_frame=START_FRAME, fs_override=None):
    cache = build_cache(calc_dir, pz, start_frame)
    d = np.load(cache, allow_pickle=True)
    A, H = float(d["Lx"] * d["Ly"]), float(d["H"])
    fb = d["frames_in_block"]
    sites = list(d["sites"])
    radii = dict(zip(sites, d["radii"]))
    ok = fb > 0

    dens = {s: d[f"hist_{s}"][ok] / (fb[ok, None] * 2 * A * BIN) for s in sites}
    n_contact = {s: np.array([contact_value(dens[s][b]) for b in range(ok.sum())])
                 for s in sites}
    p_contact = sum(KT * n_contact[s] for s in sites)

    n_mid = {}
    for s in sites:
        nb = dens[s].shape[1]
        dist_to_mid = (H / 2 - radii[s]) - (np.arange(nb) + 0.5) * BIN
        sel = dist_to_mid < MID_HALF_WIDTH
        n_mid[s] = dens[s][:, sel].mean(axis=1)

    df = pd.read_csv(osp.join(calc_dir, "simulation_data.csv"))
    fs = df["solvation_force"].values[start_frame:]
    fs_blocks = np.array([b.mean() for b in np.array_split(fs, N_BLOCKS)])[ok]
    fs_logged = fs_blocks.copy()
    if fs_override is not None:
        fs_blocks = np.full_like(fs_blocks, float(fs_override))

    nNa = n_mid.get("Na", 0 * fs_blocks)
    nCl = n_mid.get("Cl", 0 * fs_blocks)
    nG = n_mid.get("G", 0 * fs_blocks)
    Pb = KT * (2 * np.sqrt(np.clip(nNa * nCl, 0, None)) + nG)
    PN = p_contact - fs_blocks
    Pi = PN - Pb

    # Second, independent route to the disjoining pressure, asked for
    # explicitly in the task: Pi = kT (sqrt(n+) - sqrt(n-))^2 at mid-plane.
    # It uses only mid-pore densities -- no contact extrapolation and no wall
    # force -- so comparing it with `Pi` above is a strong internal
    # consistency check on the whole force route.
    Pi_mid = KT * (np.sqrt(np.clip(nNa, 0, None))
                   - np.sqrt(np.clip(nCl, 0, None))) ** 2

    sigma_C_m2 = float(d["q_wb"]) * E_CHARGE / (A * 1e-18)

    # Contact-theorem balance quoted in the task:
    #   kT sum_i n_i(contact) - sigma^2/(2 eps0 eps)  ==  kT sum_i n_i(mid)
    p_smeared_ = sigma_C_m2 ** 2 / (2 * EPS0 * EPS_W) / 1e6 / TO_MPA
    p_mid_all = sum(KT * n_mid[s] for s in sites)
    balance = p_contact - p_smeared_ - p_mid_all

    def ms(x):
        x = np.asarray(x) * TO_MPA
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))
                                      if len(x) > 1 else np.nan)

    out = dict(dir=calc_dir, H_box=H, n_blocks_used=int(ok.sum()),
               n_frames_total=int(d["n_frames_total"]),
               sigma_eff=float(d["q_wb"]) * 16.0217663 / A,
               p_smeared=sigma_C_m2 ** 2 / (2 * EPS0 * EPS_W) / 1e6)
    for name, arr in [("P_contact", p_contact), ("f_s", fs_blocks),
                      ("f_s_logged", fs_logged), ("P_N", PN), ("P_b", Pb),
                      ("Pi", Pi), ("Pi_mid", Pi_mid), ("balance", balance),
                      ("p_mid_all", p_mid_all)]:
        out[name], out[name + "_err"] = ms(arr)
    for s in sites:
        out[f"nc_{s}"] = float(np.mean(n_contact[s]))
        out[f"nc_{s}_err"] = float(np.std(n_contact[s], ddof=1)
                                   / np.sqrt(len(n_contact[s])))
        out[f"nmid_{s}"] = float(np.mean(n_mid[s]))
        out[f"nmid_{s}_err"] = float(np.std(n_mid[s], ddof=1)
                                     / np.sqrt(len(n_mid[s])))
    out["_dens"] = {s: dens[s].mean(axis=0) for s in sites}
    out["_radii"] = radii
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--label", nargs="+", default=None,
                    help="one label per dir (default: basename)")
    ap.add_argument("--pz", type=float, default=17.0)
    ap.add_argument("--start-frame", type=int, default=START_FRAME)
    ap.add_argument("--fs-override", nargs="+", type=float, default=None,
                    help="replace the logged solvation_force (kJ/mol/nm^3)")
    ap.add_argument("--tag", default="step5")
    args = ap.parse_args()

    labels = args.label or [osp.basename(d.rstrip("/")) for d in args.dirs]
    ovr = args.fs_override or [None] * len(args.dirs)
    res = []
    for d, lab, o in zip(args.dirs, labels, ovr):
        pz = parse_folder(osp.basename(d.rstrip("/"))).get("pz", args.pz)
        r = analyze(d, pz, args.start_frame, fs_override=o)
        r["label"] = lab
        res.append(r)
        print(f"\n### {lab}   ({r['n_frames_total']} frames, "
              f"{r['n_blocks_used']} blocks, H={r['H_box']})")
        for k in ("P_contact", "f_s", "f_s_logged", "P_N", "P_b", "Pi",
                  "Pi_mid", "p_mid_all", "balance"):
            print(f"   {k:>12} = {r[k]:9.4f} +- {r[k+'_err']:.4f} MPa")
        print(f"   {'p_smeared':>12} = {r['p_smeared']:9.4f} MPa")
        for s in r["_radii"]:
            print(f"   n_contact[{s:>2}] = {r['nc_'+s]:9.4f} +- {r['nc_'+s+'_err']:.4f}"
                  f"   n_mid[{s:>2}] = {r['nmid_'+s]:8.5f} +- {r['nmid_'+s+'_err']:.5f} nm^-3")

    os.makedirs(FIG, exist_ok=True)
    tab = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                        for r in res])
    tab.to_csv(osp.join(OUT, f"observables_{args.tag}.csv"), index=False)
    with open(osp.join(OUT, f"observables_{args.tag}.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in res], f, indent=1)

    # profiles: one panel per site present in all runs
    common = [s for s in SITES if all(s in r["_dens"] for r in res)]
    if common:
        fig, axes = plt.subplots(1, len(common), figsize=(3.4 * len(common), 3.8),
                                 squeeze=False)
        for ax, s in zip(axes[0], common):
            for r in res:
                n = r["_dens"][s]
                x = (np.arange(len(n)) + 0.5) * BIN
                ax.plot(x, n, lw=1.2, label=r["label"])
            ax.set(xlabel=r"$d=\min(z,H-z)-r_i$, nm", ylabel=r"$n$, nm$^{-3}$",
                   yscale="log", title=s, xlim=(0, 1.0))
            ax.grid(alpha=.3)
        axes[0][0].legend(fontsize=6)
        fig.suptitle("Density profiles at the wall: old vs corrected Ewald")
        fig.tight_layout()
        fig.savefig(osp.join(FIG, f"profiles_{args.tag}.png"), dpi=150)
        print(f"\n-> {osp.join(FIG, f'profiles_{args.tag}.png')}")
    print(f"-> {osp.join(OUT, f'observables_{args.tag}.csv')}")


if __name__ == "__main__":
    main()
