#!/usr/bin/env python3
"""Aggregate the per-run JSONs from frame_audit.py into tables and figures."""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
FIG = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(FIG, exist_ok=True)

MOVES = ["Na_wall", "Na_bulk", "Cl", "dipole"]
MPA = 1.66054


def parse_tag(tag):
    out = {}
    m = re.search(r"pz_(27p6|17)", tag)
    out["pz"] = 27.6 if (m and m.group(1) == "27p6") else 17.0
    for key, pat in (("H", r"_H_(\d+)_"), ("sigma", r"_sigma_(-?\d+)_"),
                     ("C", r"_C_([\d.]+)_Cz"), ("Cz", r"_Cz_([\d.]+)_\d{8}")):
        m = re.search(pat, tag)
        out[key] = float(m.group(1)) if m else None
    out["neutral"] = "results_prod_neutral" in tag
    out["sweep"] = ("H" if "__H__" in tag else "sigma" if "__sigma__" in tag
                    else "C" if "__C__" in tag else "Cz" if "__Cz__" in tag
                    else "?")
    return out


def mean_over_frames(frames, path):
    vals = []
    for fr in frames:
        node = fr
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node and node[key] is not None:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (int, float)):
            vals.append(float(node))
    return float(np.mean(vals)) if vals else None


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "frames", "*.json"))):
        d = json.load(open(p))
        frames = d["frames"]
        if not frames:
            continue
        r = dict(tag=d["tag"], run=d["run_dir"], N=d["N_atoms"], **parse_tag(d["tag"]))
        r["H"] = d["H"]
        r["ref_z_scale"] = d["ref_ewald"]["z_scale"]
        r["ref_n_k"] = d["ref_ewald"]["n_k"]
        r["n_frames_used"] = len(frames)
        # --- energies
        for k in ("err_total_kT", "err_per_atom_kT", "err_real_kT",
                  "err_recip_kT", "err_slab_kT", "prod_total_kT",
                  "ref_total_no_self_kT", "ref_recip_kT", "ref_slab_kT"):
            r[k] = mean_over_frames(frames, ["energy", k])
        # --- moves
        for mv in MOVES:
            for amp in ("large", "small"):
                key = f"{mv}/{amp}"
                for stat, name in (("rms_err_kT", "rms"), ("max_err_kT", "max"),
                                   ("mean_dacc", "dacc"),
                                   ("frac_dacc_gt_0p1", "f01"),
                                   ("rms_dE_ref_kT", "dEref"),
                                   ("rms_err_wo_slab_kT", "rmsnoslab"),
                                   ("rms_slab_delta_kT", "slabdelta")):
                    r[f"{name}_{mv}_{amp}"] = mean_over_frames(
                        frames, ["moves", key, stat])
        # worst case over move types (large amplitude = production amplitude)
        r["rms_worst_large"] = max(
            [r[f"rms_{m}_large"] for m in MOVES if r.get(f"rms_{m}_large") is not None]
            or [np.nan])
        r["f01_worst_large"] = max(
            [r[f"f01_{m}_large"] for m in MOVES if r.get(f"f01_{m}_large") is not None]
            or [np.nan])
        r["rms_worst_small"] = max(
            [r[f"rms_{m}_small"] for m in MOVES if r.get(f"rms_{m}_small") is not None]
            or [np.nan])
        # --- wall
        for k in ("prod", "prod_real_sign_flipped", "ref", "ideal", "logged",
                  "sigma_eff_e_per_nm2"):
            r["wall_" + k] = mean_over_frames(frames, ["wall", k])
        if r["wall_ref"]:
            r["wall_err_frac"] = (r["wall_prod"] - r["wall_ref"]) / r["wall_ref"]
            r["wall_err_frac_fixed"] = (
                r["wall_prod_real_sign_flipped"] - r["wall_ref"]) / r["wall_ref"]
        r["wall_log_equals_recompute"] = (
            abs((r["wall_logged"] or 0) - (r["wall_prod"] or 0)) < 1e-6
            if r["wall_logged"] is not None else None)
        # --- fd force check
        fd = frames[0].get("fd_force") or []
        if fd:
            r["fd_err_as_coded"] = float(np.mean([x["err_as_coded"] / max(x["scale"], 1e-12) for x in fd]))
            r["fd_err_if_flipped"] = float(np.mean([x["err_if_real_flipped"] / max(x["scale"], 1e-12) for x in fd]))
        # --- reference self-consistency
        conv = frames[0].get("ref_convergence") or {}
        r["ref_selfcheck_max_kT"] = (
            max(abs(v["dE_kT"]) for v in conv.values()) if conv else None)
        rows.append(r)
    return rows


def verdict(r):
    """Verdict on the MC sampling quality (electrostatics only)."""
    rms = r.get("rms_worst_large")
    f01 = r.get("f01_worst_large")
    if rms is None or np.isnan(rms):
        return "?"
    if rms < 0.05 and f01 < 0.01:
        return "godится"
    if rms < 0.30:
        return "godится s ogovorkami"
    return "pereschityvat"


def main():
    rows = load_all()
    if not rows:
        print("no results yet")
        return
    rows.sort(key=lambda r: (r["pz"], r["sweep"], r["H"], r["sigma"] or 0))
    for r in rows:
        r["verdict_sampling"] = verdict(r)

    path = os.path.join(RES, "summary.csv")
    keys = sorted({k for r in rows for k in r})
    lead = ["tag", "pz", "sweep", "H", "sigma", "C", "Cz", "N", "neutral",
            "rms_worst_large", "f01_worst_large", "rms_worst_small",
            "err_total_kT", "err_per_atom_kT", "verdict_sampling"]
    keys = lead + [k for k in keys if k not in lead]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}\n")

    # ------------------------------------------------------------- printed table
    hdr = (f"{'pz':>5} {'sweep':>6} {'H':>4} {'sig':>5} {'C':>5} {'Cz':>4} "
           f"{'N':>5} | {'rms_lg':>7} {'f>0.1':>6} {'rms_sm':>7} | "
           f"{'dE_tot':>9} {'dE/at':>8} | {'wall_prod':>9} {'wall_fix':>9} "
           f"{'wall_ref':>9} {'ideal':>8} | verdict")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['pz']:>5} {r['sweep']:>6} {r['H']:>4.0f} "
              f"{(r['sigma'] if r['sigma'] is not None else 0):>5.0f} "
              f"{(r['C'] if r['C'] is not None else 0):>5.2f} "
              f"{(r['Cz'] if r['Cz'] is not None else 0):>4.1f} {r['N']:>5} | "
              f"{r['rms_worst_large']:>7.3f} {r['f01_worst_large']:>6.3f} "
              f"{r['rms_worst_small']:>7.3f} | "
              f"{r['err_total_kT']:>9.2f} {r['err_per_atom_kT']:>8.4f} | "
              f"{(r['wall_prod'] or 0):>9.3f} "
              f"{(r['wall_prod_real_sign_flipped'] or 0):>9.3f} "
              f"{(r['wall_ref'] or 0):>9.3f} {(r['wall_ideal'] or 0):>8.3f} | "
              f"{r['verdict_sampling']}")

    # ------------------------------------------------------------------ figures
    def subset(sweep, pz, neutral=True):
        return [r for r in rows if r["sweep"] == sweep and r["pz"] == pz
                and r["neutral"] == neutral]

    # 1. dE error vs H
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for pz, c in ((17.0, "C0"), (27.6, "C1")):
        s = sorted(subset("H", pz) + [r for r in rows if r["sweep"] == "sigma"
                                      and r["sigma"] == -20 and r["pz"] == pz
                                      and r["neutral"]],
                   key=lambda r: r["H"])
        if not s:
            continue
        ax[0].plot([r["H"] for r in s], [r["rms_worst_large"] for r in s],
                   "o-", color=c, label=f"pz={pz} D, large moves")
        ax[0].plot([r["H"] for r in s], [r["rms_worst_small"] for r in s],
                   "s--", color=c, alpha=0.6, label=f"pz={pz} D, small moves")
        ax[1].plot([r["H"] for r in s], [r["f01_worst_large"] for r in s],
                   "o-", color=c, label=f"pz={pz} D")
    ax[0].axhline(0.05, color="g", ls=":", label="0.05 kT threshold")
    ax[0].axhline(0.30, color="r", ls=":", label="0.30 kT threshold")
    ax[0].set_xlabel("H, nm"); ax[0].set_ylabel(r"RMS $|\Delta E_{prod}-\Delta E_{ref}|$, kT")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].axhline(0.01, color="g", ls=":", label="1% threshold")
    ax[1].set_xlabel("H, nm"); ax[1].set_ylabel(r"fraction of moves with $|\Delta P_{acc}|>0.1$")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.suptitle("Ewald error in MC energy differences vs slit width")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dE_error_vs_H.png"), dpi=150)

    # 2. dE error vs sigma
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for pz, c in ((17.0, "C0"), (27.6, "C1")):
        s = sorted(subset("sigma", pz), key=lambda r: r["sigma"])
        if not s:
            continue
        ax.plot([abs(r["sigma"]) for r in s], [r["rms_worst_large"] for r in s],
                "o-", color=c, label=f"pz={pz} D, large")
        ax.plot([abs(r["sigma"]) for r in s], [r["rms_worst_small"] for r in s],
                "s--", color=c, alpha=.6, label=f"pz={pz} D, small")
    ax.axhline(0.05, color="g", ls=":"); ax.axhline(0.30, color="r", ls=":")
    ax.set_xlabel(r"$|\sigma|$, $\mu$C/cm$^2$"); ax.set_ylabel("RMS dE error, kT")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dE_error_vs_sigma.png"), dpi=150)

    # 3. wall force
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    s = sorted([r for r in rows if r["sweep"] == "sigma" and r["pz"] == 17.0
                and r["neutral"]], key=lambda r: abs(r["sigma"]))
    if s:
        x = [abs(r["sigma"]) for r in s]
        ax[0].plot(x, [r["wall_ref"] * MPA for r in s], "k-o", label="reference")
        ax[0].plot(x, [r["wall_prod"] * MPA for r in s], "r-s", label="production (logged)")
        ax[0].plot(x, [r["wall_prod_real_sign_flipped"] * MPA for r in s], "b--^",
                   label="production, real-force sign fixed")
        ax[0].plot(x, [r["wall_ideal"] * MPA for r in s], "g:.",
                   label=r"$\sigma_{eff}^2/(2\varepsilon_0\varepsilon)$")
        ax[0].set_xlabel(r"$|\sigma|$, $\mu$C/cm$^2$")
        ax[0].set_ylabel("force on wall per area, MPa")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    s = sorted([r for r in rows if r["sweep"] == "H" and r["pz"] == 17.0
                and r["neutral"]], key=lambda r: r["H"])
    if s:
        x = [r["H"] for r in s]
        ax[1].plot(x, [r["wall_ref"] * MPA for r in s], "k-o", label="reference")
        ax[1].plot(x, [r["wall_prod"] * MPA for r in s], "r-s", label="production (logged)")
        ax[1].plot(x, [r["wall_prod_real_sign_flipped"] * MPA for r in s], "b--^",
                   label="real-force sign fixed")
        ax[1].set_xlabel("H, nm"); ax[1].set_ylabel("force on wall per area, MPa")
        ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.suptitle("Force on the charged walls: production vs reference")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "wall_force.png"), dpi=150)

    # 4. error attribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    s = [r for r in rows if r["neutral"]]
    s.sort(key=lambda r: (r["pz"], r["H"], r["sigma"] or 0))
    idx = np.arange(len(s))
    ax.bar(idx - 0.2, [abs(r["err_real_kT"]) for r in s], 0.2, label="real")
    ax.bar(idx, [abs(r["err_recip_kT"]) for r in s], 0.2, label="reciprocal")
    ax.bar(idx + 0.2, [abs(r["err_slab_kT"]) for r in s], 0.2, label="slab")
    ax.set_yscale("log"); ax.set_ylabel("|error| in total energy, kT")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"pz{r['pz']:.0f} H{r['H']:.0f} s{r['sigma']:.0f}" for r in s],
                       rotation=90, fontsize=6)
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "error_attribution.png"), dpi=150)

    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
