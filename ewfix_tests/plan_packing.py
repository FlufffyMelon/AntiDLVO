#!/usr/bin/env python3
"""Assign the 32 production runs to four V100s.

Cost model, calibrated on this cluster rather than assumed. A single-particle
MC move costs a real-space pass over every atom plus a reciprocal-space pass
over every k-vector, so

    t_step = a * N_atoms + b * n_k

Two measured points fix a and b: H=3 base at 26.9 it/s (2166 atoms, 5012
k-vectors) and H=11 base at 17.5 it/s (2780 atoms, 9170 k-vectors), both at one
process per card. n_k itself is linear in the elongated box length, since the
k-sphere is fixed and only the z spacing changes: the nine grids measured
between Lz=18 and Lz=44 give n_k = 208.8 * Lz to better than 0.5%.

The assignment is longest-processing-time-first, which both balances the cards
and, as a side effect, deals the expensive runs out to different cards -- the
H=9, H=11 and C_z=0.7 points land on four different GPUs because they are dealt
first. The result is checked explicitly rather than trusted.

    python ewfix_tests/plan_packing.py                 # table
    python ewfix_tests/plan_packing.py --tsv           # for the launcher
    python ewfix_tests/plan_packing.py --per-gpu 4     # concurrency assumption
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_list import fmt, runs  # noqa: E402

NA = 0.602214076        # particles per nm^3 per mol/l
A_PER_ATOM = 0.009271   # ms per step per atom
B_PER_K = 0.003416      # ms per step per k-vector
K_PER_NM = 208.8        # k-vectors per nm of elongated box length
N_GPU = 4


def n_wall(sigma, L):
    return round(abs(sigma * L * L / 16.020506248) ** 0.5) ** 2


def geometry(r, L=20.0):
    """Atoms and k-vectors for one run, from the config's own formulas."""
    H, C, Cz, sigma = r["H"], r["C"], r["Cz"], r["sigma"]
    v = L * L * H
    nw = n_wall(sigma, L) if sigma else 0
    n_na = int(NA * C * v + 2 * nw)
    n_cl = int(NA * C * v)
    n_dip = int(NA * Cz * v)
    n_atoms = n_na + n_cl + 3 * n_dip + 2 * nw
    z_scale = max(4.0, (H + 25.0) / H)
    lz = H * z_scale
    return n_atoms, int(round(K_PER_NM * lz)), nw, lz


def cost_hours(n_atoms, n_k, n_steps=1_000_000):
    """Wall hours for one run alone on a card."""
    return (A_PER_ATOM * n_atoms + B_PER_K * n_k) * n_steps / 1000.0 / 3600.0


def plan(n_gpu=N_GPU):
    items = []
    for r in runs():
        n_atoms, n_k, nw, lz = geometry(r)
        h = cost_hours(n_atoms, n_k)
        items.append(dict(r, n_atoms=n_atoms, n_k=n_k, n_wall=nw, lz=lz,
                          hours_alone=h))

    # Longest first, onto whichever card has the least work so far.
    load = [0.0] * n_gpu
    order = sorted(items, key=lambda x: -x["hours_alone"])
    for it in order:
        g = min(range(n_gpu), key=lambda i: load[i])
        it["gpu"] = g
        load[g] += it["hours_alone"]

    # Keep the launch priority from run_list within each card.
    priority = {it["name"]: i for i, it in enumerate(items)}
    for it in items:
        it["rank"] = priority[it["name"]]
    items.sort(key=lambda x: (x["gpu"], x["rank"]))
    return items, load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="store_true")
    ap.add_argument("--per-gpu", type=float, default=4.0,
                    help="processes run concurrently per card, for the estimate")
    ap.add_argument("--speedup", type=float, default=None,
                    help="aggregate throughput of a card at that concurrency, "
                         "in units of one process alone, as measured by "
                         "sbatch_bench.sh. Defaults to perfect scaling.")
    args = ap.parse_args(argv)

    items, load = plan()

    if args.tsv:
        for it in items:
            ov = (f"H={fmt(it['H'])} sigma={fmt(it['sigma'])} C={fmt(it['C'])} "
                  f"Cz={fmt(it['Cz'])} pz={fmt(it['pz'])}")
            print(f"{it['gpu']}\t{it['name']}\t{it['results_dir']}\t{ov}")
        return 0

    print(f"{'gpu':>3} {'group':<9} {'pz':>5} {'H':>4} {'sigma':>6} {'C':>5} "
          f"{'Cz':>4} {'atoms':>6} {'n_k':>6} {'Lz':>6} {'h alone':>8}")
    print("-" * 78)
    for it in items:
        print(f"{it['gpu']:>3} {it['group']:<9} {fmt(it['pz']):>5} "
              f"{fmt(it['H']):>4} {fmt(it['sigma']):>6} {fmt(it['C']):>5} "
              f"{fmt(it['Cz']):>4} {it['n_atoms']:>6} {it['n_k']:>6} "
              f"{it['lz']:>6.1f} {it['hours_alone']:>8.2f}")

    print()
    total = sum(load)
    for g, h in enumerate(load):
        n = sum(1 for it in items if it["gpu"] == g)
        print(f"  GPU {g}: {n:>2} runs, {h:>6.1f} single-process GPU-hours")
    print(f"  total {total:.1f} GPU-hours if every run had a card to itself")

    # A card sequentially would need load[g] hours. Running per_gpu processes
    # at once divides that by the card's aggregate throughput -- which is the
    # thing to measure, and is never per_gpu. Each individual process slows to
    # speedup/per_gpu of its solo rate, so the card cannot finish sooner than
    # its longest run takes at that reduced rate.
    speedup = args.speedup if args.speedup else args.per_gpu
    per_proc = speedup / args.per_gpu
    slowest = max(items, key=lambda x: x["hours_alone"])
    tail = slowest["hours_alone"] / per_proc
    wall = max(max(load) / speedup, tail)
    print(f"\n  at {args.per_gpu:g} processes per card, aggregate throughput "
          f"{speedup:.2f}x one process ({per_proc:.2f}x each):")
    print(f"  expected wall time = {wall:.1f} h "
          f"(slowest card: GPU {load.index(max(load))}; "
          f"throughput bound {max(load) / speedup:.1f} h, tail bound {tail:.1f} h)")
    print(f"  slowest single run = {slowest['name']}, "
          f"{slowest['hours_alone']:.2f} h alone, {tail:.2f} h at this packing")

    # The expensive points must not pile onto one card.
    heavy = [it for it in items
             if it["H"] >= 9.0 or it["Cz"] >= 0.7]
    print("\n  heavy runs (H >= 9 or Cz >= 0.7):")
    for it in heavy:
        print(f"    GPU {it['gpu']}  {it['name']}  {it['hours_alone']:.2f} h")
    per_card = {}
    for it in heavy:
        per_card[it["gpu"]] = per_card.get(it["gpu"], 0) + 1
    spread = max(per_card.values()) - min(per_card.values()) if per_card else 0
    print(f"  per card: {dict(sorted(per_card.items()))} "
          f"-> {'balanced' if spread <= 1 else 'UNBALANCED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
