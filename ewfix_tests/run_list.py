#!/usr/bin/env python3
"""The 32 production runs, in one place.

The launcher, the manifest and the QA pass all read the list from here, so
there is exactly one definition of what "the production set" means.

Naming follows the old scheme exactly, so the analysis notebooks keep working:

    results_ewfix/pz_<17|27p6>/<group>/ions_dipole_<group>_pz_<pz>_H_<H>
        _sigma_<sigma>_C_<C>_Cz_<Cz>_<timestamp>/

    python ewfix_tests/run_list.py            # human-readable table
    python ewfix_tests/run_list.py --tsv      # machine-readable, for bash
"""
from __future__ import annotations

import argparse
import sys

# Base point, shared by every sweep.
BASE = dict(H=3.0, sigma=-20.0, C=0.01, Cz=0.1)

PZ_VALUES = (17.0, 27.6)

# Launch priority, as set by the task: the points the paper leans on first.
GROUP_ORDER = ("base", "base_rep", "Cz", "H", "sigma", "C")

SWEEPS = {
    "sigma": ("sigma", (0.0, -5.0, -10.0, -15.0)),
    "C": ("C", (0.0, 0.04)),
    "Cz": ("Cz", (0.0, 0.3, 0.5, 0.7)),
    "H": ("H", (5.0, 7.0, 9.0, 11.0)),
}


def fmt(v: float) -> str:
    """Render a parameter the way the old run directories do (0.01, 0, -20, 3)."""
    return f"{v:g}"


def pz_tag(pz: float) -> str:
    return fmt(pz).replace(".", "p")


def runs():
    """Yield the 32 runs in launch order."""
    for group in GROUP_ORDER:
        for pz in PZ_VALUES:
            if group in ("base", "base_rep"):
                points = [dict(BASE)]
            else:
                key, values = SWEEPS[group]
                points = []
                for v in values:
                    p = dict(BASE)
                    p[key] = v
                    points.append(p)
            for p in points:
                name = (f"ions_dipole_{group}_pz_{pz_tag(pz)}"
                        f"_H_{fmt(p['H'])}_sigma_{fmt(p['sigma'])}"
                        f"_C_{fmt(p['C'])}_Cz_{fmt(p['Cz'])}")
                yield dict(
                    group=group, pz=pz, name=name,
                    results_dir=f"results_ewfix/pz_{pz_tag(pz)}/{group}",
                    **p,
                )


def base_symlinks():
    """Where each sweep expects to find the base point.

    The old layout has the base point physically present inside every sweep
    directory. Rerunning it four times would waste GPU-days and, worse, give
    four differently-seeded "base points", so the sweeps get symlinks to the
    single base run instead.
    """
    for group in ("H", "sigma", "C", "Cz"):
        for pz in PZ_VALUES:
            link = (f"ions_dipole_{group}_pz_{pz_tag(pz)}"
                    f"_H_{fmt(BASE['H'])}_sigma_{fmt(BASE['sigma'])}"
                    f"_C_{fmt(BASE['C'])}_Cz_{fmt(BASE['Cz'])}")
            yield dict(pz=pz, group=group,
                       link_dir=f"results_ewfix/pz_{pz_tag(pz)}/{group}",
                       link_name=link,
                       target_dir=f"results_ewfix/pz_{pz_tag(pz)}/base")


FIELDS = ("group", "pz", "H", "sigma", "C", "Cz", "name", "results_dir")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="store_true", help="tab-separated, no header")
    ap.add_argument("--links", action="store_true", help="list base symlinks")
    args = ap.parse_args(argv)

    if args.links:
        for s in base_symlinks():
            if args.tsv:
                print(f"{s['link_dir']}\t{s['link_name']}\t{s['target_dir']}")
            else:
                print(f"{s['link_dir']}/{s['link_name']} -> {s['target_dir']}/<base>")
        return 0

    rows = list(runs())
    if args.tsv:
        for r in rows:
            print("\t".join(fmt(r[f]) if isinstance(r[f], float) else str(r[f])
                            for f in FIELDS))
    else:
        print(f"{'#':>3} {'group':<9} {'pz':>5} {'H':>5} {'sigma':>6} "
              f"{'C':>5} {'Cz':>4}  name")
        for i, r in enumerate(rows, 1):
            print(f"{i:>3} {r['group']:<9} {fmt(r['pz']):>5} {fmt(r['H']):>5} "
                  f"{fmt(r['sigma']):>6} {fmt(r['C']):>5} {fmt(r['Cz']):>4}  "
                  f"{r['name']}")
        print(f"\n{len(rows)} runs")
        assert len(rows) == 32, f"expected 32 runs, got {len(rows)}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
