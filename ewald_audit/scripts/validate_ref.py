#!/usr/bin/env python3
"""Step 3: validate the reference Ewald implementation.

1. NaCl Madelung constant (3D, no slab correction)      -> 1.7475646
2. Convergence of the slab reference: k_max x1.5, z_scale 4 / 6 / 8
3. Independent check against the exact Ewald2D (Parry/Heyes) for a pair and
   for a small random set of charges in a slab.
"""
import os
import sys
import math
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_ewald import RefEwald3DSlab, ewald2d_energy, madelung_nacl, KC, KB

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

USE_GPU = os.environ.get("AUDIT_GPU", "1") == "1"
report = {}


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- 1. Madelung
sec("1. NaCl Madelung constant (3D periodic, slab correction off)")
exact = 1.747564594633
res = {}
for n in (1, 2):
    M, terms = madelung_nacl(n_cell=n)
    res[f"n_cell={n}"] = M
    print(f"  n_cell={n}  N={8*n**3:4d}  M = {M:.9f}   error = {M-exact:+.2e}")
report["madelung"] = dict(values=res, exact=exact)

# ------------------------------------------------------- 2. slab convergence
sec("2. Convergence of the slab reference")
rng = np.random.default_rng(7)


def make_slab_system(N, Lx, Ly, H, seed=3):
    r = np.random.default_rng(seed)
    pos = np.column_stack([
        r.random(N) * Lx, r.random(N) * Ly, 0.3 + r.random(N) * (H - 0.6)
    ])
    q = np.ones(N)
    q[N // 2:] = -1.0
    return pos, q


rows = []
for H in (3.0, 11.0):
    pos, q = make_slab_system(120, 20.0, 20.0, H)
    base = None
    for z_scale in (4.0, 6.0, 8.0):
        for kfac in (1.0, 1.5):
            rc = min(0.45 * H * z_scale, 5.0)
            a = math.sqrt(-math.log(1e-8)) / rc
            alpha = a * a
            kmax = 2.0 * math.sqrt(alpha * (-math.log(1e-8))) * kfac
            ew = RefEwald3DSlab(20.0, 20.0, H, z_scale=z_scale, alpha=alpha,
                                r_cut=rc, k_max=kmax, lB=KC / 78.0,
                                use_gpu=USE_GPU)
            t = ew.energy_terms(pos, q)
            E = t["total"] / (KB * 298.0)
            if base is None:
                base = E
            rows.append(dict(H=H, z_scale=z_scale, kfac=kfac, n_k=ew.n_k,
                             r_cut=rc, alpha=alpha, k_max=kmax,
                             E_kT=E, dE_kT=E - base))
            print(f"  H={H:4.1f} z_scale={z_scale:.0f} k_max x{kfac:.1f} "
                  f"n_k={ew.n_k:8d} r_c={rc:.2f} E={E:16.6f} kT  "
                  f"dE={E-base:+.3e} kT ({(E-base)/len(pos):+.2e} kT/atom)")
report["slab_convergence"] = rows

# ------------------------------------------- 3. independent check vs Ewald2D
sec("3. Reference vs exact Ewald2D (independent)")
lB = 1.0  # work in q^2/nm units here
cases = []

# (a) a +/- pair at several separations, H = 3 and H = 11
for H in (3.0, 11.0):
    for label, dr in (("vertical 0.33", np.array([0.0, 0.0, 0.33])),
                      ("lateral 0.66", np.array([0.66, 0.0, 0.0])),
                      ("lateral 2.0", np.array([2.0, 0.0, 0.0]))):
        z0 = 0.5 if H > 1.5 else 0.3
        p1 = np.array([5.0, 5.0, z0])
        p2 = p1 + dr
        pos = np.array([p1, p2])
        q = np.array([1.0, -1.0])
        cases.append((f"pair {label}, H={H}", pos, q, H))

# (b) a small random neutral set
for H in (3.0, 11.0):
    pos, q = make_slab_system(40, 20.0, 20.0, H, seed=11)
    cases.append((f"40 charges, H={H}", pos, q, H))

rows = []
for name, pos, q, H in cases:
    # exact Ewald2D
    a2d = 5.0 / 20.0
    e2d = ewald2d_energy(pos, q, 20.0, 20.0, a=a2d, r_img=2, eps=1e-12)
    # reference 3D + slab
    vals = {}
    for z_scale in (4.0, 8.0):
        rc = min(0.45 * H * z_scale, 5.0)
        a = math.sqrt(-math.log(1e-10)) / rc
        alpha = a * a
        kmax = 2.0 * math.sqrt(alpha * (-math.log(1e-10)))
        ew = RefEwald3DSlab(20.0, 20.0, H, z_scale=z_scale, alpha=alpha,
                            r_cut=rc, k_max=kmax, lB=lB, use_gpu=USE_GPU)
        t = ew.energy_terms(pos, q)
        vals[z_scale] = t["total"]
    print(f"  {name:28s}  Ewald2D = {e2d['total']:14.8f}   "
          f"3D+slab(z4) = {vals[4.0]:14.8f} ({vals[4.0]-e2d['total']:+.2e})   "
          f"3D+slab(z8) = {vals[8.0]:14.8f} ({vals[8.0]-e2d['total']:+.2e})")
    rows.append(dict(case=name, ewald2d=e2d["total"],
                     ref_z4=vals[4.0], ref_z8=vals[8.0],
                     err_z4=vals[4.0] - e2d["total"],
                     err_z8=vals[8.0] - e2d["total"]))
report["vs_ewald2d"] = rows

with open(os.path.join(OUT, "validate_ref.json"), "w") as f:
    json.dump(report, f, indent=2, default=float)
print(f"\nwritten: {os.path.join(OUT, 'validate_ref.json')}")
