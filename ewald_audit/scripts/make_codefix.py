#!/usr/bin/env python3
"""Build ewald_audit/codefix/ -- a patched *copy* of the production code.

The production src/ and configs_prod/ are never touched (task constraint).
Three defects are fixed, each marked with an `AUDIT FIX` comment:

  1. src/ewald_handler.py  initialize_k_vectors
     isotropic integer grid -n_c..n_c on all three axes  ->  anisotropic grid
     n_i = ceil(k_max L_i / 2pi) with a spherical cutoff |k| <= k_max, where
     k_max is now derived from n_c and the *longest* box side.

  2. src/ewald_handler.py  delta_dipole_correction_energy_forces
     the "new" energy was evaluated at the OLD positions, so the slab dipole
     correction contributed exactly zero to every MC move.

  3. src/forces/ewald_particle.py  _real_space_part
     the real-space pair force had the wrong sign (r_ij = r_j - r_i is added
     to the force on i, see src/topology.py:128 and :184).

Usage:  python ewald_audit/scripts/make_codefix.py [--force]
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "ewald_audit", "codefix")


def patch(path, old, new, label):
    with open(path) as f:
        text = f.read()
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[{label}] anchor found {n} times in {path}")
    with open(path, "w") as f:
        f.write(text.replace(old, new))
    print(f"  patched: {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(DST):
        if not args.force:
            raise SystemExit(f"{DST} already exists (use --force)")
        shutil.rmtree(DST)
    os.makedirs(DST)
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(DST, "src"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.bak*"))
    shutil.copy2(os.path.join(ROOT, "mc_main.py"), os.path.join(DST, "mc_main.py"))
    print(f"copied production code -> {DST}")

    eh = os.path.join(DST, "src", "ewald_handler.py")
    ep = os.path.join(DST, "src", "forces", "ewald_particle.py")

    # ---------------------------------------------------------------- fix 1
    patch(eh, """        # Build integer grid from -n_c..n_c and exclude zero vector
        rng = np.arange(-self.n_c, self.n_c + 1, dtype=int)
        ms = np.array(np.meshgrid(rng, rng, rng, indexing="ij"))
        ms = ms.reshape(3, -1).T  # (M,3)
        mask_nonzero = ~np.all(ms == 0, axis=1)
        ms = ms[mask_nonzero]

        # k vectors
        kvecs = np.zeros_like(ms, dtype=np.float64)
        kvecs[:, 0] = ms[:, 0] * b1
        kvecs[:, 1] = ms[:, 1] * b2
        kvecs[:, 2] = ms[:, 2] * b3
        k_sq = np.sum(kvecs * kvecs, axis=1)
""", """        # AUDIT FIX 1: anisotropic k-grid with a spherical cutoff.
        #
        # The original code used the same integer range -n_c..n_c on all three
        # axes.  Because b_i = 2*pi/L_i differ by more than an order of
        # magnitude for a slit pore (L_x = L_y = 20 nm, L_z*z_scale = 12 nm),
        # that truncates the lateral k-sum at k_max_xy = n_c*2*pi/L_x, far
        # below what the Gaussian requires.
        #
        # n_c is now interpreted as the number of k-shells along the *longest*
        # box side, i.e. k_max = n_c * 2*pi / max(L_i); the per-axis ranges
        # follow as n_i = ceil(k_max / b_i) and a spherical cutoff |k| <= k_max
        # is applied so that the resolution is isotropic in k, not in indices.
        k_max = self.n_c * 2.0 * np.pi / max(Lx, Ly, Lz_scaled)
        self.k_max = k_max
        nx = int(np.ceil(k_max / b1))
        ny = int(np.ceil(k_max / b2))
        nz = int(np.ceil(k_max / b3))
        self.n_grid = (nx, ny, nz)
        ms = np.array(
            np.meshgrid(
                np.arange(-nx, nx + 1, dtype=int),
                np.arange(-ny, ny + 1, dtype=int),
                np.arange(-nz, nz + 1, dtype=int),
                indexing="ij",
            )
        )
        ms = ms.reshape(3, -1).T  # (M,3)
        mask_nonzero = ~np.all(ms == 0, axis=1)
        ms = ms[mask_nonzero]

        # k vectors
        kvecs = np.zeros_like(ms, dtype=np.float64)
        kvecs[:, 0] = ms[:, 0] * b1
        kvecs[:, 1] = ms[:, 1] * b2
        kvecs[:, 2] = ms[:, 2] * b3
        k_sq = np.sum(kvecs * kvecs, axis=1)

        keep = k_sq <= k_max * k_max
        ms, kvecs, k_sq = ms[keep], kvecs[keep], k_sq[keep]
        print(
            f"[EWALD] anisotropic k-grid: k_max={k_max:.4f} 1/nm "
            f"n=({nx},{ny},{nz}) -> {len(k_sq)} k-vectors"
        )
""", "fix 1: anisotropic k-grid")

    # ---------------------------------------------------------------- fix 2
    patch(eh, """        new_energy, new_forces = self.compute_dipole_correction_energy_forces(
            positions, charges, volume
        )""", """        # AUDIT FIX 2: this used to pass `positions` (the OLD configuration),
        # so the slab dipole correction contributed exactly zero to every
        # MC energy difference and to the incremental forces.
        new_energy, new_forces = self.compute_dipole_correction_energy_forces(
            updated_positions, charges, volume
        )""", "fix 2: slab correction in delta moves")

    # ---------------------------------------------------------------- fix 3
    patch(ep, """        forces = np.zeros_like(r_ij)
        forces_sel = (
            prefactor * bracket[:, np.newaxis] * r_ij[mask, :] * inv_r2[:, np.newaxis]
        )""", """        # AUDIT FIX 3: sign of the real-space pair force.
        # topology._single_pair_energy_forces builds r_ij = r_j - r_i and adds
        # the value returned here to the force on atom i, but
        #   F_i = -dU/dr_i = q_i q_j [erfc(a r)/r + 2a/sqrt(pi) e^{-a^2 r^2}]
        #                    * (r_i - r_j) / r^2
        # i.e. it points along r_i - r_j = -r_ij.  Verified by finite
        # differences (ewald_audit/scripts/frame_audit.py, fd_force section).
        forces = np.zeros_like(r_ij)
        forces_sel = (
            -prefactor * bracket[:, np.newaxis] * r_ij[mask, :] * inv_r2[:, np.newaxis]
        )""", "fix 3: real-space force sign")

    print(f"\nOK -- run it with:  .venv/bin/python {os.path.relpath(DST, ROOT)}/mc_main.py --config ...")


if __name__ == "__main__":
    main()
