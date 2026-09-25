#!/usr/bin/env python3
"""Steps 3b + 4: compare the production Ewald against a converged reference on
real trajectory frames.

For one production run directory (containing config_backup.yaml,
trajectory.xyz and simulation_data.csv) this script

  1. rebuilds the production System/Topology/EwaldHandler from config_backup.yaml,
  2. overwrites the positions with a frame from trajectory.xyz,
  3. compares the total electrostatic energy (and its real / reciprocal / slab
     split) against RefEwald3DSlab,
  4. generates MC trial moves exactly as src/sampler.py does and compares
     dE_prod (topology.get_energy_difference_translation) against dE_ref,
     reporting RMS/max error in kT and the induced error in the Metropolis
     acceptance probability,
  5. compares the force on the walls (production full recompute, reference,
     the ideal sigma^2/(2 eps0 eps) result and the logged solvation_force),
  6. runs a finite-difference check of the production forces against the
     production energy, to localise force bugs.

Output: ewald_audit/results/frames/<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # ~/AntiDLVO
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf  # noqa: E402

from ref_ewald import RefEwald3DSlab, KC, KB  # noqa: E402

from src.utils import (  # noqa: E402
    _build_z_limits,
    create_units,
    create_system,
    create_topology,
    setup_initial_configuration,
)

# trajectory.xyz is written with six decimals, so a reloaded coordinate can
# differ from the one the run held by up to half of the last digit.
XYZ_PRECISION = 5e-7  # nm

# kJ/(mol nm^3) -> MPa
PRESSURE_MPA = 1e30 / 6.02214076e23 / 1e6 * 1e3 / 1e3  # see note below
PRESSURE_MPA = 1000.0 / (6.02214076e23 * 1e-27) / 1e6  # 1.66054 MPa


# --------------------------------------------------------------------------
#  trajectory reading
# --------------------------------------------------------------------------
def scan_xyz(path):
    """Return a list of byte offsets, one per frame, plus the atom count."""
    offsets = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                n = int(line.split()[0])
            except (ValueError, IndexError):
                break
            offsets.append(pos)
            f.readline()  # comment
            for _ in range(n):
                f.readline()
    return offsets, n


def read_xyz_frame(path, offset):
    with open(path, "rb") as f:
        f.seek(offset)
        n = int(f.readline().split()[0])
        comment = f.readline().decode()
        names = []
        pos = np.empty((n, 3))
        chg = np.empty(n)
        for i in range(n):
            parts = f.readline().split()
            names.append(parts[0].decode())
            pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
            chg[i] = float(parts[4])
    step = None
    for token in ("step=", "Step="):
        if token in comment:
            step = int(float(comment.split(token)[1].split()[0].strip('"')))
            break
    return dict(n=n, names=np.array(names), pos=pos, charges=chg,
                comment=comment, step=step)


# --------------------------------------------------------------------------
#  production side
# --------------------------------------------------------------------------
def relax_initial_placement(cfg):
    """Make the initial random placement always succeed.

    The configuration built here is overwritten by a trajectory frame a few
    lines later, so only the *order* of the atoms matters -- and that is fixed
    by the insertion order of the molecule specs, not by min_distance.  The
    current src/utils.py cannot pack the densest H=3 systems (the production
    runs predate the placement rewrite, cf. src/utils.py.bak_before_init_fix),
    and switching to mode='random_hs' is not an option: it inserts molecules in
    a different order, which set_frame() rejects.
    """
    ni = cfg.get("system", {}).get("non_static_init", None)
    if ni is None:
        return
    if "min_distance" in ni:
        ni.min_distance = 0.05
    if "padding" in ni and ni.padding is not None:
        ni.padding = [0.0, 0.0, 0.05]


def build_production(run_dir, use_gpu=None):
    cfg = OmegaConf.load(os.path.join(run_dir, "config_backup.yaml"))
    if use_gpu is not None:
        cfg.ewald.use_gpu = bool(use_gpu)
    relax_initial_placement(cfg)
    units = create_units(cfg)
    system = create_system(cfg, units)
    topology = create_topology(cfg, units)
    setup_initial_configuration(system, cfg.get("types", []), topology, cfg=cfg)
    return cfg, units, system, topology


def set_frame(system, frame, z_limits=None):
    """Install a trajectory frame into the production System.

    trajectory.xyz carries six decimals, so a z that the run kept legally
    inside its slab can be written out and read back sitting exactly on the
    boundary -- and HardWall's test is strict, so the reloaded configuration
    then has an infinite energy and compute_energy_forces returns zero forces
    for every atom. That is a property of the file, not of the run. Atoms
    within the write precision of a limit are pulled back inside and the
    largest correction is reported, so a genuine excursion (which would be
    orders of magnitude larger) still shows up instead of being absorbed.
    """
    n = system.N_atoms
    if frame["n"] != n:
        raise RuntimeError(f"atom count mismatch: xyz={frame['n']} system={n}")
    names = np.asarray(system.names[:n], dtype=object).astype(str)
    if not np.array_equal(names, frame["names"]):
        bad = np.flatnonzero(names != frame["names"])[:5]
        raise RuntimeError(f"atom-name order mismatch at {bad}: "
                           f"{names[bad]} vs {frame['names'][bad]}")
    if not np.allclose(system.charges[:n], frame["charges"], atol=1e-9):
        raise RuntimeError("charge mismatch between xyz and rebuilt system")
    system.positions[:n] = frame["pos"]
    if z_limits is not None:
        z_lo, z_hi = z_limits
        types = np.asarray(system.types[:n])
        z = system.positions[:n, 2]
        lo, hi = z_lo[types], z_hi[types]
        clipped = np.clip(z, lo, hi)
        moved = np.flatnonzero(clipped != z)
        if moved.size:
            shift = float(np.max(np.abs(clipped[moved] - z[moved])))
            if shift > XYZ_PRECISION:
                raise RuntimeError(
                    f"frame has {moved.size} atoms outside their slab by up to "
                    f"{shift:.3e} nm, far more than the {XYZ_PRECISION:.0e} nm "
                    f"the xyz write precision can explain")
            print(f"    pulled {moved.size} atom(s) back inside their slab, "
                  f"largest correction {shift:.2e} nm (xyz write precision)",
                  flush=True)
            system.positions[:n, 2] = clipped
    system.ewald_handler.update_structure_factors(
        system.positions[:n], system.charges[:n]
    )
    system.potential_energy = None
    system.forces = None


def prod_terms(system, topology):
    """Production total energy and forces, split into the parts we can isolate."""
    n = system.N_atoms
    energy, forces = topology.compute_energy_forces(system)
    eh = system.ewald_handler
    e_k, f_k = eh.compute_total_kspace_energy_forces(n)
    e_slab, f_slab = eh.compute_dipole_correction_energy_forces(
        system.positions[:n], system.charges[:n], system.get_volume()
    )
    f_k = np.asarray(f_k)[:n]
    f_slab = np.asarray(f_slab)[:n]
    forces = np.asarray(forces)[:n]
    return dict(
        total=float(energy),
        recip=float(e_k),
        slab=float(e_slab),
        real=float(energy) - float(e_k) - float(e_slab),
        forces=forces,
        forces_recip=f_k,
        forces_slab=f_slab,
        forces_real=forces - f_k - f_slab,
    )


# --------------------------------------------------------------------------
#  reference
# --------------------------------------------------------------------------
def make_reference(Lx, Ly, H, eps_r, z_scale=None, r_cut=9.0, eps=1e-10,
                   use_gpu=True, gap_factor=3.0):
    """Converged reference for a slit of width H.

    z_scale is chosen so that the vacuum gap H*(z_scale-1) is at least
    gap_factor * Lx -- validate_ref.py showed z_scale=4 is *not* enough at
    H=3 (the production value).
    """
    if z_scale is None:
        z_scale = max(4.0, 1.0 + gap_factor * max(Lx, Ly) / H)
    a = math.sqrt(-math.log(eps)) / r_cut
    return RefEwald3DSlab(
        Lx, Ly, H, z_scale=z_scale, alpha=a * a, r_cut=r_cut,
        k_max=2.0 * math.sqrt(a * a * (-math.log(eps))),
        lB=KC / eps_r, use_gpu=use_gpu,
    )


# --------------------------------------------------------------------------
#  MC trial moves, mirroring src/sampler.py
# --------------------------------------------------------------------------
def rotate_like_sampler(vectors, rng, max_rotation_deg):
    """Copy of Sampler._rotate_vector_randomly (rigid dipole rotation)."""
    vector = vectors[0] - vectors[1]
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vectors
    u = vector / norm
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    perp = np.cross(u, tmp)
    perp /= np.linalg.norm(perp)
    phi = 2 * np.pi * rng.random()
    axis = np.cos(phi) * perp + np.sin(phi) * np.cross(u, perp)
    axis /= np.linalg.norm(axis)
    angle = rng.random() * max_rotation_deg * np.pi / 180.0
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = (np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis)
         + np.sin(angle) * K)
    return vectors @ R


def molecule_pools(system):
    """Group movable molecules by the kind of trial move they represent."""
    n = system.N_atoms
    names = np.asarray(system.names[:n], dtype=object).astype(str)
    pools = {"Na_wall": [], "Na_bulk": [], "Cl": [], "dipole": []}
    for mol_id in system.molecule_type_names.keys():
        idx = np.asarray(system.get_molecule_atoms(mol_id), dtype=int)
        if len(idx) == 0 or np.any(system.statics[idx]):
            continue
        if len(idx) == 1:
            nm = names[idx[0]]
            z = system.positions[idx[0], 2]
            if nm == "Na":
                pools["Na_wall" if z < 1.0 else "Na_bulk"].append(idx)
            elif nm == "Cl":
                pools["Cl"].append(idx)
        else:
            pools["dipole"].append(idx)
    return pools


def make_move(system, idx, rng, max_disp, max_rot):
    disp = (rng.random(3) - 0.5) * 2 * max_disp
    old = system.positions[idx, :].copy()
    if len(idx) > 1:
        unwrapped = system.unwrap_molecule(old)
        centroid = np.mean(unwrapped, axis=0)
        rotated = rotate_like_sampler(unwrapped - centroid, rng, max_rot)
        return old, system.apply_pbc(rotated + centroid + disp)
    return old, system.apply_pbc(old + disp)


def compare_moves(system, topology, ref, pool, rng, max_disp, max_rot,
                  n_moves, beta):
    """dE_prod vs dE_ref for n_moves trial moves drawn from `pool`."""
    if not pool:
        return None
    errs, dprod, dref, dslab, dreal, drecip = [], [], [], [], [], []
    n_inf = 0
    for _ in range(n_moves):
        idx = pool[rng.integers(0, len(pool))]
        old, new = make_move(system, idx, rng, max_disp, max_rot)
        e_prod = topology.get_energy_difference_translation(idx, new, system)[0]
        if not np.isfinite(e_prod):
            n_inf += 1
            continue
        r = ref.delta_energy(idx, new, split=True)
        dprod.append(float(e_prod))
        dref.append(r["total"])
        dreal.append(r["real"])
        drecip.append(r["recip"])
        dslab.append(r["slab"])
        errs.append(float(e_prod) - r["total"])
    if not errs:
        return dict(n=0, n_inf=n_inf)
    errs = np.array(errs) / (KB * 298.0)          # kT
    dprod = np.array(dprod) / (KB * 298.0)
    dref = np.array(dref) / (KB * 298.0)
    dslab = np.array(dslab) / (KB * 298.0)
    acc_p = np.minimum(1.0, np.exp(-np.clip(dprod, -700, 700)))
    acc_r = np.minimum(1.0, np.exp(-np.clip(dref, -700, 700)))
    dacc = np.abs(acc_p - acc_r)
    return dict(
        n=int(len(errs)), n_inf=int(n_inf),
        rms_err_kT=float(np.sqrt(np.mean(errs ** 2))),
        max_err_kT=float(np.max(np.abs(errs))),
        mean_err_kT=float(np.mean(errs)),
        median_abs_err_kT=float(np.median(np.abs(errs))),
        rms_dE_ref_kT=float(np.sqrt(np.mean(dref ** 2))),
        mean_dacc=float(np.mean(dacc)),
        frac_dacc_gt_0p1=float(np.mean(dacc > 0.1)),
        frac_dacc_gt_0p01=float(np.mean(dacc > 0.01)),
        # attribution: how much of the error is the missing slab delta?
        rms_slab_delta_kT=float(np.sqrt(np.mean(dslab ** 2))),
        rms_err_wo_slab_kT=float(np.sqrt(np.mean((errs + dslab) ** 2))),
    )


# --------------------------------------------------------------------------
#  wall force
# --------------------------------------------------------------------------
def wall_force(system, forces):
    n = system.N_atoms
    names = np.asarray(system.names[:n], dtype=object).astype(str)
    A = system.box[0] * system.box[1]
    fb = np.sum(forces[names == "Wb", 2])
    ft = np.sum(forces[names == "Wt", 2])
    return float((fb - ft) / 2.0 / A)


def fd_force_check(system, topology, atom_ids, h=2e-4):
    """Finite-difference dE from the production energy vs production forces.

    get_energy_difference_translation omits the slab term (see the audit
    report), so the FD force is compared against real+recip only.
    """
    out = []
    n = system.N_atoms
    eh = system.ewald_handler
    _, f_slab = eh.compute_dipole_correction_energy_forces(
        system.positions[:n], system.charges[:n], system.get_volume()
    )
    _, f_k = eh.compute_total_kspace_energy_forces(n)
    f_k = np.asarray(f_k)[:n]
    f_slab = np.asarray(f_slab)[:n]
    f_tot = np.asarray(system.forces)[:n]
    for aid in atom_ids:
        idx = np.array([aid])
        fd = np.zeros(3)
        ok = True
        for d in range(3):
            ep = system.positions[aid].copy()
            em = system.positions[aid].copy()
            ep[d] += h
            em[d] -= h
            dp = topology.get_energy_difference_translation(idx, ep[None, :], system)[0]
            dm = topology.get_energy_difference_translation(idx, em[None, :], system)[0]
            if not (np.isfinite(dp) and np.isfinite(dm)):
                ok = False
                break
            fd[d] = -(dp - dm) / (2 * h)
        if not ok:
            continue
        f_elec = f_tot[aid] - f_slab[aid]           # real + recip, as coded
        f_real = f_tot[aid] - f_slab[aid] - f_k[aid]
        f_flipped = f_k[aid] - f_real               # real part sign-flipped
        out.append(dict(
            atom=int(aid), name=str(system.names[aid]),
            F_fd=fd.tolist(),
            F_code=f_elec.tolist(),
            F_code_real=f_real.tolist(),
            F_code_recip=f_k[aid].tolist(),
            F_real_sign_flipped=f_flipped.tolist(),
            err_as_coded=float(np.linalg.norm(fd - f_elec)),
            err_if_real_flipped=float(np.linalg.norm(fd - f_flipped)),
            scale=float(np.linalg.norm(fd)),
        ))
    return out


# --------------------------------------------------------------------------
def audit_run(run_dir, frames, n_moves, use_gpu, out_dir, tag=None,
              ref_gap_factor=3.0, convergence_check=True):
    t0 = time.time()
    tag = tag or os.path.basename(run_dir.rstrip("/"))
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)

    cfg, units, system, topology = build_production(run_dir, use_gpu=use_gpu)
    n = system.N_atoms
    Lx, Ly, H = [float(b) for b in system.box]
    eps_r = float(cfg.ewald.dielectric)
    T = float(cfg.simulation.temperature)
    kT = KB * T
    beta = 1.0 / kT
    print(f"  N_atoms={n}  box=({Lx},{Ly},{H})  eps={eps_r}  T={T}", flush=True)
    z_limits = _build_z_limits(cfg, topology.type_name_to_id, H)

    ref = make_reference(Lx, Ly, H, eps_r, use_gpu=use_gpu,
                         gap_factor=ref_gap_factor)
    print(f"  reference: z_scale={ref.z_scale:.2f} r_cut={ref.r_cut} "
          f"alpha={ref.alpha:.4f} k_max={ref.k_max:.3f} n_k={ref.n_k} "
          f"grid={ref.n_grid}", flush=True)

    traj = os.path.join(run_dir, "trajectory.xyz")
    offsets, n_xyz = scan_xyz(traj)
    n_frames = len(offsets)
    print(f"  trajectory: {n_frames} frames of {n_xyz} atoms", flush=True)

    wanted = [f for f in frames if f < n_frames]
    if not wanted:
        wanted = [n_frames - 1]

    # logged solvation force
    logged = {}
    csvp = os.path.join(run_dir, "simulation_data.csv")
    if os.path.exists(csvp):
        import csv as _csv
        with open(csvp) as f:
            for row in _csv.DictReader(f):
                try:
                    logged[int(float(row["step"]))] = float(row["solvation_force"])
                except (KeyError, ValueError):
                    pass

    result = dict(
        tag=tag, run_dir=run_dir, N_atoms=n, Lx=Lx, Ly=Ly, H=H,
        eps_r=eps_r, T=T, n_frames=n_frames,
        prod_ewald=dict(alpha=float(cfg.ewald.alpha),
                        real_cut=float(cfg.ewald.real_cut),
                        n_c=int(cfg.ewald.n_c),
                        z_scale=float(cfg.ewald.z_scale_factor)),
        ref_ewald=dict(z_scale=ref.z_scale, r_cut=ref.r_cut, alpha=ref.alpha,
                       k_max=ref.k_max, n_k=ref.n_k, n_grid=list(ref.n_grid)),
        frames=[],
    )

    for fi, frame_idx in enumerate(wanted):
        frame = read_xyz_frame(traj, offsets[frame_idx])
        set_frame(system, frame, z_limits=z_limits)
        pos = system.positions[:n].copy()
        q = system.charges[:n].copy()

        # ---- production totals
        t = time.time()
        pt = prod_terms(system, topology)
        if not math.isfinite(pt["total"]):
            # compute_energy_forces answers an infinite energy with an all-zero
            # force array, which would quietly turn into a wall force of zero.
            raise RuntimeError(
                f"frame {frame_idx} has infinite production energy: a hard "
                f"sphere or hard wall is violated, so no force comparison on "
                f"this frame would mean anything")
        system.potential_energy = pt["total"]
        system.forces = np.zeros_like(system.positions)
        system.forces[:n] = pt["forces"]
        print(f"  frame {frame_idx} (step {frame['step']}): "
              f"prod recompute {time.time()-t:.1f}s", flush=True)

        # ---- reference totals
        t = time.time()
        rt = ref.energy_terms(pos, q, want_forces=True)
        print(f"    ref energy {time.time()-t:.1f}s", flush=True)

        fr = dict(frame=frame_idx, step=frame["step"])
        fr["energy"] = dict(
            prod_total_kT=pt["total"] / kT,
            prod_real_kT=pt["real"] / kT,
            prod_recip_kT=pt["recip"] / kT,
            prod_slab_kT=pt["slab"] / kT,
            ref_total_kT=rt["total"] / kT,
            ref_total_no_self_kT=rt["total_no_self"] / kT,
            ref_real_kT=rt["real"] / kT,
            ref_recip_kT=rt["recip"] / kT,
            ref_self_kT=rt["self_"] / kT,
            ref_slab_kT=rt["slab"] / kT,
            err_total_kT=(pt["total"] - rt["total_no_self"]) / kT,
            err_per_atom_kT=(pt["total"] - rt["total_no_self"]) / kT / n,
            err_real_kT=(pt["real"] - rt["real"]) / kT,
            err_recip_kT=(pt["recip"] - rt["recip"]) / kT,
            err_slab_kT=(pt["slab"] - rt["slab"]) / kT,
        )

        # ---- reference self-consistency (first frame only)
        if convergence_check and fi == 0:
            conv = {}
            for label, kw in (("z_scale x1.5", dict(z_scale=ref.z_scale * 1.5)),
                              ("r_cut 4.5", dict(r_cut=4.5)),
                              ("eps 1e-12", dict(eps=1e-12))):
                kw2 = dict(use_gpu=use_gpu, gap_factor=ref_gap_factor)
                kw2.update(kw)
                r2 = make_reference(Lx, Ly, H, eps_r, **kw2)
                t2 = r2.energy_terms(pos, q)
                conv[label] = dict(
                    dE_kT=(t2["total"] - rt["total"]) / kT,
                    n_k=r2.n_k, z_scale=r2.z_scale, r_cut=r2.r_cut)
                print(f"    ref check [{label:12s}] dE = "
                      f"{conv[label]['dE_kT']:+.3e} kT  (n_k={r2.n_k})", flush=True)
                del r2
            fr["ref_convergence"] = conv

        # ---- wall force
        A = Lx * Ly
        names = np.asarray(system.names[:n], dtype=object).astype(str)
        n_wall = int(np.sum(names == "Wb"))
        sigma_eff = n_wall / A                       # e / nm^2
        p_ideal = 2.0 * math.pi * KC * sigma_eff ** 2 / eps_r  # kJ/(mol nm^3)
        f_prod = wall_force(system, pt["forces"])
        f_prod_flip = wall_force(
            system, pt["forces_recip"] + pt["forces_slab"] - pt["forces_real"])
        f_ref = wall_force(system, rt["forces"])
        fr["wall"] = dict(
            n_wall=n_wall, sigma_eff_e_per_nm2=sigma_eff,
            prod=f_prod, prod_MPa=f_prod * PRESSURE_MPA,
            prod_real_sign_flipped=f_prod_flip,
            prod_real_sign_flipped_MPa=f_prod_flip * PRESSURE_MPA,
            ref=f_ref, ref_MPa=f_ref * PRESSURE_MPA,
            ideal=p_ideal, ideal_MPa=p_ideal * PRESSURE_MPA,
            logged=logged.get(frame["step"]),
            logged_MPa=(logged.get(frame["step"]) * PRESSURE_MPA
                        if frame["step"] in logged else None),
        )
        print(f"    wall force  prod={f_prod:12.4f}  flip={f_prod_flip:12.4f}  "
              f"ref={f_ref:12.4f}  ideal={p_ideal:12.4f}  "
              f"logged={logged.get(frame['step'])} kJ/(mol nm^3)", flush=True)

        # ---- finite-difference force check (first frame only)
        if fi == 0:
            pools0 = molecule_pools(system)
            ids = []
            for key in ("Na_bulk", "Na_wall", "Cl"):
                if pools0[key]:
                    ids.append(int(pools0[key][0][0]))
                    ids.append(int(pools0[key][len(pools0[key]) // 2][0]))
            fr["fd_force"] = fd_force_check(system, topology, ids)
            for row in fr["fd_force"]:
                print(f"    FD atom {row['atom']:5d} {row['name']:3s} "
                      f"|F_fd|={row['scale']:.4f}  err_as_coded="
                      f"{row['err_as_coded']:.4e}  err_if_real_flipped="
                      f"{row['err_if_real_flipped']:.4e}", flush=True)

        # ---- MC trial moves
        ref.prepare(pos, q)
        pools = molecule_pools(system)
        rng = np.random.default_rng(1234 + frame_idx)
        max_disp = float(cfg.sampler.max_displacement)
        max_rot = float(cfg.sampler.max_rotation)
        moves = {}
        t = time.time()
        for amp_name, (d, r) in (("large", (max_disp, max_rot)),
                                 ("small", (0.1, 0.1 * max_rot))):
            for kind, pool in pools.items():
                key = f"{kind}/{amp_name}"
                res = compare_moves(system, topology, ref, pool, rng, d, r,
                                    n_moves, beta)
                if res is None:
                    continue
                moves[key] = res
                if res.get("n"):
                    print(f"    {key:18s} n={res['n']:4d} "
                          f"rms={res['rms_err_kT']:9.4f} kT  "
                          f"max={res['max_err_kT']:9.4f}  "
                          f"<|dacc|>={res['mean_dacc']:.4f}  "
                          f"f(>0.1)={res['frac_dacc_gt_0p1']:.3f}  "
                          f"rms(dE_ref)={res['rms_dE_ref_kT']:8.3f}",
                          flush=True)
        print(f"    moves {time.time()-t:.1f}s", flush=True)
        fr["moves"] = moves
        result["frames"].append(fr)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"  -> {path}  ({time.time()-t0:.0f}s)", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", type=int, nargs="+", default=[2000, 3000, 4000])
    ap.add_argument("--n-moves", type=int, default=250)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(HERE), "results", "frames"))
    ap.add_argument("--tag-prefix", default="")
    args = ap.parse_args()

    for run in args.runs:
        tag = args.tag_prefix + run.rstrip("/").replace("/", "__")
        try:
            audit_run(run, args.frames, args.n_moves, bool(args.gpu),
                      args.out, tag=tag)
        except Exception as exc:  # keep going over the other runs
            import traceback
            traceback.print_exc()
            print(f"  FAILED {run}: {exc}", flush=True)


if __name__ == "__main__":
    main()
