"""Reference (converged) Ewald implementations for the slab audit.

Three independent pieces:

* :class:`RefEwald3DSlab` -- Ewald3D in a box (Lx, Ly, z_scale*Lz) with the
  Yeh-Berkowitz / dos Santos-Girotto-Levin slab correction, using an
  **anisotropic** k-grid n_i = ceil(k_max L_i / 2pi) with a spherical cutoff.
  Same conventions as the production code so that terms can be compared
  one-to-one:
      real:   erfc(sqrt(alpha) r)/r,  alpha = a^2
      recip:  (2pi/V) sum_k exp(-k^2/(4 alpha)) |S(k)|^2 / k^2
      self:   -sqrt(alpha/pi) sum_i q_i^2
      slab:   (2pi/V) (M_z^2 - Q_tot G_z)
  Everything is multiplied by lB = k_c/epsilon at the end.

* :func:`ewald2d_energy` -- the exact Parry/Heyes Ewald2D energy for a system
  periodic in x,y only.  O(N^2 K) so it is meant for validation systems of at
  most a few hundred charges.  This is the ground truth the slab construction
  is supposed to approximate.

* :func:`madelung_nacl` -- NaCl Madelung constant from RefEwald3DSlab with the
  slab correction switched off.

All energies are returned in the same units as the charges/lengths given
(multiply by lB yourself, or pass lB to the constructor).
"""

from __future__ import annotations

import math
import os
import numpy as np
from scipy.special import erfc as _erfc_cpu

try:
    import cupy as cp

    CUPY = True
except Exception:  # pragma: no cover
    cp = None
    CUPY = False


KC = 138.935456  # kJ nm / mol  (Coulomb constant in GROMACS units)
KB = 8.314462618e-3  # kJ/(mol K)


def _xp(use_gpu):
    return cp if (use_gpu and CUPY) else np


def _asnumpy(a):
    if CUPY and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


class RefEwald3DSlab:
    """Converged Ewald3D + slab correction with an anisotropic k grid."""

    def __init__(
        self,
        Lx,
        Ly,
        Lz_phys,
        z_scale=4.0,
        alpha=None,      # = a^2, the production convention
        r_cut=None,
        k_max=None,
        eps=1e-8,
        lB=1.0,
        slab_correction=True,
        pbc_z=False,     # minimum image in z (False for slab geometry)
        use_gpu=True,
        k_block=200_000,
        spherical=True,
    ):
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.Lz_phys = float(Lz_phys)
        self.z_scale = float(z_scale)
        self.Lz = self.Lz_phys * self.z_scale
        self.V = self.Lx * self.Ly * self.Lz
        self.lB = float(lB)
        self.slab_correction = bool(slab_correction)
        self.pbc_z = bool(pbc_z)
        self.use_gpu = bool(use_gpu and CUPY)
        self.xp = _xp(self.use_gpu)
        self.k_block = int(k_block)
        # cap on the (k_block x N) temporaries, in bytes -- the GPU is shared
        self.mem_budget = float(os.environ.get("AUDIT_MEM_BUDGET", 2.5e8))
        self.eps = float(eps)

        # --- splitting parameter -------------------------------------------
        if r_cut is None:
            # keep the real-space cutoff inside the minimum-image limit
            r_cut = 0.45 * min(self.Lx, self.Ly, self.Lz)
        self.r_cut = float(r_cut)
        if alpha is None:
            # erfc(a r_cut) = eps  ->  a = sqrt(-ln eps)/r_cut, alpha = a^2
            a = math.sqrt(-math.log(self.eps)) / self.r_cut
            alpha = a * a
        self.alpha = float(alpha)
        self.a = math.sqrt(self.alpha)

        if k_max is None:
            # exp(-k^2/(4 alpha)) = eps
            k_max = 2.0 * math.sqrt(self.alpha * (-math.log(self.eps)))
        self.k_max = float(k_max)

        self._build_k(spherical)

        # cached structure factors
        self._Sc = None
        self._Ss = None
        self._pos = None
        self._q = None

    # ------------------------------------------------------------------ k grid
    def _build_k(self, spherical=True):
        nx = int(math.ceil(self.k_max * self.Lx / (2 * math.pi)))
        ny = int(math.ceil(self.k_max * self.Ly / (2 * math.pi)))
        nz = int(math.ceil(self.k_max * self.Lz / (2 * math.pi)))
        self.n_grid = (nx, ny, nz)
        bx, by, bz = 2 * math.pi / self.Lx, 2 * math.pi / self.Ly, 2 * math.pi / self.Lz

        # half space (kx>0) or (kx==0 and ky>0) or (kx==0 and ky==0 and kz>0),
        # with a factor 2 -- halves memory and work.
        mx = np.arange(-nx, nx + 1)
        my = np.arange(-ny, ny + 1)
        mz = np.arange(-nz, nz + 1)
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing="ij")
        MX, MY, MZ = MX.ravel(), MY.ravel(), MZ.ravel()
        half = (MX > 0) | ((MX == 0) & (MY > 0)) | ((MX == 0) & (MY == 0) & (MZ > 0))
        MX, MY, MZ = MX[half], MY[half], MZ[half]

        kx, ky, kz = MX * bx, MY * by, MZ * bz
        k_sq = kx * kx + ky * ky + kz * kz
        if spherical:
            keep = k_sq <= self.k_max ** 2
            kx, ky, kz, k_sq = kx[keep], ky[keep], kz[keep], k_sq[keep]

        # factor 2 for the missing half space
        Ak = 2.0 * (2.0 * math.pi / self.V) * np.exp(-k_sq / (4.0 * self.alpha)) / k_sq

        order = np.argsort(k_sq)
        self.kvecs = np.ascontiguousarray(np.stack([kx, ky, kz], axis=1)[order])
        self.k_sq = np.ascontiguousarray(k_sq[order])
        self.Ak = np.ascontiguousarray(Ak[order])
        self.n_k = len(self.Ak)

        xp = self.xp
        self._kvecs = xp.asarray(self.kvecs)
        self._Ak = xp.asarray(self.Ak)

    def _kb(self, N):
        """k-block size that keeps the (kb x N) temporaries inside the budget.

        recip_energy_forces holds ~4 such arrays at once, so budget/(4*8*N).
        """
        kb = int(self.mem_budget / (32.0 * max(N, 1)))
        return max(64, min(self.k_block, kb))

    # --------------------------------------------------------------- distances
    def _mic(self, dr):
        dr = dr.copy()
        dr[..., 0] -= self.Lx * np.round(dr[..., 0] / self.Lx)
        dr[..., 1] -= self.Ly * np.round(dr[..., 1] / self.Ly)
        if self.pbc_z:
            dr[..., 2] -= self.Lz * np.round(dr[..., 2] / self.Lz)
        return dr

    # ------------------------------------------------------------- real space
    def real_energy_forces(self, pos, q, want_forces=True, block=512):
        """Full O(N^2) real-space sum with minimum image in x,y."""
        N = len(pos)
        E = 0.0
        F = np.zeros((N, 3)) if want_forces else None
        rc2 = self.r_cut ** 2
        a = self.a
        two_a_sqrtpi = 2.0 * a / math.sqrt(math.pi)
        for s in range(0, N, block):
            e = min(s + block, N)
            dr = self._mic(pos[s:e, None, :] - pos[None, :, :])  # (b,N,3) = r_i - r_j
            r2 = np.einsum("ijk,ijk->ij", dr, dr)
            idx = np.arange(s, e)
            r2[np.arange(e - s), idx] = np.inf  # exclude self
            mask = r2 <= rc2
            r = np.sqrt(np.where(mask, r2, 1.0))
            qq = q[s:e, None] * q[None, :]
            pot = np.where(mask, _erfc_cpu(a * r) / r, 0.0)
            E += 0.5 * np.sum(qq * pot)
            if want_forces:
                r2s = np.where(mask, r2, 1.0)
                coef = np.where(
                    mask,
                    qq * (_erfc_cpu(a * r) / r + two_a_sqrtpi * np.exp(-self.alpha * r2s)) / r2s,
                    0.0,
                )
                F[s:e] += np.einsum("ij,ijk->ik", coef, dr)
        return E, F

    # -------------------------------------------------------- reciprocal space
    def _structure_factors(self, pos, q):
        """Return Sc(K,), Ss(K,) for the whole system (blocked over k)."""
        xp = self.xp
        p = xp.asarray(pos)
        qq = xp.asarray(q)
        Sc = xp.empty(self.n_k)
        Ss = xp.empty(self.n_k)
        kb = self._kb(len(pos))
        for s in range(0, self.n_k, kb):
            e = min(s + kb, self.n_k)
            kr = self._kvecs[s:e] @ p.T
            Sc[s:e] = xp.sum(qq * xp.cos(kr), axis=1)
            Ss[s:e] = xp.sum(qq * xp.sin(kr), axis=1)
            del kr
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        return Sc, Ss

    def recip_energy_forces(self, pos, q, want_forces=True):
        xp = self.xp
        p = xp.asarray(pos)
        qq = xp.asarray(q)
        N = len(pos)
        E = 0.0
        F = xp.zeros((N, 3)) if want_forces else None
        kb = self._kb(N)
        for s in range(0, self.n_k, kb):
            e = min(s + kb, self.n_k)
            kr = self._kvecs[s:e] @ p.T          # (kb, N)
            c, sn = xp.cos(kr), xp.sin(kr)
            Sc = xp.sum(qq * c, axis=1)
            Ss = xp.sum(qq * sn, axis=1)
            A = self._Ak[s:e]
            E += float(xp.sum(A * (Sc ** 2 + Ss ** 2)))
            if want_forces:
                # F_i = 2 * sum_k A(k) k q_i [Sc sin(k r_i) - Ss cos(k r_i)]
                w = (Sc[:, None] * sn - Ss[:, None] * c)      # (kb,N)
                F += 2.0 * (w.T @ (A[:, None] * self._kvecs[s:e])) * qq[:, None]
            del kr, c, sn
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()
        return E, (_asnumpy(F) if want_forces else None)

    # ---------------------------------------------------------- other terms
    def self_energy(self, q):
        return -math.sqrt(self.alpha / math.pi) * float(np.sum(q * q))

    def background_energy(self, q):
        """Neutralising-background term for a non-neutral cell."""
        Q = float(np.sum(q))
        return -math.pi * Q * Q / (2.0 * self.alpha * self.V)

    def slab_energy_forces(self, pos, q, want_forces=True):
        if not self.slab_correction:
            return 0.0, np.zeros_like(pos)
        Mz = float(np.sum(q * pos[:, 2]))
        Q = float(np.sum(q))
        Gz = float(np.sum(q * pos[:, 2] ** 2))
        E = 2.0 * math.pi * (Mz * Mz - Q * Gz) / self.V
        F = np.zeros_like(pos)
        if want_forces:
            F[:, 2] = -4.0 * math.pi * (Mz * q - Q * q * pos[:, 2]) / self.V
        return E, F

    # ------------------------------------------------------------------ totals
    def energy_terms(self, pos, q, want_forces=False):
        pos = np.ascontiguousarray(np.asarray(pos, dtype=float))
        q = np.ascontiguousarray(np.asarray(q, dtype=float))
        Er, Fr = self.real_energy_forces(pos, q, want_forces)
        Ek, Fk = self.recip_energy_forces(pos, q, want_forces)
        Es = self.self_energy(q)
        Eb = self.background_energy(q)
        El, Fl = self.slab_energy_forces(pos, q, want_forces)
        out = dict(
            real=Er * self.lB,
            recip=Ek * self.lB,
            self_=Es * self.lB,
            background=Eb * self.lB,
            slab=El * self.lB,
        )
        out["total"] = sum(out.values())
        out["total_no_self"] = out["total"] - out["self_"] - out["background"]
        if want_forces:
            out["forces"] = (Fr + Fk + Fl) * self.lB
            out["forces_real"] = Fr * self.lB
            out["forces_recip"] = Fk * self.lB
            out["forces_slab"] = Fl * self.lB
        return out

    def energy(self, pos, q):
        return self.energy_terms(pos, q)["total"]

    # ----------------------------------------------------------- MC-style dE
    def prepare(self, pos, q):
        """Cache S(k) for the current configuration (for fast delta moves)."""
        self._pos = np.ascontiguousarray(np.asarray(pos, float))
        self._q = np.ascontiguousarray(np.asarray(q, float))
        self._Sc, self._Ss = self._structure_factors(self._pos, self._q)

    def delta_energy(self, indices, new_pos, split=False):
        """Energy change for moving atoms `indices` to `new_pos`.

        Uses the cached S(k); the configuration itself is left unchanged.
        """
        xp = self.xp
        pos, q = self._pos, self._q
        idx = np.asarray(indices, dtype=int)
        old_pos = pos[idx]
        new_pos = np.asarray(new_pos, dtype=float)
        qi = q[idx]

        # ---- real space: moved atoms vs everything else, plus intra pairs
        mask = np.ones(len(pos), dtype=bool)
        mask[idx] = False
        others = pos[mask]
        qo = q[mask]
        a, alpha, rc2 = self.a, self.alpha, self.r_cut ** 2

        def _pair_sum(P):
            dr = self._mic(P[:, None, :] - others[None, :, :])
            r2 = np.einsum("ijk,ijk->ij", dr, dr)
            m = (r2 <= rc2) & (r2 > 0)
            r = np.sqrt(np.where(m, r2, 1.0))
            return float(np.sum(np.where(m, (qi[:, None] * qo[None, :]) * _erfc_cpu(a * r) / r, 0.0)))

        def _intra(P):
            if len(idx) < 2:
                return 0.0
            dr = self._mic(P[:, None, :] - P[None, :, :])
            r2 = np.einsum("ijk,ijk->ij", dr, dr)
            iu = np.triu_indices(len(idx), 1)
            r2 = r2[iu]
            m = (r2 <= rc2) & (r2 > 0)
            r = np.sqrt(np.where(m, r2, 1.0))
            qq = (qi[:, None] * qi[None, :])[iu]
            return float(np.sum(np.where(m, qq * _erfc_cpu(a * r) / r, 0.0)))

        dE_real = (_pair_sum(new_pos) + _intra(new_pos)) - (_pair_sum(old_pos) + _intra(old_pos))

        # ---- reciprocal space via delta structure factors
        dE_k = 0.0
        pn = xp.asarray(new_pos)
        po = xp.asarray(old_pos)
        qb = xp.asarray(qi)
        kb = self._kb(4 * len(idx))
        for s in range(0, self.n_k, kb):
            e = min(s + kb, self.n_k)
            K = self._kvecs[s:e]
            krn = K @ pn.T
            kro = K @ po.T
            dSc = xp.sum(qb * (xp.cos(krn) - xp.cos(kro)), axis=1)
            dSs = xp.sum(qb * (xp.sin(krn) - xp.sin(kro)), axis=1)
            Sc, Ss = self._Sc[s:e], self._Ss[s:e]
            dE_k += float(xp.sum(self._Ak[s:e] * (2.0 * (Sc * dSc + Ss * dSs) + dSc ** 2 + dSs ** 2)))
            del krn, kro, dSc, dSs
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()

        # ---- slab correction
        dE_slab = 0.0
        if self.slab_correction:
            Q = float(np.sum(q))
            Mz = float(np.sum(q * pos[:, 2]))
            Gz = float(np.sum(q * pos[:, 2] ** 2))
            dMz = float(np.sum(qi * (new_pos[:, 2] - old_pos[:, 2])))
            dGz = float(np.sum(qi * (new_pos[:, 2] ** 2 - old_pos[:, 2] ** 2)))
            dE_slab = 2.0 * math.pi * ((Mz + dMz) ** 2 - Mz ** 2 - Q * dGz) / self.V

        if split:
            return dict(real=dE_real * self.lB, recip=dE_k * self.lB,
                        slab=dE_slab * self.lB,
                        total=(dE_real + dE_k + dE_slab) * self.lB)
        return (dE_real + dE_k + dE_slab) * self.lB


# --------------------------------------------------------------------------
#  Exact Ewald2D (Parry / Heyes) -- independent ground truth, O(N^2 K)
# --------------------------------------------------------------------------
def ewald2d_energy(pos, q, Lx, Ly, a=None, n_max=None, r_img=1, eps=1e-10):
    """Electrostatic energy of a system periodic in x,y only.

    Formulation (Heyes/Parry, see Frenkel & Smit App. B):

      U = 1/2 sum_{i!=j,n} q_i q_j erfc(a r)/r   (real, lateral images n)
        - a/sqrt(pi) sum_i q_i^2                  (self)
        + (pi/2A) sum_{h!=0} (1/h) sum_{ij} q_i q_j cos(h.s_ij) *
              [ e^{ h z_ij} erfc(a z_ij + h/2a)
              + e^{-h z_ij} erfc(-a z_ij + h/2a) ]
        - (sqrt(pi)/A) sum_{ij} q_i q_j [ z_ij erf(a z_ij)
              + exp(-a^2 z_ij^2)/(a sqrt(pi)) ]   (h = 0 term)

    Valid for a charge-neutral cell.  Returns the energy in units of q^2/length.
    """
    from scipy.special import erf as _erf

    pos = np.asarray(pos, float)
    q = np.asarray(q, float)
    N = len(pos)
    A = Lx * Ly
    if a is None:
        a = 5.0 / min(Lx, Ly)
    if n_max is None:
        n_max = int(math.ceil(2.0 * a * math.sqrt(-math.log(eps)) * max(Lx, Ly) / (2 * math.pi)))

    # ---- real space with lateral images
    E_real = 0.0
    for nx in range(-r_img, r_img + 1):
        for ny in range(-r_img, r_img + 1):
            shift = np.array([nx * Lx, ny * Ly, 0.0])
            dr = pos[:, None, :] - pos[None, :, :] + shift
            r = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))
            qq = q[:, None] * q[None, :]
            if nx == 0 and ny == 0:
                np.fill_diagonal(r, np.inf)
            E_real += 0.5 * np.sum(qq * _erfc_cpu(a * r) / r)

    E_self = -a / math.sqrt(math.pi) * np.sum(q * q)

    # ---- reciprocal (2D h vectors)
    sx = pos[:, 0]
    sy = pos[:, 1]
    z = pos[:, 2]
    zij = z[:, None] - z[None, :]
    qq = q[:, None] * q[None, :]
    E_rec = 0.0
    bx, by = 2 * math.pi / Lx, 2 * math.pi / Ly
    for mx in range(-n_max, n_max + 1):
        for my in range(-n_max, n_max + 1):
            if mx == 0 and my == 0:
                continue
            hx, hy = mx * bx, my * by
            h = math.hypot(hx, hy)
            if h > 2.0 * a * math.sqrt(-math.log(eps)) * 1.5:
                continue
            phase = np.cos(hx * (sx[:, None] - sx[None, :]) + hy * (sy[:, None] - sy[None, :]))
            t1 = np.exp(np.clip(h * zij, -700, 700)) * _erfc_cpu(a * zij + h / (2 * a))
            t2 = np.exp(np.clip(-h * zij, -700, 700)) * _erfc_cpu(-a * zij + h / (2 * a))
            E_rec += (math.pi / (2.0 * A)) / h * np.sum(qq * phase * (t1 + t2))

    # ---- h = 0 term
    # h = 0 term: the planar-average potential of a sheet, -2 pi sigma |z|,
    # smeared by the Ewald Gaussian.  Full double sum => prefactor -pi/A.
    E_h0 = -(math.pi / A) * np.sum(
        qq * (zij * _erf(a * zij) + np.exp(-(a ** 2) * zij ** 2) / (a * math.sqrt(math.pi)))
    )

    return dict(real=E_real, self_=E_self, recip=E_rec, h0=E_h0,
                total=E_real + E_self + E_rec + E_h0)


def madelung_nacl(n_cell=1, **kw):
    """Madelung constant of NaCl from RefEwald3DSlab (pbc in all 3 dims)."""
    a0 = 1.0  # nearest-neighbour distance
    pts, chg = [], []
    L = 2 * a0 * n_cell
    for i in range(2 * n_cell):
        for j in range(2 * n_cell):
            for k in range(2 * n_cell):
                pts.append([i * a0, j * a0, k * a0])
                chg.append(1.0 if (i + j + k) % 2 == 0 else -1.0)
    pts = np.array(pts, float)
    chg = np.array(chg, float)
    kwargs = dict(z_scale=1.0, slab_correction=False, pbc_z=True,
                  eps=1e-12, use_gpu=False)
    kwargs.update(kw)
    ew = RefEwald3DSlab(L, L, L, **kwargs)
    t = ew.energy_terms(pts, chg)
    # U = -M * N/2 * q^2/a0  =>  M = -2 U / N
    return -2.0 * t["total"] / len(chg), t
