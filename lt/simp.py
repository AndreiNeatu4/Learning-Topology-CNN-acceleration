"""Optimizare topologica SIMP (port vectorizat al codului "88-line", Andreassen et al. 2011).

Domeniu: consola 2D (latura stanga incastrata), forta punctuala intr-un nod oarecare
din jumatatea dreapta, cu directie data de un unghi. Minimizeaza complianta la
fractie de volum impusa, cu filtru de densitate.

Conventii:
  - imaginea de densitati are forma (nely, nelx); randul 0 este SUS
  - axa y fizica este in SUS; unghiul 0 = forta spre dreapta, 90 = forta in sus
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solveh_banded
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------------
# Parametri impusi (ficsi pentru tot proiectul)
# ----------------------------------------------------------------------------
NELX, NELY = 64, 32          # rezolutia grilei (elemente)
E0, EMIN, NU = 1.0, 1e-9, 0.3
PENAL = 3.0                  # penalizarea SIMP
RMIN = 1.5                   # raza filtrului de densitate (elemente)
MOVE = 0.2                   # pasul maxim OC
MAX_ITER = 150
TOL = 0.01                   # criteriu de convergenta (schimbare maxima)

# domeniul de variatie al conditiilor de intrare
PX_RANGE = (0.5, 1.0)        # pozitia fortei pe x (fractie din lungime)
PY_RANGE = (0.0, 1.0)        # pozitia fortei pe y (fractie din inaltime, 0 = jos)
ANGLE_RANGE = (-180.0, 180.0)
VOLFRAC_RANGE = (0.25, 0.55)


@dataclass
class Conditions:
    """Conditiile de intrare ale unui caz (variabilele modelului ML)."""
    px: float        # 0..1, pozitia pe x a fortei
    py: float        # 0..1, pozitia pe y a fortei (0 = jos)
    angle: float     # grade, 0 = dreapta, 90 = sus
    volfrac: float   # fractia de volum impusa

    def as_array(self) -> np.ndarray:
        return np.array([self.px, self.py, self.angle, self.volfrac], dtype=np.float32)

    @staticmethod
    def from_array(a) -> "Conditions":
        return Conditions(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


def load_node(c: Conditions, nelx: int = NELX, nely: int = NELY) -> tuple[int, int]:
    """Nodul (ix, iy) in care se aplica forta; iy = 0 este sus."""
    ix = int(round(c.px * nelx))
    iy = int(round((1.0 - c.py) * nely))
    ix = min(max(ix, 1), nelx)       # niciodata pe latura incastrata
    iy = min(max(iy, 0), nely)
    return ix, iy


# ----------------------------------------------------------------------------
# Matrici precalculate (o singura data per rezolutie)
# ----------------------------------------------------------------------------
def _element_stiffness(nu: float = NU) -> np.ndarray:
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8,
                  -1 / 4 + nu / 12, -1 / 8 - nu / 8, nu / 6, 1 / 8 - 3 * nu / 8])
    idx = [[0, 1, 2, 3, 4, 5, 6, 7],
           [1, 0, 7, 6, 5, 4, 3, 2],
           [2, 7, 0, 5, 6, 3, 4, 1],
           [3, 6, 5, 0, 7, 2, 1, 4],
           [4, 5, 6, 7, 0, 1, 2, 3],
           [5, 4, 3, 2, 1, 0, 7, 6],
           [6, 3, 4, 1, 2, 7, 0, 5],
           [7, 2, 1, 4, 3, 6, 5, 0]]
    return 1.0 / (1 - nu ** 2) * k[np.array(idx)]


@dataclass
class _Mesh:
    nelx: int
    nely: int
    ndof: int
    KE: np.ndarray
    edof: np.ndarray        # (nel, 8)
    free: np.ndarray        # dof-uri libere
    iKr: np.ndarray         # indici reindexati in sistemul redus
    jKr: np.ndarray
    keep: np.ndarray        # masca intrari K care raman in sistemul redus
    H: sp.csr_matrix        # matricea filtrului
    Hs: np.ndarray
    low: np.ndarray         # intrari din triunghiul inferior (pentru forma banda)
    band_idx: np.ndarray    # index plat in matricea banda (bw+1, nfree)
    bw: int                 # latimea benzii


@lru_cache(maxsize=4)
def get_mesh(nelx: int = NELX, nely: int = NELY, rmin: float = RMIN) -> _Mesh:
    nel = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = _element_stiffness()

    elx, ely = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    elx, ely = elx.ravel(), ely.ravel()                  # el = ely + elx*nely
    n1 = (nely + 1) * elx + ely
    n2 = (nely + 1) * (elx + 1) + ely
    edof = np.stack([2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
                     2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], axis=1)

    iK = np.kron(edof, np.ones((8, 1), dtype=int)).ravel()
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).ravel()

    fixed = np.arange(2 * (nely + 1))                   # latura stanga, ambele directii
    free = np.setdiff1d(np.arange(ndof), fixed)
    remap = -np.ones(ndof, dtype=int)
    remap[free] = np.arange(free.size)
    keep = (remap[iK] >= 0) & (remap[jK] >= 0)
    iKr, jKr = remap[iK[keep]], remap[jK[keep]]

    # filtru de densitate (con liniar de raza rmin)
    r = int(np.ceil(rmin)) - 1
    rows, cols, vals = [], [], []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            fac = rmin - np.hypot(di, dj)
            if fac <= 0:
                continue
            k, l = elx + di, ely + dj
            ok = (k >= 0) & (k < nelx) & (l >= 0) & (l < nely)
            rows.append((elx * nely + ely)[ok])
            cols.append((k * nely + l)[ok])
            vals.append(np.full(ok.sum(), fac))
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nel, nel)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()

    # numerotarea pe coloane da o matrice banda -> Cholesky banda (rapid)
    low = iKr >= jKr
    d = (iKr - jKr)[low]
    bw = int(d.max())
    band_idx = d * free.size + jKr[low]
    return _Mesh(nelx, nely, ndof, KE, edof, free, iKr, jKr, keep, H, Hs, low, band_idx, bw)


def _to_image(v: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    return v.reshape(nelx, nely).T


def _from_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img.T).ravel()


def _force_vector(c: Conditions, m: _Mesh) -> np.ndarray:
    f = np.zeros(m.ndof)
    ix, iy = load_node(c, m.nelx, m.nely)
    n = (m.nely + 1) * ix + iy
    a = np.deg2rad(c.angle)
    f[2 * n] = np.cos(a)
    f[2 * n + 1] = np.sin(a)
    return f


def _solve(xphys: np.ndarray, f: np.ndarray, m: _Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Rezolva K(x) u = f; intoarce deplasarile si energia elementala ce."""
    E = EMIN + xphys ** PENAL * (E0 - EMIN)
    sK = (m.KE.ravel()[:, None] * E[None, :]).ravel(order="F")[m.keep]
    n = m.free.size
    u = np.zeros(m.ndof)
    try:
        ab = np.bincount(m.band_idx, weights=sK[m.low],
                         minlength=(m.bw + 1) * n).reshape(m.bw + 1, n)
        u[m.free] = solveh_banded(ab, f[m.free], lower=True, check_finite=False)
    except np.linalg.LinAlgError:   # rezerva: solver rar general
        K = sp.coo_matrix((sK, (m.iKr, m.jKr)), shape=(n, n)).tocsc()
        u[m.free] = spsolve(K, f[m.free])
    ue = u[m.edof]
    ce = np.einsum("ij,jk,ik->i", ue, m.KE, ue)
    return u, ce


# ----------------------------------------------------------------------------
# API public
# ----------------------------------------------------------------------------
def run_simp(c: Conditions, nelx: int = NELX, nely: int = NELY,
             max_iter: int = MAX_ITER, tol: float = TOL, callback=None) -> dict:
    """Ruleaza optimizarea topologica SIMP pentru conditiile date.

    Returneaza: density (nely, nelx), compliance, iterations, time_s, history.
    """
    t0 = time.perf_counter()
    m = get_mesh(nelx, nely)
    nel = nelx * nely
    f = _force_vector(c, m)

    x = np.full(nel, c.volfrac)
    xphys = x.copy()
    history = []
    it, change, obj = 0, 1.0, np.inf
    while change > tol and it < max_iter:
        it += 1
        _, ce = _solve(xphys, f, m)
        obj = float(((EMIN + xphys ** PENAL * (E0 - EMIN)) * ce).sum())
        dc = -PENAL * xphys ** (PENAL - 1) * (E0 - EMIN) * ce
        dv = np.ones(nel)
        dc = m.H @ (dc / m.Hs)
        dv = m.H @ (dv / m.Hs)

        # Optimality Criteria cu bisectie pe multiplicatorul Lagrange
        l1, l2 = 0.0, 1e9
        scale = np.sqrt(np.maximum(-dc, 0) / dv)
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(x * scale / np.sqrt(lmid), np.maximum(0.0, x - MOVE),
                           np.minimum(1.0, x + MOVE))
            xphys_new = (m.H @ xnew) / m.Hs
            if xphys_new.sum() > c.volfrac * nel:
                l1 = lmid
            else:
                l2 = lmid
        change = float(np.abs(xnew - x).max())
        x, xphys = xnew, xphys_new
        history.append(obj)
        if callback is not None:
            callback(it, obj, _to_image(xphys, nelx, nely))

    # complianta finala, evaluata pe layout-ul final
    _, ce = _solve(xphys, f, m)
    obj = float(((EMIN + xphys ** PENAL * (E0 - EMIN)) * ce).sum())
    return {
        "density": _to_image(xphys, nelx, nely).astype(np.float32),
        "compliance": obj,
        "iterations": it,
        "time_s": time.perf_counter() - t0,
        "history": history,
    }


def compliance(density_img: np.ndarray, c: Conditions) -> float:
    """Complianta (analiza FE) unui camp de densitati oarecare, ex. prezis de CNN."""
    nely, nelx = density_img.shape
    m = get_mesh(nelx, nely)
    x = np.clip(_from_image(density_img.astype(np.float64)), 0.0, 1.0)
    _, ce = _solve(x, _force_vector(c, m), m)
    return float(((EMIN + x ** PENAL * (E0 - EMIN)) * ce).sum())


def enforce_volume(density_img: np.ndarray, volfrac: float) -> np.ndarray:
    """Post-procesare: deplaseaza densitatile (bisectie) ca media sa fie exact volfrac."""
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(density_img + mid, 0, 1).mean() > volfrac:
            hi = mid
        else:
            lo = mid
    return np.clip(density_img + 0.5 * (lo + hi), 0, 1).astype(np.float32)


def sample_conditions(n: int, seed: int = 0) -> list[Conditions]:
    """Esantionare Latin Hypercube peste domeniul conditiilor de intrare."""
    from scipy.stats import qmc
    s = qmc.LatinHypercube(d=4, seed=seed).random(n)
    lo = np.array([PX_RANGE[0], PY_RANGE[0], ANGLE_RANGE[0], VOLFRAC_RANGE[0]])
    hi = np.array([PX_RANGE[1], PY_RANGE[1], ANGLE_RANGE[1], VOLFRAC_RANGE[1]])
    pts = qmc.scale(s, lo, hi)
    return [Conditions.from_array(p) for p in pts]
