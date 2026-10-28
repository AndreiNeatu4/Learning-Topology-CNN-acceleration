"""Codificarea conditiilor de intrare ca imagini (canale) pentru CNN.

Canale (C = 6), fiecare de forma (nely, nelx):
  0  Fx  - pata gaussiana in punctul de aplicare a fortei * cos(unghi)
  1  Fy  - aceeasi pata * sin(unghi)
  2  rezemare - 1 pe coloana incastrata (stanga), 0 in rest
  3  fractia de volum - constanta pe toata imaginea
  4  coordonata x normalizata (CoordConv)
  5  coordonata y normalizata (CoordConv)
"""
from __future__ import annotations

import numpy as np

from .simp import NELX, NELY

N_CHANNELS = 6
SIGMA = 1.5   # latimea petei de forta (elemente)


def encode(conds: np.ndarray, nelx: int = NELX, nely: int = NELY) -> np.ndarray:
    """conds: (N, 4) = [px, py, unghi_grade, volfrac] -> (N, 6, nely, nelx) float32."""
    conds = np.atleast_2d(np.asarray(conds, dtype=np.float64))
    n = conds.shape[0]
    px, py, ang, vf = conds.T

    # acelasi nod ca in simp.load_node
    ix = np.clip(np.round(px * nelx), 1, nelx)
    iy = np.clip(np.round((1.0 - py) * nely), 0, nely)

    ex = np.arange(nelx) + 0.5          # centrele elementelor (in coordonate de noduri)
    ey = np.arange(nely) + 0.5
    dx = ex[None, None, :] - ix[:, None, None]
    dy = ey[None, :, None] - iy[:, None, None]
    blob = np.exp(-(dx ** 2 + dy ** 2) / (2 * SIGMA ** 2))

    a = np.deg2rad(ang)
    out = np.zeros((n, N_CHANNELS, nely, nelx), dtype=np.float32)
    out[:, 0] = blob * np.cos(a)[:, None, None]
    out[:, 1] = blob * np.sin(a)[:, None, None]
    out[:, 2, :, 0] = 1.0
    out[:, 3] = vf[:, None, None]
    out[:, 4] = (ex / nelx)[None, None, :]
    out[:, 5] = (ey / nely)[None, :, None]
    return out


def flip_vertical(conds: np.ndarray, density: np.ndarray):
    """Simetrie a problemei: oglindire sus-jos (py -> 1-py, unghi -> -unghi)."""
    c = np.array(conds, dtype=np.float32, copy=True)
    c[:, 1] = 1.0 - c[:, 1]
    c[:, 2] = -c[:, 2]
    return c, density[:, ::-1, :].copy()
