"""Generarea setului de date: esantionare LHS + rulari SIMP in paralel.

Utilizare din linia de comanda:
    python -m lt.dataset --n 1500 --workers 4 --out data/dataset.npz
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .simp import NELX, NELY, Conditions, run_simp, sample_conditions

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.npz"


def _worker(args):
    idx, arr = args
    r = run_simp(Conditions.from_array(arr))
    return idx, r["density"].astype(np.float16), r["compliance"], r["iterations"], r["time_s"]


def generate(n: int = 1500, seed: int = 0, workers: int | None = None,
             out: str | Path = DEFAULT_PATH, progress=None) -> Path:
    """Genereaza n perechi (conditii, layout optim SIMP) si le salveaza in .npz.

    progress(done, total, eta_s) este apelat dupa fiecare caz (pentru GUI / consola).
    """
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    conds = np.stack([c.as_array() for c in sample_conditions(n, seed)])
    dens = np.zeros((n, NELY, NELX), dtype=np.float16)
    comp = np.zeros(n, dtype=np.float64)
    iters = np.zeros(n, dtype=np.int32)
    times = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    jobs = list(enumerate(conds))
    done = 0

    def _store(res):
        nonlocal done
        i, d, c, it, t = res
        dens[i], comp[i], iters[i], times[i] = d, c, it, t
        done += 1
        if progress is not None:
            el = time.perf_counter() - t0
            progress(done, n, el / done * (n - done))

    if workers == 1:
        for j in jobs:
            _store(_worker(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_worker, j) for j in jobs]):
                _store(fut.result())

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, conds=conds, density=dens, compliance=comp,
                        iterations=iters, time_s=times, nelx=NELX, nely=NELY, seed=seed)
    return out


def load(path: str | Path = DEFAULT_PATH) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def split_indices(n: int, seed: int = 0, fractions=(0.70, 0.15, 0.15)):
    """Imparte indicii in train / val / test (aleator, reproductibil)."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    a = int(fractions[0] * n)
    b = a + int(fractions[1] * n)
    return p[:a], p[a:b], p[b:]


def main():
    ap = argparse.ArgumentParser(description="Genereaza datasetul SIMP")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(DEFAULT_PATH))
    a = ap.parse_args()

    def prog(d, t, eta):
        if d % 25 == 0 or d == t:
            print(f"  {d}/{t} cazuri, ETA {eta / 60:.1f} min", flush=True)

    print(f"Generez {a.n} cazuri SIMP ({NELX}x{NELY})...")
    p = generate(a.n, a.seed, a.workers, a.out, prog)
    print(f"Salvat: {p}")


if __name__ == "__main__":
    main()
