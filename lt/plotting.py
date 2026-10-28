"""Figuri reutilizabile (raport + GUI)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from .simp import NELX, NELY, Conditions, load_node

ARROW_LEN = 9.0


def draw_layout(ax, density: np.ndarray, cond: Conditions, title: str = "", show_load: bool = True):
    """Deseneaza un layout (negru = material), forta (sageata rosie) si incastrarea (hasura)."""
    nely, nelx = density.shape
    ax.imshow(1.0 - density, cmap="gray", vmin=0, vmax=1, interpolation="nearest",
              extent=(-0.5, nelx - 0.5, nely - 0.5, -0.5))
    # incastrare pe latura stanga
    for y in np.linspace(-0.5, nely - 0.5, 9):
        ax.plot([-0.5, -2.5], [y, y + 1.5], color="#3b6fd8", lw=1.2, clip_on=False)
    ax.plot([-0.5, -0.5], [-0.5, nely - 0.5], color="#3b6fd8", lw=2.5, clip_on=False)
    if show_load:
        ix, iy = load_node(cond, nelx, nely)
        hx, hy = ix - 0.5, iy - 0.5
        a = np.deg2rad(cond.angle)
        tx, ty = hx - ARROW_LEN * np.cos(a), hy + ARROW_LEN * np.sin(a)
        ax.add_patch(FancyArrowPatch((tx, ty), (hx, hy), arrowstyle="-|>", mutation_scale=14,
                                     color="#d62728", lw=2, clip_on=False, zorder=5))
    ax.set_xlim(-4, nelx + 3)
    ax.set_ylim(nely + 3, -4)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def compare_figure(cond: Conditions, simp_d: np.ndarray | None, cnn_d: np.ndarray,
                   simp_title: str = "SIMP (referinta)", cnn_title: str = "CNN (U-Net)"):
    """Figura SIMP vs CNN (sau doar CNN daca simp_d este None)."""
    if simp_d is None:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6))
        draw_layout(ax, cnn_d, cond, cnn_title)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6),
                                gridspec_kw={"width_ratios": [1, 1, 1]})
        draw_layout(axs[0], simp_d, cond, simp_title)
        draw_layout(axs[1], cnn_d, cond, cnn_title)
        diff = cnn_d - simp_d
        im = axs[2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        axs[2].set_title("Diferenta CNN - SIMP", fontsize=10)
        axs[2].axis("off")
        fig.colorbar(im, ax=axs[2], fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def learning_curve(hist: dict, ylabel: str = "loss (MSE + termen volum)"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="validare")
    ax.set_yscale("log")
    ax.set_xlabel("epoca")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def dc_histogram(dc: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(dc, bins=40, color="#3b6fd8", alpha=0.85)
    ax.axvline(np.median(dc), color="#d62728", ls="--", label=f"mediana {np.median(dc):.1f}%")
    ax.set_xlabel("Δc = (c_CNN - c_SIMP) / c_SIMP  [%]")
    ax.set_ylabel("numar cazuri test")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def dc_vs_volfrac(volfrac: np.ndarray, dc: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter(volfrac, dc, s=10, alpha=0.6, color="#3b6fd8")
    ax.set_xlabel("fractia de volum impusa")
    ax.set_ylabel("Δc [%]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def examples_figure(rows: list[tuple[str, Conditions, np.ndarray, np.ndarray, float]]):
    """rows: (eticheta, conditii, simp, cnn, dc%)."""
    fig, axs = plt.subplots(len(rows), 2, figsize=(10, 2.9 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, (lab, c, s, p, dc) in enumerate(rows):
        draw_layout(axs[r, 0], s, c, f"{lab} - SIMP")
        draw_layout(axs[r, 1], p, c, f"{lab} - CNN (Δc = {dc:+.1f}%)")
    fig.tight_layout()
    return fig


def dataset_grid(conds: np.ndarray, dens: np.ndarray, n: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(conds), size=min(n, len(conds)), replace=False)
    cols = 4
    rws = int(np.ceil(len(idx) / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(14, 2.3 * rws))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, i in zip(np.ravel(axs), idx):
        c = Conditions.from_array(conds[i])
        draw_layout(ax, dens[i].astype(np.float32), c,
                    f"unghi {c.angle:.0f}°, vf {c.volfrac:.2f}")
    fig.tight_layout()
    return fig
