"""Integrarea dataset-ului public TOP (Sosnovik & Oseledets, "Neural Networks for Topology
Optimization", 2017) - https://github.com/ISosnovik/top

Datasetul: 10.000 de probleme generate cu ToPy (SIMP), grila 40x40. Fiecare obiect este un tensor
(100, 40, 40) = istoria a 100 de iteratii SIMP, in fisiere .npz citite cu np.load(f)['arr_0'].
Nu contine fortele / rezemarile, deci sarcina este cea din articol:

    CNN-ul primeste starea SIMP de la iteratia k (densitatea x_k si gradientul x_k - x_{k-1})
    si prezice direct layout-ul final  ->  SIMP "terminat" dupa doar k iteratii.

Comparatia de referinta (baseline) este pragul naiv: binarizarea lui x_k la 0.5.

Utilizare din linia de comanda:
    python -m lt.sosnovik import --src C:\\cale\\catre\\TOP4040
    python -m lt.sosnovik train --epochs 30
    python -m lt.sosnovik evaluate
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import dataset as ds
from .unet import UNet, pick_device

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "sosnovik_raw"          # aici pui arhiva dezarhivata
PACKED = ROOT / "data" / "sosnovik.npz"           # format compact, creat la import
MODEL = ROOT / "models" / "sosnovik" / "unet_sosnovik.pt"
REPORTS = ROOT / "reports" / "sosnovik"

K_VALUES = (3, 5, 10, 20, 30)   # iteratiile k pastrate la import (+ k-1 si starea finala)
IN_CH = 2


# ----------------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------------
def find_files(src: str | Path) -> list[Path]:
    return sorted(Path(src).rglob("*.npz"))


def _iter_objects(path: Path):
    """Un fisier poate contine un obiect (T, H, W) sau mai multe (N, T, H, W)."""
    with np.load(path) as z:
        a = z["arr_0"] if "arr_0" in z.files else z[z.files[0]]
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        raise ValueError(f"{path.name}: forma neasteptata {a.shape}")
    yield from a


def import_dataset(src: str | Path = RAW_DIR, out: str | Path = PACKED,
                   max_samples: int | None = None, progress=None) -> dict:
    """Citeste fisierele originale si salveaza doar cadrele necesare (uint8, ~15 MB / 1000 cazuri).

    progress(done_files, total_files) - pentru GUI / consola.
    """
    files = find_files(src)
    if not files:
        raise FileNotFoundError(f"Nu am gasit fisiere .npz in {src}")
    need = max(K_VALUES) + 1
    frames, skipped = [], 0
    for i, f in enumerate(files, 1):
        for hist in _iter_objects(f):
            if hist.shape[0] < need:
                skipped += 1
                continue
            sel = []
            for k in K_VALUES:
                sel += [hist[k - 1], hist[k]]
            sel.append(hist[-1])
            frames.append(np.round(np.clip(np.stack(sel), 0, 1) * 255).astype(np.uint8))
            if max_samples and len(frames) >= max_samples:
                break
        if progress is not None:
            progress(i, len(files))
        if max_samples and len(frames) >= max_samples:
            break
    if not frames:
        raise ValueError("Niciun obiect valid (sunt necesare cel putin "
                         f"{need} iteratii per obiect).")
    frames = np.stack(frames)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, k_values=np.array(K_VALUES))
    return {"n_samples": int(len(frames)), "n_files": len(files), "skipped": skipped,
            "shape": list(frames.shape[2:]), "out": str(out)}


def load_packed(path: str | Path = PACKED) -> dict:
    with np.load(path) as z:
        return {"frames": z["frames"], "k_values": tuple(int(k) for k in z["k_values"])}


def make_inputs(frames: np.ndarray, kidx: np.ndarray | int):
    """frames (N, 2K+1, H, W) uint8 -> intrari (N, 2, H, W) si tinta (N, 1, H, W), float32."""
    n = len(frames)
    kidx = np.broadcast_to(np.asarray(kidx), (n,))
    rows = np.arange(n)
    xk = frames[rows, 2 * kidx + 1].astype(np.float32) / 255
    xk1 = frames[rows, 2 * kidx].astype(np.float32) / 255
    y = frames[:, -1].astype(np.float32) / 255
    return np.stack([xk, xk - xk1], 1), y[:, None]


def _dihedral(x: torch.Tensor, y: torch.Tensor):
    """Augmentare: una din cele 8 simetrii ale patratului (ca in articol)."""
    r = int(torch.randint(4, (1,)))
    x, y = torch.rot90(x, r, (2, 3)), torch.rot90(y, r, (2, 3))
    if torch.rand(1) < 0.5:
        x, y = torch.flip(x, (3,)), torch.flip(y, (3,))
    return x, y


# ----------------------------------------------------------------------------
# Antrenare
# ----------------------------------------------------------------------------
def train(data_path=PACKED, out_path=MODEL, epochs: int = 30, batch_size: int = 64,
          lr: float = 2e-3, lam_vol: float = 1.0, base: int = 32, seed: int = 0,
          device: str | None = None, progress=None, should_stop=None) -> dict:
    """La fiecare pas, k este ales aleator dintre K_VALUES (model robust pentru orice k)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    d = load_packed(data_path)
    frames, kv = d["frames"], d["k_values"]
    tr, va, _ = ds.split_indices(len(frames), seed)

    model = UNet(in_ch=IN_CH, base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    # validare fixa: fiecare caz la fiecare k
    xv, yv = zip(*[make_inputs(frames[va], j) for j in range(len(kv))])
    Xva, Yva = torch.from_numpy(np.concatenate(xv)), torch.from_numpy(np.concatenate(yv))

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "time_s": []}
    best, t0 = np.inf, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            idx = np.sort(perm[i:i + batch_size])
            x, y = make_inputs(frames[idx], rng.integers(len(kv), size=len(idx)))
            xb, yb = _dihedral(torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev))
            p = model(xb)
            vol = ((p.mean((1, 2, 3)) - xb[:, 0].mean((1, 2))) ** 2).mean()
            loss = F.binary_cross_entropy(p, yb) + lam_vol * vol
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            vl, acc = 0.0, 0.0
            for i in range(0, len(Xva), 512):
                xb, yb = Xva[i:i + 512].to(dev), Yva[i:i + 512].to(dev)
                p = model(xb)
                vl += F.binary_cross_entropy(p, yb, reduction="sum").item() / yb[0].numel()
                acc += ((p > 0.5) == (yb > 0.5)).float().mean((1, 2, 3)).sum().item()
        vl, acc = vl / len(Xva), acc / len(Xva)
        for key, v in [("epoch", ep), ("train_loss", tot / len(tr)), ("val_loss", vl),
                       ("val_acc", acc), ("time_s", time.perf_counter() - t0)]:
            hist[key].append(v)

        if vl < best:
            best = vl
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "config": {"task": "sosnovik", "in_ch": IN_CH, "base": base,
                                   "k_values": list(kv), "lam_vol": lam_vol, "epochs": epochs,
                                   "best_epoch": ep, "n_train": int(len(tr))},
                        "history": hist}, out_path)
        if progress is not None:
            progress(ep, epochs, hist)
        if should_stop is not None and should_stop():
            break

    ck = torch.load(out_path, map_location="cpu", weights_only=False)
    ck["history"] = hist
    torch.save(ck, out_path)
    return hist


# ----------------------------------------------------------------------------
# Evaluare
# ----------------------------------------------------------------------------
def _binary_metrics(p: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P, T = p > 0.5, t > 0.5
    acc = (P == T).mean((1, 2))
    inter = (P & T).sum((1, 2))
    union = (P | T).sum((1, 2))
    return acc, np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def evaluate(model_path=MODEL, data_path=PACKED, out_dir=REPORTS, seed: int = 0,
             progress=None) -> dict:
    from . import plotting
    from .unet import Predictor
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_packed(data_path)
    frames, kv = d["frames"], d["k_values"]
    _, _, te = ds.split_indices(len(frames), seed)
    pr = Predictor(model_path)

    rows, ex = [], None
    for j, k in enumerate(kv):
        x, y = make_inputs(frames[te], j)
        t0 = time.perf_counter()
        p = pr.predict_raw(x)
        ms = (time.perf_counter() - t0) / len(x) * 1000
        t = y[:, 0]
        acc_cnn, iou_cnn = _binary_metrics(p, t)
        acc_thr, iou_thr = _binary_metrics(x[:, 0], t)
        rows.append({"k": k, "acc_cnn": float(acc_cnn.mean()), "acc_prag": float(acc_thr.mean()),
                     "iou_cnn": float(iou_cnn.mean()), "iou_prag": float(iou_thr.mean()),
                     "vol_err_cnn": float(np.abs(p.mean((1, 2)) - t.mean((1, 2))).mean()),
                     "ms_per_caz": ms})
        if j == 0:
            order = np.argsort(iou_cnn)
            ex = [(lab, x[i, 0], p[i], t[i], iou_cnn[i]) for lab, i in
                  [("Cel mai bun", order[-1]), ("Median", order[len(order) // 2]),
                   ("Cel mai slab", order[0])]]
        if progress is not None:
            progress(j + 1, len(kv))

    summary = {"n_test": int(len(te)), "per_k": rows, "n_params": pr.n_params,
               "device": str(pr.device)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plotting.plt.subplots(figsize=(6.4, 3.4))
    ks = [r["k"] for r in rows]
    ax.plot(ks, [r["acc_cnn"] for r in rows], "o-", label="CNN")
    ax.plot(ks, [r["acc_prag"] for r in rows], "s--", label="prag naiv pe x_k")
    ax.set_xlabel("iteratia k la care se opreste SIMP")
    ax.set_ylabel("acuratete binara fata de final")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "acc_vs_k.png", dpi=120)
    plotting.plt.close(fig)

    fig = examples_figure(ex, kv[0])
    fig.savefig(out_dir / "examples.png", dpi=120)
    plotting.plt.close(fig)
    if pr.history:
        fig = plotting.learning_curve(pr.history, "loss (BCE + termen volum)")
        fig.savefig(out_dir / "learning_curve.png", dpi=120)
        plotting.plt.close(fig)
    return summary


def examples_figure(rows, k: int):
    from . import plotting
    fig, axs = plotting.plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, (lab, xk, p, t, iou) in enumerate(rows):
        for c, (img, tit) in enumerate([(xk, f"{lab}: SIMP la k={k}"),
                                        (p, f"CNN (IoU {iou:.2f})"), (t, "SIMP final")]):
            axs[r, c].imshow(1 - img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axs[r, c].set_title(tit, fontsize=10)
            axs[r, c].axis("off")
    fig.tight_layout()
    return fig


def history_figure(frames: np.ndarray, kv, n: int = 4, seed: int = 0):
    """Previzualizare: pentru cateva cazuri, cadrele k din K_VALUES si starea finala."""
    from . import plotting
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frames), size=min(n, len(frames)), replace=False)
    cols = len(kv) + 1
    fig, axs = plotting.plt.subplots(len(idx), cols, figsize=(2.2 * cols, 2.3 * len(idx)))
    axs = np.atleast_2d(axs)
    for r, i in enumerate(idx):
        for c in range(cols):
            img = frames[i, 2 * c + 1] if c < len(kv) else frames[i, -1]
            axs[r, c].imshow(255 - img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axs[r, c].set_title(f"k = {kv[c]}" if c < len(kv) else "final", fontsize=9)
            axs[r, c].axis("off")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Dataset Sosnovik & Oseledets (TOP)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_imp = sub.add_parser("import")
    a_imp.add_argument("--src", default=str(RAW_DIR))
    a_imp.add_argument("--max", type=int, default=None)
    a_tr = sub.add_parser("train")
    a_tr.add_argument("--epochs", type=int, default=30)
    a_tr.add_argument("--batch-size", type=int, default=64)
    sub.add_parser("evaluate")
    a = ap.parse_args()

    if a.cmd == "import":
        r = import_dataset(a.src, PACKED, a.max,
                           lambda d, n: print(f"  {d}/{n} fisiere", flush=True)
                           if d % 500 == 0 or d == n else None)
        print(json.dumps(r, indent=2))
    elif a.cmd == "train":
        train(PACKED, MODEL, a.epochs, a.batch_size,
              progress=lambda e, n, h: print(f"epoca {e}/{n} val {h['val_loss'][-1]:.4f} "
                                             f"acc {h['val_acc'][-1]:.4f}", flush=True))
    else:
        print(json.dumps(evaluate(), indent=2))


if __name__ == "__main__":
    main()
