"""Antrenarea U-Net pe datele SIMP.

Functia de cost: MSE pe pixel + lambda * (media densitatii prezise - fractia de volum)^2
(termenul de volum este hook-ul "physics-informed": impune constrangerea problemei).

Utilizare:
    python -m lt.train --epochs 60
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
from .encoding import encode, flip_vertical
from .unet import DEFAULT_MODEL, UNet, pick_device


def _tensors(conds, dens):
    return torch.from_numpy(encode(conds)), torch.from_numpy(dens.astype(np.float32))[:, None]


def loss_fn(pred, y, volfrac, lam_vol):
    mse = F.mse_loss(pred, y)
    vol = ((pred.mean(dim=(1, 2, 3)) - volfrac) ** 2).mean()
    return mse + lam_vol * vol, mse


def train(data_path=ds.DEFAULT_PATH, out_path=DEFAULT_MODEL, epochs: int = 60,
          batch_size: int = 32, lr: float = 2e-3, lam_vol: float = 1.0, base: int = 32,
          seed: int = 0, augment: bool = True, device: str | None = None,
          progress=None, should_stop=None) -> dict:
    """Antreneaza modelul si salveaza cel mai bun checkpoint (dupa loss-ul de validare).

    progress(epoch, epochs, history) - apelat la final de epoca (GUI / consola)
    should_stop() -> bool              - oprire anticipata ceruta din GUI
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = pick_device(device)

    data = ds.load(data_path)
    conds, dens = data["conds"], data["density"].astype(np.float32)
    tr, va, _ = ds.split_indices(len(conds), seed)

    c_tr, d_tr = conds[tr], dens[tr]
    if augment:   # simetria sus-jos dubleaza datele de antrenare
        c_f, d_f = flip_vertical(c_tr, d_tr)
        c_tr, d_tr = np.concatenate([c_tr, c_f]), np.concatenate([d_tr, d_f])
    Xtr, Ytr = _tensors(c_tr, d_tr)
    Vtr = torch.from_numpy(c_tr[:, 3].astype(np.float32))
    Xva, Yva = _tensors(conds[va], dens[va])
    Vva = torch.from_numpy(conds[va, 3].astype(np.float32))

    model = UNet(base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xtr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": [], "lr": [], "time_s": []}
    best, t0 = np.inf, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, vb = Xtr[idx].to(dev), Ytr[idx].to(dev), Vtr[idx].to(dev)
            loss, _ = loss_fn(model(xb), yb, vb, lam_vol)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            vl, vm = 0.0, 0.0
            for i in range(0, len(Xva), 256):
                xb, yb, vb = Xva[i:i + 256].to(dev), Yva[i:i + 256].to(dev), Vva[i:i + 256].to(dev)
                p = model(xb)
                l, _ = loss_fn(p, yb, vb, lam_vol)
                vl += l.item() * len(xb)
                vm += (p - yb).abs().mean(dim=(1, 2, 3)).sum().item()
        vl, vm = vl / len(Xva), vm / len(Xva)

        hist["epoch"].append(ep)
        hist["train_loss"].append(tot / len(Xtr))
        hist["val_loss"].append(vl)
        hist["val_mae"].append(vm)
        hist["lr"].append(sched.get_last_lr()[0])
        hist["time_s"].append(time.perf_counter() - t0)

        if vl < best:
            best = vl
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "config": {"base": base, "lam_vol": lam_vol, "lr": lr, "epochs": epochs,
                                   "batch_size": batch_size, "seed": seed, "augment": augment,
                                   "n_train": int(len(Xtr)), "best_epoch": ep},
                        "history": hist}, out_path)
        if progress is not None:
            progress(ep, epochs, hist)
        if should_stop is not None and should_stop():
            break

    # salveaza istoricul complet si in checkpoint
    ck = torch.load(out_path, map_location="cpu", weights_only=False)
    ck["history"] = hist
    torch.save(ck, out_path)
    return hist


def main():
    ap = argparse.ArgumentParser(description="Antreneaza U-Net pe datele SIMP")
    ap.add_argument("--data", default=str(ds.DEFAULT_PATH))
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam-vol", type=float, default=1.0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    def prog(ep, n, h):
        print(f"epoca {ep:3d}/{n}  train {h['train_loss'][-1]:.5f}  val {h['val_loss'][-1]:.5f}  "
              f"MAE val {h['val_mae'][-1]:.4f}  ({h['time_s'][-1]:.0f}s)", flush=True)

    h = train(a.data, a.out, a.epochs, a.batch_size, a.lr, a.lam_vol, a.base,
              augment=not a.no_augment, device=a.device, progress=prog)
    print(json.dumps({"best_val_loss": min(h["val_loss"])}, indent=2))


if __name__ == "__main__":
    main()
