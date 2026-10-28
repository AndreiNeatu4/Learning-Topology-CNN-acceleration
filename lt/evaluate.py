"""Evaluarea modelului pe setul de test (cazuri nevazute la antrenare).

Trei niveluri (vezi specificatia, cap. 2.5):
  1. asemanare structurala: MAE, IoU (prag 0.5)
  2. metrica fizica: complianta layout-ului prezis (analiza FE) vs SIMP -> Δc%
  3. respectarea constrangerii de volum
plus timpul: o predictie CNN vs o rulare SIMP completa.

Utilizare:
    python -m lt.evaluate
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import dataset as ds
from . import plotting
from .simp import Conditions, compliance, enforce_volume
from .unet import DEFAULT_MODEL, Predictor

DEFAULT_REPORTS = Path(__file__).resolve().parent.parent / "reports"


def iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    A, B = a > thr, b > thr
    u = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / u) if u else 1.0


def evaluate(model_path=DEFAULT_MODEL, data_path=ds.DEFAULT_PATH, out_dir=DEFAULT_REPORTS,
             progress=None, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = ds.load(data_path)
    _, _, te = ds.split_indices(len(data["conds"]), seed)
    conds = data["conds"][te]
    true = data["density"][te].astype(np.float32)
    c_simp = data["compliance"][te]

    pred_model = Predictor(model_path)
    t0 = time.perf_counter()
    pred = pred_model.predict(conds)
    batch_ms = (time.perf_counter() - t0) / len(conds) * 1000

    pred_model.predict_one(conds[0])            # incalzire
    single = [pred_model.predict_one(conds[i % len(conds)])[1] for i in range(30)]
    single_ms = float(np.median(single) * 1000)

    rows = []
    for k in range(len(conds)):
        c = Conditions.from_array(conds[k])
        p = pred[k]
        pv = enforce_volume(p, c.volfrac)
        c_raw = compliance(p, c)
        c_vol = compliance(pv, c)
        rows.append({
            "idx": int(te[k]), "px": c.px, "py": c.py, "angle": c.angle, "volfrac": c.volfrac,
            "mae": float(np.abs(p - true[k]).mean()),
            "iou": iou(p, true[k]),
            "vol_err": float(p.mean() - c.volfrac),
            "c_simp": float(c_simp[k]),
            "c_cnn_raw": c_raw,
            "c_cnn": c_vol,
            "dc_pct": (c_vol - c_simp[k]) / c_simp[k] * 100,
        })
        if progress is not None:
            progress(k + 1, len(conds))

    with open(out_dir / "metrics_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    dc = np.array([r["dc_pct"] for r in rows])
    simp_s = float(data["time_s"].mean())
    summary = {
        "n_test": len(rows),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "iou_mean": float(np.mean([r["iou"] for r in rows])),
        "vol_err_abs_mean": float(np.mean([abs(r["vol_err"]) for r in rows])),
        "dc_mean_pct": float(dc.mean()),
        "dc_median_pct": float(np.median(dc)),
        "dc_p90_pct": float(np.percentile(dc, 90)),
        "frac_dc_below_5pct": float((dc < 5).mean()),
        "frac_dc_below_10pct": float((dc < 10).mean()),
        "cnn_ms_single": single_ms,
        "cnn_ms_batch_per_case": float(batch_ms),
        "simp_s_mean": simp_s,
        "simp_iterations_mean": float(data["iterations"].mean()),
        "speedup_single": simp_s / (single_ms / 1000),
        "n_params": int(pred_model.n_params),
        "device": str(pred_model.device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figuri
    order = np.argsort(dc)
    pick = [("Cel mai bun", order[0]), ("Median", order[len(order) // 2]), ("Cel mai slab", order[-1])]
    ex = [(lab, Conditions.from_array(conds[i]), true[i], pred[i], dc[i]) for lab, i in pick]
    figs = {
        "examples.png": plotting.examples_figure(ex),
        "hist_dc.png": plotting.dc_histogram(dc),
        "dc_vs_volfrac.png": plotting.dc_vs_volfrac(conds[:, 3], dc),
    }
    if pred_model.history:
        figs["learning_curve.png"] = plotting.learning_curve(pred_model.history)
    for name, fig in figs.items():
        fig.savefig(out_dir / name, dpi=120)
        plotting.plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evalueaza modelul pe setul de test")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--data", default=str(ds.DEFAULT_PATH))
    ap.add_argument("--out", default=None, help="implicit: reports/<nume model>")
    a = ap.parse_args()
    out = a.out or DEFAULT_REPORTS / Path(a.model).stem
    s = evaluate(a.model, a.data, out,
                 progress=lambda d, n: print(f"  {d}/{n}", flush=True) if d % 50 == 0 else None)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
