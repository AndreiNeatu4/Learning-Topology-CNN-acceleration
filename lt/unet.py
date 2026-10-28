"""Modelul ML: U-Net (encoder-decoder cu skip connections) pentru mapare image-to-image."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_CHANNELS, encode

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "unet.pt"


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net cu 3 niveluri de downsampling (intrare H, W divizibile cu 8)."""

    def __init__(self, in_ch: int = N_CHANNELS, base: int = 32):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = DoubleConv(in_ch, b), DoubleConv(b, 2 * b), DoubleConv(2 * b, 4 * b)
        self.mid = DoubleConv(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2), DoubleConv(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2), DoubleConv(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2), DoubleConv(2 * b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        m = self.mid(F.max_pool2d(x3, 2))
        y = self.d3(torch.cat([self.u3(m), x3], 1))
        y = self.d2(torch.cat([self.u2(y), x2], 1))
        y = self.d1(torch.cat([self.u1(y), x1], 1))
        return torch.sigmoid(self.head(y))


def pick_device(pref: str | None = None) -> torch.device:
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Predictor:
    """Incarca un checkpoint si prezice layout-ul optim pentru conditii date."""

    def __init__(self, path: str | Path = DEFAULT_MODEL, device: str | None = None):
        self.device = pick_device(device)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ck["config"]
        self.history = ck.get("history", {})
        self.model = UNet(in_ch=self.config.get("in_ch", N_CHANNELS),
                          base=self.config.get("base", 32)).to(self.device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def predict(self, conds: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """conds (N, 4) -> densitati (N, nely, nelx)."""
        return self.predict_raw(encode(conds), batch_size)

    @torch.no_grad()
    def predict_raw(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Intrare deja codificata (N, C, H, W) -> (N, H, W)."""
        x = np.asarray(x, dtype=np.float32)
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            out.append(self.model(xb)[:, 0].cpu().numpy())
        return np.concatenate(out)

    def predict_one(self, cond) -> tuple[np.ndarray, float]:
        """Prezice un singur caz; intoarce (densitate, timp_s)."""
        t0 = time.perf_counter()
        d = self.predict(np.asarray(cond, dtype=np.float32)[None])[0]
        return d, time.perf_counter() - t0
