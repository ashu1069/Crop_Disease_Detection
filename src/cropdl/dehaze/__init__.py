"""Dehazing methods + a unified batch interface and image-quality metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .aodnet import AODNet
from .dcp import dehaze_dcp

_AOD_CACHE: dict[str, AODNet] = {}


def load_aodnet(weights: str | None, device) -> AODNet:
    key = weights or "untrained"
    if key not in _AOD_CACHE:
        net = AODNet().to(device).eval()
        if weights and Path(weights).exists():
            sd = torch.load(weights, map_location=device)
            net.load_state_dict(sd["model"] if "model" in sd else sd)
        _AOD_CACHE[key] = net
    return _AOD_CACHE[key]


def dehaze_batch(img: torch.Tensor, method: str, aod_weights: str | None = None
                 ) -> torch.Tensor:
    """Dehaze a batch of images in [0,1], shape (B,3,H,W). Returns [0,1]."""
    if method == "none":
        return img
    if method == "dcp":
        out = torch.empty_like(img)
        arr = img.detach().cpu().permute(0, 2, 3, 1).numpy()
        for i in range(arr.shape[0]):
            out[i] = torch.from_numpy(
                dehaze_dcp(arr[i]).transpose(2, 0, 1)).to(img.device)
        return out
    if method == "aodnet":
        net = load_aodnet(aod_weights, img.device)
        with torch.no_grad():
            return net(img)
    raise ValueError(f"Unknown dehaze method '{method}'")


# ---- image quality metrics (reviewer #4: quantitative dehazing evaluation) ----

def psnr(restored: np.ndarray, clean: np.ndarray) -> float:
    mse = np.mean((restored - clean) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10 * np.log10(1.0 / mse))


def ssim(restored: np.ndarray, clean: np.ndarray) -> float:
    """Mean SSIM over channels for float RGB images in [0,1] (Wang et al. 2004).

    Self-contained Gaussian-windowed implementation (no scikit-image dependency).
    """
    import cv2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(restored.shape[2]):
        x = restored[..., c].astype(np.float64)
        y = clean[..., c].astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        sig_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sig_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        s = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2))
        vals.append(s.mean())
    return float(np.mean(vals))
