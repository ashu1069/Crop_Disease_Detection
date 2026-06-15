"""Inference + evaluation routines shared by training and standalone eval.

Supports an optional haze/dehaze pipeline applied to the input batch so the very
same model and loader can be scored on clean, hazy, and dehazed test sets
(reviewer #4/#5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..data.haze import add_haze_tensor

_MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _denorm(x):
    return (x * _STD.to(x.device) + _MEAN.to(x.device)).clamp(0, 1)


def _renorm(x):
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def _apply_degradation(x, haze_eval, beta, airlight, dehaze_method, aod_weights=None):
    """x is normalised. Returns normalised, optionally hazed then dehazed."""
    if not haze_eval and dehaze_method == "none":
        return x
    img = _denorm(x)
    if haze_eval:
        img = add_haze_tensor(img, beta, airlight)
    if dehaze_method != "none":
        from ..dehaze import dehaze_batch
        img = dehaze_batch(img, dehaze_method, aod_weights=aod_weights)
    return _renorm(img)


@torch.no_grad()
def run_inference(model, loader, device, *, haze_eval=False, beta=1.5,
                  airlight=0.8, dehaze_method="none", aod_weights=None,
                  collect_paths=False):
    """Return (y_true, y_prob, paths?) over a loader."""
    model.eval()
    ys, probs, paths = [], [], []
    for batch in loader:
        if len(batch) == 3:
            x, y, p = batch
        else:
            x, y = batch
            p = None
        x = x.to(device, non_blocking=True)
        x = _apply_degradation(x, haze_eval, beta, airlight, dehaze_method,
                               aod_weights)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(x)
        probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(np.asarray(y))
        if collect_paths and p is not None:
            paths.extend(list(p))
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    if collect_paths:
        return y_true, y_prob, paths
    return y_true, y_prob
