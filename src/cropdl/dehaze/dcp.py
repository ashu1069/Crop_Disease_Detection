"""Dark Channel Prior dehazing (He et al., CVPR 2009) with guided filtering.

A classical, training-free baseline for the dehazing comparison (reviewer #4).
Operates on float RGB images in [0,1].
"""
from __future__ import annotations

import cv2
import numpy as np


def _dark_channel(img: np.ndarray, patch: int = 15) -> np.ndarray:
    mn = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(mn, kernel)


def _atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    h, w = dark.shape
    n = max(int(0.001 * h * w), 1)
    idx = np.argsort(dark.ravel())[-n:]
    flat = img.reshape(-1, 3)
    return flat[idx].max(axis=0)


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 60,
                   eps: float = 1e-3) -> np.ndarray:
    mean_g = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
    mean_s = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
    mean_gs = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))
    cov = mean_gs - mean_g * mean_s
    mean_gg = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
    var = mean_gg - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    return mean_a * guide + mean_b


def dehaze_dcp(img: np.ndarray, omega: float = 0.95, t0: float = 0.1,
               patch: int = 15) -> np.ndarray:
    """Return dehazed RGB float image in [0,1]."""
    img = img.astype(np.float64)
    dark = _dark_channel(img, patch)
    A = _atmospheric_light(img, dark)
    A = np.clip(A, 1e-3, 1.0)
    norm = img / A
    trans = 1 - omega * _dark_channel(norm, patch)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    trans = _guided_filter(gray, trans)
    trans = np.clip(trans, t0, 1.0)[..., None]
    J = (img - A) / trans + A
    return np.clip(J, 0, 1)
