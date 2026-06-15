"""Quantitative image-quality evaluation of dehazing methods (reviewer #4).

For each test image: clean -> add synthetic haze -> dehaze with {none, dcp,
aodnet} -> measure PSNR/SSIM against the clean reference. Reports mean PSNR/SSIM
per method so the manuscript can tabulate dehazing quality, not just downstream
accuracy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image, ImageOps

from ..data.haze import add_haze
from . import dehaze_batch, psnr, ssim


def evaluate(manifest, crop, methods, beta, airlight, size, n_images, device,
             aod_weights, wandb_mode, project, entity):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(manifest)
    df = df[(df["crop"] == crop) & (df["split"] == "test")]
    paths = df["path"].tolist()[:n_images]

    run = wandb.init(project=project, entity=entity, mode=wandb_mode,
                     name=f"dehaze-quality-{crop}", job_type="dehaze-eval",
                     config=dict(crop=crop, beta=beta, airlight=airlight,
                                 n_images=len(paths), methods=methods))
    results = {m: {"psnr": [], "ssim": []} for m in (["hazy"] + methods)}
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size))
        clean = np.asarray(im, dtype=np.float32) / 255.0
        hazy = add_haze(clean, beta=beta, airlight=airlight, seed=7)
        # baseline: hazy vs clean (no dehazing)
        results["hazy"]["psnr"].append(psnr(hazy, clean))
        results["hazy"]["ssim"].append(ssim(hazy, clean))
        hazy_t = torch.from_numpy(hazy.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        for m in methods:
            out = dehaze_batch(hazy_t, m, aod_weights=aod_weights)
            out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            results[m]["psnr"].append(psnr(out_np, clean))
            results[m]["ssim"].append(ssim(out_np, clean))

    table = wandb.Table(columns=["method", "PSNR", "SSIM"])
    print(f"\n=== Dehazing quality on {crop} (beta={beta}) ===")
    summary = {}
    for m, d in results.items():
        mp, ms = float(np.mean(d["psnr"])), float(np.mean(d["ssim"]))
        print(f"  {m:8s}  PSNR {mp:6.2f}  SSIM {ms:.4f}")
        table.add_data(m, mp, ms)
        summary[f"{m}/psnr"] = mp
        summary[f"{m}/ssim"] = ms
    run.log({"dehaze_quality": table})
    run.summary.update(summary)
    run.finish()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--crop", required=True)
    ap.add_argument("--methods", nargs="+", default=["dcp", "aodnet"])
    ap.add_argument("--beta", type=float, default=1.5)
    ap.add_argument("--airlight", type=float, default=0.8)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aod-weights", default=None)
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--project", default="crop_disease")
    ap.add_argument("--entity", default="ashu-1069-rochester-institute-of-technology")
    a = ap.parse_args()
    evaluate(a.manifest, a.crop, a.methods, a.beta, a.airlight, a.size,
             a.n_images, a.device, a.aod_weights, a.wandb_mode, a.project, a.entity)


if __name__ == "__main__":
    main()
