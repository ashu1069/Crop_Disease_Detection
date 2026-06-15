"""Self-supervised AOD-Net training on synthetically-hazed crop images.

We sample clean crop images, apply the atmospheric-scattering haze model to make
(hazy, clean) pairs, and train AOD-Net to invert it (L2 + 0.1*L1). The trained
weights are then reused at classification-eval time as the learned dehazer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from ..data.haze import add_haze
from ..seed import set_seed
from . import psnr, ssim
from .aodnet import AODNet


class HazePairs(Dataset):
    def __init__(self, paths, size=224, beta_range=(0.8, 2.0), air_range=(0.7, 0.95)):
        self.paths = paths
        self.size = size
        self.beta_range = beta_range
        self.air_range = air_range

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize(
                (self.size, self.size))
        clean = np.asarray(im, dtype=np.float32) / 255.0
        beta = np.random.uniform(*self.beta_range)
        air = np.random.uniform(*self.air_range)
        hazy = add_haze(clean, beta=beta, airlight=air, seed=i)
        to_t = lambda a: torch.from_numpy(a.transpose(2, 0, 1)).float()
        return to_t(hazy), to_t(clean)


def train(manifest, crop, out, epochs, batch_size, lr, size, device, wandb_mode,
          project, entity):
    set_seed(42)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(manifest)
    df = df[(df["crop"] == crop) & (df["split"] == "trainval")]
    paths = df["path"].tolist()
    ds = HazePairs(paths, size=size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8,
                    pin_memory=True, drop_last=True)

    net = AODNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    l2, l1 = nn.MSELoss(), nn.L1Loss()
    run = wandb.init(project=project, entity=entity, mode=wandb_mode,
                     name=f"aodnet-{crop}", job_type="dehaze-train",
                     config=dict(crop=crop, epochs=epochs, lr=lr, size=size))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for hazy, clean in dl:
            hazy, clean = hazy.to(dev), clean.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = net(hazy)
            loss = l2(pred, clean) + 0.1 * l1(pred, clean)
            loss.backward()
            opt.step()
            tot += loss.item()
        # quick quality probe on one batch
        net.eval()
        with torch.no_grad():
            hazy, clean = next(iter(dl))
            pred = net(hazy.to(dev)).cpu().numpy().transpose(0, 2, 3, 1)
            cl = clean.numpy().transpose(0, 2, 3, 1)
            p = np.mean([psnr(pred[i], cl[i]) for i in range(len(pred))])
            s = np.mean([ssim(pred[i], cl[i]) for i in range(len(pred))])
        run.log({"epoch": ep, "loss": tot / len(dl), "probe/psnr": p, "probe/ssim": s})
        print(f"[aodnet-{crop}] ep {ep} loss {tot/len(dl):.4f} psnr {p:.2f} ssim {s:.3f}")
    torch.save({"model": net.state_dict()}, out)
    run.summary["weights"] = str(out)
    run.finish()
    print(f"Saved AOD-Net weights -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--crop", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--project", default="crop_disease")
    ap.add_argument("--entity", default="ashu-1069-rochester-institute-of-technology")
    a = ap.parse_args()
    out = a.out or f"outputs/dehaze/aodnet_{a.crop}.pt"
    train(a.manifest, a.crop, out, a.epochs, a.batch_size, a.lr, a.size,
          a.device, a.wandb_mode, a.project, a.entity)


if __name__ == "__main__":
    main()
