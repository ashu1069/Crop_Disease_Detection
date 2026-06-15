"""Grad-CAM visualisations (reviewer #8) for any trained model.

Loads a checkpoint, picks correctly- and incorrectly-classified test samples, and
saves Grad-CAM overlays. Works for CNNs, ViT and Swin via the per-architecture
target-layer map in models/factory.py (with a transformer reshape).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from ..config import load_config
from ..data.dataset import (IMAGENET_MEAN, IMAGENET_STD, build_transforms,
                            load_manifest, split_frames)
from ..models.factory import build_model, gradcam_target_layer


def _reshape_transform_vit(tensor, height=14, width=14):
    # ViT: drop CLS token, reshape tokens to a spatial grid
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def _reshape_transform_swin(tensor):
    # timm Swin works channels-last: norm1 output is (B, H, W, C). Some variants
    # emit a flattened (B, L, C). Handle both, returning (B, C, H, W) for CAM.
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    b, l, c = tensor.shape
    s = int(round(l ** 0.5))
    return tensor.reshape(b, s, s, c).permute(0, 3, 1, 2)


def _denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def run(config_path, ckpt, n_per_class, device, overrides):
    cfg = load_config(config_path, overrides)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=dev)
    class_names = state["class_names"]
    model = build_model(cfg.model, len(class_names)).to(dev).eval()
    model.load_state_dict(state["model"])

    df = load_manifest(cfg.data)
    _, _, test_df = split_frames(df, cfg.data.fold)
    tfm = build_transforms(cfg.data.image_size, cfg.data, train=False)

    name = cfg.model.name
    targets_layers = gradcam_target_layer(model, name)
    reshape = None
    if name.startswith(("vit", "deit")):
        g = cfg.data.image_size // 16
        reshape = lambda t: _reshape_transform_vit(t, g, g)
    elif name.startswith("swin"):
        reshape = _reshape_transform_swin

    cam = GradCAM(model=model, target_layers=targets_layers,
                  reshape_transform=reshape)

    out_dir = Path(cfg.output_dir) / cfg.run_name / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     mode=cfg.wandb.mode, name=f"gradcam-{cfg.run_name}",
                     job_type="explain")

    images_logged = []
    for label, cname in enumerate(class_names):
        sub = test_df[test_df["label"] == label].head(n_per_class)
        for _, row in sub.iterrows():
            with Image.open(row["path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            x = tfm(im).unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = int(model(x).argmax(1))
            grayscale = cam(input_tensor=x)[0]
            rgb = _denorm(x.squeeze(0).cpu())
            overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
            tag = "correct" if pred == label else "wrong"
            fname = out_dir / f"{cname}_pred-{class_names[pred]}_{tag}_{Path(row['path']).stem}.png"
            Image.fromarray(overlay).save(fname)
            images_logged.append(wandb.Image(
                str(fname), caption=f"true={cname} pred={class_names[pred]}"))
    run.log({"gradcam": images_logged})
    run.finish()
    print(f"Saved {len(images_logged)} Grad-CAM overlays -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides key=value")
    a = ap.parse_args()
    from ..config import parse_cli_overrides
    run(a.config, a.ckpt, a.n_per_class, a.device, parse_cli_overrides(a.set))


if __name__ == "__main__":
    main()
