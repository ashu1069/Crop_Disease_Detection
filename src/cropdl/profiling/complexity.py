"""Computational complexity: parameters, FLOPs/MACs, and inference latency
(reviewer #9). Produces one CSV row per model for a comparison table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import ModelConfig
from ..models.factory import REGISTRY, build_model, input_size_for


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, size, device) -> float:
    """GFLOPs for a single forward pass. Tries fvcore, falls back to ptflops."""
    x = torch.randn(1, 3, size, size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, (3, size, size), as_strings=False,
                print_per_layer_stat=False, verbose=False)
            return 2 * macs / 1e9  # MACs -> FLOPs
        except Exception:
            return float("nan")


@torch.no_grad()
def measure_latency(model, size, device, batch_size=1, warmup=10, iters=50):
    model.eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1000.0  # ms/image at this batch size, per-sample


def profile_model(name, num_classes, device, pretrained=False):
    cfg = ModelConfig(name=name, pretrained=pretrained)
    size = input_size_for(name)
    model = build_model(cfg, num_classes).to(device)
    total, trainable = count_params(model)
    gflops = measure_flops(model, size, device)
    lat1 = measure_latency(model, size, device, batch_size=1)
    lat32 = measure_latency(model, size, device, batch_size=32) / 32
    return {
        "model": name, "input_size": size,
        "params_M": round(total / 1e6, 3),
        "trainable_M": round(trainable / 1e6, 3),
        "gflops": round(gflops, 3),
        "latency_ms_bs1": round(lat1, 3),
        "latency_ms_bs32_peritem": round(lat32, 4),
        "throughput_img_s_bs32": round(1000.0 / lat32, 1) if lat32 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "densenet121", "densenet161", "densenet169", "densenet201",
        "densexnet", "densexnet_cbam", "resnet50", "vgg16", "inception_v3",
        "xception", "efficientnet_b0", "vit_small", "vit_base", "swin_tiny",
        "convnext_tiny", "deit_small"])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/complexity.csv")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--project", default="crop_disease")
    ap.add_argument("--entity", default="ashu-1069-rochester-institute-of-technology")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    rows = []
    for name in a.models:
        if name not in REGISTRY:
            print(f"skip unknown {name}")
            continue
        try:
            r = profile_model(name, a.num_classes, dev)
            rows.append(r)
            print(r)
        except Exception as e:
            print(f"FAILED {name}: {e}")
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nWrote {a.out}")

    if a.wandb:
        import wandb
        run = wandb.init(project=a.project, entity=a.entity, name="complexity",
                         job_type="profile")
        run.log({"complexity": wandb.Table(dataframe=df)})
        run.finish()


if __name__ == "__main__":
    main()
