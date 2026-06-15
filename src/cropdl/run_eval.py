"""Evaluate a trained checkpoint on the test set under clean / hazy / dehazed
conditions (reviewer #4/#5). Logs metrics + confusion matrices to W&B.

    python -m cropdl.run_eval --config configs/dehazing.yaml \
        --ckpt outputs/dehazing-okra-densexnet-f0/best.pt \
        --conditions clean hazy hazy_dcp hazy_aodnet \
        --aod-weights outputs/dehaze/aodnet_okra.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from .config import load_config, parse_cli_overrides
from .data.dataset import build_loaders
from .engine.evaluate import run_inference
from .engine.metrics import compute_metrics, confusion, save_confusion_plot, text_report
from .models.factory import build_model
from .seed import set_seed

CONDITIONS = {
    "clean": dict(haze_eval=False, dehaze_method="none"),
    "hazy": dict(haze_eval=True, dehaze_method="none"),
    "hazy_dcp": dict(haze_eval=True, dehaze_method="dcp"),
    "hazy_aodnet": dict(haze_eval=True, dehaze_method="aodnet"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--aod-weights", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, parse_cli_overrides(args.set))
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    _, _, test_loader, meta = build_loaders(cfg.data)
    state = torch.load(args.ckpt, map_location=device)
    model = build_model(cfg.model, meta["num_classes"]).to(device)
    model.load_state_dict(state["model"])

    # make aod weights discoverable by the dehaze batch interface
    if args.aod_weights:
        import cropdl.dehaze as dz
        dz.load_aodnet(args.aod_weights, device)

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     mode=cfg.wandb.mode, name=f"eval-{cfg.run_name}",
                     job_type="robustness-eval", tags=["dehazing-eval"],
                     config={**cfg.to_dict(), "ckpt": args.ckpt})
    out_dir = Path(cfg.output_dir) / cfg.run_name / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = wandb.Table(columns=["condition", "accuracy", "balanced_accuracy",
                                 "f1_weighted", "f1_macro"])
    for cond in args.conditions:
        if cond not in CONDITIONS:
            print(f"skip unknown condition {cond}")
            continue
        kw = CONDITIONS[cond]
        method = kw["dehaze_method"]
        y_true, y_prob = run_inference(
            model, test_loader, device, haze_eval=kw["haze_eval"],
            beta=cfg.data.haze_beta, airlight=cfg.data.haze_airlight,
            dehaze_method=method, aod_weights=args.aod_weights)
        m = compute_metrics(y_true, y_prob, meta["class_names"])
        y_pred = y_prob.argmax(1)
        cm = confusion(y_true, y_pred, meta["num_classes"])
        cm_path = save_confusion_plot(cm, meta["class_names"],
                                      out_dir / f"confusion_{cond}.png",
                                      title=f"{cfg.run_name} [{cond}]")
        (out_dir / f"report_{cond}.txt").write_text(
            text_report(y_true, y_pred, meta["class_names"]))
        print(f"\n[{cond}] acc={m['accuracy']:.4f} bacc={m['balanced_accuracy']:.4f} "
              f"f1w={m['f1_weighted']:.4f}")
        run.log({f"{cond}/{k}": v for k, v in m.items()})
        run.log({f"{cond}/confusion": wandb.Image(str(cm_path))})
        table.add_data(cond, m["accuracy"], m["balanced_accuracy"],
                       m["f1_weighted"], m["f1_macro"])
    run.log({"robustness_summary": table})
    run.finish()


if __name__ == "__main__":
    main()
