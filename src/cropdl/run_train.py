"""Top-level training entry point.

    python -m cropdl.run_train --config configs/densenet_variants.yaml \
        --set model.name=densenet201 data.crop=potato data.fold=0

Runs a single fold and logs everything to W&B. For a full K-fold sweep use
--all-folds, which loops folds 0..num_folds-1 in one process and prints the
aggregated mean +/- std at the end.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import wandb

from .config import load_config, parse_cli_overrides
from .engine.train import train_one_fold


def _init_wandb(cfg, fold):
    tags = list(cfg.wandb.tags) + [cfg.model.name, cfg.data.crop, cfg.experiment]
    return wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity, mode=cfg.wandb.mode,
        name=cfg.run_name, group=cfg.wandb.group or f"{cfg.experiment}-{cfg.data.crop}-{cfg.model.name}",
        job_type="train", tags=tags, config=cfg.to_dict(), reinit=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value ...")
    ap.add_argument("--all-folds", action="store_true",
                    help="train every CV fold sequentially and aggregate")
    args = ap.parse_args()
    overrides = parse_cli_overrides(args.set)

    base = load_config(args.config, overrides)
    folds = range(base.data.num_folds) if args.all_folds else [base.data.fold]

    fold_dirs, fold_results = [], []
    for fold in folds:
        cfg = load_config(args.config, {**overrides, "data.fold": fold})
        run = _init_wandb(cfg, fold)
        res = train_one_fold(cfg, run=run)
        run.finish()
        fold_results.append(res)
        fold_dirs.append(res["out_dir"])

    if args.all_folds and len(fold_dirs) > 1:
        from .stats.analyze import aggregate_folds
        agg = aggregate_folds([Path(d) for d in fold_dirs])
        print("\n=== K-FOLD AGGREGATE ===")
        print(json.dumps(agg, indent=2))
        # log a small summary run
        cfg = base
        summ = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                          mode=cfg.wandb.mode, name=f"{cfg.experiment}-{cfg.data.crop}-{cfg.model.name}-AGG",
                          job_type="aggregate", reinit=True)
        summ.summary.update({
            "cv/acc_mean": agg["accuracy"]["mean"], "cv/acc_std": agg["accuracy"]["std"],
            "cv/acc_ci95": agg["accuracy"]["ci95"],
            "cv/f1w_mean": agg["f1_weighted"]["mean"], "cv/f1w_std": agg["f1_weighted"]["std"],
            "cv/bacc_mean": agg["balanced_accuracy"]["mean"],
            "cv/bacc_std": agg["balanced_accuracy"]["std"],
        })
        summ.finish()
        out = Path(base.output_dir) / f"{base.experiment}-{base.data.crop}-{base.model.name}-agg.json"
        out.write_text(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
