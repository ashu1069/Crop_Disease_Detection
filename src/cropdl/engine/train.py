"""Training engine: one fold end-to-end with W&B logging, AMP, cosine schedule,
class-weighted loss, early stopping, and best-checkpoint selection on val
balanced-accuracy (robust to imbalance).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from ..data.dataset import build_loaders
from ..models.factory import build_model
from ..seed import seed_worker, set_seed
from .evaluate import run_inference
from .metrics import compute_metrics, confusion, save_confusion_plot, text_report


def _build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9,
                               weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _lr_at(cfg, epoch, steps_per_epoch, step):
    """Warmup + cosine; returns multiplier on base lr."""
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch
    cur = epoch * steps_per_epoch + step
    if cfg.scheduler == "none":
        return 1.0
    if cur < warm:
        return cur / max(1, warm)
    if cfg.scheduler == "cosine":
        prog = (cur - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return 1.0


def train_one_fold(cfg, run=None) -> dict:
    set_seed(cfg.seed + cfg.data.fold)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    gen = torch.Generator()
    gen.manual_seed(cfg.seed + cfg.data.fold)
    train_loader, val_loader, test_loader, meta = build_loaders(
        cfg.data, seed_worker=seed_worker, generator=gen)

    model = build_model(cfg.model, meta["num_classes"]).to(device)

    weights = meta["class_weights"].to(device) if cfg.optim.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=cfg.optim.label_smoothing)
    optimizer = _build_optimizer(cfg.optim, model)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.optim.mixed_precision)
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)
        if cfg.optim.scheduler == "plateau" else None)

    steps = len(train_loader)
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    best_metric, best_epoch, since_improve = -1.0, -1, 0
    for epoch in range(cfg.optim.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if plateau is None:
                mult = _lr_at(cfg.optim, epoch, steps, step)
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.optim.lr * mult
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=cfg.optim.mixed_precision):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if cfg.optim.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        # ---- validation ----
        y_true, y_prob = run_inference(model, val_loader, device)
        vm = compute_metrics(y_true, y_prob, meta["class_names"])
        train_loss = running / max(1, steps)
        cur_lr = optimizer.param_groups[0]["lr"]
        if plateau is not None:
            plateau.step(vm["balanced_accuracy"])

        log = {"epoch": epoch, "train/loss": train_loss, "lr": cur_lr,
               "time/epoch_s": time.time() - t0}
        log.update({f"val/{k}": v for k, v in vm.items()})
        if run is not None:
            run.log(log)
        print(f"[{cfg.run_name}] ep {epoch:03d} loss {train_loss:.4f} "
              f"val_bacc {vm['balanced_accuracy']:.4f} val_f1w {vm['f1_weighted']:.4f}")

        if vm["balanced_accuracy"] > best_metric:
            best_metric = vm["balanced_accuracy"]
            best_epoch = epoch
            since_improve = 0
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "class_names": meta["class_names"], "epoch": epoch}, ckpt_path)
        else:
            since_improve += 1
            if since_improve >= cfg.optim.early_stop_patience:
                print(f"Early stop at epoch {epoch} (best {best_epoch}).")
                break

    # ---- final test eval with best checkpoint ----
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    y_true, y_prob, paths = run_inference(model, test_loader, device, collect_paths=True)
    tm = compute_metrics(y_true, y_prob, meta["class_names"])
    y_pred = y_prob.argmax(1)
    cm = confusion(y_true, y_pred, meta["num_classes"])
    cm_path = save_confusion_plot(cm, meta["class_names"],
                                  out_dir / "confusion_test.png",
                                  title=f"{cfg.run_name} (test)")
    report = text_report(y_true, y_pred, meta["class_names"])
    (out_dir / "report_test.txt").write_text(report)

    # save raw test predictions for later error analysis / significance tests
    np.savez(out_dir / "test_preds.npz", y_true=y_true, y_prob=y_prob,
             paths=np.array(paths))

    if run is not None:
        run.log({f"test/{k}": v for k, v in tm.items()})
        run.summary.update({f"test/{k}": v for k, v in tm.items()})
        run.summary["best_epoch"] = best_epoch
        run.log({"test/confusion_matrix": wandb.Image(str(cm_path))})

    print("\n=== TEST REPORT ===\n" + report)
    return {"val_best_balanced_acc": best_metric, "best_epoch": best_epoch,
            "test": tm, "meta": {k: meta[k] for k in ("n_train", "n_val", "n_test",
                                                       "class_names")},
            "out_dir": str(out_dir)}
