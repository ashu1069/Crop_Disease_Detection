"""Evaluation metrics + plots (confusion matrix, classification report)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)


def compute_metrics(y_true, y_prob, class_names) -> dict:
    """Full metric suite. y_prob: (N, C) softmax probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    num_classes = len(class_names)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    # AUC: handle binary vs multiclass, guard against single-class batches
    try:
        if num_classes == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["auroc"] = float("nan")

    # per-class f1
    per_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)),
                      zero_division=0)
    for name, val in zip(class_names, per_f1):
        metrics[f"f1/{name}"] = float(val)
    return metrics


def text_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, target_names=class_names,
                                 digits=4, zero_division=0)


def confusion(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def save_confusion_plot(cm, class_names, out_path: Path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_norm = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(class_names),
                                    1.2 + 1.0 * len(class_names)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
