"""Statistical validation (reviewer #3).

Two capabilities:
  1. Aggregate K-fold test metrics into mean +/- std with 95% CIs.
  2. Compare two models on the *same* test set with McNemar's test (paired,
     correct vs wrong) and a paired t-test across folds.

Consumes the test_preds.npz files written per run by engine/train.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_folds(run_dirs: list[Path]) -> dict:
    """Mean/std/95%CI of test accuracy & f1 across fold runs."""
    accs, f1s, baccs = [], [], []
    for d in run_dirs:
        npz = np.load(d / "test_preds.npz", allow_pickle=True)
        y, p = npz["y_true"], npz["y_prob"]
        pred = p.argmax(1)
        accs.append((pred == y).mean())
        from sklearn.metrics import balanced_accuracy_score, f1_score
        f1s.append(f1_score(y, pred, average="weighted", zero_division=0))
        baccs.append(balanced_accuracy_score(y, pred))

    def summ(xs):
        xs = np.asarray(xs)
        n = len(xs)
        mean, sd = xs.mean(), xs.std(ddof=1) if n > 1 else 0.0
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        return {"mean": float(mean), "std": float(sd), "ci95": float(ci),
                "values": [float(v) for v in xs]}

    return {"accuracy": summ(accs), "f1_weighted": summ(f1s),
            "balanced_accuracy": summ(baccs), "n_folds": len(run_dirs)}


def mcnemar(run_a: Path, run_b: Path) -> dict:
    """McNemar's test between two models on the identical test set."""
    a = np.load(run_a / "test_preds.npz", allow_pickle=True)
    b = np.load(run_b / "test_preds.npz", allow_pickle=True)
    # align by path so the comparison is truly paired
    pa = {p: int(pr.argmax()) for p, pr in zip(a["paths"], a["y_prob"])}
    ya = {p: int(y) for p, y in zip(a["paths"], a["y_true"])}
    pb = {p: int(pr.argmax()) for p, pr in zip(b["paths"], b["y_prob"])}
    common = sorted(set(pa) & set(pb))
    ca = np.array([pa[p] == ya[p] for p in common])
    cb = np.array([pb[p] == ya[p] for p in common])
    n01 = int(np.sum(ca & ~cb))   # A right, B wrong
    n10 = int(np.sum(~ca & cb))   # A wrong, B right
    # exact binomial McNemar (robust for small discordant counts)
    n = n01 + n10
    if n == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"n_common": len(common), "a_only_correct": n01,
            "b_only_correct": n10, "p_value": p_value,
            "significant_0.05": p_value < 0.05}


def paired_ttest(dirs_a: list[Path], dirs_b: list[Path]) -> dict:
    """Paired t-test of per-fold accuracy between two models."""
    def fold_acc(dirs):
        out = []
        for d in dirs:
            npz = np.load(d / "test_preds.npz", allow_pickle=True)
            out.append((npz["y_prob"].argmax(1) == npz["y_true"]).mean())
        return np.array(out)

    aa, bb = fold_acc(dirs_a), fold_acc(dirs_b)
    t, p = stats.ttest_rel(aa, bb)
    return {"mean_a": float(aa.mean()), "mean_b": float(bb.mean()),
            "t_stat": float(t), "p_value": float(p),
            "significant_0.05": bool(p < 0.05)}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("--dirs", nargs="+", required=True)
    m = sub.add_parser("mcnemar")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    t = sub.add_parser("ttest")
    t.add_argument("--a-dirs", nargs="+", required=True)
    t.add_argument("--b-dirs", nargs="+", required=True)
    a = ap.parse_args()

    if a.cmd == "aggregate":
        import json
        print(json.dumps(aggregate_folds([Path(d) for d in a.dirs]), indent=2))
    elif a.cmd == "mcnemar":
        import json
        print(json.dumps(mcnemar(Path(a.a), Path(a.b)), indent=2))
    elif a.cmd == "ttest":
        import json
        print(json.dumps(paired_ttest([Path(d) for d in a.a_dirs],
                                      [Path(d) for d in a.b_dirs]), indent=2))


if __name__ == "__main__":
    main()
