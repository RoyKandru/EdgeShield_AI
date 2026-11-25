"""Binary-classification metrics with bootstrap confidence intervals.

Positive class = malicious (scam / phishing / attack).
FPR = FP / (FP + TN): the share of benign items wrongly flagged.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "tp": int((y_true & y_pred).sum()), "fp": int((~y_true & y_pred).sum()),
        "tn": int((~y_true & ~y_pred).sum()), "fn": int((y_true & ~y_pred).sum()),
    }


def rates(c: dict) -> dict:
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / max(1, tp + fp + tn + fn),
    }


def evaluate(y_true, y_pred, scores=None, n_boot: int = 300, seed: int = 0) -> dict:
    """Point estimates + 95% bootstrap CIs (percentile) for P/R/F1/FPR."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    c = confusion(y_true, y_pred)
    out = {"n": int(len(y_true)), "n_pos": int(y_true.sum()), **c, **rates(c)}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in ("precision", "recall", "f1", "fpr")}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = rates(confusion(y_true[idx], y_pred[idx]))
        for k in boots:
            boots[k].append(r[k])
    out["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for k, v in boots.items()}
    if scores is not None and 0 < y_true.sum() < n:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def pick_threshold(y_true, scores, max_fpr: float = 0.01) -> float:
    """Validation-set operating point: best F1 among thresholds with FPR <= max_fpr.

    Choosing this on validation data only (never on test) keeps the reported
    test metrics honest.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    pos, neg = max(1, ys.sum()), max(1, (~ys).sum())
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / pos
    f1 = 2 * prec * rec / np.maximum(1e-12, prec + rec)
    ok = (fp / neg) <= max_fpr
    # only cut between distinct score values
    distinct = np.r_[ss[1:] != ss[:-1], True]
    ok &= distinct
    if not ok.any():
        return float(ss[0]) + 1e-9  # nothing satisfies the FPR cap: flag nothing
    best = np.argmax(np.where(ok, f1, -1))
    return float(ss[best])


def fmt_row(name: str, m: dict) -> str:
    ci = m["ci95"]
    return (f"| {name} | {m['precision']:.4f} [{ci['precision'][0]:.3f}-{ci['precision'][1]:.3f}] "
            f"| {m['recall']:.4f} [{ci['recall'][0]:.3f}-{ci['recall'][1]:.3f}] "
            f"| {m['f1']:.4f} | {m['fpr']:.4f} [{ci['fpr'][0]:.4f}-{ci['fpr'][1]:.4f}] "
            f"| {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
