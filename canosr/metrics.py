"""Metrics. Threshold-free where possible, everything reported per fold."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _safe_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target: float = 0.95) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    index = int(np.searchsorted(tpr, target))
    index = min(index, len(fpr) - 1)
    return float(fpr[index])


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Positive class = 'not benign'."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "acc": (tp + tn) / max(len(y_true), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def closed_set_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class = {}
    support = {}
    precisions, recalls, f1s = [], [], []
    for index in range(n_classes):
        mask = y_true == index
        pred_mask = y_pred == index
        support[f"n_{index}"] = int(mask.sum())
        if not mask.any():
            per_class[f"recall_{index}"] = float("nan")
            continue
        tp = int((mask & pred_mask).sum())
        recall = float(tp / int(mask.sum()))
        precision = float(tp / int(pred_mask.sum())) if pred_mask.any() else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[f"recall_{index}"] = recall
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    return {
        "acc": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "macro_precision": float(np.mean(precisions)) if precisions else float("nan"),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class_recall": per_class,
        "per_class_support": support,
    }


def openness(n_known: int, n_unknown: int) -> float:
    if n_known + n_unknown == 0:
        return float("nan")
    return 1.0 - math.sqrt(2.0 * n_known / (n_known + n_unknown))


def oscr_score(
    y_true: np.ndarray,
    known_pred: np.ndarray,
    scores: np.ndarray,
    n_known: int,
) -> float:
    """Open-Set Classification Rate (Dhamija et al. 2018): AUC of CCR vs FPR."""
    y_true = np.asarray(y_true)
    known_pred = np.asarray(known_pred)
    scores = np.asarray(scores, dtype=float)
    known_mask = y_true < n_known
    unknown_mask = ~known_mask
    n_k = int(known_mask.sum())
    n_u = int(unknown_mask.sum())
    if n_k == 0 or n_u == 0:
        return float("nan")

    order = np.argsort(scores, kind="stable")
    is_unknown = unknown_mask[order]
    is_correct_known = known_mask[order] & (known_pred[order] == y_true[order])

    fpr = np.r_[0.0, np.cumsum(is_unknown) / n_u]
    ccr = np.r_[0.0, np.cumsum(is_correct_known) / n_k]
    return float(np.trapezoid(ccr, fpr) if hasattr(np, "trapezoid") else np.trapz(ccr, fpr))


def open_set_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_known: int,
    n_unknown_classes: int,
    scores: np.ndarray | None = None,
    threshold: float | None = None,
    known_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """y_true uses ``n_known`` for the unknown class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    known_mask = y_true < n_known
    unknown_mask = ~known_mask

    correct_known = int((y_pred[known_mask] == y_true[known_mask]).sum())
    correct_unknown = int((y_pred[unknown_mask] == n_known).sum())
    total = len(y_true)

    report: dict[str, Any] = {
        "osr_acc": (correct_known + correct_unknown) / max(total, 1),
        "oscr": float("nan"),
        "openness": openness(n_known, n_unknown_classes),
        "known_acc": correct_known / max(int(known_mask.sum()), 1),
        "unknown_recall": correct_unknown / max(int(unknown_mask.sum()), 1),
        "n_known": int(known_mask.sum()),
        "n_unknown": int(unknown_mask.sum()),
    }

    if scores is not None:
        binary_true = unknown_mask.astype(int)
        report["oscr"] = oscr_score(
            y_true,
            y_pred if known_pred is None else known_pred,
            scores,
            n_known,
        )
        report["auroc"] = _safe_auroc(binary_true, scores)
        report["aupr"] = _safe_aupr(binary_true, scores)
        report["fpr_at_95tpr"] = fpr_at_tpr(binary_true, scores, 0.95)
        if threshold is not None:
            predicted_unknown = np.asarray(scores) > threshold
            report["unknown_precision"] = (
                float((binary_true[predicted_unknown] == 1).mean()) if predicted_unknown.any() else 0.0
            )
            report["unknown_recall_at_threshold"] = (
                float(predicted_unknown[binary_true == 1].mean()) if (binary_true == 1).any() else float("nan")
            )
    return report


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """mean/std/ci95 across seeds for every numeric key."""
    keys = {k for row in rows for k, v in row.items() if isinstance(v, (int, float))}
    out: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = np.array([row[key] for row in rows if key in row], dtype=float)
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out[key] = {
            "mean": float(values.mean()),
            "std": std,
            "ci95": float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0,
            "n": int(values.size),
        }
    return out
