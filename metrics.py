"""Per-label AUROC and variance-normalized MSE, with NaN-masking for uncertain val examples."""
from __future__ import annotations

import math
from typing import List, Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def per_label_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    *,
    min_positives: int = 10,
) -> Dict[str, float]:
    """Compute per-label AUROC, masking NaN entries (uncertain labels).

    y_true : (N, num_labels) with values in {0, 1, nan}. nan = uncertain, skipped.
    y_pred : (N, num_labels) real-valued scores (logits or probabilities).

    Returns a dict mapping each label to its AUROC, plus a "mean" entry.
    Labels whose val positive count is below ``min_positives`` get a
    ``nan`` AUROC (treated as unreliable) and are skipped in the mean.
    """
    assert y_true.shape == y_pred.shape, f"{y_true.shape} vs {y_pred.shape}"
    out: Dict[str, float] = {}
    valid: List[float] = []
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = ~np.isnan(yt)
        yt_m = yt[mask]
        yp_m = yp[mask]
        n_pos = int((yt_m == 1).sum())
        n_neg = int((yt_m == 0).sum())
        if n_pos < min_positives or n_neg < min_positives:
            out[name] = math.nan
            continue
        try:
            auc = float(roc_auc_score(yt_m, yp_m))
        except ValueError:
            auc = math.nan
        out[name] = auc
        if not math.isnan(auc):
            valid.append(auc)
    out["mean"] = float(np.mean(valid)) if valid else math.nan
    return out


def per_label_nmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    *,
    min_samples: int = 10,
) -> Dict[str, float]:
    """Compute per-label variance-normalized MSE (NMSE = MSE / Var(y)).

    y_true : (N, num_labels) with values in {0, 1, nan}. nan = uncertain, skipped.
    y_pred : (N, num_labels) real-valued scores in [0, 1] (sigmoid probabilities).

    NMSE semantics:
      0.0  = perfect predictions
      1.0  = no better than always predicting the per-label mean
      >1.0 = worse than the constant-mean baseline
    Lower is better.

    Labels with fewer than ``min_samples`` clean entries, or zero
    variance, get ``nan`` and are skipped in the mean.
    """
    assert y_true.shape == y_pred.shape, f"{y_true.shape} vs {y_pred.shape}"
    out: Dict[str, float] = {}
    valid: List[float] = []
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mask = ~np.isnan(yt)
        yt_m = yt[mask]
        yp_m = yp[mask]
        if len(yt_m) < min_samples:
            out[name] = math.nan
            continue
        mse = float(np.mean((yt_m - yp_m) ** 2))
        var = float(np.var(yt_m))
        if var <= 0:
            out[name] = math.nan
            continue
        nmse = mse / var
        out[name] = nmse
        valid.append(nmse)
    out["mean"] = float(np.mean(valid)) if valid else math.nan
    return out
