"""Find optimal per-label temperature scaling and No Finding alpha on val set.

Given a trained checkpoint, runs inference on the validation split and
optimizes post-processing parameters to minimize NMSE.

Usage::

    uv run python calibrate.py --ckpt /path/to/ckpt_best.pt

Outputs the optimal --temp-scale and --derive-no-finding args to pass to submit.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar, minimize
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    CheXpertDataset,
    build_val_transform,
    load_and_split,
)
from metrics import per_label_nmse
from submit import load_model


def get_val_logits_and_labels(
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on val set, return (logits, y_val) as numpy arrays."""
    _, df_val, _, y_val = load_and_split(cfg)
    val_ds = CheXpertDataset(df_val, y_val, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_logits = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def nmse_for_label(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NMSE for a single label column, masking NaN."""
    mask = ~np.isnan(y_true)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 10:
        return float("nan")
    mse = np.mean((yt - yp) ** 2)
    var = np.var(yt)
    if var <= 0:
        return float("nan")
    return float(mse / var)


def optimize_temperatures(
    logits: np.ndarray,
    y_val: np.ndarray,
    label_names: list[str],
) -> np.ndarray:
    """Find optimal per-label temperature to minimize NMSE on val set."""
    num_labels = logits.shape[1]
    temperatures = np.ones(num_labels)

    for i, name in enumerate(label_names):
        yt = y_val[:, i]

        def neg_nmse(t):
            probs = 1.0 / (1.0 + np.exp(-logits[:, i] / t))
            return nmse_for_label(yt, probs)

        result = minimize_scalar(neg_nmse, bounds=(0.1, 10.0), method="bounded")
        temperatures[i] = result.x
        base_nmse = nmse_for_label(yt, 1.0 / (1.0 + np.exp(-logits[:, i])))
        opt_nmse = result.fun
        delta = base_nmse - opt_nmse
        print(f"  {name:30s}  T={result.x:.4f}  NMSE: {base_nmse:.4f} -> {opt_nmse:.4f}  (delta={delta:+.4f})")

    return temperatures


def optimize_no_finding_alpha(
    logits: np.ndarray,
    y_val: np.ndarray,
    label_names: list[str],
    temperatures: np.ndarray,
) -> float:
    """Find optimal alpha for blending model No Finding with derived signal."""
    nf_idx = label_names.index("No Finding")
    other_indices = [i for i in range(len(label_names)) if i != nf_idx]
    yt_nf = y_val[:, nf_idx]

    # Apply temperature scaling first
    probs_all = 1.0 / (1.0 + np.exp(-logits / temperatures[np.newaxis, :]))
    model_nf = probs_all[:, nf_idx]
    other_probs = probs_all[:, other_indices]
    derived_nf = np.prod(1.0 - other_probs, axis=1)

    def nf_nmse(alpha):
        blended = alpha * model_nf + (1.0 - alpha) * derived_nf
        return nmse_for_label(yt_nf, blended)

    result = minimize_scalar(nf_nmse, bounds=(0.0, 1.0), method="bounded")
    base_nmse = nmse_for_label(yt_nf, model_nf)
    opt_nmse = result.fun
    derived_only_nmse = nmse_for_label(yt_nf, derived_nf)

    print(f"\n  No Finding alpha optimization:")
    print(f"    model only (alpha=1.0):   NMSE={base_nmse:.4f}")
    print(f"    derived only (alpha=0.0): NMSE={derived_only_nmse:.4f}")
    print(f"    optimal (alpha={result.x:.3f}):    NMSE={opt_nmse:.4f}")

    return float(result.x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model, cfg, meta = load_model(args.ckpt, device)
    print(f"  best_{meta['primary_metric']}={meta['best_metric']}", flush=True)

    print(f"\nRunning inference on val set...", flush=True)
    logits, y_val = get_val_logits_and_labels(
        model, cfg, device, args.batch_size, args.num_workers,
    )
    print(f"  val samples: {logits.shape[0]:,}", flush=True)

    # Baseline NMSE
    probs_base = 1.0 / (1.0 + np.exp(-logits))
    base_metrics = per_label_nmse(y_val, probs_base, cfg.label_names)
    print(f"\nBaseline val NMSE: {base_metrics['mean']:.4f}")
    for name in cfg.label_names:
        print(f"  {name:30s}  {base_metrics[name]:.4f}")

    # Optimize temperatures
    print(f"\nOptimizing per-label temperatures...")
    temperatures = optimize_temperatures(logits, y_val, cfg.label_names)

    # Optimized NMSE
    probs_opt = 1.0 / (1.0 + np.exp(-logits / temperatures[np.newaxis, :]))
    opt_metrics = per_label_nmse(y_val, probs_opt, cfg.label_names)
    print(f"\nPost-temperature val NMSE: {opt_metrics['mean']:.4f} (was {base_metrics['mean']:.4f})")

    # Optimize No Finding alpha
    print(f"\nOptimizing No Finding derivation...")
    alpha = optimize_no_finding_alpha(logits, y_val, cfg.label_names, temperatures)

    # Final NMSE with both
    probs_final = probs_opt.copy()
    nf_idx = cfg.label_names.index("No Finding")
    other_indices = [i for i in range(len(cfg.label_names)) if i != nf_idx]
    derived_nf = np.prod(1.0 - probs_final[:, other_indices], axis=1)
    probs_final[:, nf_idx] = alpha * probs_final[:, nf_idx] + (1.0 - alpha) * derived_nf
    final_metrics = per_label_nmse(y_val, probs_final, cfg.label_names)
    print(f"\nFinal val NMSE (temp + derived NF): {final_metrics['mean']:.4f}")
    for name in cfg.label_names:
        delta = base_metrics[name] - final_metrics[name]
        print(f"  {name:30s}  {final_metrics[name]:.4f}  (delta={delta:+.4f})")

    # Print the submit.py command
    temp_str = " ".join(f"{t:.4f}" for t in temperatures)
    print(f"\n{'='*70}")
    print(f"To use these settings with submit.py:")
    print(f"  uv run python submit.py \\")
    print(f"    --ckpt {args.ckpt} \\")
    print(f"    --out submission.csv \\")
    print(f"    --temp-scale {temp_str} \\")
    print(f"    --derive-no-finding {alpha:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
