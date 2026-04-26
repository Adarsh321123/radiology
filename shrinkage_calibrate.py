"""Shrinkage calibration: blend affine predictions with train raw mean.

Uses cached val/test logits from raw_calibrate.py. No inference needed.

Usage:
    uv run python -u shrinkage_calibrate.py \
        --config configs/hpc_densenet_v1.yaml \
        --cache-dir /path/to/calib_cache \
        --out-shrinkage submission_shrinkage.csv \
        --out-override submission_override.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import Config
from dataset import _drop_junk_cols, _extract_pid


def load_raw_val_labels(cfg: Config) -> np.ndarray:
    df = pd.read_csv(cfg.labels_csv)
    df = _drop_junk_cols(df)
    df["pid"] = _extract_pid(df["Path"])
    df = df[df["pid"].notna()].reset_index(drop=True)

    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * cfg.val_frac))
    val_pids = set(pids[:n_val].tolist())

    in_val = df["pid"].isin(val_pids)
    df_val = df[in_val].reset_index(drop=True)
    return df_val[cfg.label_names].to_numpy(dtype=np.float32)


def raw_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 10:
        return float("nan")
    mse = np.mean((yt - yp) ** 2)
    var = np.var(yt)
    if var <= 0:
        return float("nan")
    return mse / var


def fit_affine(x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    from scipy.optimize import minimize
    mask = ~np.isnan(y_true)
    x_m = x[mask]
    y_m = y_true[mask]

    def objective(params):
        a, b = params
        pred = np.clip(a * x_m + b, -1, 1)
        return np.mean((y_m - pred) ** 2)

    best_result = None
    best_mse = float("inf")
    for a_init in [-2, -1, 0, 1, 2, 4]:
        for b_init in [-1, -0.5, 0, 0.5, 1]:
            result = minimize(objective, [a_init, b_init], method="Nelder-Mead",
                              options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result
    return float(best_result.x[0]), float(best_result.x[1])


def write_csv(path: Path, ids: list, output: np.ndarray, label_names: list):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for idx, row in zip(ids, output):
            writer.writerow([int(idx)] + [f"{v:.6f}" for v in row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-shrinkage", required=True, type=Path)
    ap.add_argument("--out-override", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    label_names = cfg.label_names

    # Load cached logits
    val_data = np.load(str(args.cache_dir / "val_ensemble_cache.npz"))
    test_data = np.load(str(args.cache_dir / "test_ensemble_cache.npz"))
    val_logits = val_data["logits"]
    val_probs = val_data["probs"]
    test_logits = test_data["logits"]
    test_probs = test_data["probs"]
    test_ids = test_data["ids"].tolist()

    # Load raw val labels
    y_val_raw = load_raw_val_labels(cfg)

    print("=" * 90, flush=True)
    print("SHRINKAGE CALIBRATION", flush=True)
    print("=" * 90, flush=True)

    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    shrinkage_output = np.zeros_like(test_probs)
    override_output = np.zeros_like(test_probs)

    print(f"\n{'Label':30s}  {'Best Method':25s}  {'Val NMSE':>10s}  {'Test Mean':>10s}  {'Lambda':>8s}", flush=True)
    print("-" * 90, flush=True)

    for i, name in enumerate(label_names):
        yt = y_val_raw[:, i]
        mask = ~np.isnan(yt)
        train_raw_mean = float(np.mean(yt[mask]))

        # Fit affine on both prob and logit
        a_p, b_p = fit_affine(val_probs[:, i], yt)
        a_l, b_l = fit_affine(val_logits[:, i], yt)

        # Affine predictions
        val_affine_p = np.clip(a_p * val_probs[:, i] + b_p, -1, 1)
        val_affine_l = np.clip(a_l * val_logits[:, i] + b_l, -1, 1)
        test_affine_p = np.clip(a_p * test_probs[:, i] + b_p, -1, 1)
        test_affine_l = np.clip(a_l * test_logits[:, i] + b_l, -1, 1)

        # Pick better affine base
        if raw_nmse(yt, val_affine_p) <= raw_nmse(yt, val_affine_l):
            val_affine = val_affine_p
            test_affine = test_affine_p
            affine_type = "prob"
        else:
            val_affine = val_affine_l
            test_affine = test_affine_l
            affine_type = "logit"

        # Try shrinkage: pred = lambda * affine + (1-lambda) * train_mean
        best_lam = 1.0
        best_nmse = raw_nmse(yt, val_affine)
        best_method = f"affine_{affine_type}"

        for lam in lambdas:
            val_blend = lam * val_affine + (1 - lam) * train_raw_mean
            val_blend = np.clip(val_blend, -1, 1)
            nmse = raw_nmse(yt, val_blend)

            if nmse < best_nmse:
                best_nmse = nmse
                best_lam = lam
                best_method = f"shrink_{affine_type}_lam{lam:.2f}"

        # Also try pure constant mean
        const_nmse = raw_nmse(yt, np.full_like(val_affine, train_raw_mean))
        if const_nmse < best_nmse:
            best_nmse = const_nmse
            best_lam = 0.0
            best_method = "constant_mean"

        # Apply best to test
        if best_method == "constant_mean":
            shrinkage_output[:, i] = train_raw_mean
        else:
            test_blend = best_lam * test_affine + (1 - best_lam) * train_raw_mean
            shrinkage_output[:, i] = np.clip(test_blend, -1, 1)

        test_mean = float(np.mean(shrinkage_output[:, i]))
        print(f"  {name:28s}  {best_method:25s}  {best_nmse:10.4f}  {test_mean:+10.4f}  {best_lam:8.2f}", flush=True)

        # Override candidate: use calibrated for strong labels,
        # constant mean for Fracture and Pleural Other
        if name in ("Fracture", "Pleural Other"):
            override_output[:, i] = train_raw_mean
        else:
            override_output[:, i] = shrinkage_output[:, i]

    # Write both CSVs
    write_csv(args.out_shrinkage, test_ids, shrinkage_output, label_names)
    print(f"\nwrote {args.out_shrinkage}", flush=True)

    write_csv(args.out_override, test_ids, override_output, label_names)
    print(f"wrote {args.out_override}", flush=True)

    # Sanity check both
    for name, output in [("SHRINKAGE", shrinkage_output), ("OVERRIDE", override_output)]:
        print(f"\n=== {name} SANITY CHECK ===", flush=True)
        print(f"{'Label':30s}  {'Test Mean':>10s}  {'Min':>8s}  {'Max':>8s}", flush=True)
        for i, label in enumerate(label_names):
            col = output[:, i]
            print(f"  {label:28s}  {np.mean(col):+10.4f}  {np.min(col):+8.4f}  {np.max(col):+8.4f}", flush=True)


if __name__ == "__main__":
    main()
