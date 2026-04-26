"""Build a per-label hybrid submission from two model families.

For each label, picks the best prediction source by val NMSE:
  - Calibrated binary ensemble (from calib_cache)
  - Calibrated raw-MSE ensemble (from raw_mse_calib_cache)
  - Constant train raw mean

Usage:
    uv run python -u hybrid_submit.py \
        --config configs/hpc_densenet_v1.yaml \
        --binary-cache /path/to/calib_cache \
        --raw-cache /path/to/raw_mse_calib_cache \
        --out submission_hybrid_best.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def fit_affine(x: np.ndarray, y_true: np.ndarray):
    mask = ~np.isnan(y_true)
    x_m = x[mask]
    y_m = y_true[mask]

    def objective(params):
        a, b = params
        return np.mean((y_m - np.clip(a * x_m + b, -1, 1)) ** 2)

    best = None
    best_mse = float("inf")
    for a0 in [-2, -1, 0, 1, 2, 4]:
        for b0 in [-1, -0.5, 0, 0.5, 1]:
            r = minimize(objective, [a0, b0], method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
            if r.fun < best_mse:
                best_mse = r.fun
                best = r
    return float(best.x[0]), float(best.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--binary-cache", required=True, type=Path)
    ap.add_argument("--raw-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists")

    cfg = Config.from_yaml(args.config)
    label_names = cfg.label_names

    # Load caches
    bin_val = np.load(str(args.binary_cache / "val_ensemble_cache.npz"))
    bin_test = np.load(str(args.binary_cache / "test_ensemble_cache.npz"))
    raw_val = np.load(str(args.raw_cache / "val_ensemble_cache.npz"))
    raw_test = np.load(str(args.raw_cache / "test_ensemble_cache.npz"))

    test_ids = bin_test["ids"].tolist()
    y_val_raw = load_raw_val_labels(cfg)

    print("=" * 100, flush=True)
    print("PER-LABEL HYBRID SELECTION", flush=True)
    print("=" * 100, flush=True)
    print(f"\n{'Label':30s}  {'Source':30s}  {'Val NMSE':>10s}  {'Test Mean':>10s}", flush=True)
    print("-" * 85, flush=True)

    test_output = np.zeros((len(test_ids), len(label_names)))

    for i, name in enumerate(label_names):
        yt = y_val_raw[:, i]
        mask = ~np.isnan(yt)
        train_raw_mean = float(np.mean(yt[mask]))

        candidates = {}

        # 1. Constant mean
        candidates["constant_mean"] = (
            np.full(len(yt), train_raw_mean),
            np.full(len(test_ids), train_raw_mean),
            raw_nmse(yt, np.full(len(yt), train_raw_mean)),
        )

        # 2. Binary ensemble: affine on prob
        bp = bin_val["probs"][:, i]
        bt = bin_test["probs"][:, i]
        a, b = fit_affine(bp, yt)
        vp = np.clip(a * bp + b, -1, 1)
        tp = np.clip(a * bt + b, -1, 1)
        candidates["binary_affine_prob"] = (vp, tp, raw_nmse(yt, vp))

        # 3. Binary ensemble: affine on logit
        bl = bin_val["logits"][:, i]
        btl = bin_test["logits"][:, i]
        a, b = fit_affine(bl, yt)
        vl = np.clip(a * bl + b, -1, 1)
        tl = np.clip(a * btl + b, -1, 1)
        candidates["binary_affine_logit"] = (vl, tl, raw_nmse(yt, vl))

        # 4. Raw-MSE ensemble: affine on output
        # For raw-MSE, "logits" are the raw model outputs (not sigmoided)
        ro = raw_val["logits"][:, i]
        rto = raw_test["logits"][:, i]
        a, b = fit_affine(ro, yt)
        vo = np.clip(a * ro + b, -1, 1)
        to_ = np.clip(a * rto + b, -1, 1)
        candidates["raw_mse_affine"] = (vo, to_, raw_nmse(yt, vo))

        # 5. Raw-MSE ensemble: identity (clipped)
        vo_id = np.clip(ro, -1, 1)
        to_id = np.clip(rto, -1, 1)
        candidates["raw_mse_identity"] = (vo_id, to_id, raw_nmse(yt, vo_id))

        # Select best
        best_name = min(candidates, key=lambda k: candidates[k][2])
        best = candidates[best_name]
        test_output[:, i] = best[1]

        for cname in sorted(candidates, key=lambda k: candidates[k][2]):
            c = candidates[cname]
            marker = " <-- SELECTED" if cname == best_name else ""
            tmean = float(np.mean(c[1]))
            print(f"  {name:28s}  {cname:30s}  {c[2]:10.4f}  {tmean:+10.4f}{marker}", flush=True)
        print(flush=True)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for idx, row in zip(test_ids, test_output):
            writer.writerow([int(idx)] + [f"{v:.6f}" for v in row])

    print(f"wrote {args.out} ({len(test_ids)} rows)", flush=True)

    # Sanity
    print(f"\n{'Label':30s}  {'Test Mean':>10s}  {'Min':>8s}  {'Max':>8s}", flush=True)
    for i, name in enumerate(label_names):
        col = test_output[:, i]
        print(f"  {name:28s}  {np.mean(col):+10.4f}  {np.min(col):+8.4f}  {np.max(col):+8.4f}", flush=True)


if __name__ == "__main__":
    main()
