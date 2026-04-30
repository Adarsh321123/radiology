"""Gated blend: use umask model only when it's confident, binary model otherwise.

For Pneumonia (and optionally Pleural Other), the umask model has better
+1/-1 discrimination but can't predict uncertain (0) cases. This script
blends umask predictions into the binary model's predictions using a
confidence gate.

Smooth gate version:
  gate = sigmoid(k * (abs(u) - threshold))
  pred = (1 - alpha * gate) * base + (alpha * gate) * u

Grid-searches threshold, alpha, and k on raw val NMSE including 0s.
Then rebuilds the mega hybrid with blended Pneumonia.

Usage:
    uv run python -u gated_blend.py \
        --config configs/hpc_densenet_v1.yaml \
        --base-cache /path/to/calib_cache \
        --umask-cache /path/to/raw_umask_calib_cache \
        --mega-csv /path/to/submission_mega_hybrid_v3_umask.csv \
        --out /path/to/submission_gated_blend.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from config import Config
from dataset import _drop_junk_cols, _extract_pid


def load_raw_val_labels(cfg: Config) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(cfg.labels_csv)
    df = _drop_junk_cols(df)
    df["pid"] = _extract_pid(df["Path"])
    df = df[df["pid"].notna()].reset_index(drop=True)
    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * cfg.val_frac))
    val_pids = set(pids[:n_val].tolist())
    df_val = df[df["pid"].isin(val_pids)].reset_index(drop=True)
    return df_val[cfg.label_names].to_numpy(dtype=np.float32)


def raw_nmse(y_true, y_pred):
    mask = ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 10:
        return float("nan")
    var = np.var(yt)
    return np.mean((yt - yp) ** 2) / var if var > 0 else float("nan")


def fit_affine(x, y_true):
    mask = ~np.isnan(y_true)
    xm, ym = x[mask], y_true[mask]
    def obj(p):
        return np.mean((ym - np.clip(p[0] * xm + p[1], -1, 1)) ** 2)
    best, best_mse = None, float("inf")
    for a0 in [-2, -1, 0, 1, 2, 4]:
        for b0 in [-1, -0.5, 0, 0.5, 1]:
            r = minimize(obj, [a0, b0], method="Nelder-Mead",
                         options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
            if r.fun < best_mse:
                best_mse, best = r.fun, r
    return float(best.x[0]), float(best.x[1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def gated_blend(base, umask, threshold, alpha, k):
    gate = sigmoid(k * (np.abs(umask) - threshold))
    return (1.0 - alpha * gate) * base + (alpha * gate) * umask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--base-cache", required=True, type=Path)
    ap.add_argument("--umask-cache", required=True, type=Path)
    ap.add_argument("--mega-csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--labels", nargs="+", default=["Pneumonia"],
                    help="Labels to try gated blend on (default: Pneumonia)")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists")

    cfg = Config.from_yaml(args.config)
    label_names = cfg.label_names
    y_val_raw = load_raw_val_labels(cfg)

    # Load caches
    base_val = np.load(str(args.base_cache / "val_ensemble_cache.npz"))
    base_test = np.load(str(args.base_cache / "test_ensemble_cache.npz"))
    umask_val = np.load(str(args.umask_cache / "val_ensemble_cache.npz"))
    umask_test = np.load(str(args.umask_cache / "test_ensemble_cache.npz"))

    # Load the mega hybrid CSV as our starting point
    import pandas as pd
    mega_df = pd.read_csv(args.mega_csv)
    test_ids = mega_df["Id"].tolist()
    test_output = mega_df[label_names].to_numpy(dtype=np.float32)

    print("=" * 90, flush=True)
    print("GATED BLEND: UMASK AS AUXILIARY SIGNAL", flush=True)
    print("=" * 90, flush=True)

    for label in args.labels:
        i = label_names.index(label)
        yt = y_val_raw[:, i]

        # Get base predictions (binary affine-calibrated on val)
        # Fit affine on binary logits against raw targets
        base_vl = base_val["logits"][:, i]
        base_tl = base_test["logits"][:, i]
        a_base, b_base = fit_affine(base_vl, yt)
        base_val_pred = np.clip(a_base * base_vl + b_base, -1, 1)
        base_test_pred = np.clip(a_base * base_tl + b_base, -1, 1)
        base_nmse = raw_nmse(yt, base_val_pred)

        # Also try affine on probs
        base_vp = base_val["probs"][:, i]
        base_tp = base_test["probs"][:, i]
        a_bp, b_bp = fit_affine(base_vp, yt)
        base_val_pred_p = np.clip(a_bp * base_vp + b_bp, -1, 1)
        base_test_pred_p = np.clip(a_bp * base_tp + b_bp, -1, 1)
        base_nmse_p = raw_nmse(yt, base_val_pred_p)

        if base_nmse_p < base_nmse:
            base_val_pred = base_val_pred_p
            base_test_pred = base_test_pred_p
            base_nmse = base_nmse_p
            base_type = "affine_prob"
        else:
            base_type = "affine_logit"

        print(f"\n{label}:", flush=True)
        print(f"  base ({base_type}): val NMSE = {base_nmse:.4f}", flush=True)

        # Get umask predictions (raw logits, clipped)
        umask_vl = np.clip(umask_val["logits"][:, i], -1, 1)
        umask_tl = np.clip(umask_test["logits"][:, i], -1, 1)

        # Also try affine-calibrated umask (on +1/-1 only for the fit)
        mask_nonzero = ~np.isnan(yt) & (yt != 0.0)
        yt_nonzero = yt.copy()
        yt_nonzero[~mask_nonzero] = np.nan
        a_um, b_um = fit_affine(umask_val["logits"][:, i], yt_nonzero)
        umask_val_calib = np.clip(a_um * umask_val["logits"][:, i] + b_um, -1, 1)
        umask_test_calib = np.clip(a_um * umask_test["logits"][:, i] + b_um, -1, 1)

        # Grid search: threshold, alpha, k
        best_nmse = base_nmse
        best_params = None
        best_val_pred = base_val_pred
        best_test_pred = base_test_pred

        for umask_pred_name, uv, ut in [
            ("identity", umask_vl, umask_tl),
            ("calib_+-only", umask_val_calib, umask_test_calib),
        ]:
            for threshold in np.arange(0.0, 0.8, 0.05):
                for alpha in np.arange(0.1, 1.05, 0.1):
                    for k in [5, 10, 20, 50]:
                        blended = gated_blend(base_val_pred, uv, threshold, alpha, k)
                        blended = np.clip(blended, -1, 1)
                        nmse = raw_nmse(yt, blended)
                        if nmse < best_nmse:
                            best_nmse = nmse
                            best_params = (umask_pred_name, threshold, alpha, k)
                            best_val_pred = blended
                            best_test_pred = np.clip(
                                gated_blend(base_test_pred, ut, threshold, alpha, k),
                                -1, 1,
                            )

        # Also try simple hard gate (Codex's first suggestion)
        for umask_pred_name, uv, ut in [
            ("identity", umask_vl, umask_tl),
            ("calib_+-only", umask_val_calib, umask_test_calib),
        ]:
            for threshold in np.arange(0.1, 0.8, 0.05):
                for alpha in np.arange(0.1, 1.05, 0.1):
                    mask_confident = np.abs(uv) >= threshold
                    blended = base_val_pred.copy()
                    blended[mask_confident] = (
                        (1 - alpha) * base_val_pred[mask_confident]
                        + alpha * uv[mask_confident]
                    )
                    blended = np.clip(blended, -1, 1)
                    nmse = raw_nmse(yt, blended)
                    if nmse < best_nmse:
                        best_nmse = nmse
                        best_params = (f"hard_{umask_pred_name}", threshold, alpha, "hard")
                        best_val_pred = blended
                        mask_test = np.abs(ut) >= threshold
                        t_blend = base_test_pred.copy()
                        t_blend[mask_test] = (
                            (1 - alpha) * base_test_pred[mask_test]
                            + alpha * ut[mask_test]
                        )
                        best_test_pred = np.clip(t_blend, -1, 1)

        if best_params is not None:
            print(f"  IMPROVED: val NMSE = {best_nmse:.4f} (was {base_nmse:.4f})", flush=True)
            print(f"  params: umask={best_params[0]} threshold={best_params[1]:.2f} "
                  f"alpha={best_params[2]:.1f} k={best_params[3]}", flush=True)
            test_output[:, i] = best_test_pred
            print(f"  test mean: {np.mean(best_test_pred):+.4f}", flush=True)
        else:
            print(f"  NO IMPROVEMENT over base {base_nmse:.4f}", flush=True)
            print(f"  keeping mega hybrid prediction for {label}", flush=True)

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for idx, row in zip(test_ids, test_output):
            writer.writerow([int(idx)] + [f"{v:.6f}" for v in row])

    print(f"\nwrote {args.out} ({len(test_ids)} rows)", flush=True)

    # Sanity check
    print(f"\n{'Label':30s}  {'Mean':>10s}  {'Min':>8s}  {'Max':>8s}", flush=True)
    for i, name in enumerate(label_names):
        col = test_output[:, i]
        print(f"  {name:28s}  {np.mean(col):+10.4f}  {np.min(col):+8.4f}  {np.max(col):+8.4f}", flush=True)


if __name__ == "__main__":
    main()
