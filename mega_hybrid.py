"""Mega hybrid: pick best prediction source per label from ALL model families.

Usage:
    uv run python -u mega_hybrid.py \
        --config configs/hpc_densenet_v1.yaml \
        --caches binary=/path/to/cache raw224=/path raw320=/path rawwt=/path ... \
        --out submission_mega_hybrid.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--caches", required=True, nargs="+",
                    help="name=path pairs, e.g. binary=/path/to/cache raw224=/path")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists")

    cfg = Config.from_yaml(args.config)
    label_names = cfg.label_names
    y_val_raw = load_raw_val_labels(cfg)

    # Load all caches
    families = {}
    for pair in args.caches:
        name, path = pair.split("=", 1)
        cache_dir = Path(path)
        val_f = cache_dir / "val_ensemble_cache.npz"
        test_f = cache_dir / "test_ensemble_cache.npz"
        if not val_f.exists() or not test_f.exists():
            print(f"  SKIP {name}: cache not ready at {cache_dir}", flush=True)
            continue
        vd = np.load(str(val_f))
        td = np.load(str(test_f))
        families[name] = {
            "val_logits": vd["logits"], "val_probs": vd.get("probs", None),
            "test_logits": td["logits"], "test_probs": td.get("probs", None),
            "test_ids": td["ids"].tolist(),
        }
        # Compute probs if not cached (raw-MSE models don't have sigmoid probs)
        if families[name]["val_probs"] is None:
            families[name]["val_probs"] = 1.0 / (1.0 + np.exp(-vd["logits"]))
            families[name]["test_probs"] = 1.0 / (1.0 + np.exp(-td["logits"]))
        print(f"  loaded {name}: {cache_dir}", flush=True)

    if not families:
        raise RuntimeError("no caches loaded")

    test_ids = list(families.values())[0]["test_ids"]

    print("\n" + "=" * 110, flush=True)
    print("MEGA HYBRID: PER-LABEL SELECTION FROM ALL FAMILIES", flush=True)
    print("=" * 110, flush=True)

    test_output = np.zeros((len(test_ids), len(label_names)))

    for i, name in enumerate(label_names):
        yt = y_val_raw[:, i]
        mask = ~np.isnan(yt)
        train_raw_mean = float(np.mean(yt[mask]))

        candidates = {}
        candidates["constant_mean"] = (
            np.full(len(yt), train_raw_mean),
            np.full(len(test_ids), train_raw_mean),
            raw_nmse(yt, np.full(len(yt), train_raw_mean)),
        )

        for fname, fdata in families.items():
            is_binary = "binary" in fname
            # Affine on logits/outputs
            vl = fdata["val_logits"][:, i]
            tl = fdata["test_logits"][:, i]
            a, b = fit_affine(vl, yt)
            vp = np.clip(a * vl + b, -1, 1)
            tp = np.clip(a * tl + b, -1, 1)
            candidates[f"{fname}_affine_logit"] = (vp, tp, raw_nmse(yt, vp))

            # Affine on probs (only meaningful for binary models)
            if is_binary and fdata["val_probs"] is not None:
                vpr = fdata["val_probs"][:, i]
                tpr = fdata["test_probs"][:, i]
                a, b = fit_affine(vpr, yt)
                vpp = np.clip(a * vpr + b, -1, 1)
                tpp = np.clip(a * tpr + b, -1, 1)
                candidates[f"{fname}_affine_prob"] = (vpp, tpp, raw_nmse(yt, vpp))

            # Identity (clipped) for raw models
            if not is_binary:
                vi = np.clip(vl, -1, 1)
                ti = np.clip(tl, -1, 1)
                candidates[f"{fname}_identity"] = (vi, ti, raw_nmse(yt, vi))

        best_name = min(candidates, key=lambda k: candidates[k][2])
        best = candidates[best_name]
        test_output[:, i] = best[1]

        # Print top 5 candidates
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1][2])
        for j, (cname, (_, tpred, nmse_val)) in enumerate(sorted_cands[:5]):
            marker = " <-- SELECTED" if cname == best_name else ""
            tmean = float(np.mean(tpred))
            print(f"  {name:28s}  {cname:35s}  {nmse_val:8.4f}  {tmean:+8.4f}{marker}", flush=True)
        print(flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for idx, row in zip(test_ids, test_output):
            writer.writerow([int(idx)] + [f"{v:.6f}" for v in row])

    print(f"wrote {args.out} ({len(test_ids)} rows)", flush=True)
    print(f"\n{'Label':30s}  {'Mean':>10s}  {'Min':>8s}  {'Max':>8s}", flush=True)
    for i, name in enumerate(label_names):
        col = test_output[:, i]
        print(f"  {name:28s}  {np.mean(col):+10.4f}  {np.min(col):+8.4f}  {np.max(col):+8.4f}", flush=True)


if __name__ == "__main__":
    main()
