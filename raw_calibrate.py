"""Per-label affine calibration against raw -1/0/1 targets.

Runs the 5-checkpoint DenseNet ensemble on val and test, then fits per-label
affine transforms to minimize raw NMSE on validation. Generates a calibrated
submission CSV.

Usage:
    uv run python -u raw_calibrate.py \
        --config configs/hpc_densenet_v1.yaml \
        --ckpt ckpt1.pt ckpt2.pt ... \
        --out submission_raw_calibrated.csv \
        --cache-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar, minimize
from torch.utils.data import DataLoader

from config import Config, LABEL_NAMES
from dataset import (
    CheXpertDataset,
    build_val_transform,
    load_and_split,
    _drop_junk_cols,
    _extract_pid,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from model import CheXpertModel
from submit import SubmitDataset, run_inference, _set_deterministic


def load_raw_labels(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw -1/0/1 labels for val split, with blanks as NaN."""
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

    cols = df_val[cfg.label_names].to_numpy(dtype=np.float32)
    # Raw: 1.0 -> 1, 0.0 -> 0, -1.0 -> -1, blank (NaN) -> NaN (masked)
    return df_val, cols


def run_ensemble_inference(
    ckpt_paths: List[Path],
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Run ensemble inference, return (ids, mean_logits, mean_probs)."""
    all_logits = None
    all_ids = None

    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"  loading ckpt {i+1}/{len(ckpt_paths)}: {ckpt_path.name}", flush=True)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        known = {f.name for f in Config.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
        cfg_i = Config(**cfg_dict)
        model = CheXpertModel(cfg_i, pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device).eval()

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
        ids, logits = run_inference(model, loader, device)

        if all_logits is None:
            all_logits = logits
            all_ids = ids
        else:
            all_logits += logits

        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_logits /= len(ckpt_paths)
    all_probs = 1.0 / (1.0 + np.exp(-all_logits))
    return all_ids, all_logits, all_probs


def raw_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NMSE against raw -1/0/1 labels, masking NaN."""
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


def fit_affine(x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
    """Fit clip(a*x + b, -1, 1) to minimize raw NMSE. Returns (a, b, nmse)."""
    mask = ~np.isnan(y_true)
    x_m = x[mask]
    y_m = y_true[mask]

    def objective(params):
        a, b = params
        pred = np.clip(a * x_m + b, -1, 1)
        mse = np.mean((y_m - pred) ** 2)
        return mse

    # Try multiple starting points
    best_result = None
    best_mse = float("inf")
    for a_init in [-2, -1, 0, 1, 2, 4]:
        for b_init in [-1, -0.5, 0, 0.5, 1]:
            result = minimize(objective, [a_init, b_init], method="Nelder-Mead",
                              options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8})
            if result.fun < best_mse:
                best_mse = result.fun
                best_result = result

    a, b = best_result.x
    var = np.var(y_m)
    nmse = best_mse / var if var > 0 else float("nan")
    return float(a), float(b), nmse


def main():
    _set_deterministic()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists. Use --force to overwrite.")

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    # --- Load raw val labels ---
    print("loading raw val labels...", flush=True)
    df_val, y_val_raw = load_raw_labels(cfg)
    print(f"  val samples: {len(df_val)}", flush=True)

    # --- Load test ids ---
    df_test = pd.read_csv(cfg.test_ids_csv)
    print(f"  test samples: {len(df_test)}", flush=True)

    # --- Cache paths ---
    cache_dir = args.cache_dir or Path(".")
    cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache = cache_dir / "val_ensemble_cache.npz"
    test_cache = cache_dir / "test_ensemble_cache.npz"

    transform = build_val_transform(cfg)

    # --- Val inference ---
    if val_cache.exists():
        print(f"loading cached val predictions from {val_cache}", flush=True)
        data = np.load(str(val_cache))
        val_logits = data["logits"]
        val_probs = data["probs"]
    else:
        print("running val inference...", flush=True)
        val_ds = CheXpertDataset(df_val, y_val_raw, cfg.data_root, transform)
        # CheXpertDataset returns (image, label), but we only need images
        # We'll use it directly since run_inference expects (id, image)
        # Need to use SubmitDataset-like approach for val
        # Actually, let's just run model forward manually
        val_logits_list = []
        for ckpt_path in args.ckpt:
            print(f"  ckpt: {ckpt_path.name}", flush=True)
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            cfg_dict = ckpt["config"]
            known = {f.name for f in Config.__dataclass_fields__.values()}
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
            cfg_i = Config(**cfg_dict)
            model = CheXpertModel(cfg_i, pretrained=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device).eval()

            loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            logit_chunks = []
            with torch.no_grad():
                for x, _y in loader:
                    x = x.to(device, non_blocking=True)
                    logits = model(x)
                    logit_chunks.append(logits.float().cpu().numpy())
            val_logits_list.append(np.concatenate(logit_chunks, axis=0))
            del model, ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_logits = np.mean(val_logits_list, axis=0)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        np.savez(str(val_cache), logits=val_logits, probs=val_probs)
        print(f"  cached val predictions to {val_cache}", flush=True)

    # --- Test inference ---
    if test_cache.exists():
        print(f"loading cached test predictions from {test_cache}", flush=True)
        data = np.load(str(test_cache))
        test_logits = data["logits"]
        test_probs = data["probs"]
        test_ids = data["ids"].tolist()
    else:
        print("running test inference...", flush=True)
        test_ds = SubmitDataset(df_test, cfg.data_root, transform)
        test_ids_list = []
        test_logits_list = []
        for ckpt_path in args.ckpt:
            print(f"  ckpt: {ckpt_path.name}", flush=True)
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            cfg_dict = ckpt["config"]
            known = {f.name for f in Config.__dataclass_fields__.values()}
            cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
            cfg_i = Config(**cfg_dict)
            model = CheXpertModel(cfg_i, pretrained=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device).eval()

            loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            ids, logits = run_inference(model, loader, device)
            if not test_ids_list:
                test_ids_list = ids
            test_logits_list.append(logits)
            del model, ckpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        test_logits = np.mean(test_logits_list, axis=0)
        test_probs = 1.0 / (1.0 + np.exp(-test_logits))
        test_ids = test_ids_list
        np.savez(str(test_cache), logits=test_logits, probs=test_probs,
                 ids=np.array(test_ids))
        print(f"  cached test predictions to {test_cache}", flush=True)

    # --- Per-label calibration ---
    print("\n" + "=" * 70, flush=True)
    print("PER-LABEL CALIBRATION", flush=True)
    print("=" * 70, flush=True)

    label_names = cfg.label_names
    test_output = np.zeros_like(test_probs)

    print(f"\n{'Label':30s}  {'Candidate':20s}  {'Val NMSE':>10s}  {'Test Mean':>10s}  {'a':>8s}  {'b':>8s}", flush=True)
    print("-" * 90, flush=True)

    for i, name in enumerate(label_names):
        yt = y_val_raw[:, i]
        vp = val_probs[:, i]
        vl = val_logits[:, i]
        tp = test_probs[:, i]
        tl = test_logits[:, i]

        mask = ~np.isnan(yt)
        train_raw_mean = np.mean(yt[mask])

        candidates = {}

        # 1. Raw model probability
        candidates["model_prob"] = (vp, tp, raw_nmse(yt, vp), "-", "-")

        # 2. Constant train raw mean
        const_pred = np.full_like(vp, train_raw_mean)
        const_test = np.full_like(tp, train_raw_mean)
        candidates["constant_mean"] = (const_pred, const_test, raw_nmse(yt, const_pred),
                                        "-", f"{train_raw_mean:.3f}")

        # 3. Affine on probability
        a_p, b_p, nmse_p = fit_affine(vp, yt)
        pred_p = np.clip(a_p * vp + b_p, -1, 1)
        test_p = np.clip(a_p * tp + b_p, -1, 1)
        candidates["affine_prob"] = (pred_p, test_p, nmse_p, f"{a_p:.3f}", f"{b_p:.3f}")

        # 4. Affine on logit
        a_l, b_l, nmse_l = fit_affine(vl, yt)
        pred_l = np.clip(a_l * vl + b_l, -1, 1)
        test_l = np.clip(a_l * tl + b_l, -1, 1)
        candidates["affine_logit"] = (pred_l, test_l, nmse_l, f"{a_l:.3f}", f"{b_l:.3f}")

        # Select best
        best_name = min(candidates, key=lambda k: candidates[k][2])
        best = candidates[best_name]

        for cand_name, (_, t_pred, nmse_val, a_str, b_str) in sorted(
            candidates.items(), key=lambda x: x[1][2]
        ):
            marker = " <-- SELECTED" if cand_name == best_name else ""
            test_mean = np.mean(t_pred)
            print(
                f"  {name:28s}  {cand_name:20s}  {nmse_val:10.4f}  {test_mean:+10.4f}  {a_str:>8s}  {b_str:>8s}{marker}",
                flush=True,
            )

        test_output[:, i] = best[1]
        print(flush=True)

    # --- Write submission ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + label_names)
        for idx, row in zip(test_ids, test_output):
            writer.writerow([idx] + [f"{v:.6f}" for v in row])

    print(f"\nwrote {args.out} ({len(test_ids)} rows)", flush=True)

    # --- Sanity check ---
    print("\n=== SANITY CHECK ===", flush=True)
    print(f"{'Label':30s}  {'Test Mean':>10s}  {'Min':>8s}  {'Max':>8s}", flush=True)
    for i, name in enumerate(label_names):
        col = test_output[:, i]
        print(f"  {name:28s}  {np.mean(col):+10.4f}  {np.min(col):+8.4f}  {np.max(col):+8.4f}", flush=True)


if __name__ == "__main__":
    main()
