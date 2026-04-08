"""Compute MSE / variance-normalized MSE on the val split for a trained ckpt.

Run from project root::

    CUDA_VISIBLE_DEVICES=0 uv run python tests/eval_val_nmse.py \
        --ckpt /data/artifacts/frank/misc/runs/v1/ckpts/ckpt_best.pt

Metrics reported per-label and macro-averaged:
  MSE             = mean((y - p)^2)
  Var(y)          = variance of clean {0, 1} labels for this column
  NMSE_macro      = MSE / Var(y)    (per label, then averaged across labels)
  NMSE_pooled     = total SSE / total SS_tot across the flattened (row, label) matrix

Uncertain labels (raw "0", stored as NaN in val) are masked per-label.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    CheXpertDataset,
    build_val_transform,
    load_and_split,
)
from metrics import per_label_auroc
from model import CheXpertModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ckpt → config → model
    print(f"loading {args.ckpt}", flush=True)
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    print(f"  run_name={cfg.run_name}  step={ckpt['step']}  best_mean_auc={ckpt['best_mean_auc']:.4f}")

    model = CheXpertModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # Rebuild the *same* train/val split as training used
    _, df_val, _, y_val = load_and_split(cfg)
    val_ds = CheXpertDataset(df_val, y_val, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"val rows: {len(val_ds):,}", flush=True)

    # Run inference
    t0 = time.time()
    logits_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits_list.append(logits.float().cpu())
            y_list.append(y)
    all_logits = torch.cat(logits_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    yp = torch.sigmoid(all_logits).numpy()     # (N, 9) probabilities in [0, 1]
    yt = all_y.numpy()                          # (N, 9) labels with NaN for uncertain
    print(f"inference done in {time.time() - t0:.1f}s  shape={yp.shape}", flush=True)

    # AUROC as a sanity check — should match what was printed at the end of training
    auc = per_label_auroc(yt, yp, cfg.label_names)
    print()
    print("=" * 78)
    print(f"{'label':30s}  {'AUROC':>8s}  {'n':>7s}  {'MSE':>9s}  {'Var(y)':>9s}  {'NMSE':>9s}")
    print("-" * 78)

    mses = []
    vars_ = []
    nmses = []
    ssre_total = 0.0
    sstot_total = 0.0
    total_n = 0

    for i, name in enumerate(cfg.label_names):
        col_y = yt[:, i]
        col_p = yp[:, i]
        mask = ~np.isnan(col_y)
        y_clean = col_y[mask]
        p_clean = col_p[mask]
        n = len(y_clean)
        mse = float(np.mean((y_clean - p_clean) ** 2))
        ybar = float(np.mean(y_clean))
        var = float(np.mean((y_clean - ybar) ** 2))     # same as np.var(y_clean)
        nmse = mse / var if var > 0 else float("nan")

        mses.append(mse)
        vars_.append(var)
        nmses.append(nmse)

        ssre_total += float(np.sum((y_clean - p_clean) ** 2))
        sstot_total += float(np.sum((y_clean - ybar) ** 2))
        total_n += n

        print(f"{name:30s}  {auc.get(name, float('nan')):>8.4f}  {n:>7,}  {mse:>9.5f}  {var:>9.5f}  {nmse:>9.5f}")

    mean_auc = auc.get("mean", float("nan"))
    macro_mse = float(np.mean(mses))
    macro_nmse = float(np.mean(nmses))
    pooled_nmse = ssre_total / sstot_total if sstot_total > 0 else float("nan")
    pooled_mse = ssre_total / total_n if total_n > 0 else float("nan")

    print("-" * 78)
    print(f"{'MACRO MEAN':30s}  {mean_auc:>8.4f}          {macro_mse:>9.5f}             {macro_nmse:>9.5f}")
    print(f"{'POOLED (flattened)':30s}                    {pooled_mse:>9.5f}             {pooled_nmse:>9.5f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
