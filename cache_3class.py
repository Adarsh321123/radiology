"""Cache 3-class model logits as expected values for mega_hybrid.

Handles the (N, 9, 3) -> (N, 9) conversion that raw_calibrate.py can't do.
Runs val and test inference, computes P(+1) - P(-1), saves as npz.

Usage:
    uv run python -u cache_3class.py \
        --config configs/hpc_3class_448_s0.yaml \
        --ckpt ckpt0.pt ckpt1.pt ckpt2.pt \
        --cache-dir /path/to/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import (
    CheXpertDataset, build_val_transform, load_and_split,
    _drop_junk_cols, _extract_pid,
)
from model import CheXpertModel
from submit import SubmitDataset, _set_deterministic


def main():
    _set_deterministic()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path, nargs="+")
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    val_cache = args.cache_dir / "val_ensemble_cache.npz"
    test_cache = args.cache_dir / "test_ensemble_cache.npz"

    # Load val data
    _, df_val, _, y_val = load_and_split(cfg)
    val_transform = build_val_transform(cfg)
    val_ds = CheXpertDataset(df_val, y_val, cfg.data_root, val_transform)

    # Load test data
    df_test = pd.read_csv(cfg.test_ids_csv)
    test_ds = SubmitDataset(df_test, cfg.data_root, val_transform)

    print(f"val: {len(val_ds)}, test: {len(test_ds)}", flush=True)

    def run_val_inference(model):
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
        chunks = []
        model.eval()
        with torch.no_grad():
            for x, _y in loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)  # (B, 9, 3) for 3-class
                chunks.append(logits.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def run_test_inference(model):
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
        all_ids = []
        chunks = []
        model.eval()
        with torch.no_grad():
            for ids, x in loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                chunks.append(logits.float().cpu().numpy())
                all_ids.extend([int(i) for i in ids])
        return all_ids, np.concatenate(chunks, axis=0)

    def logits_to_expected(logits_3c):
        """Convert (N, 9, 3) logits to (N, 9) expected value P(+1) - P(-1)."""
        if logits_3c.ndim == 2:
            logits_3c = logits_3c.reshape(-1, cfg.num_labels, 3)
        exp = np.exp(logits_3c - logits_3c.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return np.clip(probs[:, :, 2] - probs[:, :, 0], -1, 1)

    # Ensemble val inference
    print("=== VAL INFERENCE ===", flush=True)
    val_logits_sum = None
    for i, ckpt_path in enumerate(args.ckpt):
        print(f"  ckpt {i+1}/{len(args.ckpt)}: {ckpt_path.name}", flush=True)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        known = {f.name for f in Config.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
        cfg_i = Config(**cfg_dict)
        model = CheXpertModel(cfg_i, pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        logits = run_val_inference(model)
        if val_logits_sum is None:
            val_logits_sum = logits
        else:
            val_logits_sum += logits
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_logits = val_logits_sum / len(args.ckpt)
    val_expected = logits_to_expected(val_logits)
    np.savez(str(val_cache), logits=val_expected, probs=val_expected)
    print(f"  saved {val_cache} shape={val_expected.shape}", flush=True)

    # Ensemble test inference
    print("=== TEST INFERENCE ===", flush=True)
    test_logits_sum = None
    test_ids = None
    for i, ckpt_path in enumerate(args.ckpt):
        print(f"  ckpt {i+1}/{len(args.ckpt)}: {ckpt_path.name}", flush=True)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg_dict = ckpt["config"]
        known = {f.name for f in Config.__dataclass_fields__.values()}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
        cfg_i = Config(**cfg_dict)
        model = CheXpertModel(cfg_i, pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        ids, logits = run_test_inference(model)
        if test_logits_sum is None:
            test_logits_sum = logits
            test_ids = ids
        else:
            test_logits_sum += logits
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_logits = test_logits_sum / len(args.ckpt)
    test_expected = logits_to_expected(test_logits)
    np.savez(str(test_cache), logits=test_expected, probs=test_expected,
             ids=np.array(test_ids))
    print(f"  saved {test_cache} shape={test_expected.shape}", flush=True)

    # Sanity check
    print("\n=== SANITY CHECK ===", flush=True)
    for i, name in enumerate(cfg.label_names):
        vm = np.mean(val_expected[:, i])
        tm = np.mean(test_expected[:, i])
        print(f"  {name:30s}  val_mean={vm:+.4f}  test_mean={tm:+.4f}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
