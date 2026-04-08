"""Inference over test_ids.csv → submission CSV with sigmoid probabilities.

Usage::

    uv run python submit.py --ckpt /path/to/ckpt_best.pt --out submission.csv

The submission format matches the sample shown in the judge:

    Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,
    Pneumonia,Pleural Effusion,Pleural Other,Fracture,Support Devices
    18,0.012,0.048,...
    ...

Submits **sigmoid probabilities** rather than hard 0/1. If the judge
rejects floats we can threshold at 0.5 and re-submit.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config
from dataset import build_val_transform
from model import CheXpertModel


class SubmitDataset(Dataset):
    """Minimal dataset that iterates rows of test_ids.csv."""

    def __init__(self, df: pd.DataFrame, data_root: str, transform) -> None:
        self.ids: List[int] = df["Id"].tolist()
        self.paths: List[str] = df["Path"].tolist()
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor]:
        with Image.open(self.root / self.paths[idx]) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        return self.ids[idx], x


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[CheXpertModel, Config, dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    # Drop unknown keys so older checkpoints still load after the config
    # schema grows.
    known = {f.name for f in Config.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k in known}
    cfg = Config(**cfg_dict)
    model = CheXpertModel(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    # Support both new ("best_metric" + "primary_metric") and old
    # ("best_mean_auc") checkpoint formats.
    best_metric = ckpt.get("best_metric", ckpt.get("best_mean_auc"))
    primary = ckpt.get("primary_metric", "auroc")
    meta = {
        "step": ckpt.get("step"),
        "epoch": ckpt.get("epoch"),
        "primary_metric": primary,
        "best_metric": best_metric,
    }
    return model, cfg, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help="path to ckpt_best.pt")
    ap.add_argument("--out", required=True, type=Path, help="output submission CSV path")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(
            f"{args.out} already exists. Pass --force to overwrite, or pick a different --out path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ckpt: {args.ckpt}", flush=True)
    model, cfg, meta = load_model(args.ckpt, device)
    print(
        f"  trained step={meta['step']}  epoch={meta['epoch']}  "
        f"best_{meta['primary_metric']}={meta['best_metric']}",
        flush=True,
    )

    # test set
    df = pd.read_csv(cfg.test_ids_csv)
    print(f"test rows: {len(df):,}", flush=True)

    ds = SubmitDataset(df, cfg.data_root, build_val_transform(cfg))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_ids: List[int] = []
    all_probs: List[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for ids, x in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_ids.extend([int(i) for i in ids])
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    print(f"inference done in {time.time()-t0:.1f}s  shape={probs.shape}", flush=True)

    # Integrity checks: every test row produced one prediction, ids unique.
    if len(all_ids) != len(df):
        raise RuntimeError(
            f"id count mismatch: predicted {len(all_ids)} but test_ids.csv has {len(df)}"
        )
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("duplicate Ids in submission — DataLoader order bug?")
    if probs.shape[0] != len(all_ids):
        raise RuntimeError(f"probs rows={probs.shape[0]} != ids={len(all_ids)}")

    # write submission CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + cfg.label_names)
        for i, p in zip(all_ids, probs):
            writer.writerow([i] + [f"{v:.6f}" for v in p])
    print(f"wrote {args.out}  ({len(all_ids)} rows)", flush=True)


if __name__ == "__main__":
    main()
