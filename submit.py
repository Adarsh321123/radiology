"""Inference over test_ids.csv → submission CSV with sigmoid probabilities.

Usage::

    uv run python submit.py --ckpt /path/to/ckpt_best.pt --out submission.csv

    # With TTA (averages predictions over augmented views):
    uv run python submit.py --ckpt /path/to/ckpt_best.pt --out submission.csv --tta 5

    # With multiple checkpoints (ensemble):
    uv run python submit.py --ckpt ckpt1.pt ckpt2.pt ckpt3.pt --out submission.csv

    # With post-processing (temperature scaling + derived No Finding):
    uv run python submit.py --ckpt /path/to/ckpt_best.pt --out submission.csv \\
        --temp-scale 1.2 --derive-no-finding 0.5

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
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from config import Config, LABEL_NAMES
from dataset import build_val_transform, IMAGENET_MEAN, IMAGENET_STD
from model import CheXpertModel


# --------------------------------------------------------------------------- #
# TTA transforms
# --------------------------------------------------------------------------- #
def build_tta_transforms(cfg: Config) -> List:
    """Return a list of deterministic augmentation transforms for TTA.

    Each transform produces a different view of the same image.
    The first is always the standard val transform (identity-like).
    """
    base = build_val_transform(cfg)
    sz = cfg.image_size
    transforms = [base]  # view 0: standard center resize

    # view 1: horizontal flip
    transforms.append(v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(sz, sz), antialias=True),
        v2.RandomHorizontalFlip(p=1.0),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]))

    # view 2: slight center crop (95%)
    crop = int(sz * 0.95)
    transforms.append(v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(sz, sz), antialias=True),
        v2.CenterCrop(crop),
        v2.Resize(size=(sz, sz), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]))

    # view 3: small rotation +5
    transforms.append(v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(sz, sz), antialias=True),
        v2.RandomRotation(degrees=(5, 5)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]))

    # view 4: small rotation -5
    transforms.append(v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(sz, sz), antialias=True),
        v2.RandomRotation(degrees=(-5, -5)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]))

    return transforms


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


class SubmitDatasetRaw(Dataset):
    """Returns PIL images (not transformed) so TTA can apply multiple transforms."""

    def __init__(self, df: pd.DataFrame, data_root: str) -> None:
        self.ids: List[int] = df["Id"].tolist()
        self.paths: List[str] = df["Path"].tolist()
        self.root = Path(data_root)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[int, Image.Image]:
        img = Image.open(self.root / self.paths[idx]).convert("RGB")
        return self.ids[idx], img


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


def run_inference(
    model: CheXpertModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], np.ndarray]:
    """Run inference, return (ids, logits) — logits not sigmoided yet.

    Runs in native FP32 (no autocast) for bit-reproducible outputs. Slower,
    but eliminates a source of run-to-run drift in submission CSVs.
    """
    all_ids: List[int] = []
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for ids, x in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.float().cpu().numpy())
            all_ids.extend([int(i) for i in ids])
    return all_ids, np.concatenate(all_logits, axis=0)


def run_inference_tta(
    model: CheXpertModel,
    df: pd.DataFrame,
    cfg: Config,
    device: torch.device,
    num_tta: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[int], np.ndarray]:
    """Run inference with TTA, averaging logits across augmented views."""
    tta_transforms = build_tta_transforms(cfg)[:num_tta]
    print(f"TTA: using {len(tta_transforms)} views", flush=True)

    all_logits_sum = None
    all_ids = None

    for view_idx, tfm in enumerate(tta_transforms):
        ds = SubmitDataset(df, cfg.data_root, tfm)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        ids, logits = run_inference(model, loader, device)
        if all_logits_sum is None:
            all_logits_sum = logits
            all_ids = ids
        else:
            all_logits_sum += logits
        print(f"  view {view_idx+1}/{len(tta_transforms)} done", flush=True)

    return all_ids, all_logits_sum / len(tta_transforms)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def apply_temperature_scaling(
    logits: np.ndarray,
    temperatures: np.ndarray,
) -> np.ndarray:
    """Apply per-label temperature scaling: sigmoid(logits / T).

    temperatures: shape (num_labels,) — one T per label.
    """
    return logits / temperatures[np.newaxis, :]


def derive_no_finding(
    probs: np.ndarray,
    label_names: List[str],
    alpha: float,
) -> np.ndarray:
    """Replace No Finding with a blend of model prediction and derived signal.

    derived = product(1 - p_i) for all pathology labels (not No Finding).
    final_no_finding = alpha * model_pred + (1 - alpha) * derived

    alpha=0 means fully derived, alpha=1 means fully model prediction.
    """
    nf_idx = label_names.index("No Finding")
    other_indices = [i for i in range(len(label_names)) if i != nf_idx]
    other_probs = probs[:, other_indices]
    derived = np.prod(1.0 - other_probs, axis=1)
    probs[:, nf_idx] = alpha * probs[:, nf_idx] + (1.0 - alpha) * derived
    return probs


def _set_deterministic() -> None:
    """Force bit-reproducible inference: same ckpt + same args = same CSV.

    Without these, CuDNN/cuBLAS heuristics and TF32 pick different kernels
    run-to-run, which compounds through a 840M ViT into per-row prediction
    drift visible in submission CSVs.
    """
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    _set_deterministic()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, nargs="+",
                    help="path(s) to checkpoint(s). Multiple = ensemble.")
    ap.add_argument("--out", required=True, type=Path, help="output submission CSV path")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    # TTA
    ap.add_argument("--tta", type=int, default=1,
                    help="number of TTA views (1=no TTA, up to 5)")
    # Post-processing
    ap.add_argument("--temp-scale", type=float, nargs="+", default=None,
                    help="temperature(s) for scaling. Single value = global, "
                         "9 values = per-label.")
    ap.add_argument("--derive-no-finding", type=float, default=None,
                    help="alpha for blending model No Finding with derived signal. "
                         "0=fully derived, 1=fully model, 0.5=50/50 blend.")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(
            f"{args.out} already exists. Pass --force to overwrite, or pick a different --out path."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model(s) ---
    models_and_cfgs = []
    for ckpt_path in args.ckpt:
        print(f"loading ckpt: {ckpt_path}", flush=True)
        model, cfg, meta = load_model(ckpt_path, device)
        print(
            f"  trained step={meta['step']}  epoch={meta['epoch']}  "
            f"best_{meta['primary_metric']}={meta['best_metric']}",
            flush=True,
        )
        models_and_cfgs.append((model, cfg, meta))

    # Use first checkpoint's config for data loading
    cfg = models_and_cfgs[0][1]

    # --- Test set ---
    df = pd.read_csv(cfg.test_ids_csv)
    print(f"test rows: {len(df):,}", flush=True)

    # --- Inference (with optional TTA and ensembling) ---
    t0 = time.time()
    ensemble_logits = None

    for model_idx, (model, model_cfg, _) in enumerate(models_and_cfgs):
        if len(models_and_cfgs) > 1:
            print(f"\n--- model {model_idx+1}/{len(models_and_cfgs)} ---", flush=True)

        if args.tta > 1:
            all_ids, logits = run_inference_tta(
                model, df, model_cfg, device,
                num_tta=args.tta,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        else:
            ds = SubmitDataset(df, model_cfg.data_root, build_val_transform(model_cfg))
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            all_ids, logits = run_inference(model, loader, device)

        if ensemble_logits is None:
            ensemble_logits = logits
        else:
            ensemble_logits += logits

    # Average across ensemble members
    logits = ensemble_logits / len(models_and_cfgs)
    print(f"\ninference done in {time.time()-t0:.1f}s  shape={logits.shape}", flush=True)
    if len(models_and_cfgs) > 1:
        print(f"  ensembled {len(models_and_cfgs)} models", flush=True)
    if args.tta > 1:
        print(f"  TTA with {args.tta} views", flush=True)

    # --- Post-processing ---
    # Temperature scaling (applied to logits before sigmoid)
    if args.temp_scale is not None:
        if len(args.temp_scale) == 1:
            temperatures = np.full(cfg.num_labels, args.temp_scale[0])
        elif len(args.temp_scale) == cfg.num_labels:
            temperatures = np.array(args.temp_scale)
        else:
            raise ValueError(
                f"--temp-scale expects 1 or {cfg.num_labels} values, got {len(args.temp_scale)}"
            )
        logits = apply_temperature_scaling(logits, temperatures)
        print(f"  applied temperature scaling: {temperatures.tolist()}", flush=True)

    # Sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Derived No Finding (applied after sigmoid)
    if args.derive_no_finding is not None:
        alpha = args.derive_no_finding
        probs = derive_no_finding(probs, cfg.label_names, alpha)
        print(f"  derived No Finding with alpha={alpha}", flush=True)

    # --- Integrity checks ---
    if len(all_ids) != len(df):
        raise RuntimeError(
            f"id count mismatch: predicted {len(all_ids)} but test_ids.csv has {len(df)}"
        )
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("duplicate Ids in submission — DataLoader order bug?")
    if probs.shape[0] != len(all_ids):
        raise RuntimeError(f"probs rows={probs.shape[0]} != ids={len(all_ids)}")

    # --- Write submission CSV ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id"] + cfg.label_names)
        for i, p in zip(all_ids, probs):
            writer.writerow([i] + [f"{v:.6f}" for v in p])
    print(f"wrote {args.out}  ({len(all_ids)} rows)", flush=True)


if __name__ == "__main__":
    main()
