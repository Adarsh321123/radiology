"""CheXpert 2023 dataset + patient-wise train/val split.

Label conventions (verified against the 2023 CSV, which uses a convention
that does NOT match the upstream Stanford paper):

    raw csv value   meaning            ones  zeros  ignore
    -------------   ----------------   ----  -----  ------
    1.0             positive             1      1      1
    0.0             UNCERTAIN            1      0    (mask)
    -1.0            NEGATIVE             0      0      0
    blank           unmentioned          0      0    (mask, when configured)

Train uses U-Ones by default (uncertain → positive, blank → negative).
Score-aligned configs can instead mask uncertain and blank labels with
``nan``. Val always masks uncertain labels, and optionally masks blanks.

Images are loaded as grayscale JPEGs at native resolution (commonly 2022²),
duplicated across 3 channels via PIL's ``convert("RGB")``, and fed through
a torchvision.v2 transform pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from config import Config, LABEL_NAMES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def build_train_transform(cfg: Config) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(cfg.image_size, cfg.image_size),
            scale=(cfg.crop_scale_min, cfg.crop_scale_max),
            antialias=True,
        ),
        v2.RandomRotation(degrees=cfg.rotation_deg),
        v2.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform(cfg: Config) -> Callable:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(cfg.image_size, cfg.image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# CSV loading and split
# --------------------------------------------------------------------------- #
def _drop_junk_cols(df: pd.DataFrame) -> pd.DataFrame:
    """train2023.csv has two leading index columns; drop them."""
    drop = [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    return df.drop(columns=drop, errors="ignore")


def _extract_pid(path_series: pd.Series) -> pd.Series:
    # Paths look like 'train/pid50512/study1/view1_frontal.jpg'.
    # train2023.csv contains exactly one stray row whose Path uses the
    # upstream 'CheXpert-v1.0/train/patientXXXXX/...' format; callers
    # should drop any NaN pids produced here.
    return path_series.str.extract(r"(pid\d+)", expand=False)


def _labels_to_array(
    df: pd.DataFrame,
    label_names: List[str],
    *,
    mode: str,
    uncertain_strategy: dict | None = None,
    default_uncertain_strategy: str = "ones",
    blank_strategy: str = "zeros",
) -> np.ndarray:
    """Return (N, num_labels) float32 label array.

    mode="u_ones"  : 1→1, 0(unc)→1, -1(neg)→0, blank per blank_strategy
    mode="u_zeros" : 1→1, 0(unc)→0, -1(neg)→0, blank per blank_strategy
    mode="val"     : 1→1, 0(unc)→nan, -1(neg)→0, blank per blank_strategy
    mode="per_label": uses uncertain_strategy dict to handle each label differently.
                      Labels not in the dict default to default_uncertain_strategy.
                      "ignore" maps uncertain→nan (masked from loss via loss_mask).

    uncertain_strategy: dict mapping label name → "ones"/"zeros"/"ignore".
    default_uncertain_strategy: fallback for uncertain labels not listed in the dict.
    blank_strategy: "zeros" for legacy blank-as-negative, "ignore" to mask blanks.
    """
    cols = df[label_names].copy()
    arr = cols.to_numpy(dtype=np.float32)  # blank → NaN

    # normalize the 4 raw states explicitly
    is_pos   = (arr == 1.0)
    is_unc   = (arr == 0.0)
    is_neg   = (arr == -1.0)
    is_blank = np.isnan(arr)

    def apply_strategy(out_col: np.ndarray, mask: np.ndarray, strategy: str, label: str) -> None:
        if strategy == "ones":
            out_col[mask] = 1.0
        elif strategy == "zeros":
            out_col[mask] = 0.0
        elif strategy == "ignore":
            out_col[mask] = np.nan
        else:
            raise ValueError(f"unknown label strategy for {label}: {strategy!r}")

    out = np.zeros_like(arr, dtype=np.float32)
    out[is_pos] = 1.0
    # is_neg and legacy blanks -> 0 (default)

    for i, name in enumerate(label_names):
        apply_strategy(out[:, i], is_blank[:, i], blank_strategy, f"{name} blank")

    if mode == "u_ones":
        out[is_unc] = 1.0
    elif mode == "u_zeros":
        out[is_unc] = 0.0  # already 0
    elif mode == "val":
        out[is_unc] = np.nan  # masked during AUROC
    elif mode == "per_label":
        if uncertain_strategy is None:
            uncertain_strategy = {}
        for i, name in enumerate(label_names):
            strategy = uncertain_strategy.get(name, default_uncertain_strategy)
            apply_strategy(out[:, i], is_unc[:, i], strategy, name)
    else:
        raise ValueError(f"unknown labeling mode: {mode}")

    return out.astype(np.float32)


def _labels_to_raw_array(
    df: pd.DataFrame,
    label_names: List[str],
) -> np.ndarray:
    """Return (N, num_labels) with raw -1/0/1 values. Blanks -> NaN (masked)."""
    cols = df[label_names].copy()
    arr = cols.to_numpy(dtype=np.float32)  # blank -> NaN, keeps -1/0/1
    return arr


def load_and_split(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load train2023.csv and return (df_train, df_val, y_train, y_val).

    The returned dataframes have an added ``pid`` column. ``y_train`` uses
    U-Ones; ``y_val`` keeps -1 for later masking.
    """
    df = pd.read_csv(cfg.labels_csv)
    df = _drop_junk_cols(df)
    df["pid"] = _extract_pid(df["Path"])
    unparseable = int(df["pid"].isna().sum())
    if unparseable:
        print(f"[dataset] dropping {unparseable} row(s) with unparseable Path")
        df = df[df["pid"].notna()].reset_index(drop=True)

    missing = set(cfg.label_names) - set(df.columns)
    if missing:
        raise ValueError(f"train2023.csv is missing expected label cols: {missing}")

    pids = df["pid"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(cfg.split_seed)
    rng.shuffle(pids)
    n_val = int(round(len(pids) * cfg.val_frac))
    val_pids = set(pids[:n_val].tolist())

    in_val = df["pid"].isin(val_pids)
    df_val = df[in_val].reset_index(drop=True)
    df_train = df[~in_val].reset_index(drop=True)

    if cfg.target_type == "raw":
        y_train = _labels_to_raw_array(df_train, cfg.label_names)
        y_val = _labels_to_raw_array(df_val, cfg.label_names)
    else:
        custom_train_labels = (
            cfg.uncertain_strategy
            or cfg.default_uncertain_strategy != "ones"
            or cfg.blank_strategy != "zeros"
        )
        if custom_train_labels:
            y_train = _labels_to_array(
                df_train, cfg.label_names,
                mode="per_label",
                uncertain_strategy=cfg.uncertain_strategy,
                default_uncertain_strategy=cfg.default_uncertain_strategy,
                blank_strategy=cfg.blank_strategy,
            )
        else:
            y_train = _labels_to_array(df_train, cfg.label_names, mode="u_ones")
        y_val = _labels_to_array(
            df_val, cfg.label_names,
            mode="val",
            blank_strategy=cfg.blank_strategy,
        )
    return df_train, df_val, y_train, y_val


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CheXpertDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        data_root: str | Path,
        transform: Callable,
    ) -> None:
        assert len(df) == len(y), f"{len(df)} rows vs {len(y)} labels"
        self.paths: List[str] = df["Path"].tolist()
        self.y = y.astype(np.float32)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        full = self.data_root / self.paths[idx]
        with Image.open(full) as img:
            img = img.convert("RGB")  # grayscale → 3-channel
            x = self.transform(img)
        y = torch.from_numpy(self.y[idx])
        return x, y
