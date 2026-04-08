"""Training configuration for CheXpert 2023 × DINOv3 ViT-H+/16.

Source of truth is a YAML file passed to ``train.py --config <path>``.
This module defines the schema and defaults; any field not set in YAML
falls back to the default below. The resolved config is copied into the
run directory at training start for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# The 9 labels in train2023.csv, in the exact order expected by the submission CSV.
LABEL_NAMES: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


@dataclass
class Config:
    # --- run identity (REQUIRED in YAML; drives output paths) ---
    run_name: str = "default"
    runs_root: str = "/data/artifacts/frank/misc/runs"

    # --- data ---
    data_root: str = "/data/artifacts/frank/misc"
    labels_csv: str = "/data/artifacts/frank/misc/labels/train2023.csv"
    test_ids_csv: str = "/data/artifacts/frank/misc/labels/test_ids.csv"
    val_frac: float = 0.1
    split_seed: int = 42

    # --- image ---
    image_size: int = 768
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.00
    rotation_deg: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2

    # --- labels ---
    label_names: List[str] = field(default_factory=lambda: list(LABEL_NAMES))
    num_labels: int = 9
    # Train: U-Ones (map -1 → 1, blank → 0).
    # Val: keep -1 as-is; mask per-label during AUROC so only {0, 1} contribute.

    # --- model ---
    # Native DINOv3 loader: we call torch.hub.load on a local clone of
    # facebookresearch/dinov3 and point at a downloaded .pth. The HF repo
    # is gated; bypassing HF avoids the access barrier.
    dinov3_repo: str = "/data/artifacts/frank/misc/dinov3_repo"
    dinov3_arch: str = "dinov3_vith16plus"
    dinov3_weights: str = "/data/artifacts/frank/misc/labels/dino/dinov3_hplus.pth"
    # Head type: "attention" (attention pooling over CLS + storage + patches)
    # or "cls" (linear on CLS token only).
    head_type: str = "attention"
    attn_pool_heads: int = 8

    # --- optimization ---
    epochs: int = 5
    batch_size_per_gpu: int = 8  # global 32 on 4× H100 (bs=16 OOMs at 768²)
    num_workers: int = 8
    lr_head: float = 2e-5
    lr_backbone: float = 1e-6
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    mixup_alpha: float = 0.1  # Beta(a, a); 0 disables

    # --- eval / ckpt ---
    eval_every_steps: int = 500
    keep_last_n_ckpts: int = 2

    # --- misc ---
    seed: int = 0
    log_every_steps: int = 20

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
