"""Training configuration for CheXpert 2023 × DINOv3 ViT-H+/16.

Source of truth is a YAML file passed to ``train.py --config <path>``.
This module defines the schema and defaults; any field not set in YAML
falls back to the default below. The resolved config is copied into the
run directory at training start for provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


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
    hflip: bool = False

    # --- labels ---
    # label_names: every label the MODEL is trained on. Default is the 9 scored
    # labels. For 14-label aux training, set this to the full 14-label list
    # (9 scored + 5 aux). num_labels must match len(label_names).
    label_names: List[str] = field(default_factory=lambda: list(LABEL_NAMES))
    num_labels: int = 9
    # scored_label_names: the subset of label_names that the LEADERBOARD cares
    # about. Used for val metric computation and submission CSV output. If
    # None, falls back to label_names (existing 9-label behavior). For 14-label
    # aux training, set this to the 9 scored labels.
    scored_label_names: Optional[List[str]] = None
    # Train defaults to the original U-Ones setup: uncertain -> 1,
    # blank/unmentioned -> 0. For score-aligned experiments, set
    # default_uncertain_strategy="ignore" and blank_strategy="ignore".
    #
    # Val always masks uncertain labels. blank_strategy controls whether
    # blank/unmentioned val labels are counted as negatives ("zeros", legacy)
    # or excluded from metrics ("ignore", score-aligned).
    #
    # Per-label uncertain handling: a dict mapping label name to strategy.
    # Strategies: "ones" (uncertain→1), "zeros" (uncertain→0),
    #             "ignore" (uncertain→nan, masked from loss).
    # Labels not listed default to default_uncertain_strategy.
    # Example: {"Pneumonia": "ignore", "Pleural Other": "ignore"}
    default_uncertain_strategy: str = "ones"
    blank_strategy: str = "zeros"
    uncertain_strategy: Optional[dict] = None

    # --- model ---
    # model_type: "dinov3" (original ViT-H+) or "densenet121" (torchvision).
    model_type: str = "dinov3"
    # DenseNet-121 / ConvNeXt settings
    dropout: float = 0.5
    convnext_weights: str = ""
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
    # "micro" matches the original code: average BCE across every valid
    # (sample, label) element. "macro" averages valid elements per label,
    # then averages labels equally; useful when blanks are masked and label
    # density differs by orders of magnitude.
    loss_reduction: str = "micro"
    # "binary": train targets are 0/1, loss is BCE, submit sigmoid probs.
    # "raw": train targets are -1/0/1, loss is MSE, submit clipped linear outputs.
    # "3class": train targets are class indices {0=-1, 1=0, 2=+1}, loss is per-label CE,
    #           submit P(+1) - P(-1).
    target_type: str = "binary"
    # "bce" (default for binary), "mse" (default for raw), "smooth_l1", "ce" (for 3class)
    loss_fn: str = "bce"
    # Per-label loss weights. If set, multiply each label's loss by its weight.
    # Map from label name to float. Labels not listed get weight 1.0.
    label_weights: Optional[dict] = None
    # In raw target mode: list of label names whose uncertain (0) values
    # should be masked from loss. Useful for labels like Pneumonia where
    # uncertain dominates and raw MSE learns "predict 0" instead of
    # useful pos/neg separation.
    raw_uncertain_mask: Optional[List[str]] = None
    # Optimizer: "adamw" (default) or "rmsprop"
    optimizer: str = "adamw"
    rmsprop_momentum: float = 0.9
    rmsprop_alpha: float = 0.99
    # EMA (exponential moving average of weights)
    ema: bool = False
    ema_decay: float = 0.999

    # --- eval / ckpt ---
    eval_every_steps: int = 500
    # "nmse" (lower is better, matches judge's scoring) or "auroc"
    # (higher is better, standard CheXpert convention).
    primary_metric: str = "nmse"

    # --- misc ---
    seed: int = 0
    log_every_steps: int = 20
    # Smoke / debug knobs. Leave 0 for full runs.
    max_train_samples: int = 0  # if >0, slice train set to first N rows
    max_val_samples: int = 0    # if >0, slice val set to first N rows
    max_steps: int = 0          # if >0, stop training after this many steps

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def run_dir(self) -> Path:
        return Path(self.runs_root) / self.run_name

    @property
    def effective_scored_labels(self) -> List[str]:
        """Labels used for val metrics and submission output.

        Falls back to label_names when scored_label_names is not set, so
        existing 9-label configs behave identically to before.
        """
        return list(self.scored_label_names) if self.scored_label_names else list(self.label_names)

    @property
    def scored_indices(self) -> List[int]:
        """Indices of effective_scored_labels within label_names.

        Used to slice model outputs (num_labels wide) down to the scored
        subset for metrics and submission.
        """
        scored = self.effective_scored_labels
        name_to_idx = {n: i for i, n in enumerate(self.label_names)}
        missing = [n for n in scored if n not in name_to_idx]
        if missing:
            raise ValueError(
                f"scored_label_names contains labels not in label_names: {missing}"
            )
        return [name_to_idx[n] for n in scored]

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
