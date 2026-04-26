"""Model backbones for CheXpert classification.

Supports two model types (selected via cfg.model_type):

  - "dinov3": DINOv3 ViT-H+/16 (840M params) with attention pool or CLS head.
  - "densenet121": torchvision DenseNet-121 (7M params) with ImageNet pretraining,
    GAP + dropout + linear head.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# --------------------------------------------------------------------------- #
# backbone loader
# --------------------------------------------------------------------------- #
def _load_dinov3_backbone(cfg: Config) -> nn.Module:
    """Build the DINOv3 architecture from the local repo clone and load
    weights from a local .pth. Bypasses both ``torch.hub.load`` (which
    triggers hubconf.py → torchmetrics) and the factory's built-in
    weights loader (which routes every path through
    ``load_state_dict_from_url`` and hits a read-only cache dir).
    """
    import sys, importlib
    repo_dir = str(Path(cfg.dinov3_repo).expanduser().resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    backbones_mod = importlib.import_module("dinov3.hub.backbones")
    factory = getattr(backbones_mod, cfg.dinov3_arch)

    model = factory(pretrained=False)

    weights_path = Path(cfg.dinov3_weights).expanduser().resolve()
    sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv3 state_dict mismatch: missing={missing} unexpected={unexpected}"
        )
    return model


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class ClsHead(nn.Module):
    """Linear classifier on the CLS token only."""

    def __init__(self, dim: int, num_labels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, num_labels)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"]  # (B, D)
        return self.proj(cls)


class AttentionPoolHead(nn.Module):
    """Single-query multi-head attention pool over all backbone tokens.

    Tokens supplied to the pool = concat(CLS, storage_tokens, patch_tokens).
    A single learned query attends across them; the (1, D) output is
    LayerNormed and fed to a Linear classifier.
    """

    def __init__(self, dim: int, num_labels: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, num_labels)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: dict) -> torch.Tensor:
        cls = feats["x_norm_clstoken"].unsqueeze(1)      # (B, 1, D)
        storage = feats["x_storage_tokens"]               # (B, n_storage, D)
        patches = feats["x_norm_patchtokens"]             # (B, N_patch, D)
        tokens = torch.cat([cls, storage, patches], dim=1)  # (B, 1+n_storage+N_patch, D)

        kv = self.norm_kv(tokens)
        q = self.query.expand(tokens.size(0), -1, -1)    # (B, 1, D)
        pooled, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        pooled = self.norm_out(pooled.squeeze(1))        # (B, D)
        return self.proj(pooled)


def _build_head(cfg: Config, dim: int) -> nn.Module:
    if cfg.head_type == "cls":
        return ClsHead(dim, cfg.num_labels)
    elif cfg.head_type == "attention":
        return AttentionPoolHead(dim, cfg.num_labels, num_heads=cfg.attn_pool_heads)
    else:
        raise ValueError(f"unknown head_type: {cfg.head_type!r}")


# --------------------------------------------------------------------------- #
# DenseNet-121 backbone
# --------------------------------------------------------------------------- #
class DenseNet121Model(nn.Module):
    """DenseNet-121 with ImageNet pretrained weights, GAP + dropout + linear head."""

    def __init__(self, cfg: Config, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        import torchvision.models as models
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.densenet121(weights=weights)
        self.features = base.features
        self.hidden_dim = base.classifier.in_features  # 1024
        self.drop = nn.Dropout(p=cfg.dropout)
        self.classifier = nn.Linear(self.hidden_dim, cfg.num_labels)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.features(pixel_values)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = torch.flatten(feats, 1)
        feats = self.drop(feats)
        return self.classifier(feats)

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.features.named_parameters())
        hd_decay, hd_no_decay = split_decay(
            list(self.classifier.named_parameters())
        )

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# top-level wrapper (DINOv3)
# --------------------------------------------------------------------------- #
class CheXpertDINOv3Model(nn.Module):
    """DINOv3 backbone + classification head."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _load_dinov3_backbone(cfg)
        hidden_dim: int = int(self.backbone.embed_dim)
        self.hidden_dim = hidden_dim
        self.head = _build_head(cfg, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(pixel_values)
        return self.head(feats)

    def param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float,
    ) -> List[dict]:
        _NO_DECAY_SUFFIXES = (
            "cls_token",
            "storage_tokens",
            "mask_token",
            "query",
        )

        def split_decay(named: Iterable[tuple[str, nn.Parameter]]) -> tuple[list, list]:
            decay, no_decay = [], []
            for name, p in named:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(_NO_DECAY_SUFFIXES):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bb_decay, bb_no_decay = split_decay(self.backbone.named_parameters())
        hd_decay, hd_no_decay = split_decay(self.head.named_parameters())

        groups = [
            {"params": bb_decay,    "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
            {"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0,          "name": "backbone_nodecay"},
            {"params": hd_decay,    "lr": lr_head,     "weight_decay": weight_decay, "name": "head_decay"},
            {"params": hd_no_decay, "lr": lr_head,     "weight_decay": 0.0,          "name": "head_nodecay"},
        ]
        return [g for g in groups if len(g["params"]) > 0]


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
# Backward-compatible alias used by train.py and submit.py
def CheXpertModel(cfg: Config, pretrained: bool = True) -> nn.Module:
    if cfg.model_type == "densenet121":
        return DenseNet121Model(cfg, pretrained=pretrained)
    elif cfg.model_type == "dinov3":
        return CheXpertDINOv3Model(cfg)
    else:
        raise ValueError(f"unknown model_type: {cfg.model_type!r}")
