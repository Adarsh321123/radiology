"""Measure peak GPU memory for one full train step at various batch sizes.

Helps choose batch_size_per_gpu before launching the full run. Run from
the project root::

    CUDA_VISIBLE_DEVICES=0 uv run python tests/mem_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when this script is invoked from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import time
import torch
import torch.nn.functional as F

from config import Config
from model import CheXpertModel


def probe_one(cfg: Config, bs: int, device: torch.device) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = CheXpertModel(cfg).to(device)
    optim = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
    )
    x = torch.randn(bs, 3, cfg.image_size, cfg.image_size, device=device)
    y = (torch.rand(bs, cfg.num_labels, device=device) > 0.5).float()
    model.train()

    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"  bs={bs:>3}  peak_alloc={peak:.1f}GB  peak_reserved={reserved:.1f}GB  step_time={dt:.2f}s", flush=True)

    del model, optim, x, y, logits, loss
    torch.cuda.empty_cache()


def main() -> None:
    cfg = Config.from_yaml(_PROJECT_ROOT / "configs" / "default.yaml")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device: {torch.cuda.get_device_name(0)}  total={total:.1f}GB  img_size={cfg.image_size}")

    for bs in [4, 8, 12, 16]:
        try:
            probe_one(cfg, bs, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  bs={bs:>3}  OOM: {str(e)[:200]}")
            torch.cuda.empty_cache()
            break


if __name__ == "__main__":
    main()
