"""Single-GPU smoke test: load model, run 2 train steps + 1 eval on tiny
subsets. Catches bugs before burning SLURM time on the full 840M ViT-H+
run. Run from the project root::

    CUDA_VISIBLE_DEVICES=0 uv run python tests/smoke_test.py
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
from torch.utils.data import DataLoader, Subset

from config import Config
from dataset import (
    CheXpertDataset,
    build_train_transform,
    build_val_transform,
    load_and_split,
)
from metrics import per_label_auroc
from model import CheXpertModel
from train import mixup_batch


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    cfg = Config.from_yaml(_PROJECT_ROOT / "configs" / "default.yaml")
    # small batch just to verify pipeline
    cfg.batch_size_per_gpu = 2
    cfg.num_workers = 2

    print("loading data …", flush=True)
    df_train, df_val, y_train, y_val = load_and_split(cfg)
    train_ds = CheXpertDataset(df_train, y_train, cfg.data_root, build_train_transform(cfg))
    val_ds = CheXpertDataset(df_val, y_val, cfg.data_root, build_val_transform(cfg))
    train_ds_small = Subset(train_ds, list(range(8)))
    val_ds_small = Subset(val_ds, list(range(8)))
    print(f"  train total={len(train_ds):,}  val total={len(val_ds):,}", flush=True)
    print(f"  using {len(train_ds_small)} train / {len(val_ds_small)} val samples for smoke", flush=True)

    train_loader = DataLoader(train_ds_small, batch_size=cfg.batch_size_per_gpu, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds_small, batch_size=cfg.batch_size_per_gpu, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    print("building model (this downloads nothing; weights loaded from disk) …", flush=True)
    t0 = time.time()
    model = CheXpertModel(cfg).to(device)
    print(f"  model on device in {time.time()-t0:.1f}s", flush=True)

    print("allocating optimizer …", flush=True)
    optim = torch.optim.AdamW(
        model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    model.train()
    print("running 2 train steps …", flush=True)
    for step, (x, y) in enumerate(train_loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x, y = mixup_batch(x, y, cfg.mixup_alpha)
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  step {step}  loss={loss.item():.4f}  grad_norm={float(gn):.3f}  step_time={dt:.2f}s  peak_mem={mem:.1f}GB  logits={tuple(logits.shape)}", flush=True)
        if step == 2:
            break

    print("running eval …", flush=True)
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits_list.append(logits.float())
            y_list.append(y)
    all_logits = torch.cat(logits_list, dim=0)
    all_y = torch.cat(y_list, dim=0)
    yp = torch.sigmoid(all_logits).cpu().numpy()
    yt = all_y.cpu().numpy()
    try:
        metrics = per_label_auroc(yt, yp, cfg.label_names, min_positives=1)  # tiny subset; relax threshold
        print("  eval metrics (8 val samples, AUCs largely meaningless):", flush=True)
        for k, v in metrics.items():
            print(f"    {k}: {v}", flush=True)
    except Exception as e:
        print(f"  metric computation failed: {e}", flush=True)
    print("smoke test OK", flush=True)


if __name__ == "__main__":
    main()
