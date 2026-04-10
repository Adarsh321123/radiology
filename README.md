# CheXpert 2023 - Chest X-Ray Classification

CS/CNS/EE 156b (Learning Systems) at Caltech. Team of 3 competing on the CheXpert dataset.

## Task

Given a chest X-ray image, predict the probability of 9 pathologies:
No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Pneumonia, Pleural Effusion, Pleural Other, Fracture, Support Devices.

## Dataset

- **224k chest X-rays** across 65k patients (frontal + lateral views)
- Labels are NLP auto-generated from radiologist reports (noisy)
- Label values: positive (1), uncertain (0), negative (-1), blank (unmentioned)
- Heavily imbalanced (e.g., No Finding is ~9% positive, ~91% negative)
- Data split: 80% train, 10% public test, 10% private test
- Data lives on Caltech HPC at `/data/artifacts/frank/misc`

## Scoring

**Normalized MSE** per category: `NMSE = MSE / variance`. Blanks excluded from scoring. Lower is better.
- NMSE < 1.0 = model beats predicting the mean (learning real signal)
- NMSE = 1.0 = no better than the mean
- NMSE > 1.0 = worse than the mean

## Leaderboard

- **Public phase**: scored on public test set (10%), scores visible
- **Private phase**: scored on hidden holdout (10%), scores hidden until challenge ends (final ranking)

## Current Standing (as of 2026-04-09)

**2nd place** (average NMSE 1.18), 4 submissions.

| Label | Us (2nd) | 1st Place | Status |
|---|---|---|---|
| Pleural Effusion | **0.602** | 0.763 | We lead |
| Cardiomegaly | **0.748** | 0.845 | We lead |
| Support Devices | 0.914 | 0.993 | We lead |
| Pneumonia | 1.012 | 0.995 | They lead |
| Lung Opacity | 1.084 | 0.940 | They lead |
| Fracture | 1.162 | 0.932 | They lead |
| Enlarged Cardiom. | 1.169 | 1.165 | ~Tied |
| Pleural Other | 1.603 | 0.986 | They lead |
| No Finding | 2.326 | 2.326 | Tied |
| **Average** | **1.18** | **1.105** | Gap: 0.075 |

Biggest improvement opportunities: **Pleural Other** (1.603 vs their 0.986) and **Fracture** (1.162 vs 0.932).
No Finding is equally bad for both teams (2.326).

## Approach

Fine-tuning **DINOv3 ViT-H+/16** (840M params) with an attention-pool classification head.

- Backbone loaded from local clone, pretrained on ImageNet21K + IN-1k
- Attention pool head: learned query over all tokens (CLS + storage + patches)
- U-Ones label strategy: uncertain -> positive during training, masked during eval
- Patient-wise train/val split (no data leakage)
- Mixup augmentation (Beta 0.1)
- Differential LR: backbone 1e-6, head 2e-5
- Cosine annealing with 5% linear warmup
- 4x H100 GPUs, batch size 8/GPU, 768x768 images

## Training

```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/v1.yaml
```

## Inference

```bash
uv run python submit.py --ckpt /path/to/ckpt_best.pt --out submission.csv
```

## Submission Format

CSV with columns: `Id, No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Pneumonia, Pleural Effusion, Pleural Other, Fracture, Support Devices`

## Key Deadlines

- First submission: April 5 (done)
- At least one pathology NMSE <= 0.99: May 1 (done - Pleural Effusion at 0.602)
- Presentations: weeks 3 (April 14-17), 6, 9

## Rules

- No CheXpert-specific papers or external CheXpert data
- Any other technique, package, or non-CheXpert paper is allowed
- Grading: presentations + ideas/techniques + leaderboard score
