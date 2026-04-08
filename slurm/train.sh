#!/bin/bash
#SBATCH --job-name=chexpert
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/home/fxiao/misc/slurm_logs/%j.out

# Launch: sbatch slurm/train.sh [config_path]
# Default config: configs/v1.yaml
# Single-node DDP across 4 GPUs via torchrun.

set -euo pipefail

CONFIG="${1:-configs/v1.yaml}"
cd /home/fxiao/misc

echo "=============================================="
echo "job id     : ${SLURM_JOB_ID}"
echo "node       : $(hostname)"
echo "config     : ${CONFIG}"
echo "cwd        : $(pwd)"
echo "cuda devs  : ${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=============================================="

# NCCL / torch settings for cleaner logs and faster startup on H100
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
# Flush Python stdout immediately so live `tail -f` on the slurm log is responsive.
export PYTHONUNBUFFERED=1
export TORCH_HOME="${HOME}/.cache/torch"
mkdir -p "${TORCH_HOME}"

uv run torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py --config "${CONFIG}"

echo "=============================================="
echo "job ${SLURM_JOB_ID} finished at $(date)"
