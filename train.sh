#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -A cs156b
#SBATCH -p gpu
#SBATCH -J cs156b_train
#SBATCH --mail-user=akumarap@caltech.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o train_%j.out

module load anaconda3
# conda activate cs156b  # uncomment once env is set up

# Default: configs/hpc.yaml. Override: sbatch train.sh configs/hpc_v2_perlabel.yaml
CONFIG="${1:-configs/hpc.yaml}"

torchrun --standalone --nproc_per_node=4 train.py --config "$CONFIG"
