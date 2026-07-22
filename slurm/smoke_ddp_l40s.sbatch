#!/usr/bin/env bash
#SBATCH --job-name=dfw-ddp-smoke
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export TORCH_HOME=${ARTIFACT_ROOT}/cache/torch
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/driftworld"
"${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node=2 main_train.py \
    --config-name=pushT_driftflow \
    train.max_steps=2 train.ckpt_every=1 model.n_neg=2 data.batch_size=1 \
    output_dir="${ARTIFACT_ROOT}/checkpoints/smoke-ddp" \
    wandb_info.name=smoke-ddp
