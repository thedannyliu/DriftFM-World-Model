#!/usr/bin/env bash
#SBATCH --job-name=dfw-smoke
#SBATCH --partition=gpu-l40s
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --qos=embers
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" scripts/smoke_model.py \
    --output "${ARTIFACT_ROOT}/runs/smoke/model-${SLURM_JOB_ID}.json"
