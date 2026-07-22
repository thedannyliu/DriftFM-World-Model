#!/usr/bin/env bash
#SBATCH --job-name=dfw-rollout
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
NFES=(1 2 4)
NUM_VIDEOS=${EVAL_NUM_VIDEOS:-1000}
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export TORCH_HOME=${ARTIFACT_ROOT}/cache/torch
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
cd "${REPO_ROOT}/driftworld"
"${ENV_PREFIX}/bin/python" main_eval_metrics.py --config-name=pushT_driftflow \
    eval.nfe="${NFES[${SLURM_ARRAY_TASK_ID}]}" eval.num_videos="${NUM_VIDEOS}"
