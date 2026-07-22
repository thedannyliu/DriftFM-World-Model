#!/usr/bin/env bash
#SBATCH --job-name=dfw-posttrain
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
CONFIGS=(pushT_driftworld_continue pushT_driftflow)
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
cd "${REPO_ROOT}/driftworld"
"${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node=2 main_train.py \
    --config-name="${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"
