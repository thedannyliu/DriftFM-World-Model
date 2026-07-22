#!/usr/bin/env bash
#SBATCH --job-name=dw-baseline
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%j.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
NUM_VIDEOS=${EVAL_NUM_VIDEOS:-10}
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export TORCH_HOME=${ARTIFACT_ROOT}/cache/torch
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/driftworld"
"${ENV_PREFIX}/bin/python" main_eval_metrics.py --config-name=pushT_driftworld \
    data.dataset_path_dir="${ARTIFACT_ROOT}/data/world_model_data/dataset_domain/all_data" \
    output_dir="${ARTIFACT_ROOT}/checkpoints/official/pusht_checkpoints/pushT_driftworld" \
    +eval.num_videos="${NUM_VIDEOS}" \
    +eval.metrics_dir="${ARTIFACT_ROOT}/runs/metrics/driftworld-official-smoke"
