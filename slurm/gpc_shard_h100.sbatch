#!/usr/bin/env bash
#SBATCH --job-name=dfw-gpc
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8
#SBATCH --output=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.out
#SBATCH --error=/storage/scratch1/9/eliu354/driftflowworld/slurm_logs/%x-%A_%a.err

set -euo pipefail
REPO_ROOT=/storage/project/r-agarg35-0/eliu354/projects/driftworld
ARTIFACT_ROOT=/storage/scratch1/9/eliu354/driftflowworld
ENV_PREFIX=${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1
CONFIG_NAME=${CONFIG_NAME:-gpc_rank_driftflow_for_ep100}
PLANNING_STRATEGY=${PLANNING_STRATEGY:-coarse_to_fine}
NUM_PROPOSALS=${NUM_PROPOSALS:-100}
NFE=${NFE:-1}
REFINE_NFE=${REFINE_NFE:-4}
REFINE_RATIO=${REFINE_RATIO:-0.2}
TOTAL_SEEDS=${TOTAL_SEEDS:-100}
SHARDS=8
PER_SHARD=$(( (TOTAL_SEEDS + SHARDS - 1) / SHARDS ))
START_SEED=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))
END_SEED=$(( START_SEED + PER_SHARD ))
if (( END_SEED > TOTAL_SEEDS )); then END_SEED=${TOTAL_SEEDS}; fi
if (( START_SEED >= TOTAL_SEEDS )); then exit 0; fi

export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export TORCH_HOME=${ARTIFACT_ROOT}/cache/torch
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
export PYTHONNOUSERSITE=1
cd "${REPO_ROOT}/driftworld"
"${ENV_PREFIX}/bin/python" main_gpc_rank.py --config-name="${CONFIG_NAME}" \
    planning.strategy="${PLANNING_STRATEGY}" \
    planning.num_proposals="${NUM_PROPOSALS}" planning.nfe="${NFE}" \
    planning.refine_nfe="${REFINE_NFE}" planning.refine_ratio="${REFINE_RATIO}" \
    +start_number_test="${START_SEED}" +end_number_test="${END_SEED}"
