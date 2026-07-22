#!/usr/bin/env bash
#SBATCH --job-name=dfw-resume-eq
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
RUN_ROOT=${ARTIFACT_ROOT}/checkpoints/resume-equivalence-${SLURM_JOB_ID}
export HF_HOME=${ARTIFACT_ROOT}/cache/huggingface
export TORCH_HOME=${ARTIFACT_ROOT}/cache/torch
export WANDB_DIR=${ARTIFACT_ROOT}/runs/wandb
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1

run_train() {
    local output_dir=$1
    local max_steps=$2
    local run_name=$3
    cd "${REPO_ROOT}/driftworld"
    "${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node=2 main_train.py \
        --config-name=pushT_driftflow \
        train.max_steps="${max_steps}" train.ckpt_every=1 model.n_neg=2 data.batch_size=1 \
        output_dir="${output_dir}" wandb_info.name="${run_name}"
}

run_train "${RUN_ROOT}/continuous" 3 "resume-eq-continuous-${SLURM_JOB_ID}"
run_train "${RUN_ROOT}/resumed" 2 "resume-eq-resumed-${SLURM_JOB_ID}"
run_train "${RUN_ROOT}/resumed" 3 "resume-eq-resumed-${SLURM_JOB_ID}"

cd "${REPO_ROOT}"
"${ENV_PREFIX}/bin/python" scripts/compare_training_checkpoints.py \
    "${RUN_ROOT}/continuous/ckpt_save/ckpt-latest.pth" \
    "${RUN_ROOT}/resumed/ckpt_save/ckpt-latest.pth" \
    --world-size 2
