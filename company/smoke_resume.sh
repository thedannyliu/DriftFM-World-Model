#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
ENV_PREFIX=${DRIFTFLOWWORLD_ENV_PREFIX:-${RUNTIME_ROOT}/envs/driftfm-ngc24.06-py310}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_ROOT=${ASSET_ROOT}/checkpoints/smoke/resume-equivalence-${TIMESTAMP}
LOG_DIR=${RUNTIME_ROOT}/logs/smoke-resume/${TIMESTAMP}
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
INIT_CHECKPOINT=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/pushT_driftworld/ckpt_save/ckpt-step1180500.pth
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
for ARM in continuous resumed; do
    mkdir -p "${RUN_ROOT}/${ARM}" "${RUNTIME_ROOT}/wandb/smoke-${ARM}-${TIMESTAMP}"
    ln -s "${RUNTIME_ROOT}/wandb/smoke-${ARM}-${TIMESTAMP}" \
        "${RUN_ROOT}/${ARM}/wandb"
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export WANDB_DIR=${RUNTIME_ROOT}/wandb
export WANDB_MODE=offline
export PYTHONNOUSERSITE=1

run_train() {
    local output_dir=$1
    local max_steps=$2
    local label=$3
    local log_file=${LOG_DIR}/${label}-${max_steps}.log
    local statuses
    local train_status
    echo "[smoke] start arm=${label} max_steps=${max_steps} output=${output_dir}"
    cd "${REPO_ROOT}/driftworld"
    set +e
    "${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node="${GPUS_PER_NODE}" \
        main_train.py --config-name=pushT_driftflow \
        train.max_steps="${max_steps}" train.ckpt_every=1 \
        train.init_checkpoint="${INIT_CHECKPOINT}" \
        model.n_neg=2 data.batch_size=1 data.dataset_path_dir="${DATA_DIR}" \
        dataloader.num_workers=2 output_dir="${output_dir}" \
        wandb_info.name="company-resume-${label}-${TIMESTAMP}" \
        hydra.run.dir="${LOG_DIR}/hydra-${label}-${max_steps}" \
        2>&1 | tee "${log_file}" | grep --line-buffered -E \
        'Started new wandb|Resuming wandb|loss_backprop:|Saving latest ckpt|Saving final checkpoint'
    statuses=("${PIPESTATUS[@]}")
    set -e
    train_status=${statuses[0]}
    if (( train_status != 0 )); then
        echo "[smoke] failed arm=${label} max_steps=${max_steps} exit=${train_status}; last 40 log lines:" >&2
        tail -n 40 "${log_file}" >&2
        exit "${train_status}"
    fi
    if [[ ! -f ${output_dir}/ckpt-latest.pth ]]; then
        echo "[smoke] failed: checkpoint missing at ${output_dir}/ckpt-latest.pth" >&2
        exit 1
    fi
    echo "[smoke] complete arm=${label} max_steps=${max_steps} checkpoint=${output_dir}/ckpt-latest.pth"
}

echo "[smoke] gpus=${GPUS_PER_NODE} data=${DATA_DIR} init_checkpoint=${INIT_CHECKPOINT}"
echo "[smoke] full_logs=${LOG_DIR}"
run_train "${RUN_ROOT}/continuous" 3 continuous
run_train "${RUN_ROOT}/resumed" 2 resumed
run_train "${RUN_ROOT}/resumed" 3 resumed

cd "${REPO_ROOT}"
echo "[smoke] comparing uninterrupted and resumed checkpoints"
"${ENV_PREFIX}/bin/python" scripts/compare_training_checkpoints.py \
    "${RUN_ROOT}/continuous/ckpt-latest.pth" \
    "${RUN_ROOT}/resumed/ckpt-latest.pth" \
    --world-size "${GPUS_PER_NODE}"
echo "[smoke] status=complete full_logs=${LOG_DIR}"
