#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != control && $1 != driftflow ) ]]; then
    echo "Usage: $0 {control|driftflow}" >&2
    exit 2
fi
ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
ENV_PREFIX=${DRIFTFLOWWORLD_ENV_PREFIX:-${RUNTIME_ROOT}/envs/driftfm-ngc24.06-py310}
export PYTHONPATH="${ENV_PREFIX}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
BATCH_PER_GPU=${BATCH_PER_GPU:-1}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
MAX_STEPS=${MAX_STEPS:-10000}
SEED=${SEED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

if [[ ${ROLE} == control ]]; then
    CONFIG=pushT_driftworld_continue
    RUN_NAME=company-control-seed${SEED}
    OUTPUT_DIR=${ASSET_ROOT}/checkpoints/experiments/pushT_driftworld_continue_seed${SEED}
else
    CONFIG=pushT_driftflow
    RUN_NAME=company-driftflow-seed${SEED}
    OUTPUT_DIR=${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed${SEED}
fi

DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
INIT_CHECKPOINT=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/pushT_driftworld/ckpt_save/ckpt-step1180500.pth
LOG_DIR=${RUNTIME_ROOT}/logs/pilot/${ROLE}-${TIMESTAMP}
FULL_LOG=${LOG_DIR}/train.log
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RUNTIME_ROOT}/wandb/${RUN_NAME}"
if [[ ! -e ${OUTPUT_DIR}/wandb && ! -L ${OUTPUT_DIR}/wandb ]]; then
    ln -s "${RUNTIME_ROOT}/wandb/${RUN_NAME}" "${OUTPUT_DIR}/wandb"
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export WANDB_DIR=${RUNTIME_ROOT}/wandb
export PYTHONNOUSERSITE=1

WANDB_ARGS=(wandb_info.project="${WANDB_PROJECT}" wandb_info.name="${RUN_NAME}")
if [[ -n ${WANDB_ENTITY:-} ]]; then
    WANDB_ARGS+=(wandb_info.entity="${WANDB_ENTITY}")
fi

cd "${REPO_ROOT}/driftworld"
echo "[pilot] dependency preflight"
"${ENV_PREFIX}/bin/python" -c \
    'import hydra, omegaconf, torch, wandb, zarr; import train, utils_model; print("[pilot] dependency_preflight=pass")'
echo "[pilot] role=${ROLE} gpus=${GPUS_PER_NODE} batch_per_gpu=${BATCH_PER_GPU} max_steps=${MAX_STEPS} seed=${SEED}"
echo "[pilot] output=${OUTPUT_DIR} full_log=${FULL_LOG} wandb_project=${WANDB_PROJECT} wandb_run=${RUN_NAME}"
if [[ -f ${OUTPUT_DIR}/ckpt-latest.pth ]]; then
    echo "[pilot] resume_checkpoint=${OUTPUT_DIR}/ckpt-latest.pth"
else
    echo "[pilot] init_checkpoint=${INIT_CHECKPOINT}"
fi
set +e
"${ENV_PREFIX}/bin/python" -m torch.distributed.run \
    --standalone --nproc_per_node="${GPUS_PER_NODE}" main_train.py \
    --config-name="${CONFIG}" \
    train.seed="${SEED}" train.max_steps="${MAX_STEPS}" \
    train.init_checkpoint="${INIT_CHECKPOINT}" \
    data.dataset_path_dir="${DATA_DIR}" data.batch_size="${BATCH_PER_GPU}" \
    dataloader.num_workers="${WORKERS_PER_GPU}" \
    output_dir="${OUTPUT_DIR}" hydra.run.dir="${LOG_DIR}/hydra" \
    "${WANDB_ARGS[@]}" 2>&1 | tee "${FULL_LOG}" | awk '
        /Started new wandb|Resuming wandb|Saving latest ckpt|Saving final checkpoint/ {
            print; fflush(); next
        }
        /loss_backprop:/ {
            losses += 1
            if (losses == 1 || losses % 100 == 0) { print; fflush() }
        }
    '
PIPE_STATUSES=("${PIPESTATUS[@]}")
STATUS=${PIPE_STATUSES[0]}
set -e
if (( STATUS != 0 )); then
    echo "Training failed with exit ${STATUS}; first error context:" >&2
    grep -n -m 1 -B 5 -A 50 -E \
        'Traceback \(most recent call last\):|ModuleNotFoundError:|ImportError:|FileNotFoundError:|RuntimeError:' \
        "${FULL_LOG}" >&2 || tail -n 60 "${FULL_LOG}" >&2
    echo "full_log=${FULL_LOG}" >&2
    exit "${STATUS}"
fi

"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/company/summarize_checkpoint.py" \
    --role "${ROLE}" --output-dir "${OUTPUT_DIR}" --log "${FULL_LOG}"
