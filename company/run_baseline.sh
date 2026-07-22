#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
ENV_PREFIX=${DRIFTFLOWWORLD_ENV_PREFIX:-${RUNTIME_ROOT}/envs/driftfm-ngc24.06-py310}
export PYTHONPATH="${ENV_PREFIX}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
NUM_VIDEOS=${EVAL_NUM_VIDEOS:-10}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
MODEL_DIR=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/pushT_driftworld
METRICS_DIR=${RUNTIME_ROOT}/results/baseline
LOG_DIR=${RUNTIME_ROOT}/logs/baseline/${TIMESTAMP}
FULL_LOG=${LOG_DIR}/eval.log
mkdir -p "${METRICS_DIR}" "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export PYTHONNOUSERSITE=1

cd "${REPO_ROOT}/driftworld"
echo "[baseline] videos=${NUM_VIDEOS} gpu=${CUDA_VISIBLE_DEVICES}"
echo "[baseline] data=${DATA_DIR} checkpoint=${MODEL_DIR}/ckpt_save/ckpt-step1180500.pth"
echo "[baseline] full_log=${FULL_LOG}"
if ! "${ENV_PREFIX}/bin/python" main_eval_metrics.py --config-name=pushT_driftworld \
    data.dataset_path_dir="${DATA_DIR}" output_dir="${MODEL_DIR}" \
    +eval.num_videos="${NUM_VIDEOS}" +eval.metrics_dir="${METRICS_DIR}" \
    hydra.run.dir="${LOG_DIR}/hydra" >"${FULL_LOG}" 2>&1; then
    echo "Baseline failed; last 40 log lines:" >&2
    tail -n 40 "${FULL_LOG}" >&2
    exit 1
fi
"${ENV_PREFIX}/bin/python" "${REPO_ROOT}/company/summarize_eval.py" \
    --baseline-dir "${METRICS_DIR}"
echo "[baseline] status=complete full_log=${FULL_LOG}"
