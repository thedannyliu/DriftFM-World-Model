#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
SEED=${SEED:-1}
EVAL_STEP=${EVAL_STEP:-latest}
EVAL_CHECKPOINT_KIND=${EVAL_CHECKPOINT_KIND:-latest}
if [[ ${EVAL_CHECKPOINT_KIND} != latest && ${EVAL_CHECKPOINT_KIND} != best ]]; then
    echo "EVAL_CHECKPOINT_KIND must be latest or best" >&2
    exit 2
fi
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
CONTROL_OUT=${ASSET_ROOT}/checkpoints/experiments/pushT_driftworld_continue_seed${SEED}
DRIFTFLOW_OUT=${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed${SEED}
REWARD_DIR=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/reward
CONTROL_CHECKPOINT=${CONTROL_OUT}/ckpt-${EVAL_CHECKPOINT_KIND}.pth
DRIFTFLOW_CHECKPOINT=${DRIFTFLOW_OUT}/ckpt-${EVAL_CHECKPOINT_KIND}.pth
CONTROL_METRICS=${RUNTIME_ROOT}/results/pilot-seed${SEED}-step${EVAL_STEP}-${EVAL_CHECKPOINT_KIND}-control
DRIFTFLOW_METRICS=${RUNTIME_ROOT}/results/pilot-seed${SEED}-step${EVAL_STEP}-${EVAL_CHECKPOINT_KIND}-driftflow
LOG_DIR=${RUNTIME_ROOT}/logs/pilot-eval/${TIMESTAMP}
mkdir -p "${CONTROL_METRICS}" "${DRIFTFLOW_METRICS}" "${LOG_DIR}"

export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
cd "${REPO_ROOT}/driftworld"
echo "[eval] videos=${NUM_VIDEOS} seed=${SEED} step=${EVAL_STEP} checkpoint=${EVAL_CHECKPOINT_KIND} logs=${LOG_DIR}"
echo "[eval] gpu0=control gpu1=driftflow-nfe1 gpu2=driftflow-nfe2 gpu3=driftflow-nfe4"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" main_eval_metrics.py \
    --config-name=pushT_driftworld_continue \
    data.dataset_path_dir="${DATA_DIR}" output_dir="${CONTROL_OUT}" \
    +eval.checkpoint="${CONTROL_CHECKPOINT}" \
    +eval.num_videos="${NUM_VIDEOS}" +eval.metrics_dir="${CONTROL_METRICS}" \
    hydra.run.dir="${LOG_DIR}/hydra-control" >"${LOG_DIR}/control.log" 2>&1 &
PIDS=($!)
NAMES=(control)

for INDEX in 0 1 2; do
    NFE=$(( 2 ** INDEX ))
    GPU=$(( INDEX + 1 ))
    CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" main_eval_metrics.py \
        --config-name=pushT_driftflow \
        data.dataset_path_dir="${DATA_DIR}" output_dir="${DRIFTFLOW_OUT}" \
        eval.checkpoint="${DRIFTFLOW_CHECKPOINT}" \
        eval.num_videos="${NUM_VIDEOS}" eval.nfe="${NFE}" \
        eval.metrics_dir="${DRIFTFLOW_METRICS}" \
        eval.reward_predictor_xy_checkpoint="${REWARD_DIR}/reward_predictor_xy.pth" \
        eval.reward_predictor_angle_checkpoint="${REWARD_DIR}/reward_predictor_angle.pth" \
        hydra.run.dir="${LOG_DIR}/hydra-nfe${NFE}" >"${LOG_DIR}/nfe${NFE}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("nfe${NFE}")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
    if ! wait "${PIDS[${INDEX}]}"; then
        echo "${NAMES[${INDEX}]} failed; last 30 log lines:" >&2
        tail -n 30 "${LOG_DIR}/${NAMES[${INDEX}]}.log" >&2
        FAILED=1
    else
        echo "[eval] complete ${NAMES[${INDEX}]}"
    fi
done
if (( FAILED )); then
    exit 1
fi

"${PYTHON_BIN}" "${REPO_ROOT}/company/summarize_eval.py" \
    --control-dir "${CONTROL_METRICS}" --driftflow-dir "${DRIFTFLOW_METRICS}" \
    | tee "${ASSET_ROOT}/checkpoints/experiments/eval-seed${SEED}-step${EVAL_STEP}-${EVAL_CHECKPOINT_KIND}.json"
echo "full_logs=${LOG_DIR}"
