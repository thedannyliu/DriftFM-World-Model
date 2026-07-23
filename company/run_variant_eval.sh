#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 EXPERIMENT_TAG [latest|best]" >&2
    exit 2
fi
EXPERIMENT_TAG=$1
CHECKPOINT_KIND=${2:-latest}
if [[ ! ${EXPERIMENT_TAG} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EXPERIMENT_TAG contains unsupported characters" >&2
    exit 2
fi
if [[ ${CHECKPOINT_KIND} != latest && ${CHECKPOINT_KIND} != best ]]; then
    echo "Checkpoint kind must be latest or best" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
SEED=${SEED:-1}
TRANSPORT_PARAMETERIZATION=${DRIFTFLOW_TRANSPORT_PARAMETERIZATION:-residual}
WANDB_LOG_EVAL=${WANDB_LOG_EVAL:-0}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
if [[ ${TRANSPORT_PARAMETERIZATION} != residual && ${TRANSPORT_PARAMETERIZATION} != endpoint_normalized ]]; then
    echo "Unsupported DRIFTFLOW_TRANSPORT_PARAMETERIZATION: ${TRANSPORT_PARAMETERIZATION}" >&2
    exit 2
fi
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
OUTPUT_DIR=${ASSET_ROOT}/checkpoints/experiments/${EXPERIMENT_TAG}_seed${SEED}
CHECKPOINT=${OUTPUT_DIR}/ckpt-${CHECKPOINT_KIND}.pth
REWARD_DIR=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/reward
EVAL_LABEL=${CHECKPOINT_KIND}
if [[ ${TRANSPORT_PARAMETERIZATION} != residual ]]; then
    EVAL_LABEL=${CHECKPOINT_KIND}-${TRANSPORT_PARAMETERIZATION}
fi
METRICS_DIR=${RUNTIME_ROOT}/results/${EXPERIMENT_TAG}-seed${SEED}-${EVAL_LABEL}
LOG_DIR=${RUNTIME_ROOT}/logs/variant-eval/${EXPERIMENT_TAG}-${EVAL_LABEL}-${TIMESTAMP}
MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-${EXPERIMENT_TAG}-seed${SEED}-${EVAL_LABEL}.json

if [[ ! -s ${CHECKPOINT} ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
mkdir -p "${METRICS_DIR}" "${LOG_DIR}"
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch

cd "${REPO_ROOT}/driftworld"
echo "[variant-eval] tag=${EXPERIMENT_TAG} checkpoint=${CHECKPOINT_KIND} videos=${NUM_VIDEOS} transport_parameterization=${TRANSPORT_PARAMETERIZATION}"
echo "[variant-eval] gpu0=nfe1 gpu1=nfe2 gpu2=nfe4 logs=${LOG_DIR}"

PIDS=()
NAMES=()
for INDEX in 0 1 2; do
    NFE=$(( 2 ** INDEX ))
    CUDA_VISIBLE_DEVICES=${INDEX} "${PYTHON_BIN}" main_eval_metrics.py \
        --config-name=pushT_driftflow \
        data.dataset_path_dir="${DATA_DIR}" output_dir="${OUTPUT_DIR}" \
        eval.checkpoint="${CHECKPOINT}" eval.num_videos="${NUM_VIDEOS}" \
        eval.nfe="${NFE}" eval.metrics_dir="${METRICS_DIR}" \
        model.drift_flow.transport_parameterization="${TRANSPORT_PARAMETERIZATION}" \
        eval.reward_predictor_xy_checkpoint="${REWARD_DIR}/reward_predictor_xy.pth" \
        eval.reward_predictor_angle_checkpoint="${REWARD_DIR}/reward_predictor_angle.pth" \
        hydra.run.dir="${LOG_DIR}/hydra-nfe${NFE}" \
        >"${LOG_DIR}/nfe${NFE}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("nfe${NFE}")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
    if wait "${PIDS[${INDEX}]}"; then
        echo "[variant-eval] complete ${NAMES[${INDEX}]}"
    else
        echo "[variant-eval] failed ${NAMES[${INDEX}]}; last 30 lines:" >&2
        tail -n 30 "${LOG_DIR}/${NAMES[${INDEX}]}.log" >&2
        FAILED=1
    fi
done
if (( FAILED )); then
    exit 1
fi

SUMMARY_ARGS=(--variant-dir "${METRICS_DIR}" --output "${MARKER}")
if [[ ${WANDB_LOG_EVAL} == 1 ]]; then
    SUMMARY_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-name "company-rollout-${EXPERIMENT_TAG}-${EVAL_LABEL}-seed${SEED}"
    )
    if [[ -n ${WANDB_ENTITY:-} ]]; then
        SUMMARY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi
"${PYTHON_BIN}" "${REPO_ROOT}/company/summarize_eval.py" \
    "${SUMMARY_ARGS[@]}"
echo "full_logs=${LOG_DIR}"
