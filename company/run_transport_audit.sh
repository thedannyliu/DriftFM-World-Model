#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_BATCHES=${AUDIT_NUM_BATCHES:-8}
PARTICLES=${AUDIT_PARTICLES:-16}
SEED=${SEED:-1}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
RESULT_DIR=${RUNTIME_ROOT}/results/transport-audit/${NODE_ROLE}-${TIMESTAMP}
LOG_DIR=${RUNTIME_ROOT}/logs/transport-audit/${NODE_ROLE}-${TIMESTAMP}

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
if (( NUM_BATCHES < 1 || PARTICLES < 1 )); then
    echo "AUDIT_NUM_BATCHES and AUDIT_PARTICLES must be positive" >&2
    exit 2
fi

if [[ ${NODE_ROLE} == node-a ]]; then
    NAMES=(
        driftflow-main-latest
        driftflow-main-best
        driftflow-uniform-best
        driftflow-kpos16-best
    )
    CHECKPOINTS=(
        "${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed${SEED}/ckpt-latest.pth"
        "${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed${SEED}/ckpt-best.pth"
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-uniform_seed${SEED}/ckpt-best.pth"
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-kpos16_seed${SEED}/ckpt-best.pth"
    )
else
    NAMES=(
        driftflow-replay50-latest
        driftflow-replay50-best
        driftflow-grid25-latest
        driftflow-grid25-best
    )
    CHECKPOINTS=(
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-replay50_seed${SEED}/ckpt-latest.pth"
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-replay50_seed${SEED}/ckpt-best.pth"
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-grid25_seed${SEED}/ckpt-latest.pth"
        "${ASSET_ROOT}/checkpoints/experiments/driftflow-grid25_seed${SEED}/ckpt-best.pth"
    )
fi

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    if [[ ! -s ${CHECKPOINT} ]]; then
        echo "Checkpoint not found: ${CHECKPOINT}" >&2
        exit 1
    fi
done

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${RUNTIME_ROOT}/wandb"
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export WANDB_DIR=${RUNTIME_ROOT}/wandb
export WANDB_MODE=online

echo "[audit] node=${NODE_ROLE} batches=${NUM_BATCHES} particles=${PARTICLES}"
echo "[audit] gpu0=${NAMES[0]} gpu1=${NAMES[1]} gpu2=${NAMES[2]} gpu3=${NAMES[3]}"
echo "[audit] results=${RESULT_DIR} logs=${LOG_DIR} wandb_project=${WANDB_PROJECT}"

PIDS=()
RESULT_ARGS=()
cd "${REPO_ROOT}/driftworld"
for INDEX in 0 1 2 3; do
    NAME=${NAMES[${INDEX}]}
    CHECKPOINT=${CHECKPOINTS[${INDEX}]}
    OUTPUT=${RESULT_DIR}/${NAME}.json
    LOG=${LOG_DIR}/${NAME}.log
    WANDB_ARGS=(+audit.wandb_project="${WANDB_PROJECT}")
    if [[ -n ${WANDB_ENTITY:-} ]]; then
        WANDB_ARGS+=(+audit.wandb_entity="${WANDB_ENTITY}")
    fi
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES=${INDEX} "${PYTHON_BIN}" main_transport_audit.py \
            --config-name=pushT_driftflow \
            data.dataset_path_dir="${DATA_DIR}" \
            data.batch_size=1 dataloader.num_workers=2 \
            validation.enabled=true validation.batch_size=1 validation.num_workers=2 \
            eval.checkpoint="${CHECKPOINT}" \
            +audit.num_batches="${NUM_BATCHES}" +audit.particles="${PARTICLES}" \
            +audit.seed=271828 +audit.output="${OUTPUT}" \
            +audit.run_name="company-transport-audit-${NAME}-seed${SEED}" \
            "${WANDB_ARGS[@]}" \
            hydra.run.dir="${LOG_DIR}/hydra-${NAME}" 2>&1 \
            | tee "${LOG}" \
            | awk -v name="${NAME}" '
                /transport-audit.*batch=|wandb:.*Run data is saved|transport audit complete/ {
                    print "[audit " name "] " $0
                    fflush()
                }
            '
    ) &
    PIDS+=($!)
    RESULT_ARGS+=(--result "${NAME}=${OUTPUT}")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
    if wait "${PIDS[${INDEX}]}"; then
        echo "[audit] complete ${NAMES[${INDEX}]}"
    else
        echo "[audit] failed ${NAMES[${INDEX}]}; last 40 log lines:" >&2
        tail -n 40 "${LOG_DIR}/${NAMES[${INDEX}]}.log" >&2
        FAILED=1
    fi
done
if (( FAILED )); then
    exit 1
fi

"${PYTHON_BIN}" "${REPO_ROOT}/company/summarize_transport_audit.py" \
    "${RESULT_ARGS[@]}" | tee "${RESULT_DIR}/summary.json"
echo "[audit] status=complete node=${NODE_ROLE} full_logs=${LOG_DIR}"
