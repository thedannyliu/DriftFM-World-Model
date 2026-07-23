#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
MAX_STEPS=${MAX_STEPS:-10000}
EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
read -r -a SEEDS <<< "${OVERNIGHT_SEEDS:-1 2 3}"

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login' before starting the queue" >&2
    exit 1
fi

export WANDB_MODE=online
export WANDB_PROJECT
export MAX_STEPS

echo "[overnight] node=${NODE_ROLE} seeds=${SEEDS[*]} max_steps=${MAX_STEPS} wandb_mode=${WANDB_MODE} wandb_project=${WANDB_PROJECT}"
"${PYTHON_BIN}" -c 'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[overnight] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

wait_for_completion() {
    local marker=$1
    local label=$2
    while [[ ! -s ${marker} ]]; do
        echo "[overnight] waiting_for=${label} marker=${marker}"
        sleep 60
    done
    echo "[overnight] dependency_complete=${label}"
}

if [[ ${NODE_ROLE} == node-a ]]; then
    for seed in "${SEEDS[@]}"; do
        echo "[overnight] start control seed=${seed}"
        SEED=${seed} bash "${REPO_ROOT}/company/run_pilot.sh" control
        echo "[overnight] complete control seed=${seed}"

        if [[ ${seed} == 1 ]]; then
            DRIFTFLOW_MARKER=${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed1/complete-step${MAX_STEPS}.json
            wait_for_completion "${DRIFTFLOW_MARKER}" "driftflow-seed1"
            echo "[overnight] start paired_eval seed=1 videos=${EVAL_NUM_VIDEOS}"
            SEED=1 EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS} \
                bash "${REPO_ROOT}/company/run_pilot_eval.sh"
            echo "[overnight] complete paired_eval seed=1"
        fi
    done
else
    for seed in "${SEEDS[@]}"; do
        echo "[overnight] start driftflow seed=${seed}"
        SEED=${seed} bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
        echo "[overnight] complete driftflow seed=${seed}"
    done
fi

echo "[overnight] status=complete node=${NODE_ROLE}"
