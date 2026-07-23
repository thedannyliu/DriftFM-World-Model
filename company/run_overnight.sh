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
PRIMARY_STEPS=${OVERNIGHT_PRIMARY_STEPS:-30000}
REPLICATION_STEPS=${OVERNIGHT_REPLICATION_STEPS:-10000}
ABLATION_STEPS=${OVERNIGHT_ABLATION_STEPS:-10000}
EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
read -r -a SEEDS <<< "${OVERNIGHT_SEEDS:-1 2 3}"
read -r -a MILESTONES <<< "${OVERNIGHT_MILESTONES:-10000 20000}"
PRIMARY_INCLUDED=0
for step in "${MILESTONES[@]}"; do
    if (( step == PRIMARY_STEPS )); then
        PRIMARY_INCLUDED=1
    fi
done
if (( ! PRIMARY_INCLUDED )); then
    MILESTONES+=("${PRIMARY_STEPS}")
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login' before starting the queue" >&2
    exit 1
fi

export WANDB_MODE=online
export WANDB_PROJECT

echo "[overnight] node=${NODE_ROLE} seeds=${SEEDS[*]} milestones=${MILESTONES[*]} primary_steps=${PRIMARY_STEPS} replication_steps=${REPLICATION_STEPS} ablation_steps=${ABLATION_STEPS} wandb_mode=${WANDB_MODE} wandb_project=${WANDB_PROJECT}"
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
    for step in "${MILESTONES[@]}"; do
        if (( step > PRIMARY_STEPS )); then
            continue
        fi
        echo "[overnight] start control seed=1 stage=${step}"
        SEED=1 MAX_STEPS=${step} bash "${REPO_ROOT}/company/run_pilot.sh" control
        DRIFTFLOW_MARKER=${ASSET_ROOT}/checkpoints/experiments/pushT_driftflow_posttrain_seed1/complete-step${step}.json
        wait_for_completion "${DRIFTFLOW_MARKER}" "driftflow-seed1-step${step}"
        echo "[overnight] start paired_eval seed=1 step=${step} videos=${EVAL_NUM_VIDEOS}"
        SEED=1 EVAL_STEP=${step} EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS} \
            bash "${REPO_ROOT}/company/run_pilot_eval.sh"
        if (( step == PRIMARY_STEPS )); then
            echo "[overnight] start paired_eval seed=1 step=${step} checkpoint=best"
            SEED=1 EVAL_STEP=${step} EVAL_CHECKPOINT_KIND=best \
                EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS} \
                bash "${REPO_ROOT}/company/run_pilot_eval.sh"
        fi
    done

    for seed in "${SEEDS[@]}"; do
        if [[ ${seed} == 1 ]]; then
            continue
        fi
        echo "[overnight] start control seed=${seed} steps=${REPLICATION_STEPS}"
        SEED=${seed} MAX_STEPS=${REPLICATION_STEPS} bash "${REPO_ROOT}/company/run_pilot.sh" control
        echo "[overnight] complete control seed=${seed}"
    done

    echo "[overnight] start ablation=uniform-time seed=1 steps=${ABLATION_STEPS}"
    EXPERIMENT_TAG=driftflow-uniform DRIFTFLOW_TIME_SAMPLING=uniform \
        SEED=1 MAX_STEPS=${ABLATION_STEPS} \
        bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
else
    for step in "${MILESTONES[@]}"; do
        if (( step > PRIMARY_STEPS )); then
            continue
        fi
        echo "[overnight] start driftflow seed=1 stage=${step}"
        SEED=1 MAX_STEPS=${step} bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
        EVAL_MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-seed1-step${step}-latest.json
        wait_for_completion "${EVAL_MARKER}" "paired-eval-seed1-step${step}"
    done

    for seed in "${SEEDS[@]}"; do
        if [[ ${seed} == 1 ]]; then
            continue
        fi
        echo "[overnight] start driftflow seed=${seed} steps=${REPLICATION_STEPS}"
        SEED=${seed} MAX_STEPS=${REPLICATION_STEPS} bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
        echo "[overnight] complete driftflow seed=${seed}"
    done


    echo "[overnight] start ablation=endpoint-replay-0.5 seed=1 steps=${ABLATION_STEPS}"
    EXPERIMENT_TAG=driftflow-replay50 DRIFTFLOW_ENDPOINT_REPLAY=0.5 \
        SEED=1 MAX_STEPS=${ABLATION_STEPS} \
        bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
fi

echo "[overnight] status=complete node=${NODE_ROLE}"
