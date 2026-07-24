#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
SEED=${SEED:-1}
EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
MILESTONES=(1000 3000 10000)

if [[ ${NODE_ROLE} == node-a ]]; then
    EXPERIMENT_TAG=driftflow-endpointnorm-k1
    POSITIVE_PARTICLES=1
else
    EXPERIMENT_TAG=driftflow-endpointnorm-k16
    POSITIVE_PARTICLES=16
fi

export DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized
export DRIFTFLOW_TIME_SAMPLING=logit_normal
export DRIFTFLOW_ENDPOINT_REPLAY=0.25
export DRIFTFLOW_GRID_REPLAY=0.0
export DRIFTFLOW_POSITIVE_PARTICLES=${POSITIVE_PARTICLES}
export EXPERIMENT_TAG
export SEED
export EVAL_NUM_VIDEOS
export WANDB_LOG_EVAL=1
export WANDB_MODE=online

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

evaluate_checkpoint() {
    local checkpoint_kind=$1
    local milestone=$2
    local marker="${ASSET_ROOT}/checkpoints/experiments/eval-${EXPERIMENT_TAG}-seed${SEED}-${checkpoint_kind}-endpoint_normalized-step${milestone}.json"

    if [[ -s ${marker} ]]; then
        echo "[corrected] skip_eval checkpoint=${checkpoint_kind} milestone=${milestone} marker=${marker}"
    else
        echo "[corrected] start_eval checkpoint=${checkpoint_kind} milestone=${milestone} videos=${EVAL_NUM_VIDEOS}"
        set +e
        EVAL_RESULT_LABEL=step${milestone} \
            bash "${REPO_ROOT}/company/run_variant_eval.sh" \
                "${EXPERIMENT_TAG}" "${checkpoint_kind}" \
            | awk '!/^[{].*[}]$/ { print; fflush() }'
        local pipe_status=("${PIPESTATUS[@]}")
        local status=${pipe_status[0]}
        set -e
        if (( status != 0 )); then
            echo "[corrected] eval_failed checkpoint=${checkpoint_kind} milestone=${milestone}" >&2
            return "${status}"
        fi
    fi
    RESULT_ARGS+=(--result "${checkpoint_kind}-step${milestone}=${marker}")
}

echo "[corrected] node=${NODE_ROLE} hypothesis=endpoint-normalized-training tag=${EXPERIMENT_TAG}"
echo "[corrected] milestones=1000,3000,10000 eval=latest-at-each,best-through-10000 videos=${EVAL_NUM_VIDEOS}"
echo "[corrected] time_sampling=logit_normal endpoint_replay=0.25 grid_replay=0.0 positive_particles=${POSITIVE_PARTICLES}"
echo "[corrected] checkpoints=latest,best wandb_mode=online"

RESULT_ARGS=()
for MILESTONE in "${MILESTONES[@]}"; do
    LATEST_MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-${EXPERIMENT_TAG}-seed${SEED}-latest-endpoint_normalized-step${MILESTONE}.json
    if [[ -s ${LATEST_MARKER} ]]; then
        echo "[corrected] skip_training milestone=${MILESTONE} completed_marker=${LATEST_MARKER}"
    else
        echo "[corrected] start_training target_steps=${MILESTONE}"
        MAX_STEPS=${MILESTONE} \
            bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
        echo "[corrected] complete_training target_steps=${MILESTONE}"
    fi
    evaluate_checkpoint latest "${MILESTONE}"
done

evaluate_checkpoint best 10000

python3 "${REPO_ROOT}/company/summarize_parameterization_eval.py" \
    "${RESULT_ARGS[@]}"
echo "[corrected] status=complete node=${NODE_ROLE} tag=${EXPERIMENT_TAG}"
