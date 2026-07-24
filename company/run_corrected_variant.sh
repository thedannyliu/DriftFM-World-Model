#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 EXPERIMENT_TAG" >&2
    exit 2
fi

EXPERIMENT_TAG=$1
if [[ ! ${EXPERIMENT_TAG} =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "EXPERIMENT_TAG contains unsupported characters" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
SEED=${SEED:-1}
EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25}
read -r -a MILESTONES <<< "${STAGED_MILESTONES:-1000 3000 10000}"

if (( ${#MILESTONES[@]} < 1 )); then
    echo "STAGED_MILESTONES must not be empty" >&2
    exit 2
fi
PREVIOUS=0
for MILESTONE in "${MILESTONES[@]}"; do
    if [[ ! ${MILESTONE} =~ ^[1-9][0-9]*$ ]] || (( MILESTONE <= PREVIOUS )); then
        echo "STAGED_MILESTONES must be strictly increasing positive integers" >&2
        exit 2
    fi
    PREVIOUS=${MILESTONE}
done
FINAL_MILESTONE=${MILESTONES[$((${#MILESTONES[@]} - 1))]}

export DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized
export DRIFTFLOW_TIME_SAMPLING=${DRIFTFLOW_TIME_SAMPLING:-logit_normal}
export DRIFTFLOW_ENDPOINT_REPLAY=${DRIFTFLOW_ENDPOINT_REPLAY:-0.25}
export DRIFTFLOW_GRID_REPLAY=${DRIFTFLOW_GRID_REPLAY:-0.0}
export DRIFTFLOW_POSITIVE_PARTICLES=${DRIFTFLOW_POSITIVE_PARTICLES:-1}
export DRIFTFLOW_COMPOSED_SOURCE_REPLAY=${DRIFTFLOW_COMPOSED_SOURCE_REPLAY:-0.0}
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
        EVAL_NFES="1 2 4" EVAL_RESULT_LABEL=step${milestone} \
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

echo "[corrected] tag=${EXPERIMENT_TAG} seed=${SEED} milestones=${MILESTONES[*]}"
echo "[corrected] time_sampling=${DRIFTFLOW_TIME_SAMPLING} endpoint_replay=${DRIFTFLOW_ENDPOINT_REPLAY} grid_replay=${DRIFTFLOW_GRID_REPLAY} positive_particles=${DRIFTFLOW_POSITIVE_PARTICLES} composed_source_replay=${DRIFTFLOW_COMPOSED_SOURCE_REPLAY}"
echo "[corrected] eval=latest-at-each,best-through-${FINAL_MILESTONE} videos=${EVAL_NUM_VIDEOS} checkpoints=latest,best wandb_mode=online"

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

evaluate_checkpoint best "${FINAL_MILESTONE}"

python3 "${REPO_ROOT}/company/summarize_parameterization_eval.py" \
    "${RESULT_ARGS[@]}"
echo "[corrected] status=complete tag=${EXPERIMENT_TAG} seed=${SEED}"
