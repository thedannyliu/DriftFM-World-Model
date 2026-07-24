#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^node-[abcd]$ ]]; then
    echo "Usage: $0 {node-a|node-b|node-c|node-d}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
SCREEN_MILESTONES=${LONG_QUEUE_SCREEN_MILESTONES:-"1000 3000 10000"}
LONG_MILESTONES=${LONG_QUEUE_LONG_MILESTONES:-"30000 60000 100000"}
SCREEN_VIDEOS=${LONG_QUEUE_SCREEN_VIDEOS:-25}
FINAL_VIDEOS=${LONG_QUEUE_FINAL_VIDEOS:-100}
export PILOT_PRINT_EVERY=${PILOT_PRINT_EVERY:-100}
export WANDB_MODE=online

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

"${PYTHON_BIN}" -c \
    'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[long-queue] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

declare -a CANDIDATES=()
declare -A PARTICLES ENDPOINT_REPLAY GRID_REPLAY TIME_SAMPLING
declare -A SOURCE_REPLAY LEARNING_RATE WARMUP_STEPS
INITIAL_CORRECTED=

add_candidate() {
    local tag=$1
    CANDIDATES+=("${tag}")
    PARTICLES["${tag}"]=$2
    ENDPOINT_REPLAY["${tag}"]=$3
    GRID_REPLAY["${tag}"]=$4
    TIME_SAMPLING["${tag}"]=$5
    SOURCE_REPLAY["${tag}"]=$6
    LEARNING_RATE["${tag}"]=$7
    WARMUP_STEPS["${tag}"]=$8
}

run_variant() {
    local tag=$1
    local seed=$2
    local milestones=$3
    local endpoint_replay=$4
    echo "[long-queue] run tag=${tag} seed=${seed} milestones=${milestones} endpoint_replay=${endpoint_replay}"
    SEED=${seed} STAGED_MILESTONES="${milestones}" \
    EVAL_NUM_VIDEOS=${SCREEN_VIDEOS} \
    DRIFTFLOW_POSITIVE_PARTICLES=${PARTICLES[${tag}]} \
    DRIFTFLOW_ENDPOINT_REPLAY=${endpoint_replay} \
    DRIFTFLOW_GRID_REPLAY=${GRID_REPLAY[${tag}]} \
    DRIFTFLOW_TIME_SAMPLING=${TIME_SAMPLING[${tag}]} \
    DRIFTFLOW_COMPOSED_SOURCE_REPLAY=${SOURCE_REPLAY[${tag}]} \
    PILOT_LR=${LEARNING_RATE[${tag}]} \
        bash "${REPO_ROOT}/company/run_corrected_variant.sh" "${tag}"
}

run_screen() {
    local tag=$1
    local seed=$2
    local warmup=${WARMUP_STEPS[${tag}]}
    if (( warmup == 1000 )); then
        run_variant "${tag}" "${seed}" "1000" 1.0 || return $?
        run_variant "${tag}" "${seed}" "3000 10000" "${ENDPOINT_REPLAY[${tag}]}" || return $?
    elif (( warmup == 3000 )); then
        run_variant "${tag}" "${seed}" "1000 3000" 1.0 || return $?
        run_variant "${tag}" "${seed}" "10000" "${ENDPOINT_REPLAY[${tag}]}" || return $?
    else
        run_variant "${tag}" "${seed}" "${SCREEN_MILESTONES}" "${ENDPOINT_REPLAY[${tag}]}" || return $?
    fi
}

run_long_extension() {
    local tag=$1
    local seed=$2
    run_variant "${tag}" "${seed}" "${LONG_MILESTONES}" "${ENDPOINT_REPLAY[${tag}]}"
}

run_final_eval() {
    local tag=$1
    local seed=$2
    local kind
    for kind in latest best; do
        local label=locked${FINAL_VIDEOS}-step100000
        local marker=${ASSET_ROOT}/checkpoints/experiments/eval-${tag}-seed${seed}-${kind}-endpoint_normalized-${label}.json
        if [[ -s ${marker} ]]; then
            echo "[long-queue] skip_final_eval tag=${tag} seed=${seed} checkpoint=${kind}"
            continue
        fi
        echo "[long-queue] final_eval tag=${tag} seed=${seed} checkpoint=${kind} videos=${FINAL_VIDEOS} nfes=1,2,4,8"
        SEED=${seed} EVAL_NUM_VIDEOS=${FINAL_VIDEOS} EVAL_NFES="1 2 4 8" \
        EVAL_RESULT_LABEL=${label} WANDB_LOG_EVAL=1 \
        DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized \
            bash "${REPO_ROOT}/company/run_variant_eval.sh" "${tag}" "${kind}" \
            || return $?
    done
}

case ${NODE_ROLE} in
    node-a)
        QUESTION=time-pair-curriculum-k1
        PLANNED_UPDATES=340000
        INITIAL_CORRECTED=node-a
        add_candidate driftflow-endpointnorm-k1 1 0.25 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k1-noreplay 1 0.0 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k1-replay50 1 0.5 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k1-grid25 1 0.25 0.25 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k1-uniform 1 0.25 0.0 uniform 0.0 "" 0
        ;;
    node-b)
        QUESTION=positive-particle-scaling
        PLANNED_UPDATES=340000
        INITIAL_CORRECTED=node-b
        add_candidate driftflow-endpointnorm-k16 16 0.25 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k2 2 0.25 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k4 4 0.25 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k8 8 0.25 0.0 logit_normal 0.0 "" 0
        add_candidate driftflow-endpointnorm-k32 32 0.25 0.0 logit_normal 0.0 "" 0
        ;;
    node-c)
        QUESTION=composed-source-replay
        PLANNED_UPDATES=350000
        add_candidate driftflow-endpointnorm-k1-sr10 1 0.25 0.0 logit_normal 0.10 "" 0
        add_candidate driftflow-endpointnorm-k1-sr25 1 0.25 0.0 logit_normal 0.25 "" 0
        add_candidate driftflow-endpointnorm-k16-sr10 16 0.25 0.0 logit_normal 0.10 "" 0
        add_candidate driftflow-endpointnorm-k16-sr25 16 0.25 0.0 logit_normal 0.25 "" 0
        add_candidate driftflow-endpointnorm-k16-sr50 16 0.25 0.0 logit_normal 0.50 "" 0
        add_candidate driftflow-endpointnorm-k16-grid25-sr25 16 0.25 0.25 logit_normal 0.25 "" 0
        ;;
    node-d)
        QUESTION=endpoint-warmup-and-optimization
        PLANNED_UPDATES=340000
        add_candidate driftflow-endpointnorm-k16-warmup1k 16 0.25 0.0 logit_normal 0.0 "" 1000
        add_candidate driftflow-endpointnorm-k16-warmup3k 16 0.25 0.0 logit_normal 0.0 "" 3000
        add_candidate driftflow-endpointnorm-k16-lrhalf 16 0.25 0.0 logit_normal 0.0 0.00000125 0
        add_candidate driftflow-endpointnorm-k16-lrdouble 16 0.25 0.0 logit_normal 0.0 0.000005 0
        add_candidate driftflow-endpointnorm-k16-grid50 16 0.25 0.50 logit_normal 0.0 "" 0
        ;;
esac

echo "[long-queue] node=${NODE_ROLE} question=${QUESTION} candidates=${#CANDIDATES[@]} planned_updates=${PLANNED_UPDATES}"
echo "[long-queue] screen=${SCREEN_MILESTONES} long=${LONG_MILESTONES} final_videos=${FINAL_VIDEOS} print_every=${PILOT_PRINT_EVERY}"

if [[ -n ${INITIAL_CORRECTED} ]]; then
    if ! bash "${REPO_ROOT}/company/run_corrected_training.sh" "${INITIAL_CORRECTED}"; then
        echo "[long-queue] initial_corrected_failed=${INITIAL_CORRECTED}; retrying in candidate sweep" >&2
    fi
fi

SUCCESSFUL_CANDIDATES=()
FAILURES=()
for TAG in "${CANDIDATES[@]}"; do
    if run_screen "${TAG}" 1; then
        SUCCESSFUL_CANDIDATES+=("${TAG}")
    else
        echo "[long-queue] screen_failed tag=${TAG} seed=1; continuing" >&2
        FAILURES+=("${TAG}-seed1-screen")
    fi
done

if (( ${#SUCCESSFUL_CANDIDATES[@]} == 0 )); then
    echo "[long-queue] no candidate completed the 10k screen" >&2
    exit 1
fi

SELECTION_ARGS=()
for TAG in "${SUCCESSFUL_CANDIDATES[@]}"; do
    MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-${TAG}-seed1-latest-endpoint_normalized-step10000.json
    SELECTION_ARGS+=(--result "${TAG}=${MARKER}")
done
SELECTION_FILE=${ASSET_ROOT}/checkpoints/experiments/selection-${NODE_ROLE}-step10000.json
"${PYTHON_BIN}" "${REPO_ROOT}/company/select_corrected_variant.py" \
    "${SELECTION_ARGS[@]}" --output "${SELECTION_FILE}"
SELECTED_TAG=$("${PYTHON_BIN}" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["selected"])' \
    "${SELECTION_FILE}")
echo "[long-queue] selected=${SELECTED_TAG} selection=${SELECTION_FILE}"

ACTIVE_SEEDS=(1)
for SEED_VALUE in 2 3; do
    if run_screen "${SELECTED_TAG}" "${SEED_VALUE}"; then
        ACTIVE_SEEDS+=("${SEED_VALUE}")
    else
        echo "[long-queue] replication_failed tag=${SELECTED_TAG} seed=${SEED_VALUE}; continuing" >&2
        FAILURES+=("${SELECTED_TAG}-seed${SEED_VALUE}-screen")
    fi
done

for SEED_VALUE in "${ACTIVE_SEEDS[@]}"; do
    if run_long_extension "${SELECTED_TAG}" "${SEED_VALUE}"; then
        if ! run_final_eval "${SELECTED_TAG}" "${SEED_VALUE}"; then
            echo "[long-queue] final_eval_failed tag=${SELECTED_TAG} seed=${SEED_VALUE}; continuing" >&2
            FAILURES+=("${SELECTED_TAG}-seed${SEED_VALUE}-final-eval")
        fi
    else
        echo "[long-queue] extension_failed tag=${SELECTED_TAG} seed=${SEED_VALUE}; continuing" >&2
        FAILURES+=("${SELECTED_TAG}-seed${SEED_VALUE}-extension")
    fi
done

if (( ${#FAILURES[@]} )); then
    echo "[long-queue] status=complete_with_failures node=${NODE_ROLE} selected=${SELECTED_TAG} failures=${FAILURES[*]}"
else
    echo "[long-queue] status=complete node=${NODE_ROLE} selected=${SELECTED_TAG}"
fi
