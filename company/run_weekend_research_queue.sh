#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^node-[abcd]$ ]]; then
    echo "Usage: $0 {node-a|node-b|node-c|node-d} [--print-plan]" >&2
    exit 2
fi
if [[ $# -eq 2 && $2 != --print-plan ]]; then
    echo "Second argument must be --print-plan" >&2
    exit 2
fi

NODE_ROLE=$1
PRINT_PLAN=0
if [[ ${2:-} == --print-plan ]]; then
    PRINT_PLAN=1
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
SCREEN_MILESTONES=${WEEKEND_SCREEN_MILESTONES:-"10000 30000"}
WINNER_MILESTONES=${WEEKEND_WINNER_MILESTONES:-"60000 100000 150000 200000 250000 300000 400000"}
RUNNER_UP_MILESTONES=${WEEKEND_RUNNER_UP_MILESTONES:-"60000 100000 150000 200000"}
CONFIRM_MILESTONES=${WEEKEND_CONFIRM_MILESTONES:-"10000 30000 60000 100000 150000 200000 250000 300000"}
SCREEN_VIDEOS=${WEEKEND_SCREEN_VIDEOS:-25}
FINAL_VIDEOS=${WEEKEND_FINAL_VIDEOS:-100}
export PILOT_PRINT_EVERY=${PILOT_PRINT_EVERY:-100}
export WANDB_MODE=online

declare -a CANDIDATES=()
declare -A PARTICLES ENDPOINT_REPLAY GRID_REPLAY GRID_MAX_NFE
declare -A SOURCE_REPLAY SOURCE_STEPS TIME_SAMPLING LEARNING_RATE

add_candidate() {
    local tag=$1
    CANDIDATES+=("${tag}")
    PARTICLES["${tag}"]=$2
    ENDPOINT_REPLAY["${tag}"]=$3
    GRID_REPLAY["${tag}"]=$4
    GRID_MAX_NFE["${tag}"]=$5
    SOURCE_REPLAY["${tag}"]=$6
    SOURCE_STEPS["${tag}"]=$7
    TIME_SAMPLING["${tag}"]=$8
    LEARNING_RATE["${tag}"]=$9
}

run_variant() {
    local tag=$1
    local seed=$2
    local milestones=$3
    echo "[weekend] run tag=${tag} seed=${seed} milestones=${milestones}"
    SEED=${seed} STAGED_MILESTONES="${milestones}" \
    EVAL_NUM_VIDEOS=${SCREEN_VIDEOS} CORRECTED_EVAL_NFES="1 2 4 8" \
    ROLLOUT_BEST=1 \
    DRIFTFLOW_POSITIVE_PARTICLES=${PARTICLES[${tag}]} \
    DRIFTFLOW_ENDPOINT_REPLAY=${ENDPOINT_REPLAY[${tag}]} \
    DRIFTFLOW_GRID_REPLAY=${GRID_REPLAY[${tag}]} \
    DRIFTFLOW_GRID_MAX_NFE=${GRID_MAX_NFE[${tag}]} \
    DRIFTFLOW_TIME_SAMPLING=${TIME_SAMPLING[${tag}]} \
    DRIFTFLOW_COMPOSED_SOURCE_REPLAY=${SOURCE_REPLAY[${tag}]} \
    DRIFTFLOW_COMPOSED_SOURCE_STEPS=${SOURCE_STEPS[${tag}]} \
    PILOT_LR=${LEARNING_RATE[${tag}]} \
        bash "${REPO_ROOT}/company/run_corrected_variant.sh" "${tag}"
}

run_final_eval() {
    local tag=$1
    local seed=$2
    local target_step=$3
    local kind
    local label=weekend-locked${FINAL_VIDEOS}-step${target_step}
    for kind in latest best; do
        local marker=${ASSET_ROOT}/checkpoints/experiments/eval-${tag}-seed${seed}-${kind}-endpoint_normalized-${label}.json
        if [[ -s ${marker} ]]; then
            echo "[weekend] skip_final_eval tag=${tag} seed=${seed} checkpoint=${kind}"
            continue
        fi
        echo "[weekend] final_eval tag=${tag} seed=${seed} checkpoint=${kind} videos=${FINAL_VIDEOS} nfes=1,2,4,8"
        SEED=${seed} EVAL_NUM_VIDEOS=${FINAL_VIDEOS} EVAL_NFES="1 2 4 8" \
        EVAL_RESULT_LABEL=${label} WANDB_LOG_EVAL=1 \
        DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized \
            bash "${REPO_ROOT}/company/run_variant_eval.sh" "${tag}" "${kind}"
    done
}

add_interaction_candidates() {
    add_candidate wknd-a-base 16 0.25 0.0 4 0.0 2 logit_normal ""
    add_candidate wknd-a-grid25 16 0.25 0.25 4 0.0 2 logit_normal ""
    add_candidate wknd-a-sr25 16 0.25 0.0 4 0.25 2 logit_normal ""
    local grid source grid_tag source_tag
    for grid in 0.125 0.25 0.50; do
        grid_tag=${grid/0./}
        for source in 0.10 0.25 0.50; do
            source_tag=${source/0./}
            add_candidate "wknd-a-g${grid_tag}-sr${source_tag}" \
                16 0.25 "${grid}" 4 "${source}" 2 logit_normal ""
        done
    done
}

add_grid_depth_candidates() {
    local depth probability probability_tag
    for depth in 2 4 8 16; do
        for probability in 0.125 0.25 0.50; do
            probability_tag=${probability/0./}
            add_candidate "wknd-b-grid${depth}-p${probability_tag}" \
                16 0.25 "${probability}" "${depth}" 0.25 2 logit_normal ""
        done
    done
}

add_source_depth_candidates() {
    local steps probability probability_tag
    for steps in 1 2 4 8; do
        for probability in 0.10 0.25 0.50; do
            probability_tag=${probability/0./}
            add_candidate "wknd-c-compose${steps}-sr${probability_tag}" \
                16 0.25 0.25 4 "${probability}" "${steps}" logit_normal ""
        done
    done
}

add_confirmation_candidates() {
    add_candidate wknd-d-base-k16 16 0.25 0.0 4 0.0 2 logit_normal ""
    add_candidate wknd-d-grid-k16 16 0.25 0.25 4 0.0 2 logit_normal ""
    add_candidate wknd-d-source-k16 16 0.25 0.0 4 0.25 2 logit_normal ""
    add_candidate wknd-d-joint-k1 1 0.25 0.25 4 0.25 2 logit_normal ""
    add_candidate wknd-d-joint-k16 16 0.25 0.25 4 0.25 2 logit_normal ""
    add_candidate wknd-d-joint-k32 32 0.25 0.25 4 0.25 2 logit_normal ""
}

case ${NODE_ROLE} in
    node-a)
        QUESTION=grid-source-interaction-surface
        PLANNED_UPDATES=2900000
        add_interaction_candidates
        ;;
    node-b)
        QUESTION=dyadic-grid-depth
        PLANNED_UPDATES=2900000
        add_grid_depth_candidates
        ;;
    node-c)
        QUESTION=composed-source-depth
        PLANNED_UPDATES=2900000
        add_source_depth_candidates
        ;;
    node-d)
        QUESTION=three-seed-causal-confirmation
        PLANNED_UPDATES=5400000
        add_confirmation_candidates
        ;;
esac

echo "[weekend] node=${NODE_ROLE} question=${QUESTION} candidates=${#CANDIDATES[@]} planned_updates=${PLANNED_UPDATES}"
echo "[weekend] screen=${SCREEN_MILESTONES} winner=${WINNER_MILESTONES} runner_up=${RUNNER_UP_MILESTONES} confirm=${CONFIRM_MILESTONES}"
echo "[weekend] retention=latest,rollout-best eval_nfes=1,2,4,8 wandb_mode=online"
if (( PRINT_PLAN )); then
    for TAG in "${CANDIDATES[@]}"; do
        echo "PLAN tag=${TAG} particles=${PARTICLES[${TAG}]} endpoint=${ENDPOINT_REPLAY[${TAG}]} grid=${GRID_REPLAY[${TAG}]} grid_max_nfe=${GRID_MAX_NFE[${TAG}]} source_replay=${SOURCE_REPLAY[${TAG}]} source_steps=${SOURCE_STEPS[${TAG}]} time_sampling=${TIME_SAMPLING[${TAG}]} lr=${LEARNING_RATE[${TAG}]:-default}"
    done
    exit 0
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

"${PYTHON_BIN}" -c \
    'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[weekend] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

FAILURES=()
if [[ ${NODE_ROLE} == node-d ]]; then
    for TAG in "${CANDIDATES[@]}"; do
        for SEED_VALUE in 1 2 3; do
            if run_variant "${TAG}" "${SEED_VALUE}" "${CONFIRM_MILESTONES}"; then
                if ! run_final_eval "${TAG}" "${SEED_VALUE}" 300000; then
                    FAILURES+=("${TAG}-seed${SEED_VALUE}-final-eval")
                fi
            else
                echo "[weekend] confirmation_failed tag=${TAG} seed=${SEED_VALUE}; continuing" >&2
                FAILURES+=("${TAG}-seed${SEED_VALUE}-training")
            fi
        done
    done
else
    SUCCESSFUL_CANDIDATES=()
    for TAG in "${CANDIDATES[@]}"; do
        if run_variant "${TAG}" 1 "${SCREEN_MILESTONES}"; then
            SUCCESSFUL_CANDIDATES+=("${TAG}")
        else
            echo "[weekend] screen_failed tag=${TAG}; continuing" >&2
            FAILURES+=("${TAG}-seed1-screen")
        fi
    done
    if (( ${#SUCCESSFUL_CANDIDATES[@]} < 2 )); then
        echo "[weekend] fewer than two candidates completed the 30k screen" >&2
        exit 1
    fi

    SELECTION_ARGS=()
    for TAG in "${SUCCESSFUL_CANDIDATES[@]}"; do
        MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-${TAG}-seed1-latest-endpoint_normalized-step30000.json
        SELECTION_ARGS+=(--result "${TAG}=${MARKER}")
    done
    SELECTION_FILE=${ASSET_ROOT}/checkpoints/experiments/weekend-selection-${NODE_ROLE}-step30000.json
    "${PYTHON_BIN}" "${REPO_ROOT}/company/select_corrected_variant.py" \
        "${SELECTION_ARGS[@]}" --nfes 1 2 4 8 --output "${SELECTION_FILE}"
    mapfile -t SELECTED_TAGS < <("${PYTHON_BIN}" -c \
        'import json, sys; data=json.load(open(sys.argv[1])); print("\n".join(item["name"] for item in data["candidates"][:2]))' \
        "${SELECTION_FILE}")
    WINNER=${SELECTED_TAGS[0]}
    RUNNER_UP=${SELECTED_TAGS[1]}
    echo "[weekend] winner=${WINNER} runner_up=${RUNNER_UP} selection=${SELECTION_FILE}"

    if run_variant "${WINNER}" 1 "${WINNER_MILESTONES}"; then
        run_final_eval "${WINNER}" 1 400000 || FAILURES+=("${WINNER}-seed1-final-eval")
    else
        FAILURES+=("${WINNER}-seed1-extension")
    fi
    for SEED_VALUE in 2 3 4 5; do
        if run_variant "${WINNER}" "${SEED_VALUE}" "${SCREEN_MILESTONES} ${WINNER_MILESTONES}"; then
            run_final_eval "${WINNER}" "${SEED_VALUE}" 400000 \
                || FAILURES+=("${WINNER}-seed${SEED_VALUE}-final-eval")
        else
            FAILURES+=("${WINNER}-seed${SEED_VALUE}-extension")
        fi
    done

    if run_variant "${RUNNER_UP}" 1 "${RUNNER_UP_MILESTONES}"; then
        run_final_eval "${RUNNER_UP}" 1 200000 || FAILURES+=("${RUNNER_UP}-seed1-final-eval")
    else
        FAILURES+=("${RUNNER_UP}-seed1-extension")
    fi
    for SEED_VALUE in 2 3; do
        if run_variant "${RUNNER_UP}" "${SEED_VALUE}" "${SCREEN_MILESTONES} ${RUNNER_UP_MILESTONES}"; then
            run_final_eval "${RUNNER_UP}" "${SEED_VALUE}" 200000 \
                || FAILURES+=("${RUNNER_UP}-seed${SEED_VALUE}-final-eval")
        else
            FAILURES+=("${RUNNER_UP}-seed${SEED_VALUE}-extension")
        fi
    done
fi

if (( ${#FAILURES[@]} )); then
    echo "[weekend] status=complete_with_failures node=${NODE_ROLE} failures=${FAILURES[*]}"
else
    echo "[weekend] status=complete node=${NODE_ROLE}"
fi
