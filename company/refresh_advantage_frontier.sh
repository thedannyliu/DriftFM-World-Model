#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^(node-a|node-b|all)$ ]]; then
    echo "Usage: $0 {node-a|node-b|all}" >&2
    exit 2
fi

SCOPE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
PLAN=${REPO_ROOT}/company/advantage_frontier_plan.tsv
LOG_DIR=${RUNTIME_ROOT}/logs/advantage-frontier-refresh
export WANDB_MODE=online
mkdir -p "${LOG_DIR}"

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

refreshed=0
while IFS=$'\t' read -r role name tag seed checkpoint family; do
    if [[ ${role} == \#* ]]; then
        continue
    fi
    if [[ ${SCOPE} != all && ${role} != "${SCOPE}" ]]; then
        continue
    fi
    marker=${ASSET_ROOT}/checkpoints/experiments/eval-${tag}-seed${seed}-${checkpoint}-endpoint_normalized-advantage-locked1000.json
    metrics=${RUNTIME_ROOT}/results/${tag}-seed${seed}-${checkpoint}-endpoint_normalized-advantage-locked1000
    if [[ ! -s ${marker} ]]; then
        echo "[frontier-refresh] skip name=${name} reason=missing-marker"
        continue
    fi
    for nfe in 1 2 4 8; do
        if [[ ! -s ${metrics}/rollout_len-full_nfe-${nfe}.json || ! -s ${metrics}/rollout_len-64_nfe-${nfe}.json ]]; then
            echo "[frontier-refresh] skip name=${name} reason=missing-raw-metrics"
            continue 2
        fi
    done
    wandb_id=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("wandb_run_id", ""))' \
        "${marker}")
    wandb_args=()
    if [[ -n ${wandb_id} ]]; then
        wandb_args+=(--wandb-run-id "${wandb_id}")
    fi
    if [[ -n ${WANDB_ENTITY:-} ]]; then
        wandb_args+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    echo "[frontier-refresh] start name=${name} wandb=${wandb_id:-new}"
    python3 "${REPO_ROOT}/company/summarize_eval.py" \
        --variant-dir "${metrics}" --nfes 1 2 4 8 \
        --output "${marker}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-name "company-rollout-${tag}-${checkpoint}-endpoint_normalized-advantage-locked1000-seed${seed}" \
        "${wandb_args[@]}" \
        >"${LOG_DIR}/${name}.log" 2>&1
    echo "[frontier-refresh] complete name=${name} marker=${marker}"
    refreshed=$((refreshed + 1))
done < "${PLAN}"

echo "[frontier-refresh] status=complete scope=${SCOPE} refreshed=${refreshed}"
python3 "${REPO_ROOT}/company/status_advantage_frontier.py" "${SCOPE}"
