#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ! $1 =~ ^node-[ab]$ ]]; then
    echo "Usage: $0 {node-a|node-b} [--print-plan]" >&2
    exit 2
fi
if [[ $# -eq 2 && $2 != --print-plan ]]; then
    echo "Second argument must be --print-plan" >&2
    exit 2
fi

QUEUE_ROLE=$1
PRINT_PLAN=0
if [[ ${2:-} == --print-plan ]]; then
    PRINT_PLAN=1
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
PLAN=${REPO_ROOT}/company/advantage_frontier_plan.tsv
NUM_VIDEOS=${FRONTIER_NUM_VIDEOS:-1000}
RESULT_LABEL=${FRONTIER_RESULT_LABEL:-advantage-locked1000}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}

mapfile -t ROWS < <(awk -F '\t' -v role="${QUEUE_ROLE}" \
    '$1 == role { print $0 }' "${PLAN}")
if (( ${#ROWS[@]} != 8 )); then
    echo "Expected eight checkpoints for ${QUEUE_ROLE}, found ${#ROWS[@]}" >&2
    exit 1
fi

echo "[frontier] queue=${QUEUE_ROLE} checkpoints=${#ROWS[@]} videos=${NUM_VIDEOS} nfes=1,2,4,8"
echo "[frontier] question=does-depth-improve-action-risk-while-perceptual-risk-diverges"
echo "[frontier] retention=no-new-checkpoints resume=checkpoint-level-marker wandb_mode=online"
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family <<< "${row}"
    echo "PLAN name=${name} tag=${tag} seed=${seed} checkpoint=${checkpoint} family=${family}"
done
if (( PRINT_PLAN )); then
    exit 0
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
python3 -c \
    'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[frontier] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

failures=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family <<< "${row}"
    marker=${ASSET_ROOT}/checkpoints/experiments/eval-${tag}-seed${seed}-${checkpoint}-endpoint_normalized-${RESULT_LABEL}.json
    if [[ -s ${marker} ]]; then
        echo "[frontier] skip name=${name} completed_marker=${marker}"
        continue
    fi
    echo "[frontier] start name=${name} family=${family}"
    if SEED=${seed} EVAL_NUM_VIDEOS=${NUM_VIDEOS} EVAL_NFES="1 2 4 8" \
        EVAL_RESULT_LABEL=${RESULT_LABEL} WANDB_LOG_EVAL=1 \
        WANDB_PROJECT=${WANDB_PROJECT} \
        DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized \
        bash "${REPO_ROOT}/company/run_variant_eval.sh" "${tag}" "${checkpoint}"; then
        echo "[frontier] complete name=${name} marker=${marker}"
    else
        echo "[frontier] failed name=${name}; continuing" >&2
        failures+=("${name}")
    fi
done

python3 "${REPO_ROOT}/company/status_advantage_frontier.py" "${QUEUE_ROLE}"
if (( ${#failures[@]} )); then
    echo "[frontier] status=complete_with_failures queue=${QUEUE_ROLE} failures=${failures[*]}"
    exit 1
fi
echo "[frontier] status=complete queue=${QUEUE_ROLE}"
