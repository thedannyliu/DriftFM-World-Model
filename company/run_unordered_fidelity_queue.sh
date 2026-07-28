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

QUEUE_ROLE=$1
PRINT_PLAN=0
if [[ ${2:-} == --print-plan ]]; then
    PRINT_PLAN=1
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${REPO_ROOT}/company/unordered_fidelity_plan.tsv
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-unordered-fidelity-company}
export WANDB_MODE=online

mapfile -t ROWS < <(awk -F '\t' -v role="${QUEUE_ROLE}" \
    '$1 == role { print $0 }' "${PLAN}")
if (( ${#ROWS[@]} != 16 )); then
    echo "Expected 16 rows for ${QUEUE_ROLE}, found ${#ROWS[@]}" >&2
    exit 1
fi

echo "[unordered-fidelity] queue=${QUEUE_ROLE} rows=${#ROWS[@]} trials_per_row=${GPC_NUM_TRIALS:-20}"
echo "[unordered-fidelity] checkpoints=frozen retention=no-new-checkpoints resume=shared-row-markers wandb=${WANDB_PROJECT}"
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "PLAN name=${name} family=${family} policy=${policy} strategy=${strategy} proposals=${proposals} nfe=${nfe} refine=${refine_nfe}@${refine_ratio} question=${question}"
done
if (( PRINT_PLAN )); then
    exit 0
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
python3 -c \
    'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[unordered-fidelity] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

failures=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "[unordered-fidelity] start name=${name}"
    if WANDB_PROJECT=${WANDB_PROJECT} \
        bash "${REPO_ROOT}/company/run_gpc_budget_eval.sh" \
        "${name}" "${tag}" "${seed}" "${checkpoint}" "${family}" \
        "${policy}" "${strategy}" "${proposals}" "${nfe}" \
        "${refine_nfe}" "${refine_ratio}"; then
        echo "[unordered-fidelity] row_complete name=${name}"
    else
        echo "[unordered-fidelity] failed name=${name}; continuing" >&2
        failures+=("${name}")
    fi
done

python3 "${REPO_ROOT}/company/status_unordered_fidelity.py" "${QUEUE_ROLE}"
if (( ${#failures[@]} )); then
    echo "[unordered-fidelity] status=complete_with_failures queue=${QUEUE_ROLE} failures=${failures[*]}"
    exit 1
fi
echo "[unordered-fidelity] status=complete queue=${QUEUE_ROLE}"
