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
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${REPO_ROOT}/company/unordered_confirmation_plan.tsv
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-unordered-fidelity-confirmation-company}

mapfile -t ROWS < <(awk -F '\t' -v role="${QUEUE_ROLE}" \
    '$1 == role { print $0 }' "${PLAN}")
if (( ${#ROWS[@]} != 8 )); then
    echo "Expected 8 rows for ${QUEUE_ROLE}, found ${#ROWS[@]}" >&2
    exit 1
fi

echo "[unordered-confirmation] queue=${QUEUE_ROLE} rows=8 trials=80 range=20:100"
echo "[unordered-confirmation] ep100=primary ep300=policy-replication"
echo "[unordered-confirmation] wandb=${WANDB_PROJECT} resume=shared-row-markers"
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "PLAN name=${name} family=${family} policy=${policy} proposals=${proposals} nfe=${nfe} question=${question}"
done
if [[ ${2:-} == --print-plan ]]; then
    exit 0
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
python3 -c \
    'import cv2, importlib.metadata, pymunk, torch, wandb; assert hasattr(pymunk.Space, "on_collision"), "Pymunk 7 API is required"; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); pymunk_version = importlib.metadata.version("pymunk"); print(f"[unordered-confirmation] preflight=pass torch={torch.__version__} opencv={cv2.__version__} pymunk={pymunk_version} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

failures=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "[unordered-confirmation] start name=${name}"
    if GPC_NUM_TRIALS=80 \
        GPC_TRIAL_OFFSET=20 \
        WANDB_PROJECT="${WANDB_PROJECT}" \
        bash "${REPO_ROOT}/company/run_gpc_budget_eval.sh" \
        "${name}" "${tag}" "${seed}" "${checkpoint}" "${family}" \
        "${policy}" "${strategy}" "${proposals}" "${nfe}" \
        "${refine_nfe}" "${refine_ratio}"; then
        echo "[unordered-confirmation] row_complete name=${name}"
    else
        echo "[unordered-confirmation] failed name=${name}; continuing" >&2
        failures+=("${name}")
    fi
done

python3 "${REPO_ROOT}/company/status_unordered_confirmation.py" "${QUEUE_ROLE}"
if (( ${#failures[@]} )); then
    echo "[unordered-confirmation] status=complete_with_failures queue=${QUEUE_ROLE} failures=${failures[*]}"
    exit 1
fi
echo "[unordered-confirmation] status=complete queue=${QUEUE_ROLE}"
