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

ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${REPO_ROOT}/company/decision_fidelity_pusht_plan.tsv
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-decision-fidelity-pusht-company}

mapfile -t ROWS < <(awk -F '\t' -v role="${ROLE}" \
    '$1 == role { print $0 }' "${PLAN}")
EXPECTED=8
if [[ ${ROLE} == node-b ]]; then
    EXPECTED=4
fi
if (( ${#ROWS[@]} != EXPECTED )); then
    echo "Expected ${EXPECTED} rows for ${ROLE}, found ${#ROWS[@]}" >&2
    exit 1
fi

echo "[decision-fidelity] queue=${ROLE} rows=${EXPECTED} wandb=${WANDB_PROJECT}"
if [[ ${ROLE} == node-a ]]; then
    echo "[decision-fidelity] stage=N0 trials=80 range=20:100 anchor=32"
else
    echo "[decision-fidelity] stage=N1-smoke trials=16 range=100:116 execution=policy_first audits=4x(15+hold16)"
fi
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
    'import cv2, importlib.metadata, pymunk, torch, wandb; assert hasattr(pymunk.Space, "on_collision"), "Pymunk 7 API is required"; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[decision-fidelity] preflight=pass torch={torch.__version__} opencv={cv2.__version__} pymunk={importlib.metadata.version(\"pymunk\")} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

failures=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "[decision-fidelity] start name=${name}"
    COMMON_ENV=(
        WANDB_PROJECT="${WANDB_PROJECT}"
    )
    if [[ ${ROLE} == node-a ]]; then
        ROLE_ENV=(
            GPC_NUM_TRIALS=80
            GPC_TRIAL_OFFSET=20
            GPC_CANDIDATE_ANCHOR_COUNT=32
            GPC_AUDIT_CANDIDATES=false
            GPC_AUDIT_MAX_DECISIONS=1
        )
    else
        ROLE_ENV=(
            GPC_NUM_TRIALS=16
            GPC_TRIAL_OFFSET=100
            GPC_EXECUTION_STRATEGY=policy_first
            GPC_AUDIT_CANDIDATES=true
            GPC_AUDIT_MAX_DECISIONS=4
            GPC_AUDIT_CANDIDATE_STEPS=15
            GPC_AUDIT_HOLD_STEPS=16
            GPC_AUDIT_REPEAT_GROUND_TRUTH=true
        )
    fi
    if env "${COMMON_ENV[@]}" "${ROLE_ENV[@]}" \
        bash "${REPO_ROOT}/company/run_gpc_budget_eval.sh" \
        "${name}" "${tag}" "${seed}" "${checkpoint}" "${family}" \
        "${policy}" "${strategy}" "${proposals}" "${nfe}" \
        "${refine_nfe}" "${refine_ratio}"; then
        echo "[decision-fidelity] row_complete name=${name}"
    else
        echo "[decision-fidelity] failed name=${name}; continuing" >&2
        failures+=("${name}")
    fi
done

python3 "${REPO_ROOT}/company/status_decision_fidelity_pusht.py" "${ROLE}"
if (( ${#failures[@]} )); then
    echo "[decision-fidelity] status=complete_with_failures queue=${ROLE} failures=${failures[*]}"
    exit 1
fi
echo "[decision-fidelity] status=complete queue=${ROLE}"
