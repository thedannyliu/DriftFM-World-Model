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
PRINT_PLAN=${2:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
if [[ ${QUEUE_ROLE} == node-a ]]; then
    AUDIT_ROLE=node-b
else
    AUDIT_ROLE=node-d
fi

echo "[followup] queue=${QUEUE_ROLE} missing_audit=${AUDIT_ROLE}"
if [[ ${PRINT_PLAN} == --print-plan ]]; then
    bash "${REPO_ROOT}/company/run_hypothesis_audit.sh" "${AUDIT_ROLE}" --print-plan
    bash "${REPO_ROOT}/company/run_advantage_frontier_queue.sh" "${QUEUE_ROLE}" --print-plan
    exit 0
fi

mapfile -t COMPLETED_AUDITS < <(
    find "${RUNTIME_ROOT}/results/hypothesis-audit" -maxdepth 2 \
        -path "*/${AUDIT_ROLE}-*/summary.json" -type f -size +0c \
        2>/dev/null | sort
)
if (( ${#COMPLETED_AUDITS[@]} )); then
    echo "[followup] skip_audit role=${AUDIT_ROLE} summary=${COMPLETED_AUDITS[-1]}"
else
    bash "${REPO_ROOT}/company/run_hypothesis_audit.sh" "${AUDIT_ROLE}"
fi

python3 "${REPO_ROOT}/company/status_hypothesis_audit.py"
bash "${REPO_ROOT}/company/run_advantage_frontier_queue.sh" "${QUEUE_ROLE}"
echo "[followup] status=complete queue=${QUEUE_ROLE}"
