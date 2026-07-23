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
export DRIFTFLOW_TRANSPORT_PARAMETERIZATION=endpoint_normalized
export WANDB_LOG_EVAL=1
export WANDB_MODE=online

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

if [[ ${NODE_ROLE} == node-a ]]; then
    TASKS=(
        "pushT_driftflow_posttrain latest"
        "pushT_driftflow_posttrain best"
        "driftflow-uniform best"
        "driftflow-kpos16 best"
    )
else
    TASKS=(
        "driftflow-replay50 latest"
        "driftflow-replay50 best"
        "driftflow-grid25 latest"
        "driftflow-grid25 best"
    )
fi

RESULT_ARGS=()
for TASK in "${TASKS[@]}"; do
    read -r TAG KIND <<<"${TASK}"
    echo "[endpoint-normalized] start tag=${TAG} checkpoint=${KIND}"
    set +e
    bash "${REPO_ROOT}/company/run_variant_eval.sh" "${TAG}" "${KIND}" \
        | awk '!/^[{].*[}]$/ { print; fflush() }'
    PIPE_STATUSES=("${PIPESTATUS[@]}")
    STATUS=${PIPE_STATUSES[0]}
    set -e
    if (( STATUS != 0 )); then
        echo "[endpoint-normalized] failed tag=${TAG} checkpoint=${KIND}" >&2
        exit "${STATUS}"
    fi
    MARKER=${ASSET_ROOT}/checkpoints/experiments/eval-${TAG}-seed${SEED}-${KIND}-endpoint_normalized.json
    RESULT_ARGS+=(--result "${TAG}-${KIND}=${MARKER}")
    echo "[endpoint-normalized] complete tag=${TAG} checkpoint=${KIND}"
done

python3 "${REPO_ROOT}/company/summarize_parameterization_eval.py" \
    "${RESULT_ARGS[@]}"
echo "[endpoint-normalized] status=complete node=${NODE_ROLE}"
