#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi
NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MAX_STEPS=${TRANSPORT_REPAIR_STEPS:-10000}
export WANDB_MODE=online

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi

if [[ ${NODE_ROLE} == node-a ]]; then
    TAG=driftflow-kpos16
    echo "[repair] hypothesis=intermediate-positive-marginal tag=${TAG}"
    EXPERIMENT_TAG=${TAG} DRIFTFLOW_POSITIVE_PARTICLES=16 \
        SEED=1 MAX_STEPS=${MAX_STEPS} \
        bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
else
    TAG=driftflow-grid25
    echo "[repair] hypothesis=inference-grid-coverage tag=${TAG}"
    EXPERIMENT_TAG=${TAG} DRIFTFLOW_GRID_REPLAY=0.25 \
        SEED=1 MAX_STEPS=${MAX_STEPS} \
        bash "${REPO_ROOT}/company/run_pilot.sh" driftflow
fi

for CHECKPOINT_KIND in latest best; do
    EVAL_NUM_VIDEOS=${EVAL_NUM_VIDEOS:-25} \
        bash "${REPO_ROOT}/company/run_variant_eval.sh" "${TAG}" "${CHECKPOINT_KIND}"
done
echo "[repair] status=complete node=${NODE_ROLE} tag=${TAG}"
