#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ${NODE_ROLE} == node-a ]]; then
    EXPERIMENT_TAG=driftflow-endpointnorm-k1
    POSITIVE_PARTICLES=1
else
    EXPERIMENT_TAG=driftflow-endpointnorm-k16
    POSITIVE_PARTICLES=16
fi

echo "[corrected] node=${NODE_ROLE} hypothesis=endpoint-normalized-training tag=${EXPERIMENT_TAG}"
DRIFTFLOW_POSITIVE_PARTICLES=${POSITIVE_PARTICLES} \
STAGED_MILESTONES="1000 3000 10000" \
    bash "${REPO_ROOT}/company/run_corrected_variant.sh" "${EXPERIMENT_TAG}"
