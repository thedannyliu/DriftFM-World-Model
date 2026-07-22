#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/storage/project/r-agarg35-0/eliu354/projects/driftworld"
ARTIFACT_ROOT="${DRIFTFLOWWORLD_ROOT:-/storage/scratch1/9/eliu354/driftflowworld}"
ENV_PREFIX="${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1"

mkdir -p "${ARTIFACT_ROOT}/envs" "${ARTIFACT_ROOT}/cache/conda_pkgs" "${ARTIFACT_ROOT}/cache/pip"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    CONDA_PKGS_DIRS="${ARTIFACT_ROOT}/cache/conda_pkgs" \
    PIP_CACHE_DIR="${ARTIFACT_ROOT}/cache/pip" \
    conda env create --prefix "${ENV_PREFIX}" --file "${REPO_ROOT}/driftworld/environment.yml"
fi

"${ENV_PREFIX}/bin/python" -m pip freeze > "${ARTIFACT_ROOT}/envs/pace-cu128-py312-v1-pip-freeze.txt"
"${ENV_PREFIX}/bin/python" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
