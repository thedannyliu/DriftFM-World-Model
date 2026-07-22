#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
ENV_PREFIX=${DRIFTFLOWWORLD_ENV_PREFIX:-${RUNTIME_ROOT}/envs/driftfm-py312}
PYTHON_BIN=${PYTHON_BIN:-python3}
SETUP_LOG=${RUNTIME_ROOT}/logs/setup.log
export PIP_CACHE_DIR=${RUNTIME_ROOT}/cache/pip
export XDG_CACHE_HOME=${RUNTIME_ROOT}/cache

mkdir -p "${ASSET_ROOT}" "${RUNTIME_ROOT}/logs" "${RUNTIME_ROOT}/results" \
    "${PIP_CACHE_DIR}"
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${ENV_PREFIX}"
fi

if ! "${ENV_PREFIX}/bin/python" -m pip install -r "${REPO_ROOT}/company/requirements.txt" \
    >"${SETUP_LOG}" 2>&1; then
    echo "Environment installation failed; last 40 log lines:" >&2
    tail -n 40 "${SETUP_LOG}" >&2
    exit 1
fi

export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export PYTHONNOUSERSITE=1
if ! "${ENV_PREFIX}/bin/python" "${REPO_ROOT}/company/prepare_assets.py" \
    --asset-root "${ASSET_ROOT}" >>"${SETUP_LOG}" 2>&1; then
    echo "Asset preparation failed; last 40 log lines:" >&2
    tail -n 40 "${SETUP_LOG}" >&2
    exit 1
fi

"${ENV_PREFIX}/bin/python" -c \
    'import json, torch; print(json.dumps({"status":"ready","torch":torch.__version__,"cuda":torch.version.cuda,"gpu_count":torch.cuda.device_count()}))'
echo "setup_log=${SETUP_LOG}"
