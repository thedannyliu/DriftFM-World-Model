#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
DOWNLOAD_WORKERS=${DRIFTFLOWWORLD_DOWNLOAD_WORKERS:-8}
SKIP_ASSETS=${DRIFTFLOWWORLD_SKIP_ASSETS:-0}
SETUP_LOG=${RUNTIME_ROOT}/logs/setup.log
export PIP_CACHE_DIR=${RUNTIME_ROOT}/cache/pip
export XDG_CACHE_HOME=${RUNTIME_ROOT}/cache
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}

mkdir -p "${ASSET_ROOT}" "${RUNTIME_ROOT}/logs" "${RUNTIME_ROOT}/results" \
    "${PIP_CACHE_DIR}"
echo "[1/4] Checking NGC Python 3.10 / PyTorch 2.4 image"
"${PYTHON_BIN}" -c 'import sys, torch, torchvision; assert sys.version_info[:2] == (3, 10), sys.version; assert torch.__version__.split("+")[0].startswith("2.4."), torch.__version__'

echo "[2/4] Installing Python dependencies into the active container (log: ${SETUP_LOG})"
if ! "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/company/requirements.txt" \
    2>&1 | tee "${SETUP_LOG}"; then
    echo "Environment installation failed; full log: ${SETUP_LOG}" >&2
    exit 1
fi

export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
if [[ ${SKIP_ASSETS} == 1 ]]; then
    echo "[3/4] Skipping asset preparation (DRIFTFLOWWORLD_SKIP_ASSETS=1)"
else
    echo "[3/4] Downloading/resuming Hugging Face assets under ${ASSET_ROOT}"
    if ! "${PYTHON_BIN}" -u "${REPO_ROOT}/company/prepare_assets.py" \
        --asset-root "${ASSET_ROOT}" --max-workers "${DOWNLOAD_WORKERS}" \
        2>&1 | tee -a "${SETUP_LOG}"; then
        echo "Asset preparation failed; full log: ${SETUP_LOG}" >&2
        exit 1
    fi
fi

echo "[4/4] Verifying active container"
"${PYTHON_BIN}" -c \
    'import hydra, huggingface_hub, json, numpy, omegaconf, torch, torch.distributed.run, torchvision, wandb, zarr; print(json.dumps({"status":"ready","torch":torch.__version__,"torchvision":torchvision.__version__,"cuda":torch.version.cuda,"numpy":numpy.__version__,"zarr":zarr.__version__,"omegaconf":omegaconf.__version__,"huggingface_hub":huggingface_hub.__version__,"distributed_launcher":True,"gpu_count":torch.cuda.device_count()}))'
echo "setup_log=${SETUP_LOG}"
