#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^node-[ab]$ ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
HOST=$(hostname)
SETUP_LOG=${RUNTIME_ROOT}/logs/decision-fidelity/setup-${ROLE}-${HOST}.log
DATASET=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
OFFICIAL=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/pushT_driftworld/ckpt_save/ckpt-step1180500.pth

mkdir -p "$(dirname "${SETUP_LOG}")"

SKIP_ASSETS=0
if [[ -d ${DATASET} && -s ${OFFICIAL} ]]; then
    SKIP_ASSETS=1
elif [[ ${ROLE} == node-b ]]; then
    echo "Shared Push-T assets are incomplete." >&2
    echo "Run the Node A setup first, then retry Node B." >&2
    exit 1
fi

echo "[decision-fidelity-setup] role=${ROLE} host=${HOST} skip_assets=${SKIP_ASSETS}"
DRIFTFLOWWORLD_SETUP_LOG="${SETUP_LOG}" \
DRIFTFLOWWORLD_SKIP_ASSETS="${SKIP_ASSETS}" \
    bash "${REPO_ROOT}/company/setup.sh"

python3 - <<'PY'
import cv2
import importlib.metadata
import json
import pymunk
import torch

assert hasattr(pymunk.Space, "on_collision")
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print(json.dumps({
    "status": "ready",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "opencv": cv2.__version__,
    "opencv_path": cv2.__file__,
    "pymunk": importlib.metadata.version("pymunk"),
    "gpu_count": torch.cuda.device_count(),
}))
PY
echo "setup_log=${SETUP_LOG}"
