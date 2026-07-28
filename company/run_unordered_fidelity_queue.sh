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
PRINT_PLAN=0
if [[ ${2:-} == --print-plan ]]; then
    PRINT_PLAN=1
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PLAN=${REPO_ROOT}/company/unordered_fidelity_plan.tsv
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-unordered-fidelity-company}
export WANDB_MODE=online

mapfile -t ROWS < <(awk -F '\t' -v role="${QUEUE_ROLE}" \
    '$1 == role { print $0 }' "${PLAN}")
if (( ${#ROWS[@]} != 16 )); then
    echo "Expected 16 rows for ${QUEUE_ROLE}, found ${#ROWS[@]}" >&2
    exit 1
fi

echo "[unordered-fidelity] queue=${QUEUE_ROLE} rows=${#ROWS[@]} trials_per_row=${GPC_NUM_TRIALS:-20}"
echo "[unordered-fidelity] checkpoints=frozen retention=no-new-checkpoints resume=shared-row-markers wandb=${WANDB_PROJECT}"
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "PLAN name=${name} family=${family} policy=${policy} strategy=${strategy} proposals=${proposals} nfe=${nfe} refine=${refine_nfe}@${refine_ratio} question=${question}"
done
if (( PRINT_PLAN )); then
    exit 0
fi

if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
python3 -c \
    'import cv2, importlib.metadata, pymunk, torch, wandb; assert hasattr(pymunk.Space, "on_collision"), "Pymunk 7 API is required"; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); pymunk_version = importlib.metadata.version("pymunk"); print(f"[unordered-fidelity] preflight=pass torch={torch.__version__} opencv={cv2.__version__} cv2={cv2.__file__} pymunk={pymunk_version} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

mapfile -t QUEUE_POLICIES < <(awk -F '\t' -v role="${QUEUE_ROLE}" \
    '$1 == role { print $7 }' "${PLAN}" | sort -u)
for POLICY_NAME in "${QUEUE_POLICIES[@]}"; do
    POLICY_PATH=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-${POLICY_NAME}.pth
    POLICY_PATH=${POLICY_PATH} python3 - <<'PY'
import os
import torch

path = os.environ["POLICY_PATH"]
if not os.path.isfile(path):
    raise FileNotFoundError(f"Monolithic policy checkpoint not found: {path}")
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
missing = {"model", "ema"} - set(checkpoint)
if missing:
    raise RuntimeError(f"Missing policy checkpoint keys {sorted(missing)}: {path}")
print(
    f"[unordered-fidelity] policy_preflight=pass path={path} "
    f"step={checkpoint.get('step', 'unknown')}"
)
PY
done

failures=()
for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r role name tag seed checkpoint family policy strategy proposals nfe refine_nfe refine_ratio question <<< "${row}"
    echo "[unordered-fidelity] start name=${name}"
    if WANDB_PROJECT=${WANDB_PROJECT} \
        bash "${REPO_ROOT}/company/run_gpc_budget_eval.sh" \
        "${name}" "${tag}" "${seed}" "${checkpoint}" "${family}" \
        "${policy}" "${strategy}" "${proposals}" "${nfe}" \
        "${refine_nfe}" "${refine_ratio}"; then
        echo "[unordered-fidelity] row_complete name=${name}"
    else
        echo "[unordered-fidelity] failed name=${name}; continuing" >&2
        failures+=("${name}")
    fi
done

python3 "${REPO_ROOT}/company/status_unordered_fidelity.py" "${QUEUE_ROLE}"
if (( ${#failures[@]} )); then
    echo "[unordered-fidelity] status=complete_with_failures queue=${QUEUE_ROLE} failures=${failures[*]}"
    exit 1
fi
echo "[unordered-fidelity] status=complete queue=${QUEUE_ROLE}"
