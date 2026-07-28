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

NODE_ROLE=$1
PRINT_PLAN=0
if [[ ${2:-} == --print-plan ]]; then
    PRINT_PLAN=1
fi
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_BATCHES=${AUDIT_NUM_BATCHES:-64}
PARTICLES=${AUDIT_PARTICLES:-4}
PROGRESS_EVERY=${AUDIT_PROGRESS_EVERY:-8}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-world-model-company}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATA_DIR=${ASSET_ROOT}/data/world_model_data/dataset_domain/all_data
REWARD_DIR=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/reward
RESULT_DIR=${RUNTIME_ROOT}/results/hypothesis-audit/${NODE_ROLE}-${TIMESTAMP}
LOG_DIR=${RUNTIME_ROOT}/logs/hypothesis-audit/${NODE_ROLE}-${TIMESTAMP}
EXPERIMENT_ROOT=${ASSET_ROOT}/checkpoints/experiments

declare -a NAMES=()
declare -a CHECKPOINTS=()
declare -a FAMILIES=()

add_checkpoint() {
    NAMES+=("$1")
    CHECKPOINTS+=("$2")
    FAMILIES+=("$3")
}

case ${NODE_ROLE} in
    node-a)
        QUESTION=reproducible-two-step-family
        for seed in 1 2 3; do
            add_checkpoint \
                "k1-grid25-s${seed}-latest" \
                "${EXPERIMENT_ROOT}/driftflow-endpointnorm-k1-grid25_seed${seed}/ckpt-latest.pth" \
                k1-grid25
        done
        add_checkpoint \
            wknd-gridonly-s1-best \
            "${EXPERIMENT_ROOT}/wknd-a-grid25_seed1/ckpt-best.pth" \
            grid-only-k16
        ;;
    node-b)
        QUESTION=particle-scaling-versus-base
        for seed in 1 2 3; do
            add_checkpoint \
                "k32-s${seed}-latest" \
                "${EXPERIMENT_ROOT}/driftflow-endpointnorm-k32_seed${seed}/ckpt-latest.pth" \
                k32
        done
        add_checkpoint \
            wknd-base-k16-s1-best \
            "${EXPERIMENT_ROOT}/wknd-d-base-k16_seed1/ckpt-best.pth" \
            base-k16
        ;;
    node-c)
        QUESTION=source-replay-and-coupling
        for seed in 1 2 3; do
            add_checkpoint \
                "joint-k16-s${seed}-latest" \
                "${EXPERIMENT_ROOT}/driftflow-endpointnorm-k16-grid25-sr25_seed${seed}/ckpt-latest.pth" \
                joint-k16
        done
        add_checkpoint \
            wknd-sourceonly-s1-best \
            "${EXPERIMENT_ROOT}/wknd-a-sr25_seed1/ckpt-best.pth" \
            source-only-k16
        ;;
    node-d)
        QUESTION=deep-training-regression
        for seed in 1 2; do
            add_checkpoint \
                "wknd-base-k16-s${seed}-latest" \
                "${EXPERIMENT_ROOT}/wknd-d-base-k16_seed${seed}/ckpt-latest.pth" \
                deep-base-k16
            add_checkpoint \
                "wknd-base-k16-s${seed}-best" \
                "${EXPERIMENT_ROOT}/wknd-d-base-k16_seed${seed}/ckpt-best.pth" \
                rollout-best-base-k16
        done
        ;;
esac

echo "[hypothesis-audit] node=${NODE_ROLE} question=${QUESTION} checkpoints=${#CHECKPOINTS[@]}"
echo "[hypothesis-audit] batches=${NUM_BATCHES} particles=${PARTICLES} nfes=1,2,4,8 parameterization=endpoint_normalized"
for index in "${!CHECKPOINTS[@]}"; do
    echo "PLAN gpu=${index} name=${NAMES[${index}]} family=${FAMILIES[${index}]} checkpoint=${CHECKPOINTS[${index}]}"
done
if (( PRINT_PLAN )); then
    exit 0
fi

if (( NUM_BATCHES < 3 || PARTICLES < 1 || PROGRESS_EVERY < 1 )); then
    echo "AUDIT_NUM_BATCHES must be >=3; particles and progress interval must be positive" >&2
    exit 2
fi
if [[ -z ${WANDB_API_KEY:-} && ( -z ${HOME:-} || ! -f ${HOME}/.netrc ) ]]; then
    echo "W&B credentials not found; run 'wandb login --relogin' first" >&2
    exit 1
fi
for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ ! -s ${checkpoint} ]]; then
        echo "Checkpoint not found: ${checkpoint}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" -c \
    'import torch, wandb; assert torch.cuda.device_count() == 4, torch.cuda.device_count(); print(f"[hypothesis-audit] preflight=pass torch={torch.__version__} wandb={wandb.__version__} gpus={torch.cuda.device_count()}")'

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${RUNTIME_ROOT}/wandb"
export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export WANDB_DIR=${RUNTIME_ROOT}/wandb
export WANDB_MODE=online

echo "[hypothesis-audit] results=${RESULT_DIR} logs=${LOG_DIR} wandb_project=${WANDB_PROJECT}"
PIDS=()
RESULT_ARGS=()
cd "${REPO_ROOT}/driftworld"
for index in 0 1 2 3; do
    name=${NAMES[${index}]}
    checkpoint=${CHECKPOINTS[${index}]}
    family=${FAMILIES[${index}]}
    output=${RESULT_DIR}/${name}.json
    log_file=${LOG_DIR}/${name}.log
    wandb_args=(+hypothesis_audit.wandb_project="${WANDB_PROJECT}")
    if [[ -n ${WANDB_ENTITY:-} ]]; then
        wandb_args+=(+hypothesis_audit.wandb_entity="${WANDB_ENTITY}")
    fi
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES=${index} "${PYTHON_BIN}" main_hypothesis_audit.py \
            --config-name=pushT_driftflow \
            data.dataset_path_dir="${DATA_DIR}" \
            data.batch_size=1 dataloader.num_workers=2 \
            validation.enabled=true validation.batch_size=1 \
            validation.num_workers=2 \
            model.drift_flow.transport_parameterization=endpoint_normalized \
            eval.checkpoint="${checkpoint}" \
            eval.reward_predictor_xy_checkpoint="${REWARD_DIR}/reward_predictor_xy.pth" \
            eval.reward_predictor_angle_checkpoint="${REWARD_DIR}/reward_predictor_angle.pth" \
            +hypothesis_audit.num_batches="${NUM_BATCHES}" \
            +hypothesis_audit.particles="${PARTICLES}" \
            +hypothesis_audit.progress_every="${PROGRESS_EVERY}" \
            +hypothesis_audit.seed=271828 \
            +hypothesis_audit.family="${family}" \
            +hypothesis_audit.output="${output}" \
            +hypothesis_audit.run_name="company-hypothesis-audit-${name}" \
            "${wandb_args[@]}" \
            hydra.run.dir="${LOG_DIR}/hydra-${name}" 2>&1 \
            | tee "${log_file}" \
            | awk -v name="${name}" '
                /hypothesis-audit.*batch=|wandb:.*Run data is saved|hypothesis audit complete/ {
                    print "[audit " name "] " $0
                    fflush()
                }
            '
    ) &
    PIDS+=($!)
    RESULT_ARGS+=(--result "${name}=${output}")
done

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[${index}]}"; then
        echo "[hypothesis-audit] complete ${NAMES[${index}]}"
    else
        echo "[hypothesis-audit] failed ${NAMES[${index}]}; last 40 log lines:" >&2
        tail -n 40 "${LOG_DIR}/${NAMES[${index}]}.log" >&2
        failed=1
    fi
done
if (( failed )); then
    exit 1
fi

"${PYTHON_BIN}" "${REPO_ROOT}/company/summarize_hypothesis_audit.py" \
    "${RESULT_ARGS[@]}" --output "${RESULT_DIR}/summary.json"
echo "[hypothesis-audit] status=complete node=${NODE_ROLE} summary=${RESULT_DIR}/summary.json"
