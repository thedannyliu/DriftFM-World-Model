#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 11 ]]; then
    echo "Usage: $0 NAME TAG SEED CHECKPOINT FAMILY POLICY STRATEGY PROPOSALS NFE REFINE_NFE REFINE_RATIO" >&2
    exit 2
fi

NAME=$1
TAG=$2
SEED=$3
CHECKPOINT_KIND=$4
FAMILY=$5
POLICY=$6
STRATEGY=$7
PROPOSALS=$8
NFE=$9
REFINE_NFE=${10}
REFINE_RATIO=${11}

for value in "${NAME}" "${TAG}" "${FAMILY}"; do
    if [[ ! ${value} =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Names may contain only letters, numbers, dots, underscores, and dashes" >&2
        exit 2
    fi
done
if [[ ${CHECKPOINT_KIND} != latest && ${CHECKPOINT_KIND} != best ]]; then
    echo "CHECKPOINT must be latest or best" >&2
    exit 2
fi
if [[ ${POLICY} != ep100 && ${POLICY} != ep300 ]]; then
    echo "POLICY must be ep100 or ep300" >&2
    exit 2
fi
if [[ ${STRATEGY} != uniform_breadth && ${STRATEGY} != uniform_depth && ${STRATEGY} != coarse_to_fine ]]; then
    echo "Unsupported strategy: ${STRATEGY}" >&2
    exit 2
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ASSET_ROOT=${DRIFTFLOWWORLD_ASSET_ROOT:-/group-volume/danny-dataset/driftworld}
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
PYTHON_BIN=${PYTHON_BIN:-python3}
NUM_TRIALS=${GPC_NUM_TRIALS:-20}
TRIAL_OFFSET=${GPC_TRIAL_OFFSET:-0}
POLICY_SEED=${GPC_POLICY_SEED:-5}
WANDB_PROJECT=${WANDB_PROJECT:-driftfm-unordered-fidelity-company}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

if (( NUM_TRIALS < 4 || NUM_TRIALS % 4 != 0 )); then
    echo "GPC_NUM_TRIALS must be a positive multiple of four" >&2
    exit 2
fi
if [[ ! ${TRIAL_OFFSET} =~ ^[0-9]+$ ]]; then
    echo "GPC_TRIAL_OFFSET must be a non-negative integer" >&2
    exit 2
fi
if [[ ! ${POLICY_SEED} =~ ^[0-9]+$ ]]; then
    echo "GPC_POLICY_SEED must be a non-negative integer" >&2
    exit 2
fi

EXPERIMENT_ROOT=${ASSET_ROOT}/checkpoints/experiments
WORLD_DIR=${EXPERIMENT_ROOT}/${TAG}_seed${SEED}
WORLD_CHECKPOINT=${WORLD_DIR}/ckpt-${CHECKPOINT_KIND}.pth
POLICY_CHECKPOINT=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/diffusion_policy_v1/ckpt_save/ckpt-${POLICY}.pth
REWARD_DIR=${ASSET_ROOT}/checkpoints/official/pusht_checkpoints/reward
OUTPUT_DIR=${RUNTIME_ROOT}/results/unordered-fidelity/${NAME}
LOG_DIR=${RUNTIME_ROOT}/logs/unordered-fidelity/${NAME}-${TIMESTAMP}
MARKER=${EXPERIMENT_ROOT}/gpc-unordered-${NAME}.json

if [[ -s ${MARKER} ]]; then
    echo "[gpc-budget] skip name=${NAME} completed_marker=${MARKER}"
    exit 0
fi
if [[ ! -f ${POLICY_CHECKPOINT} || ! -s ${POLICY_CHECKPOINT} ]]; then
    echo "Monolithic policy checkpoint not found: ${POLICY_CHECKPOINT}" >&2
    exit 1
fi
for path in "${WORLD_CHECKPOINT}" \
    "${REWARD_DIR}/reward_predictor_xy.pth" \
    "${REWARD_DIR}/reward_predictor_angle.pth"; do
    if [[ ! -s ${path} ]]; then
        echo "Required checkpoint not found: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "$(dirname "${MARKER}")"
exec 9>"${MARKER}.lock"
if ! flock -n 9; then
    echo "[gpc-budget] another node owns name=${NAME}; skipping duplicate"
    exit 0
fi
if [[ -s ${MARKER} ]]; then
    echo "[gpc-budget] skip name=${NAME} completed_marker=${MARKER}"
    exit 0
fi

export HF_HOME=${ASSET_ROOT}/cache/huggingface
export TORCH_HOME=${ASSET_ROOT}/cache/torch
export WANDB_DIR=${RUNTIME_ROOT}/wandb
SHARD_SIZE=$((NUM_TRIALS / 4))
NUM_PARALLEL=${GPC_NUM_PARALLEL:-32}
if (( PROPOSALS < NUM_PARALLEL )); then
    NUM_PARALLEL=${PROPOSALS}
fi
AUDIT_CANDIDATES=false
if [[ ${NAME} == a-* ]]; then
    AUDIT_CANDIDATES=true
fi

echo "[gpc-budget] name=${NAME} family=${FAMILY} policy=${POLICY} strategy=${STRATEGY}"
echo "[gpc-budget] proposals=${PROPOSALS} parallel=${NUM_PARALLEL} nfe=${NFE} refine_nfe=${REFINE_NFE} refine_ratio=${REFINE_RATIO} trials=${NUM_TRIALS}"
echo "[gpc-budget] trial_range=${TRIAL_OFFSET}:$((TRIAL_OFFSET + NUM_TRIALS))"
echo "[gpc-budget] checkpoint=${WORLD_CHECKPOINT} policy_checkpoint=${POLICY_CHECKPOINT} policy_loader=monolithic policy_seed=${POLICY_SEED}"
echo "[gpc-budget] candidate_ground_truth=${AUDIT_CANDIDATES} logs=${LOG_DIR}"

PIDS=()
NAMES=()
cd "${REPO_ROOT}/driftworld"
for GPU in 0 1 2 3; do
    START=$((TRIAL_OFFSET + GPU * SHARD_SIZE))
    END=$((START + SHARD_SIZE))
    SHARD_DIR=${OUTPUT_DIR}/num_trial_${PROPOSALS}_seeds_${START}_${END}
    FINAL=${SHARD_DIR}/final_corrected_sampling_based_testing_no_simulation_planning_receding_result_from_index_f${START}.npy
    if [[ -s ${FINAL} ]]; then
        echo "[gpc-budget] shard=${START}:${END} already complete"
        continue
    fi
    CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" main_gpc_rank.py \
        --config-name="gpc_rank_driftflow_for_${POLICY}" \
        ckpt.policy_checkpoint="${POLICY_CHECKPOINT}" \
        ckpt.use_official=false \
        ckpt.world_model_checkpoint="${WORLD_CHECKPOINT}" \
        ckpt.reward_predictor_xy_checkpoint="${REWARD_DIR}/reward_predictor_xy.pth" \
        ckpt.reward_predictor_angle_checkpoint="${REWARD_DIR}/reward_predictor_angle.pth" \
        train.seed="${POLICY_SEED}" \
        output_dir="${OUTPUT_DIR}" \
        planning.strategy="${STRATEGY}" \
        planning.num_proposals="${PROPOSALS}" \
        planning.num_parallel="${NUM_PARALLEL}" \
        planning.nfe="${NFE}" \
        planning.refine_nfe="${REFINE_NFE}" \
        planning.refine_ratio="${REFINE_RATIO}" \
        +planning.save_videos=false \
        +planning.audit_candidate_ground_truth="${AUDIT_CANDIDATES}" \
        +model.drift_flow.transport_parameterization=endpoint_normalized \
        +start_number_test="${START}" \
        +end_number_test="${END}" \
        hydra.run.dir="${LOG_DIR}/hydra-gpu${GPU}" \
        >"${LOG_DIR}/gpu${GPU}.log" 2>&1 &
    PIDS+=($!)
    NAMES+=("gpu${GPU}:${START}-${END}")
done

MONITOR_PID=
if (( ${#PIDS[@]} )); then
    tail -n 0 -F "${LOG_DIR}"/gpu*.log 2>/dev/null \
        | awk '
            /\(demo [0-9]+\/[0-9]+\) reward/ {
                print "[gpc-budget] progress " $0
                fflush()
            }
        ' &
    MONITOR_PID=$!
fi

FAILED=0
for INDEX in "${!PIDS[@]}"; do
    if wait "${PIDS[${INDEX}]}"; then
        echo "[gpc-budget] complete ${NAMES[${INDEX}]}"
    else
        echo "[gpc-budget] failed ${NAMES[${INDEX}]}; last 40 lines:" >&2
        GPU_NAME=${NAMES[${INDEX}]%%:*}
        tail -n 40 "${LOG_DIR}/${GPU_NAME}.log" >&2
        FAILED=1
    fi
done
if [[ -n ${MONITOR_PID} ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
fi
if (( FAILED )); then
    exit 1
fi

SUMMARY_ARGS=(
    --name "${NAME}"
    --family "${FAMILY}"
    --policy "${POLICY}"
    --strategy "${STRATEGY}"
    --num-proposals "${PROPOSALS}"
    --nfe "${NFE}"
    --refine-nfe "${REFINE_NFE}"
    --refine-ratio "${REFINE_RATIO}"
    --expected-seeds "${NUM_TRIALS}"
    --output-dir "${OUTPUT_DIR}"
    --output "${MARKER}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-name "company-gpc-${NAME}"
)
if [[ -n ${WANDB_ENTITY:-} ]]; then
    SUMMARY_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
"${PYTHON_BIN}" "${REPO_ROOT}/company/summarize_gpc_budget.py" "${SUMMARY_ARGS[@]}"
echo "[gpc-budget] status=complete name=${NAME} marker=${MARKER}"
