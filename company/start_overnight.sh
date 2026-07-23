#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( $1 != node-a && $1 != node-b ) ]]; then
    echo "Usage: $0 {node-a|node-b}" >&2
    exit 2
fi

NODE_ROLE=$1
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_ROOT=${DRIFTFLOWWORLD_RUNTIME_ROOT:-/user-volume/driftworld}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR=${RUNTIME_ROOT}/logs/overnight
RUN_LOG=${LOG_DIR}/${NODE_ROLE}-${TIMESTAMP}.log
PID_FILE=${LOG_DIR}/${NODE_ROLE}.pid
FOLLOW_LOGS=${OVERNIGHT_FOLLOW_LOGS:-1}
mkdir -p "${LOG_DIR}"

if [[ -s ${PID_FILE} ]]; then
    EXISTING_PID=$(<"${PID_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "Overnight queue already running: node=${NODE_ROLE} pid=${EXISTING_PID}" >&2
        exit 1
    fi
fi

nohup setsid bash "${REPO_ROOT}/company/run_overnight.sh" "${NODE_ROLE}" \
    >"${RUN_LOG}" 2>&1 </dev/null &
QUEUE_PID=$!
printf '%s\n' "${QUEUE_PID}" >"${PID_FILE}"
echo "overnight_started node=${NODE_ROLE} pid=${QUEUE_PID} log=${RUN_LOG}"
if [[ ${FOLLOW_LOGS} == 1 ]]; then
    echo "streaming_training_log=on (Ctrl-C stops viewing only; queue pid ${QUEUE_PID} keeps running)"
    tail --pid="${QUEUE_PID}" -n +1 -f "${RUN_LOG}"
fi
