#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU_INDEX="${GPU_INDEX:-3}"
if [[ "${GPU_INDEX}" != "3" ]]; then
  echo "Experiment policy requires GPU 3; got GPU_INDEX=${GPU_INDEX}" >&2
  exit 2
fi

GPU_USED_MIB="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -i 3 2>/dev/null | awk '{s += $1} END {print s + 0}')"
if (( GPU_USED_MIB > 1024 )); then
  echo "GPU 3 is busy (${GPU_USED_MIB} MiB in compute processes); pipeline not started." >&2
  exit 3
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_seed${SEED:-2026}}"
RUN_ROOT="${RUN_ROOT:-logs/experiments/exp1-ksearch/formal_pipeline/${RUN_ID}}"
mkdir -p "${RUN_ROOT}"
LOG_FILE="${RUN_ROOT}/pipeline.log"
PID_FILE="${RUN_ROOT}/pipeline.pid"
CMD_FILE="${RUN_ROOT}/pipeline.cmd.sh"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "${ROOT_DIR}"
  printf "RUN_ID=%q RUN_ROOT=%q DEVICE=%q SEED=%q PYTHON_BIN=%q bash %q\n" \
    "${RUN_ID}" "${RUN_ROOT}" "cuda:3" "${SEED:-2026}" "${PYTHON_BIN:-quenv/bin/python}" \
    "exp/exp1-ksearch/run_formal_exp1_pipeline_worker.sh"
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}" exp/exp1-ksearch/run_formal_exp1_pipeline_worker.sh

setsid bash "${CMD_FILE}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started formal Experiment 1 pipeline"
echo "  run_id: ${RUN_ID}"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo "  baseline: exp/exp1-ksearch/formal_eq79_seed${SEED:-2026}"

