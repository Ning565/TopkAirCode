#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
OUT_DIR="${OUT_DIR:-exp/exp1-ksearch/final}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp1-ksearch}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/experiment1_real_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_FILE:-${LOG_FILE%.log}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_FILE%.log}.cmd.sh}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "${ROOT_DIR}"
  printf "%q " "${PYTHON_BIN}" exp/exp1-ksearch/run_real_calibration.py --output-dir "${OUT_DIR}" "$@"
  echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

setsid bash "${CMD_FILE}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started experiment-1 real calibration"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo "  out: ${OUT_DIR}"
echo "  cmd: ${CMD_FILE}"
