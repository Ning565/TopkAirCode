#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
RUN_TAG="${RUN_TAG:-snr15_eps30_seed2026}"
OUT_DIR="${OUT_DIR:-exp/exp1-ksearch/bound_direct_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp1-ksearch/bound_direct/${RUN_TAG}}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${LOG_FILE%.log}.pid"
CMD_FILE="${LOG_FILE%.log}.cmd.sh"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "${ROOT_DIR}"
  printf "%q " "${PYTHON_BIN}" exp/exp1-ksearch/run_bound_direct_ksearch.py \
    --output-dir "${OUT_DIR}" \
    --device "${DEVICE:-cuda:3}" \
    --snr-max-db "${SNR_MAX_DB:-15}" \
    --epsilon-total "${EPSILON_TOTAL:-30}" \
    --delta "${DELTA:-1e-3}" \
    --ratio-step "${RATIO_STEP:-0.01}" \
    --calib-rounds "${CALIB_ROUNDS:-8}" \
    --constant-estimation-samples "${CONSTANT_ESTIMATION_SAMPLES:-4096}" \
    --tx-coordinate-clip "${TX_COORDINATE_CLIP:-auto-q95}" \
    --adc-backoff-db "${ADC_BACKOFF_DB:-6}" \
    "$@"
  echo
} > "${CMD_FILE}"
chmod +x "${CMD_FILE}"

setsid bash "${CMD_FILE}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "Started direct-bound k search"
echo "  pid: ${PID}"
echo "  log: ${LOG_FILE}"
echo "  out: ${OUT_DIR}"
echo "  cmd: ${CMD_FILE}"
