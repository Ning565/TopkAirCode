#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-200}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EPSILON="${EPSILON:-1e8}"
SIGMA0="${SIGMA0:-0.05}"
P_MAX="${P_MAX:-1e4}"
ADC_GAMMA="${ADC_GAMMA:-2.5}"
RANDK_MASK_MODE="${RANDK_MASK_MODE:-common}"

# Defaults are the offline k_search optima for FEMNIST. Override after the ratio sweep if
# real-training validation identifies a nearby, better full-system working point.
TOPK_RATIO="${TOPK_RATIO:-0.01}"
RANDK_RATIO="${RANDK_RATIO:-0.10}"
FULL_RATIO="${FULL_RATIO:-1.0}"

OUT_DIR="${OUT_DIR:-exp/full_system/experiment2_convergence_seed${SEED}_top${TOPK_RATIO}_rand${RANDK_RATIO}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
SAFE_TOP="${TOPK_RATIO//./p}"
SAFE_RAND="${RANDK_RATIO//./p}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/experiment2_convergence_seed${SEED}_top${SAFE_TOP}_rand${SAFE_RAND}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/experiment2_convergence_seed${SEED}_top${SAFE_TOP}_rand${SAFE_RAND}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/experiment2_convergence_seed${SEED}_top${SAFE_TOP}_rand${SAFE_RAND}.cmd.sh}"

CMD=(
  "$PYTHON_BIN" exp/full_system/run.py
  --device "$DEVICE"
  --seed "$SEED"
  --datasets femnist
  --methods topk,randk,full
  --rounds "$ROUNDS"
  --ratio-overrides "femnist:topk=${TOPK_RATIO},femnist:randk=${RANDK_RATIO},femnist:full=${FULL_RATIO}"
  --epsilon "$EPSILON"
  --sigma0 "$SIGMA0"
  --p-max "$P_MAX"
  --adc-backoff-gamma "$ADC_GAMMA"
  --randk-mask-mode "$RANDK_MASK_MODE"
  --output-dir "$OUT_DIR"
  --eval-every "$EVAL_EVERY"
)

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "$ROOT_DIR"
  printf "%q " "${CMD[@]}"
  echo
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started experiment-2 convergence run"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_DIR"
echo "  cmd: $CMD_FILE"
