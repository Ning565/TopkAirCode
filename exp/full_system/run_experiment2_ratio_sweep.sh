#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-120}"
EVAL_EVERY="${EVAL_EVERY:-5}"
EPSILON="${EPSILON:-1e8}"
SIGMA0="${SIGMA0:-0.05}"
P_MAX="${P_MAX:-1e4}"
ADC_GAMMA="${ADC_GAMMA:-2.5}"
RANDK_MASK_MODE="${RANDK_MASK_MODE:-common}"

# Ratios include the k_search optima and neighboring points for real-training validation.
TOPK_RATIOS="${TOPK_RATIOS:-0.01,0.02,0.05,0.10,0.15,0.20,0.25}"
RANDK_RATIOS="${RANDK_RATIOS:-0.05,0.10,0.20,0.35,0.50,0.65}"

OUT_DIR="${OUT_DIR:-exp/full_system/experiment2_ratio_sweep_seed${SEED}_r${ROUNDS}}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/experiment2_ratio_sweep_seed${SEED}_r${ROUNDS}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/experiment2_ratio_sweep_seed${SEED}_r${ROUNDS}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/experiment2_ratio_sweep_seed${SEED}_r${ROUNDS}.cmd.sh}"

CMD=(
  "$PYTHON_BIN" exp/full_system/sweep_experiment2_ratios.py
  --execute
  --python "$PYTHON_BIN"
  --device "$DEVICE"
  --seed "$SEED"
  --rounds "$ROUNDS"
  --eval-every "$EVAL_EVERY"
  --topk-ratios "$TOPK_RATIOS"
  --randk-ratios "$RANDK_RATIOS"
  --include-full
  --epsilon "$EPSILON"
  --sigma0 "$SIGMA0"
  --p-max "$P_MAX"
  --adc-backoff-gamma "$ADC_GAMMA"
  --randk-mask-mode "$RANDK_MASK_MODE"
  --output-dir "$OUT_DIR"
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

echo "Started experiment-2 ratio sweep"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_DIR"
echo "  cmd: $CMD_FILE"
