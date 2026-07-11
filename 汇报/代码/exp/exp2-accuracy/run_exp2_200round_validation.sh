#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-200}"
EVAL_EVERY="${EVAL_EVERY:-5}"
TOPK_RATIOS="${TOPK_RATIOS:-0.12,0.15}"
RANDK_RATIOS="${RANDK_RATIOS:-0.50,0.65,0.80}"

# Predeclared power-limited profile selected from the global physical-parameter
# sweep, not from method-specific post-processing.
EPSILON="${EPSILON:-3e8}"
SIGMA0="${SIGMA0:-0.03}"
P_MAX="${P_MAX:-1e3}"
ADC_GAMMA="${ADC_GAMMA:-1.2}"
ELEMENT_CLIP="${ELEMENT_CLIP:-0.02}"

RUN_TAG="${RUN_TAG:-power_limited_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-logs/experiments/exp2-accuracy/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp2-accuracy}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"

CMD=(
  "$PYTHON_BIN" exp/exp2.0-onlinesearch/sweep_ratios.py
  --execute
  --python "$PYTHON_BIN"
  --device "$DEVICE"
  --seed "$SEED"
  --rounds "$ROUNDS"
  --eval-every "$EVAL_EVERY"
  --lr-femnist 0.05
  --lr-decay 0.992
  --min-lr 0.005
  --optimizer-momentum 0.9
  --optimizer-weight-decay 1e-4
  --topk-ratios "$TOPK_RATIOS"
  --randk-ratios "$RANDK_RATIOS"
  --include-full
  --epsilon "$EPSILON"
  --sigma0 "$SIGMA0"
  --p-max "$P_MAX"
  --adc-backoff-gamma "$ADC_GAMMA"
  --element-clip "$ELEMENT_CLIP"
  --error-feedback-methods topk
  --randk-mask-mode common
  --target-accuracy 75.0
  --tail-window 25
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

echo "Started experiment-2 200-round power-limited validation"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_DIR"
echo "  cmd: $CMD_FILE"
