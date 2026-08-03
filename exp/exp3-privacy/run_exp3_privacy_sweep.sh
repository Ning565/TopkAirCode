#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-200}"
EVAL_EVERY="${EVAL_EVERY:-5}"

# Use the experiment-2 working point selected from real online sweeps.
TOPK_RATIO="${TOPK_RATIO:-0.15}"
RANDK_RATIO="${RANDK_RATIO:-0.65}"
EPSILONS="${EPSILONS:-1e5,3e5,1e6,3e6,1e7,3e7,1e8,3e8}"

LR_FEMNIST="${LR_FEMNIST:-0.05}"
LR_DECAY="${LR_DECAY:-0.992}"
MIN_LR="${MIN_LR:-0.005}"
OPTIMIZER_MOMENTUM="${OPTIMIZER_MOMENTUM:-0.9}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-4}"

SIGMA0="${SIGMA0:-0.03}"
P_MAX="${P_MAX:-2.5e3}"
ADC_BACKOFF_GAMMA="${ADC_BACKOFF_GAMMA:-1.2}"
ELEMENT_CLIP="${ELEMENT_CLIP:-0.02}"
ERROR_FEEDBACK_METHODS="${ERROR_FEEDBACK_METHODS:-topk}"
RANDK_MASK_MODE="${RANDK_MASK_MODE:-common}"
TARGET_ACCURACY="${TARGET_ACCURACY:-75.0}"
TAIL_WINDOW="${TAIL_WINDOW:-25}"

RUN_TAG="${RUN_TAG:-privacy_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-logs/experiments/exp3-privacy/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp3-privacy}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "$ROOT_DIR"
  printf "%q " "$PYTHON_BIN" exp/exp3-privacy/sweep_epsilon.py \
    --execute \
    --python "$PYTHON_BIN" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --rounds "$ROUNDS" \
    --eval-every "$EVAL_EVERY" \
    --lr-femnist "$LR_FEMNIST" \
    --lr-decay "$LR_DECAY" \
    --min-lr "$MIN_LR" \
    --optimizer-momentum "$OPTIMIZER_MOMENTUM" \
    --optimizer-weight-decay "$OPTIMIZER_WEIGHT_DECAY" \
    --topk-ratio "$TOPK_RATIO" \
    --randk-ratio "$RANDK_RATIO" \
    --epsilons "$EPSILONS" \
    --sigma0 "$SIGMA0" \
    --p-max "$P_MAX" \
    --adc-backoff-gamma "$ADC_BACKOFF_GAMMA" \
    --element-clip "$ELEMENT_CLIP" \
    --error-feedback-methods "$ERROR_FEEDBACK_METHODS" \
    --randk-mask-mode "$RANDK_MASK_MODE" \
    --target-accuracy "$TARGET_ACCURACY" \
    --tail-window "$TAIL_WINDOW" \
    --output-dir "$OUT_DIR"
  echo
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started experiment-3 privacy sweep"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_DIR"
echo "  cmd: $CMD_FILE"
