#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-200}"
EVAL_EVERY="${EVAL_EVERY:-5}"
TAIL_WINDOW="${TAIL_WINDOW:-25}"

TOPK_RATIO="${TOPK_RATIO:-0.15}"
RANDK_RATIO="${RANDK_RATIO:-0.65}"
EPSILONS="${EPSILONS:-1e5,3e5,1e6,3e6,1e7,3e7,1e8,3e8}"
EPSILON_ANCHOR="${EPSILON_ANCHOR:-3e8}"
SIGMA0="${SIGMA0:-0.03}"
P_MAX="${P_MAX:-2500}"
ADC_GAMMA="${ADC_GAMMA:-1.2}"
GAMMAS="${GAMMAS:-0.8,1.0,1.1,1.2,1.3,1.5,2.0}"
ELEMENT_CLIP="${ELEMENT_CLIP:-0.02}"

RUN_TAG="${RUN_TAG:-aligned_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/aligned-exp3-exp4/${RUN_TAG}}"
EXP3_OUT="${EXP3_OUT:-${OUT_ROOT}/exp3-privacy}"
EXP4_OUT="${EXP4_OUT:-${OUT_ROOT}/exp4-papr-adc}"
LOG_DIR="${LOG_DIR:-logs/experiments/aligned-exp3-exp4}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

COMMON_OPTIMIZER=(
  --lr-femnist 0.05
  --lr-decay 0.992
  --min-lr 0.005
  --optimizer-momentum 0.9
  --optimizer-weight-decay 1e-4
)

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$ROOT_DIR"
  printf '%q ' "$PYTHON_BIN" exp/exp3-privacy/sweep_epsilon.py \
    --execute \
    --python "$PYTHON_BIN" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --rounds "$ROUNDS" \
    --eval-every "$EVAL_EVERY" \
    "${COMMON_OPTIMIZER[@]}" \
    --topk-ratio "$TOPK_RATIO" \
    --randk-ratio "$RANDK_RATIO" \
    --epsilons "$EPSILONS" \
    --sigma0 "$SIGMA0" \
    --p-max "$P_MAX" \
    --adc-backoff-gamma "$ADC_GAMMA" \
    --element-clip "$ELEMENT_CLIP" \
    --error-feedback-methods topk \
    --randk-mask-mode common \
    --target-accuracy 75.0 \
    --tail-window "$TAIL_WINDOW" \
    --output-dir "$EXP3_OUT"
  echo
  printf '%q ' "$PYTHON_BIN" exp/exp4-papr-adc/sweep_papr_adc.py \
    --device "$DEVICE" \
    --seed "$SEED" \
    --rounds "$ROUNDS" \
    --eval-every "$EVAL_EVERY" \
    --ratios "topk=${TOPK_RATIO},randk=${RANDK_RATIO},full=1.0" \
    --gammas "$GAMMAS" \
    --epsilon "$EPSILON_ANCHOR" \
    --sigma0 "$SIGMA0" \
    --p-max "$P_MAX" \
    --adc-backoff-gamma "$ADC_GAMMA" \
    --element-clip "$ELEMENT_CLIP" \
    --error-feedback-methods topk \
    --randk-mask-mode common \
    "${COMMON_OPTIMIZER[@]}" \
    --output-dir "$EXP4_OUT"
  echo
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Generated aligned experiment 3/4 command: $CMD_FILE"
  exit 0
fi

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started aligned experiments 3 and 4"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  exp3: $EXP3_OUT"
echo "  exp4: $EXP4_OUT"
echo "  cmd: $CMD_FILE"
