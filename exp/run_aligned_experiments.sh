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
RATIOS="${RATIOS:-0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.12,0.15,0.20,0.35,0.50,0.65,0.80}"
EPSILONS="${EPSILONS:-1e5,3e5,1e6,3e6,1e7,3e7,1e8,3e8}"

EPSILON="${EPSILON:-3e8}"
SIGMA0="${SIGMA0:-0.03}"
P_MAX="${P_MAX:-1e3}"
ADC_GAMMA="${ADC_GAMMA:-1.2}"
ELEMENT_CLIP="${ELEMENT_CLIP:-0.02}"

RUN_TAG="${RUN_TAG:-aligned_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/aligned/${RUN_TAG}}"
EXP1_OUT="${EXP1_OUT:-${OUT_ROOT}/exp1-ksearch}"
EXP2_OUT="${EXP2_OUT:-logs/experiments/exp2-accuracy/power_limited_seed2026_r200_20260710_224549}"
EXP3_OUT="${EXP3_OUT:-${OUT_ROOT}/exp3-privacy}"
LOG_DIR="${LOG_DIR:-logs/experiments/aligned}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "$ROOT_DIR"
  printf "%q " "$PYTHON_BIN" exp/exp1-ksearch/run_real_calibration.py \
    --output-dir "$EXP1_OUT" \
    --device "$DEVICE" \
    --rounds "$ROUNDS" \
    --ratios "$RATIOS,1.0" \
    --epsilon "$EPSILON" \
    --sigma0 "$SIGMA0" \
    --p-max "$P_MAX" \
    --adc-backoff-gamma "$ADC_GAMMA" \
    --element-clip "$ELEMENT_CLIP"
  echo
  printf "%q " "$PYTHON_BIN" exp/exp2.0-onlinesearch/sweep_ratios.py \
    --execute \
    --python "$PYTHON_BIN" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --rounds "$ROUNDS" \
    --eval-every "$EVAL_EVERY" \
    --lr-femnist 0.05 \
    --lr-decay 0.992 \
    --min-lr 0.005 \
    --optimizer-momentum 0.9 \
    --optimizer-weight-decay 1e-4 \
    --topk-ratios "$RATIOS" \
    --randk-ratios "$RATIOS" \
    --include-full \
    --epsilon "$EPSILON" \
    --sigma0 "$SIGMA0" \
    --p-max "$P_MAX" \
    --adc-backoff-gamma "$ADC_GAMMA" \
    --element-clip "$ELEMENT_CLIP" \
    --error-feedback-methods topk \
    --randk-mask-mode common \
    --target-accuracy 75.0 \
    --tail-window "$TAIL_WINDOW" \
    --selection-metric tail_mean_accuracy \
    --output-dir "$EXP2_OUT"
  echo
  printf "read -r TOPK_RATIO RANDK_RATIO < <(%q -c %q %q)\n" \
    "$PYTHON_BIN" \
    'import json,sys; d=json.load(open(sys.argv[1])); print(d["methods"]["topk"]["ratio"], d["methods"]["randk"]["ratio"])' \
    "$EXP2_OUT/selected_workpoints.json"
  printf 'echo "[selected] topk=$TOPK_RATIO randk=$RANDK_RATIO"\n'
  printf "%q " "$PYTHON_BIN" exp/exp3-privacy/sweep_epsilon.py \
    --execute \
    --python "$PYTHON_BIN" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --rounds "$ROUNDS" \
    --eval-every "$EVAL_EVERY" \
    --lr-femnist 0.05 \
    --lr-decay 0.992 \
    --min-lr 0.005 \
    --optimizer-momentum 0.9 \
    --optimizer-weight-decay 1e-4
  printf '%q "$TOPK_RATIO" %q "$RANDK_RATIO" ' --topk-ratio --randk-ratio
  printf "%q " \
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
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Generated aligned experiment command: $CMD_FILE"
  exit 0
fi

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started aligned experiment 1-3 pipeline"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  exp1: $EXP1_OUT"
echo "  exp2: $EXP2_OUT"
echo "  exp3: $EXP3_OUT"
echo "  cmd: $CMD_FILE"
