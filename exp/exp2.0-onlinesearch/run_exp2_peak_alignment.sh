#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-200}"
EVAL_EVERY="${EVAL_EVERY:-5}"
TAIL_WINDOW="${TAIL_WINDOW:-25}"

# Keep the successful experiment-2 optimizer and channel model fixed. Only
# global power/ADC operating points are varied, identically for all methods.
EPSILON="${EPSILON:-3e8}"
SIGMA0="${SIGMA0:-0.03}"
ELEMENT_CLIP="${ELEMENT_CLIP:-0.02}"
TOPK_RATIOS="${TOPK_RATIOS:-0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.35}"
RANDK_RATIOS="${RANDK_RATIOS:-0.35,0.50,0.65,0.80}"

# label:Pmax:ADC-gamma. The first profile is prioritized because the earlier
# 120-round sweep placed the real Top-k peak at k/d=0.15 in this regime.
PROFILES="${PROFILES:-pmax2500_g1.3:2500:1.3,pmax2000_g1.3:2000:1.3,pmax2500_g1.2:2500:1.2}"

RUN_TAG="${RUN_TAG:-peak_alignment_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/exp2.0-onlinesearch/peak_alignment/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp2.0-onlinesearch/peak_alignment}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$ROOT_DIR"
  IFS=',' read -ra profile_specs <<< "$PROFILES"
  for spec in "${profile_specs[@]}"; do
    IFS=':' read -r label p_max adc_gamma <<< "$spec"
    printf '%q ' "$PYTHON_BIN" exp/exp2.0-onlinesearch/sweep_ratios.py \
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
      --topk-ratios "$TOPK_RATIOS" \
      --randk-ratios "$RANDK_RATIOS" \
      --include-full \
      --epsilon "$EPSILON" \
      --sigma0 "$SIGMA0" \
      --p-max "$p_max" \
      --adc-backoff-gamma "$adc_gamma" \
      --element-clip "$ELEMENT_CLIP" \
      --error-feedback-methods topk \
      --randk-mask-mode common \
      --target-accuracy 75.0 \
      --tail-window "$TAIL_WINDOW" \
      --selection-metric tail_mean_accuracy \
      --output-dir "$OUT_ROOT/$label"
    echo
  done
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Generated peak-alignment command: $CMD_FILE"
  exit 0
fi

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started experiment-2 peak-alignment sweep"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_ROOT"
echo "  cmd: $CMD_FILE"
