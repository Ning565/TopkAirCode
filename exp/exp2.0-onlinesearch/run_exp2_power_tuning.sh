#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-120}"
EVAL_EVERY="${EVAL_EVERY:-5}"

# Old meeting inspired optimizer dynamics.
LR_FEMNIST="${LR_FEMNIST:-0.05}"
LR_DECAY="${LR_DECAY:-0.992}"
MIN_LR="${MIN_LR:-0.005}"
OPTIMIZER_MOMENTUM="${OPTIMIZER_MOMENTUM:-0.9}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-4}"

SIGMA0="${SIGMA0:-0.01}"
TARGET_ACCURACY="${TARGET_ACCURACY:-75.0}"
TAIL_WINDOW="${TAIL_WINDOW:-5}"

# Center Top-k at the exp1/online candidate and keep Rand-k around exp1 k*=0.5.
TOPK_RATIOS="${TOPK_RATIOS:-0.10,0.12,0.15,0.18,0.20}"
RANDK_RATIOS="${RANDK_RATIOS:-0.35,0.50,0.65,0.80}"

# Profile format:
# label:epsilon:adc_gamma:element_clip:error_feedback[:mask_mode][:pmax]
# These profiles test when Full enters the total-power bottleneck while keeping
# the old_eps3e8_g1.3_eftopk mechanism fixed.
PROFILES="${PROFILES:-pmax1e4_eps3e8_g1.3_eftopk:3e8:1.3:0.02:topk:common:1e4,pmax5e3_eps3e8_g1.3_eftopk:3e8:1.3:0.02:topk:common:5e3,pmax2.5e3_eps3e8_g1.3_eftopk:3e8:1.3:0.02:topk:common:2.5e3,pmax1e3_eps3e8_g1.3_eftopk:3e8:1.3:0.02:topk:common:1e3,pmax5e3_eps4e8_g1.3_eftopk:4e8:1.3:0.02:topk:common:5e3,pmax2.5e3_eps4e8_g1.3_eftopk:4e8:1.3:0.02:topk:common:2.5e3}"

RUN_TAG="${RUN_TAG:-power_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/exp2.0-onlinesearch/power_tuning/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp2.0-onlinesearch/power_tuning}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"

CMD=(
  "$PYTHON_BIN" exp/exp2.0-onlinesearch/tune_profiles.py
  --python "$PYTHON_BIN"
  --device "$DEVICE"
  --seed "$SEED"
  --rounds "$ROUNDS"
  --eval-every "$EVAL_EVERY"
  --lr-femnist "$LR_FEMNIST"
  --lr-decay "$LR_DECAY"
  --min-lr "$MIN_LR"
  --optimizer-momentum "$OPTIMIZER_MOMENTUM"
  --optimizer-weight-decay "$OPTIMIZER_WEIGHT_DECAY"
  --topk-ratios "$TOPK_RATIOS"
  --randk-ratios "$RANDK_RATIOS"
  --profiles "$PROFILES"
  --sigma0 "$SIGMA0"
  --target-accuracy "$TARGET_ACCURACY"
  --tail-window "$TAIL_WINDOW"
  --output-root "$OUT_ROOT"
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

echo "Started experiment-2 power tuning"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_ROOT"
echo "  cmd: $CMD_FILE"
