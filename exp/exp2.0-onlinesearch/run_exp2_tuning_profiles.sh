#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-120}"
EVAL_EVERY="${EVAL_EVERY:-5}"
LR_FEMNIST="${LR_FEMNIST:-0.05}"
LR_DECAY="${LR_DECAY:-1.0}"
MIN_LR="${MIN_LR:-0.0}"
OPTIMIZER_MOMENTUM="${OPTIMIZER_MOMENTUM:-0.0}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-0.0}"
SIGMA0="${SIGMA0:-0.01}"
P_MAX="${P_MAX:-1e6}"
TARGET_ACCURACY="${TARGET_ACCURACY:-75.0}"
TAIL_WINDOW="${TAIL_WINDOW:-5}"

# Keep the search centered on the exp1 optima and their neighbors.
TOPK_RATIOS="${TOPK_RATIOS:-0.05,0.10,0.15,0.20,0.25,0.35}"
RANDK_RATIOS="${RANDK_RATIOS:-0.35,0.50,0.65,0.80}"

# Built-in labels are defined in tune_profiles.py.  They are global, auditable
# profiles rather than per-baseline hand edits.
PROFILES="${PROFILES:-eps3e9_g1.5_clip0.02_efboth,eps1e9_g1.5_clip0.02_efboth,eps5e8_g1.5_clip0.02_efboth,eps1e9_g1.3_clip0.02_efboth,eps1e9_g1.5_clip0.025_efboth,eps1e9_g1.5_clip0.02_eftopk}"

RUN_TAG="${RUN_TAG:-seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/exp2.0-onlinesearch/profile_tuning/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp2.0-onlinesearch/profile_tuning}"
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
  --p-max "$P_MAX"
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

echo "Started experiment-2 profile tuning"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_ROOT"
echo "  cmd: $CMD_FILE"
