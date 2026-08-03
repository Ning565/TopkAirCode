#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
ROUNDS="${ROUNDS:-120}"
EVAL_EVERY="${EVAL_EVERY:-5}"

# Keep the old-meeting optimizer dynamics that gave clear Top-k/Rand-k separation.
LR_FEMNIST="${LR_FEMNIST:-0.05}"
LR_DECAY="${LR_DECAY:-0.992}"
MIN_LR="${MIN_LR:-0.005}"
OPTIMIZER_MOMENTUM="${OPTIMIZER_MOMENTUM:-0.9}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-4}"

TARGET_ACCURACY="${TARGET_ACCURACY:-75.0}"
TAIL_WINDOW="${TAIL_WINDOW:-5}"

# Narrow sweep around the useful region found so far.
TOPK_RATIOS="${TOPK_RATIOS:-0.12,0.15,0.18}"
RANDK_RATIOS="${RANDK_RATIOS:-0.50,0.65,0.80}"

# Sequential sigma sweep.  In power regime, increasing sigma0 increases
# effective aggregation noise, which should penalize Full most because it has
# the smallest b*.
SIGMA_VALUES="${SIGMA_VALUES:-0.015 0.02 0.03}"

PROFILES="${PROFILES:-pmax2.5e3_eps3e8_g1.3_eftopk:3e8:1.3:0.02:topk:common:2.5e3,pmax2.5e3_eps3e8_g1.2_eftopk:3e8:1.2:0.02:topk:common:2.5e3,pmax2.5e3_eps3e8_g1.1_eftopk:3e8:1.1:0.02:topk:common:2.5e3,pmax1e3_eps3e8_g1.2_eftopk:3e8:1.2:0.02:topk:common:1e3,pmax2.5e3_eps4e8_g1.2_eftopk:4e8:1.2:0.02:topk:common:2.5e3}"

RUN_TAG="${RUN_TAG:-full_suppression_seed${SEED}_r${ROUNDS}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-logs/experiments/exp2.0-onlinesearch/full_suppression/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp2.0-onlinesearch/full_suppression}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.pid}"
CMD_FILE="${CMD_FILE:-${LOG_DIR}/${RUN_TAG}.cmd.sh}"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "$ROOT_DIR"
  printf "SIGMA_VALUES=%q\n" "$SIGMA_VALUES"
  printf "PROFILES=%q\n" "$PROFILES"
  cat <<'SCRIPT'
for SIGMA0 in ${SIGMA_VALUES}; do
  SAFE_SIGMA="${SIGMA0//./p}"
  OUT_DIR="${OUT_ROOT}/sigma${SAFE_SIGMA}"
  echo "[sigma] ${SIGMA0}"
  "${PYTHON_BIN}" exp/exp2.0-onlinesearch/tune_profiles.py \
    --python "${PYTHON_BIN}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --rounds "${ROUNDS}" \
    --eval-every "${EVAL_EVERY}" \
    --lr-femnist "${LR_FEMNIST}" \
    --lr-decay "${LR_DECAY}" \
    --min-lr "${MIN_LR}" \
    --optimizer-momentum "${OPTIMIZER_MOMENTUM}" \
    --optimizer-weight-decay "${OPTIMIZER_WEIGHT_DECAY}" \
    --topk-ratios "${TOPK_RATIOS}" \
    --randk-ratios "${RANDK_RATIOS}" \
    --profiles "${PROFILES}" \
    --sigma0 "${SIGMA0}" \
    --target-accuracy "${TARGET_ACCURACY}" \
    --tail-window "${TAIL_WINDOW}" \
    --output-root "${OUT_DIR}"
done
SCRIPT
} > "$CMD_FILE"

# Export values consumed by the generated command file.
{
  printf "export PYTHON_BIN=%q\n" "$PYTHON_BIN"
  printf "export DEVICE=%q\n" "$DEVICE"
  printf "export SEED=%q\n" "$SEED"
  printf "export ROUNDS=%q\n" "$ROUNDS"
  printf "export EVAL_EVERY=%q\n" "$EVAL_EVERY"
  printf "export LR_FEMNIST=%q\n" "$LR_FEMNIST"
  printf "export LR_DECAY=%q\n" "$LR_DECAY"
  printf "export MIN_LR=%q\n" "$MIN_LR"
  printf "export OPTIMIZER_MOMENTUM=%q\n" "$OPTIMIZER_MOMENTUM"
  printf "export OPTIMIZER_WEIGHT_DECAY=%q\n" "$OPTIMIZER_WEIGHT_DECAY"
  printf "export TARGET_ACCURACY=%q\n" "$TARGET_ACCURACY"
  printf "export TAIL_WINDOW=%q\n" "$TAIL_WINDOW"
  printf "export TOPK_RATIOS=%q\n" "$TOPK_RATIOS"
  printf "export RANDK_RATIOS=%q\n" "$RANDK_RATIOS"
  printf "export OUT_ROOT=%q\n" "$OUT_ROOT"
  cat "$CMD_FILE"
} > "${CMD_FILE}.tmp"
mv "${CMD_FILE}.tmp" "$CMD_FILE"
chmod +x "$CMD_FILE"

setsid bash "$CMD_FILE" > "$LOG_FILE" 2>&1 < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "Started experiment-2 Full suppression tuning"
echo "  pid: $PID"
echo "  log: $LOG_FILE"
echo "  out: $OUT_ROOT"
echo "  cmd: $CMD_FILE"
