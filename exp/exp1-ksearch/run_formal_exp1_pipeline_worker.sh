#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_ROOT="${RUN_ROOT:-logs/experiments/exp1-ksearch/formal_pipeline/${RUN_ID}}"
RESULT_ROOT="${RUN_ROOT}/results"
CACHE_PATH="${RUN_ROOT}/cache/femnist_seed${SEED}_n20_tau5.pt"
BASE_OUT="${BASE_OUT:-exp/exp1-ksearch/formal_eq79_seed${SEED}}"
mkdir -p "${RESULT_ROOT}" "$(dirname "${CACHE_PATH}")" "${BASE_OUT}"

run_direct() {
  local output_dir="$1"
  shift
  "${PYTHON_BIN}" exp/exp1-ksearch/run_bound_direct_ksearch.py \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --dataset femnist \
    --num-clients 20 \
    --rounds 200 \
    --local-steps 5 \
    --batch-size 50 \
    --snr-max-db 15 \
    --epsilon-total 30 \
    --delta 1e-3 \
    --adc-backoff-db 6 \
    --ratio-step 0.01 \
    --calib-rounds 8 \
    --constant-estimation-samples 4096 \
    "$@"
}

echo "[pipeline] phase=baseline run_id=${RUN_ID}"
run_direct "${BASE_OUT}" \
  --calibrate-learning-rate \
  --lr-calibration-grid 0.001,0.005,0.01,0.02,0.05 \
  --lr-calibration-rounds 12 \
  --tx-coordinate-clip auto-q95 \
  --calibration-cache "${CACHE_PATH}"

readarray -t CALIBRATED < <("${PYTHON_BIN}" - "${BASE_OUT}/config.json" <<'PY'
import json
import sys

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
print(cfg["lr_femnist"])
print(cfg["element_clip"])
PY
)
LR="${CALIBRATED[0]}"
TX_CLIP="${CALIBRATED[1]}"
echo "[pipeline] frozen_lr=${LR} frozen_tx_clip=${TX_CLIP}"

run_cached_case() {
  local family="$1"
  local value="$2"
  shift 2
  local output_dir="${RESULT_ROOT}/${family}/${value}"
  echo "[pipeline] family=${family} value=${value}"
  run_direct "${output_dir}" \
    --learning-rate "${LR}" \
    --tx-coordinate-clip "${TX_CLIP}" \
    --calibration-cache "${CACHE_PATH}" \
    "$@"
}

for value in 0 5 10 20 25 30; do
  run_cached_case snr_db "${value}" --snr-max-db "${value}"
done

for value in 2.5 5 10 20; do
  run_cached_case epsilon_total "${value}" --epsilon-total "${value}"
done

for value in 3 9 11; do
  run_cached_case adc_backoff_db "${value}" --adc-backoff-db "${value}"
done

for value in 100 300; do
  run_cached_case rounds "${value}" --rounds "${value}"
done

run_recalibrated_case() {
  local family="$1"
  local value="$2"
  shift 2
  local output_dir="${RESULT_ROOT}/${family}/${value}"
  local cache_path="${RUN_ROOT}/cache/${family}_${value}.pt"
  echo "[pipeline] family=${family} value=${value} recalibrate_trajectory=true"
  run_direct "${output_dir}" \
    --learning-rate "${LR}" \
    --tx-coordinate-clip auto-q95 \
    --calibration-cache "${cache_path}" \
    "$@"
}

for value in 10 32 50; do
  run_recalibrated_case num_clients "${value}" --num-clients "${value}"
done

for value in 1 10; do
  run_recalibrated_case local_steps "${value}" --local-steps "${value}"
done

"${PYTHON_BIN}" exp/exp1-ksearch/summarize_formal_pipeline.py \
  --baseline "${BASE_OUT}" \
  --scan-root "${RESULT_ROOT}" \
  --output-dir "exp/exp1-ksearch/formal_applicability_seed${SEED}"

echo "[pipeline] complete"

