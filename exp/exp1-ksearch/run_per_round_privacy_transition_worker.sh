#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
DEVICE="${DEVICE:-cuda:3}"
SEED="${SEED:-2026}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
RUN_ROOT="${RUN_ROOT:-logs/experiments/exp1-ksearch/per_round_privacy/${RUN_ID}}"
RESULT_ROOT="${RUN_ROOT}/results"
CACHE_PATH="${CACHE_PATH:-logs/experiments/exp1-ksearch/formal_pipeline/20260803_015617_seed2026/cache/femnist_seed2026_n20_tau5.pt}"
BASE_OUT="${BASE_OUT:-exp/exp1-ksearch/per_round_privacy_eps5_seed${SEED}}"
TX_CLIP="${TX_CLIP:-0.0021039400063455105}"
mkdir -p "${RESULT_ROOT}" "${BASE_OUT}"

run_direct() {
  local output_dir="$1"
  shift
  "${PYTHON_BIN}" exp/exp1-ksearch/run_bound_direct_ksearch.py \
    --output-dir "${output_dir}" \
    --dataset femnist \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --num-clients 20 \
    --rounds 200 \
    --local-steps 5 \
    --batch-size 50 \
    --learning-rate 0.05 \
    --snr-max-db 15 \
    --epsilon-total 5 \
    --delta 1e-3 \
    --privacy-scope per_round_client_l2 \
    --tx-coordinate-clip "${TX_CLIP}" \
    --update-l2-clip auto-q95 \
    --adc-backoff-db 6 \
    --ratio-step 0.01 \
    --calib-rounds 8 \
    --constant-estimation-samples 4096 \
    --calibration-cache "${CACHE_PATH}" \
    "$@"
}

echo "[pipeline] phase=baseline epsilon_per_round=5"
run_direct "${BASE_OUT}"

for value in 0.5 1 2 3 4 8 10; do
  echo "[pipeline] family=epsilon_per_round value=${value}"
  run_direct "${RESULT_ROOT}/epsilon_per_round/${value}" --epsilon-total "${value}"
done

for value in 0 5 10 20 25 30; do
  echo "[pipeline] family=snr_db value=${value} epsilon_per_round=5"
  run_direct "${RESULT_ROOT}/snr_db/${value}" --snr-max-db "${value}"
done

"${PYTHON_BIN}" exp/exp1-ksearch/summarize_privacy_transition.py \
  --baseline "${BASE_OUT}" \
  --scan-root "${RESULT_ROOT}" \
  --output-dir "exp/exp1-ksearch/per_round_privacy_transition_seed${SEED}"

echo "[pipeline] complete"

