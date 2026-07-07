#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-quenv/bin/python}"
OUT_DIR="${OUT_DIR:-exp/exp1-ksearch/final}"
LOG_DIR="${LOG_DIR:-logs/experiments/exp1-ksearch}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/experiment1_real_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[实验一真实calibration] start $(date '+%F %T')"
  echo "[实验一真实calibration] python=${PYTHON_BIN}"
  echo "[实验一真实calibration] output=${OUT_DIR}"
  "${PYTHON_BIN}" exp/exp1-ksearch/run_real_calibration.py --output-dir "${OUT_DIR}" "$@"
  echo "[实验一真实calibration] done $(date '+%F %T')"
} 2>&1 | tee "${LOG_FILE}"

echo "[实验一真实calibration] log: ${LOG_FILE}"
