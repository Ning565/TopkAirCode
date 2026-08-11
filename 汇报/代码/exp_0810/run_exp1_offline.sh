#!/usr/bin/env bash
# 实验一（新场景 0810）：离线最优压缩率搜索
# 用法：bash 汇报/代码/exp_0810/run_exp1_offline.sh [device]
# 物理口径：固定参与 N=20、人工噪声补足单轮 DP、epsilon=15、独立 c_tx、
#           物理热噪声、L_os=4 功率保持过采样、理想逐轮 RMS-AGC、6 dB 径向回退。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
PYTHON_BIN="${REPO_ROOT}/quenv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] quenv python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

DEVICE="${1:-cuda:3}"
OUT="${SCRIPT_DIR}/results/exp1_offline"

"${PYTHON_BIN}" "${SCRIPT_DIR}/exp1_offline_ksearch.py" \
  --device "${DEVICE}" \
  --dataset femnist \
  --methods topk,randk,full \
  --ratios 0.00025,0.0005,0.001,0.002,0.0025,0.005,0.01,0.02,0.05,0.10,0.20,0.50 \
  --calib-rounds 8 \
  --rounds 200 \
  --epsilon 15.0 \
  --delta 1e-3 \
  --p-cap-dbm 20 \
  --adc-backoff-db 6 \
  --c-tx 0.01 \
  --dp-mode topup \
  --omega-quantile 0.10 \
  --output-dir "${OUT}"

echo "[done] exp1 offline search -> ${OUT}/summary.md"
