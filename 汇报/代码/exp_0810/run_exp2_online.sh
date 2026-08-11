#!/usr/bin/env bash
# 实验二（新场景 0810）：在线最优压缩率搜索（逐候选真实训练 sweep，可断点续跑）
# 用法：bash 汇报/代码/exp_0810/run_exp2_online.sh [device]
# 与实验一使用完全相同的物理口径；训练完成后自动叠加实验一离线 k* 做对照。
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
OUT="${SCRIPT_DIR}/results/exp2_online"
EXP1_SUMMARY="${SCRIPT_DIR}/results/exp1_offline/summary.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/exp2_online_ksearch.py" \
  --device "${DEVICE}" \
  --dataset femnist \
  --rounds 200 \
  --eval-every 5 \
  --topk-ratios 0.00025,0.0005,0.001,0.002,0.0025,0.005,0.01,0.02,0.05,0.10 \
  --randk-ratios 0.00025,0.0005,0.001,0.002,0.0025,0.005,0.01,0.02,0.05,0.10 \
  --epsilon 15.0 \
  --delta 1e-3 \
  --p-cap-dbm 20 \
  --adc-backoff-db 6 \
  --c-tx 0.01 \
  --selection-metric tail_mean_accuracy \
  --exp1-summary "${EXP1_SUMMARY}" \
  --output-dir "${OUT}"

echo "[done] exp2 online search -> ${OUT}/summary.md"
