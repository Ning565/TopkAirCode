#!/usr/bin/env bash
# 实验二（新场景 0810）：在线最优压缩率搜索（逐候选真实训练 sweep，可断点续跑）
# 用法：在 汇报/代码 目录下执行  bash exp_0810/run_exp2_online.sh [device]
# 与实验一使用完全相同的物理口径；训练完成后自动叠加实验一离线 k* 做对照。
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:3}"
OUT="exp_0810/results/exp2_online"

python exp_0810/exp2_online_ksearch.py \
  --device "${DEVICE}" \
  --dataset femnist \
  --rounds 100 \
  --eval-every 5 \
  --topk-ratios 0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50 \
  --randk-ratios 0.01,0.02,0.05,0.10,0.20,0.35,0.50,0.80 \
  --epsilon 5.0 \
  --delta 1e-3 \
  --p-cap-dbm 20 \
  --adc-backoff-db 6 \
  --c-tx 0.02 \
  --exp1-summary exp_0810/results/exp1_offline/summary.json \
  --output-dir "${OUT}"

echo "[done] exp2 online search -> ${OUT}/summary.md"
