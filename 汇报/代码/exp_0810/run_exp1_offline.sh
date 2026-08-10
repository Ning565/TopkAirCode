#!/usr/bin/env bash
# 实验一（新场景 0810）：离线最优压缩率搜索
# 用法：在 汇报/代码 目录下执行  bash exp_0810/run_exp1_offline.sh [device]
# 物理口径：固定参与 N=20、逐轮 b_t*(k)=min{B_eps_ex, B_P^t}、单轮 DP（无 sqrt(T)）、
#           sigma_sc^2=N0*Δf*F、L_os=4 功率保持过采样、理想逐轮 RMS-AGC、6 dB 径向回退。
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-cuda:3}"
OUT="exp_0810/results/exp1_offline"

python exp_0810/exp1_offline_ksearch.py \
  --device "${DEVICE}" \
  --dataset femnist \
  --methods topk,randk,full \
  --ratios 0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.80 \
  --calib-rounds 8 \
  --rounds 200 \
  --epsilon 5.0 \
  --delta 1e-3 \
  --p-cap-dbm 20 \
  --adc-backoff-db 6 \
  --c-tx 0.02 \
  --omega-quantile 0.10 \
  --output-dir "${OUT}"

echo "[done] exp1 offline search -> ${OUT}/summary.md"
