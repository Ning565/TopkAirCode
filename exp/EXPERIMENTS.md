# 实验组织规范

## 权威目录

- `exp/common/`：公共系统模型和训练模块。
- `exp/exp1-ksearch/`：实验一，离线 calibration 一维 k 搜索。
- `exp/exp2.0-onlinesearch/`：实验二前置，真实训练压缩率 sweep。
- `exp/exp2-accuracy/`：实验二正式收敛性曲线。
- `exp/exp3-privacy/`：后续隐私强度实验。

## 新旧数据判定

- `logs/archive/20260707_reorg/k_search/`：旧数据。该目录来自 2026-06-25，未被 git 跟踪；FEMNIST 给出 Top-k `0.01`、Rand-k `0.10`，不再作为论文权威结果。
- `exp/exp1-ksearch/final/`：当前权威实验一结果。该目录来自 `run_real_calibration.py`，使用真实 client update、support overlap、OFDM/IFFT/ADC surrogate 和闭式 `b*(k)`。
- `logs/archive/20260707_reorg/experiment1_objective/`：旧 proxy、strict probe 和日志，仅用于追溯，不作为主文结果。
- `logs/archive/20260707_reorg/full_system/`：旧实验二候选、多 seed/probe 和临时结果，仅用于追溯。

## 当前实验一结论

`exp/exp1-ksearch/final/summary_real.md` 给出：

- 默认权重：Top-k `k*/d=0.10`，Rand-k `k*/d=0.50`。
- 中等 ADC 权重：Top-k 最优点落在 `0.10-0.20` 区间，Rand-k 多数为 `0.50`。

因此后续实验二 sweep 应重点覆盖：

- Top-k: `0.05, 0.10, 0.15, 0.20, 0.25, 0.35`
- Rand-k: `0.20, 0.35, 0.50, 0.65, 0.80`
- Full: `1.00`

## 执行规范

长训练一律用 `setsid` 后台脚本启动，日志、pid 和原始中间输出默认写入 `logs/experiments/...`。

只有经过确认要展示的最终表格、图和说明，才复制或生成到对应的 `exp/exp*-*/final/` 或 `exp/exp*-*/results/` 中。
