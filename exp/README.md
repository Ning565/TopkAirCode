# 实验目录索引

本目录保留当前论文实验入口和确认过的结果。旧的临时结果已移到 `logs/archive/`，不要再把 probe 或候选中间过程直接堆到 `exp/` 根实验目录里。

## 目录结构

- `common/`
  - `full_system.py`：完整系统公共模块，包含 FL、DP/power、OFDM/PAPR、ADC clipping。
- `exp1-ksearch/`
  - `run_real_calibration.py`：实验一真实 calibration 版一维 k 搜索。
  - `final/`：当前权威实验一结果。
- `exp2.0-onlinesearch/`
  - `sweep_ratios.py`：真实训练压缩率 sweep 聚合脚本。
  - `run_exp2_online_sweep.sh`：后台启动真实压缩率 sweep。
  - `PIPELINE.md`：实验二从离线搜索到真实训练验证再到收敛图的执行流程。
- `exp2-accuracy/`
  - `run_exp2_accuracy.sh`：后台启动收敛性训练。
- `exp3-privacy/`
  - 预留隐私强度实验入口。

## 当前最终结果

实验一目标函数验证：

- `exp1-ksearch/final/summary_real.md`
- `exp1-ksearch/final/objective_terms_real.csv`
- `exp1-ksearch/final/real_experiment1_objective_decomposition.png`
- `exp1-ksearch/final/real_experiment1_topk_adc_ablation.png`

## 当前实验结论

- 旧 `exp/k_search` 是 6 月 25 日未入库的旧 surrogate 结果，已归档到 `logs/archive/20260707_reorg/k_search/`。
- 当前权威实验一是 `exp1-ksearch/final/`，代码使用真实 client update、support overlap、OFDM/IFFT/ADC surrogate 和闭式 `b*(k)`。
- 当前权威实验一默认权重下：Top-k `k*/d=0.10`、Rand-k `k*/d=0.50`；Top-k 在中等 ADC 权重下的稳定区间是 `0.10-0.20`。
- 实验二必须先做真实训练压缩率 sweep，再按真实最优/近最优点跑收敛性图。

## 后续规范

- 新实验的临时输出统一放到 `logs/experiments/...`，不要再创建一堆 `probe_*` 或 `latest_*`。
- 只有确认可用的结果才复制或生成到对应的 `exp/exp*-*/final/` 或 `exp/exp*-*/results/`。
- 后续新落库的 Markdown 文档默认使用中文。
