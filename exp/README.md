# 实验目录索引

本目录保留当前论文实验入口和确认过的结果。以后查最新结果，一律优先看各实验的 `final/` 或 `final_real/`。

## 目录结构

- `full_system/`
  - `run.py`：完整系统实验入口，包含 DP 模块、功率约束、OFDM/PAPR、ADC 截断和联邦学习收敛。
  - `final/`：当前最终完整系统结果。
  - `experiment2_candidate/`：严格 DP/功率约束下的实验二候选结果。
  - `build_experiment2_candidate.py`：从真实训练 CSV 合并生成实验二候选图和汇总。
  - `aggregate_experiment2_candidates.py`：汇总多个实验二候选 seed 的均值和标准差。
  - `sweep_experiment2_ratios.py`：扫描实验二 Top-k/Rand-k 压缩率与性能关系。
  - `run_experiment2_ratio_sweep.sh`：后台启动实验二真实压缩率 sweep。
  - `run_experiment2_convergence.sh`：后台启动实验二收敛性训练。
  - `EXPERIMENT2_PIPELINE.md`：实验二从离线搜索到真实训练验证再到收敛图的执行流程。
- `k_search/`
  - `run.py`：ADC-aware surrogate 一维 k 搜索入口。
  - `final/`：当前最终 k 搜索结果。
- `experiment1_objective/`
  - `run_real_calibration.py`：实验一真实 calibration 版目标函数扫描。
  - `final_real/`：当前实验一真实 calibration 结果。

## 当前最终结果

完整系统主图：

- `full_system/final/femnist_rounds_vs_accuracy.png`
- `full_system/final/summary.md`
- `full_system/final/metrics_rounds.csv`

实验二候选收敛图：

- `full_system/experiment2_candidate/femnist_rounds_vs_accuracy.png`
- `full_system/experiment2_candidate/summary.md`
- `full_system/experiment2_candidate/metrics_rounds.csv`
- `full_system/experiment2_multiseed/summary.md`
- `full_system/experiment2_multiseed/femnist_final_accuracy_multiseed.png`

k 搜索图：

- `k_search/final/mnist_topk_k_search.png`
- `k_search/final/mnist_randk_k_search.png`
- `k_search/final/femnist_topk_k_search.png`
- `k_search/final/femnist_randk_k_search.png`
- `k_search/final/summary.md`

实验一目标函数验证：

- `experiment1_objective/final_real/summary_real.md`
- `experiment1_objective/final_real/objective_terms_real.csv`
- `experiment1_objective/final_real/real_experiment1_objective_decomposition.png`
- `experiment1_objective/final_real/real_experiment1_topk_adc_ablation.png`

## 当前实验结论

- 完整系统 FEMNIST 第 120 轮排序：Top-k `79.81%` > Rand-k `77.16%` > Full `76.44%`。
- Rand-k 已启用 error feedback，因此 baseline 不再被不公平地压低。
- k 搜索使用 surrogate calibration，估计 `bar_omega`、`rho`、`ADC proxy` 和 `b*(k)`，不是用短训练准确率直接搜索。
- 实验一真实 calibration 默认权重下搜索得到 Top-k `k*/d=0.10`、Rand-k `k*/d=0.50`；Top-k 在中等 ADC 权重下的最优区间为 `0.10-0.20`。
- 实验二候选严格设置为 `epsilon=1e8`、`sigma0=0.05`、`Pmax=1e4`、`ADC gamma=2.5`，第 200 轮排序为 Top-k `83.41%` > Rand-k `81.49%` > Full `77.88%`。
- 实验二已补 `seed=2027,2028` 并生成三 seed 汇总：第 200 轮均值 Top-k `81.67%` > Rand-k `81.04%` > Full `79.14%`；但 Rand-k 的最好准确率均值 `82.64%` 略高于 Top-k `82.21%`，正式主文应避免夸大 Top-k 的最终准确率优势。

## 后续规范

- 新实验的临时输出统一放到 `exp/tmp/`，不要再创建一堆 `probe_*` 或 `latest_*`。
- 只有确认可用的结果才复制到 `full_system/final/` 或 `k_search/final/`。
- 后续新落库的 Markdown 文档默认使用中文。
