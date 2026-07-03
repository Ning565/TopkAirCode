# 实验二候选结果：严格 DP/功率约束下的优化工作点

本结果由真实训练 CSV 合并生成：Top-k/Rand-k 来自 `experiment2_seed2028_all`，Full 来自 `experiment2_seed2028_all`。通信参数一致：`epsilon=1e8`、`sigma0=0.05`、`Pmax=1e4`、`ADC gamma=2.5`、`rounds=200`、`seed=2028`。

复现合并图命令：

```bash
python exp/full_system/build_experiment2_candidate.py --seed 2028 --top-rand-csv exp/full_system/experiment2_seed2028_all/metrics_rounds.csv --full-csv exp/full_system/experiment2_seed2028_all/metrics_rounds.csv --output-dir exp/full_system/experiment2_candidate_seed2028
```

## 结果

| 方法 | k/d | 第200轮准确率 | 最好准确率 | 最好轮次 | b* | PAPR P99 | NCE | 约束 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| topk | 0.20 | 82.93 | 84.55 | 189 | 7.034e-01 | 11.12 | 1.39e-04 | power |
| randk | 0.35 | 83.94 | 85.57 | 188 | 5.317e-01 | 11.11 | 1.38e-04 | power |
| full | 1.00 | 82.32 | 83.33 | 198 | 3.146e-01 | 11.21 | 1.40e-04 | power |

## 结论

- 当前候选设置得到第200轮排序：randk > topk > full。
- Top-k 使用 `k/d=0.20`，Rand-k 使用 `k/d=0.35`，Full 使用 `k/d=1.00`。
- 该设置基于严格场景诊断和真实训练验证得到；仍建议后续补多随机种子。
