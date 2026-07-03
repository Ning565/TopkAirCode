# 实验二候选结果：严格 DP/功率约束下的优化工作点

本结果由真实训练 CSV 合并生成：Top-k/Rand-k 来自 `experiment2_strict_k020_r035`，Full 来自 `experiment2_strict`。通信参数一致：`epsilon=1e8`、`sigma0=0.05`、`Pmax=1e4`、`ADC gamma=2.5`、`rounds=200`。

复现合并图命令：

```bash
python exp/full_system/build_experiment2_candidate.py
```

## 结果

| 方法 | k/d | 第200轮准确率 | 最好准确率 | 最好轮次 | b* | PAPR P99 | NCE | 约束 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| topk | 0.20 | 83.41 | 83.41 | 200 | 7.034e-01 | 11.22 | 1.32e-04 | power |
| randk | 0.35 | 81.49 | 82.93 | 186 | 5.317e-01 | 11.09 | 1.34e-04 | power |
| full | 1.00 | 77.88 | 80.53 | 194 | 3.146e-01 | 11.16 | 1.34e-04 | power |

## 结论

- 当前候选设置得到第200轮排序：Top-k > Rand-k > Full。
- Top-k 使用 `k/d=0.20`，Rand-k 使用 `k/d=0.35`，Full 使用 `k/d=1.00`。
- 该设置基于严格场景诊断和真实训练验证得到；仍建议后续补多随机种子。
