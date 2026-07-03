# 实验二候选结果复现说明

本目录中的候选图不是手工绘制，而是由真实训练 CSV 合并生成。

## 通信参数

- `epsilon=1e8`
- `sigma0=0.05`
- `Pmax=1e4`
- `ADC gamma=2.5`
- `rounds=200`
- `seed=2026`

## 运行 Full baseline

```bash
quenv/bin/python exp/full_system/run.py \
  --device cuda:3 \
  --seed 2026 \
  --datasets femnist \
  --methods full \
  --rounds 200 \
  --ratio-overrides femnist:full=1.0 \
  --epsilon 1e8 \
  --sigma0 0.05 \
  --p-max 1e4 \
  --adc-backoff-gamma 2.5 \
  --output-dir exp/full_system/experiment2_strict \
  --eval-every 1
```

## 运行 Top-k / Rand-k

```bash
quenv/bin/python exp/full_system/run.py \
  --device cuda:3 \
  --seed 2026 \
  --datasets femnist \
  --methods topk,randk \
  --rounds 200 \
  --ratio-overrides femnist:topk=0.2,femnist:randk=0.35 \
  --epsilon 1e8 \
  --sigma0 0.05 \
  --p-max 1e4 \
  --adc-backoff-gamma 2.5 \
  --output-dir exp/full_system/experiment2_strict_k020_r035 \
  --eval-every 1
```

## 合并候选图

```bash
quenv/bin/python exp/full_system/build_experiment2_candidate.py
```

输出：

- `femnist_rounds_vs_accuracy.png`
- `metrics_rounds.csv`
- `summary.md`

当前单 seed 结果：

| 方法 | k/d | 第 200 轮准确率 | 最好准确率 |
|---|---:|---:|---:|
| Top-k | 0.20 | 83.41 | 83.41 |
| Rand-k | 0.35 | 81.49 | 82.93 |
| Full | 1.00 | 77.88 | 80.53 |

后续正式主文建议补充多随机种子，并报告均值和标准差。
