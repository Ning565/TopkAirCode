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
quenv/bin/python exp/full_system/build_experiment2_candidate.py --seed 2026
```

输出：

- `femnist_rounds_vs_accuracy.png`
- `metrics_rounds.csv`
- `summary.md`

## 多 seed 汇总

每个 seed 先用同一套真实训练命令生成 Top-k/Rand-k 和 Full 的 CSV，再用
`build_experiment2_candidate.py --seed <SEED> --output-dir exp/full_system/experiment2_candidate_seed<SEED>`
生成候选目录。随后运行：

```bash
quenv/bin/python exp/full_system/aggregate_experiment2_candidates.py
```

输出：

- `experiment2_multiseed/summary.md`
- `experiment2_multiseed/summary.json`
- `experiment2_multiseed/femnist_final_accuracy_multiseed.png`

当前已经补充 `seed=2027` 和 `seed=2028`，对应目录：

- `exp/full_system/experiment2_seed2027_top_rand`
- `exp/full_system/experiment2_seed2027_full`
- `exp/full_system/experiment2_candidate_seed2027`
- `exp/full_system/experiment2_seed2028_all`
- `exp/full_system/experiment2_candidate_seed2028`
- `exp/full_system/experiment2_multiseed`

三 seed 汇总结果：

| 方法 | seeds | 第 200 轮准确率均值 | 第 200 轮准确率 std | 最好准确率均值 |
|---|---:|---:|---:|---:|
| Top-k | 3 | 81.67 | 2.61 | 82.21 |
| Rand-k | 3 | 81.04 | 3.15 | 82.64 |
| Full | 3 | 79.14 | 2.78 | 80.52 |

结论口径：第 200 轮均值目前为 Top-k > Rand-k > Full，但 Top-k 和 Rand-k 差距只有
`0.63` 个百分点，且最好准确率均值 Rand-k 略高于 Top-k。正式主文不应夸大最终准确率优势，
更稳妥的表述是 Top-k 在更小 `k/d` 下获得相近或略高的最终均值，并具有更大的 `b*`。

当前单 seed 结果：

| 方法 | k/d | 第 200 轮准确率 | 最好准确率 |
|---|---:|---:|---:|
| Top-k | 0.20 | 83.41 | 83.41 |
| Rand-k | 0.35 | 81.49 | 82.93 |
| Full | 1.00 | 77.88 | 80.53 |

后续正式主文建议继续补充随机种子，并报告均值和标准差。
