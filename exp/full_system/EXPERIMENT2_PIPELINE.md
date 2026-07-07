# 实验二执行流程

实验二的主线是：

1. 使用 `exp/k_search/` 中的新目标函数和约束做一维离线搜索，得到候选最优压缩率。
2. 对 Top-k/Rand-k 的多个 `k/d` 做真实训练 sweep，检查真实最优点是否与离线搜索接近。
3. 根据 sweep 验证后的工作点，跑 `Accuracy vs. Rounds` 收敛图。
4. 后续再进入隐私强度、ADC/PAPR、客户端数量等实验。

## 当前离线搜索结果

`exp/k_search/final/summary.md` 中 FEMNIST 的默认结果为：

| 方法 | 离线搜索 k/d |
|---|---:|
| Top-k | 0.01 |
| Rand-k | 0.10 |
| Full | 1.00 |

注意：之前实验二候选收敛图使用过 Top-k `0.20`、Rand-k `0.35`，那是完整系统候选工作点，不是严格直接沿用离线搜索默认最优点。

## 真实压缩率 sweep

后台启动：

```bash
bash exp/full_system/run_experiment2_ratio_sweep.sh
```

默认会扫描：

- Top-k: `0.01,0.02,0.05,0.10,0.15,0.20,0.25`
- Rand-k: `0.05,0.10,0.20,0.35,0.50,0.65`
- Full: `1.00`

输出目录示例：

```text
exp/full_system/experiment2_ratio_sweep_seed2026_r120/
```

日志和 pid：

```text
logs/experiment2_ratio_sweep_seed2026_r120.log
logs/experiment2_ratio_sweep_seed2026_r120.pid
```

可改环境变量，例如：

```bash
SEED=2027 DEVICE=cuda:2 ROUNDS=200 bash exp/full_system/run_experiment2_ratio_sweep.sh
```

## 收敛性实验

先根据 sweep 的真实最优/近最优点设置压缩率，再后台运行：

```bash
TOPK_RATIO=0.01 RANDK_RATIO=0.10 bash exp/full_system/run_experiment2_convergence.sh
```

如果 sweep 显示真实训练最优点与离线搜索有偏差，可以使用附近更稳的工作点，例如：

```bash
TOPK_RATIO=0.05 RANDK_RATIO=0.20 bash exp/full_system/run_experiment2_convergence.sh
```

论文表述必须报告这个偏差，不能手工选择更差的 Rand-k 或 Full 来制造 Top-k 优势。
