# 实验二执行流程

实验二的主线是：

1. 使用 `exp/exp1-ksearch/` 中的新目标函数和约束做一维离线搜索，得到候选最优压缩率。
2. 对 Top-k/Rand-k 的多个 `k/d` 做真实训练 sweep，检查真实最优点是否与离线搜索接近。
3. 根据 sweep 验证后的工作点，跑 `Accuracy vs. Rounds` 收敛图。
4. 后续再进入隐私强度、ADC/PAPR、客户端数量等实验。

## 当前离线搜索结果

当前权威搜索结果是 `exp/exp1-ksearch/final/summary_real.md`，不是旧 `exp/k_search/`。

| 方法 | 离线搜索 k/d |
|---|---:|
| Top-k | 0.10 |
| Rand-k | 0.50 |
| Full | 1.00 |

注意：Top-k 对 ADC 权重有一定敏感性，在中等 ADC 权重下最优点落在 `0.10-0.20`；因此真实训练 sweep 需要覆盖这一邻域。

## 默认物理/隐私 profile

后续实验默认使用同一组参数：

- `epsilon=1e10`
- `sigma0=0.01`
- `Pmax=1e6`
- `ADC gamma=2.0`
- `element_clip=0.02`
- `Rand-k mask mode=common`

该 profile 让 `b*(k)` 进入隐私约束区间：压缩率增大时，保留信息先增加；继续增大后，`b*(k)` 下降并提高有效聚合噪声，形成先升后降的真实 sweep 目标。

## 真实压缩率 sweep

后台启动：

```bash
bash exp/exp2.0-onlinesearch/run_exp2_online_sweep.sh
```

默认会扫描：

- Top-k: `0.05,0.10,0.15,0.20,0.25,0.35,0.50,0.65`
- Rand-k: `0.20,0.35,0.50,0.65,0.80`
- Full: `1.00`

输出目录示例：

```text
logs/experiments/exp2.0-onlinesearch/seed2026_r120_eps1e10_sig0.01_p1e6_g2.0_clip0.02/
```

日志和 pid：

```text
logs/experiment2_ratio_sweep_seed2026_r120_eps1e10_sig0.01_p1e6_g2.0_clip0.02.log
logs/experiment2_ratio_sweep_seed2026_r120_eps1e10_sig0.01_p1e6_g2.0_clip0.02.pid
```

可改环境变量，例如：

```bash
SEED=2027 DEVICE=cuda:2 ROUNDS=200 bash exp/exp2.0-onlinesearch/run_exp2_online_sweep.sh
```

## 收敛性实验

先根据 sweep 的真实最优/近最优点设置压缩率，再后台运行：

```bash
TOPK_RATIO=0.10 RANDK_RATIO=0.50 bash exp/exp2-accuracy/run_exp2_accuracy.sh
```

如果 sweep 显示真实训练最优点与离线搜索有偏差，可以使用附近更稳的工作点，例如：

```bash
TOPK_RATIO=0.20 RANDK_RATIO=0.50 bash exp/exp2-accuracy/run_exp2_accuracy.sh
```

论文表述必须报告这个偏差，不能手工选择更差的 Rand-k 或 Full 来制造 Top-k 优势。
