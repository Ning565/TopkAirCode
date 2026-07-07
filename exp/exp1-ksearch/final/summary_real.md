# 实验一真实 calibration 扫描

本结果使用真实 client update、support mask 和 OFDM/IFFT/ADC 流程估计目标函数项。

本版使用 `calib_rounds=8`。旧版 `calib_rounds=3` 下 Top-k ADC-aware 的最优点为 `k/d=0.20`；增加 calibration 后，默认权重 `lambda_channel=0.15, lambda_adc=0.50` 下最优点移动到 `k/d=0.10`。两者目标值接近，说明 Top-k 的精确最优点对 ADC 权重有一定敏感性，但核心结论稳定：Top-k 的最优稀疏率显著小于 Rand-k，且 ADC-aware Top-k 会选择小于 ADC-unaware Top-k 的稀疏率。

## 趋势检查

- topk: learning下降=True
- topk: channel上升=True
- topk: ADC上升=False
- randk: learning下降=True
- randk: channel上升=True
- randk: ADC上升=True
- k*: Top-k=0.10, Rand-k=0.50, Top-k ADC-unaware=0.35

## 搜索结果

| 方法 | 搜索目标 | k*/d | b* | bar_omega | rho | PAPR P99 | ADC cost | J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| topk | Top-k ADC-aware | 0.10 | 3.517e+00 | 0.839 | 0.371 | 14.83 | 3.35e-04 | 0.215 |
| topk | Top-k ADC-unaware | 0.35 | 1.880e+00 | 0.986 | 0.634 | 19.57 | 7.30e-03 | 0.027 |
| randk | Rand-k ADC-aware | 0.50 | 1.573e+00 | 0.501 | 1.000 | 16.92 | 1.68e-03 | 0.332 |
| full | Full update | 1.00 | 1.112e+00 | 1.000 | 1.000 | 19.18 | 6.55e-03 | 0.000 |

## 权重敏感性

固定 `lambda_channel=0.15`，改变 `lambda_adc` 时的最优点：

| lambda_adc | Top-k k*/d | Rand-k k*/d |
|---:|---:|---:|
| 0.20 | 0.20 | 0.50 |
| 0.30 | 0.20 | 0.50 |
| 0.40 | 0.20 | 0.50 |
| 0.50 | 0.10 | 0.50 |
| 0.60 | 0.10 | 0.35 |
| 0.80 | 0.10 | 0.35 |
| 1.00 | 0.10 | 0.35 |

验收结论：实验一可以支撑论文中的优化验证图，但正文表述不应把 Top-k 的精确最优点固定死为单一数值；更稳妥的说法是默认权重下 `Top-k k*/d=0.10`、`Rand-k k*/d=0.50`，在中等 ADC 权重下 Top-k 最优点落在 `0.10-0.20` 区间。
