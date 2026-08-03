# 期刊版实验 Spec：ADC-aware Top-k Sparsified AirComp-FL

## 0. 目标定位

本文期刊版实验不再沿用会议版中“Top-k 直接降低 PAPR”的单线叙事，而是验证新版系统模型和收敛分析中的三条机制链：

1. **Sparse sensitivity-power coupling**：在相同 intrinsic DP budget 下，$k$-sparse 上传把 client-level sensitivity 从 $O(\sqrt d)$ 降到 $O(\sqrt k)$，允许更大的 AirComp power scaling $b^\star(k)$，从而降低 post-recovery channel noise。
2. **Importance retention**：在相同 $k$ 和相同 DP sensitivity 下，Top-k 通过更高有效能量保留率 $\bar\omega_A(k)$ 降低 sparsification / error-feedback 学习误差，相比 Rand-k 更稳。
3. **ADC/PAPR hardware robustness**：稀疏机制 $A$ 通过 support overlap $\rho_A(k)$、subcarrier concurrency $U_A(k)$、peak stress $\Omega_A(k)$ 改变 OFDM 时域波形峰值统计，进而影响 ADC clipping distortion 和最终精度。

最终实验主线应为：

$$
J_A(k)=
\Phi_A(k;\bar\omega_A,\beta_A)
+\frac{a_A(k)}{(b_A^\star(k))^2}
+\lambda_{\mathrm{adc}}\widehat D_{\mathrm{ADC},A}(k;\gamma),
\qquad
b_A^\star(k)=\min\{B_P(k),B_\epsilon(k)\}.
$$

实验需要证明：Top-k 的优势不是来自比 Rand-k 更低的 worst-case DP sensitivity，而是来自 **同等 sparse sensitivity 下更好的 learning retention**，以及 **在达到相同精度时可使用更小 $k$，从而改善实际资源占用和 ADC distortion**。

---

## 1. 必须对齐的理论对象

### 1.1 机制集合

实验统一使用机制索引

$$
A\in\{\mathrm{Top}\text{-}k,\mathrm{Rand}\text{-}k,\mathrm{Full}\}.
$$

- **Top-k**：proposed method，使用 error feedback，保留最大幅度 $k$ 个坐标。
- **Rand-k / PFELS-like**：同一轮或同一层共享 random support，用于对齐会议版 PFELS baseline。
- **Full**：$k=d$，无稀疏化。

Rand-k 和 Top-k 在相同 $k$、相同 element-wise clipping 阈值下具有相同 sensitivity：

$$
\Delta_{\mathrm{Top}}(k)=\Delta_{\mathrm{Rand}}(k)=2\eta\tau C\sqrt k.
$$

实验和论文表述中不能声称 Top-k 在 worst-case DP sensitivity 上优于 Rand-k。

### 1.2 功率与隐私约束

每个 $k$ 下闭式功率：

$$
B_P(k)=\frac{h_{\mathrm{th}}\sqrt{P_{\max}}}{\eta\tau C\sqrt k},
\qquad
B_\epsilon(k)=
\frac{
\sigma_0
\left(
\sqrt{\epsilon+\ln(1/\delta)}-\sqrt{\ln(1/\delta)}
\right)}
{2\eta\tau C\sqrt k\sqrt T},
$$

$$
b_A^\star(k)=\min\{B_P(k),B_\epsilon(k)\}.
$$

需要记录每个实验点处于：

- **privacy-limited regime**：$B_\epsilon(k)<B_P(k)$；
- **power-limited regime**：$B_P(k)\le B_\epsilon(k)$。

### 1.3 ADC/PAPR 链路

真实 ADC-aware 链路必须包含：

1. 频域 AirComp 聚合：

   $$
   Y_{A,\ell}^t[m;k,b_t]
   =
   b_t\sum_i S_{A,i,\ell}^t[m;k]+N_\ell^t[m].
   $$

2. 过采样 unitary IFFT 得到时域波形 $y_{A,\ell}^t[n;k,b_t]$。
3. ADC radial clipping：

   $$
   \mathcal C_{\mathrm{adc}}(y)=
   \begin{cases}
   y,& |y|\le A_{\max},\\
   A_{\max}e^{j\angle y},& |y|>A_{\max}.
   \end{cases}
   $$

4. clipping residual energy：

   $$
   E_{\mathrm{clip},A}^t(k,b_t)=
   \sum_{\ell,n}(|y_{A,\ell}^t[n;k,b_t]|-A_{\max})_+^2.
   $$

5. 等效聚合误差：

   $$
   \|e_{\mathrm{adc},A}^t\|_2^2
   \le
   \frac{E_{\mathrm{clip},A}^t(k,b_t)}{b_t^2N^2}.
   $$

正文优化层建议使用 AGC/backoff surrogate：

$$
\widehat D_{\mathrm{ADC},A}(k;\gamma)=
\frac1{T_{\mathrm{cal}}}
\sum_{t=1}^{T_{\mathrm{cal}}}\sum_{\ell,n}
(|\bar y_{A,\ell}^t[n;k]|-\gamma)_+^2.
$$

### 1.4 新版收敛界中的 ADC 截断误差项

新版收敛性分析的最终形式需要在实验设计中显式对齐。对任意机制 $A$，物理更新为

$$
\theta^{t+1}
=
\theta^t+\frac1N\sum_{i=1}^{N}s_{A,i}^t(k)
+e_{\mathrm{ch}}^t+e_{\mathrm{adc},A}^t.
$$

其中 channel noise aggregation error 满足

$$
\mathbb E\|e_{\mathrm{ch}}^t\|_2^2
=
\frac{d\sigma_0^2}{2b_t^2N^2},
$$

ADC aggregation error 由时域 clipping residual 经过 FFT 投影回数据子载波后产生，并满足

$$
\|e_{\mathrm{adc},A}^t\|_2^2
\le
\frac{E_{\mathrm{clip},A}^t(k,b_t)}{b_t^2N^2}.
$$

因此新版 convergence bound 中需要出现三类误差：

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla f(\theta^t)\|_2^2
\le
\Phi_A(k;\bar\omega_A,\beta_A)
+
\frac{8\Gamma_A(k)L\sigma_0^2d}{T\eta\tau N^2}
\sum_{t=0}^{T-1}\frac1{b_t^2}
+
\frac{16\Gamma_A(k)C_{\mathrm{adc}}}{T\eta\tau N^2}
\sum_{t=0}^{T-1}
\frac{
\mathbb E[E_{\mathrm{clip},A}^t(k,b_t)]
}{b_t^2}.
$$

这里：

- $\Phi_A(k;\bar\omega_A,\beta_A)$ 是 learning-side term，包含 local drift、heterogeneity、sparsification / error-feedback memory 和 pre-transmission clipping bias；
- 第二项是 AirComp post-recovery channel noise；
- 第三项是 ADC clipping-induced aggregation error，是新版收敛分析相比会议版必须新增建模和实验观测的部分。

正文优化时不直接对 exact $E_{\mathrm{clip},A}^t(k,b_t)$ 求闭式，而是使用 AGC/backoff 归一化 surrogate $\widehat D_{\mathrm{ADC},A}(k;\gamma)$ 影响 $k$ 的选择；但实验中需要同时保留真实 OFDM-ADC 链路下的 clipping residual / clipping energy，用来支撑该项的物理意义。

---

## 2. 数据集与训练设置

### 2.1 主数据集

期刊版实验需要扩展数据集，不能只使用会议版 FEMNIST。

- Main datasets：FEMNIST 和 CIFAR-10。
- Preliminary validation：MNIST，可用于快速调试、初步验证和小模型机制实验。
- Partition：Dirichlet non-IID，默认 $\alpha=1.0$；额外强 non-IID 用 $\alpha=0.1$。
- Model：
  - FEMNIST：CNN / StableCNN，可与当前脚本保持一致；
  - CIFAR-10：使用适合 CIFAR-10 的 CNN / ResNet 类模型，后续根据实验成本确定；
  - MNIST：SimpleMLP 或轻量 CNN，用作初步验证。
- Local training：$\tau=5$ local SGD steps per round。
- Learning rate：$\eta=0.05$ 起步，可按旧版设置衰减。
- Default clients：$N=16$。
- Total rounds：主结果建议 $T=400$；快速 calibration 可用 $T=50$ 或 $100$。

最终主文结果应至少包含 FEMNIST 和 CIFAR-10；MNIST 不作为主要说服力来源。

### 2.2 默认通信参数

以下参数只是起始设置，后续会根据模型维度、训练稳定性、ADC clipping 强度和运行成本继续调整。

- OFDM data subcarriers：先设 $M=2000$。期刊版模型参数量可能达到几十万量级，需要把大量模型参数映射到子载波上传输，$M=64$ 或 $256$ 对主实验偏小。
- Oversampling：$L_{\mathrm{os}}=4$。
- Channel inversion threshold：$h_{\mathrm{th}}=0.1$。
- Channel magnitude：可用 $|h|\sim U[0.1,0.3]$ 做仿真。
- ADC/backoff：使用 normalized clipping threshold $\gamma$ 扫描；固定阈值 $A_{\max}$ 用于 exact clipping 验证。
- DP：默认 $\epsilon=2.0,\delta=10^{-3}$，扫 $\epsilon\in\{0.2,0.5,1,2,3\}$。

---

## 3. 后续实现任务清单

1. 在现有实验脚本中补齐 DP power control：按 $B_P(k)$、$B_\epsilon(k)$、$b^\star(k)$ 设置 AirComp scaling。
2. 支持 grid sweep：Top-k、Rand-k / PFELS-like、Full。
3. 增加 calibration 模式：输出 $\bar\omega_A,\rho_A,\Omega_A,\widehat D_{\mathrm{ADC},A},b^\star,J_A$。
4. 扩展数据集管线：FEMNIST、CIFAR-10、MNIST。
5. 将 OFDM 子载波数、过采样、ADC 阈值、DP 参数、模型维度相关参数做成可配置项，便于后续调参。
6. 保存统一 JSON schema，保证后续论文图可复现。

---

## 4. 一句话实验验收标准

期刊版实验应最终证明：

$$
\boxed{
\text{Top-k does not win by smaller Rand-k sensitivity; it wins by retaining important updates, enabling smaller accuracy-equivalent sparsity, and improving the DP-power-ADC tradeoff under the revised OFDM-AirComp model.}
}
$$
