# exp_0810 隐私机制变更说明（0810：人工噪声补足设计）

本说明记录 exp_0810 套件从"纯本征噪声 DP（b_t=min{B_ε,ex, B_P}）"切换为
"满功率 + 人工噪声补足 DP"的完整口径。**物理层（拓扑、路损、热噪声、OFDM、
过采样、AGC、径向限幅、双口径统计）一律不变**；变更只发生在缩放规则与噪声
构成上。两份场景文档（`汇报8.4/通信场景/*.md`）的隐私一节尚未同步，以本文
与代码为准，待后续统一修订。

## 1. 为什么改

旧规则在真实链路（250 m / 100 mW / −174 dBm/Hz / NF 5 dB）下，隐私/功率约束
切换点为 ε\* ≈ ρ²，ρ = 2·min|g|·√(SM·P_cap)/σ_sc ≈ 7×10⁴（dB 账：链路 35 dB +
突发能量累积 10log₁₀(SM)=56 dB + 6 dB ≈ 97 dB），即 ε\* ≈ 5×10⁹。ε ≤ 30 时
b 被压到热噪声下约 50 dB，AGC/削顶统计的全是接收机热噪声，PAPR-ADC 实验失效；
且 ε 量级不符合 TWC/JSAC/TIFS 文献（调研 §2.2 已判定 10⁸–10¹⁰ 不可用）。

## 2. 新机制（Koda'20 → Wei JSAC'22 → Liu TWC'24 谱系）

记 Δ(k)=2c_tx√k，margin(ε)=√(ε+ln(1/δ))−√(ln(1/δ))，N 客户端数，d 模型维度。

1. **功率满配（含噪声功率税）**：
   b_t = B_P^t(k)/√F，F = 1 + 2d/(N·margin²)。
   税因子来自人工噪声铺满全部 d 个网格坐标（Top-k 支持集私有，噪声只铺自身
   支持集会泄露支持集），其发射能量计入 P_cap。
2. **客户端注噪（实轴、更新域、全网格）**：每客户端独立加 N(0, σ_a²)，
   σ_a² = max{0, Δ²(k)/margin² − σ_sc²/b_t²}/(2N)。
   全部为公开量（c_tx、k、N、ε、δ、σ_sc、广播的 b_t），客户端无需任何 CSI，
   信令零新增。热噪声抵扣项 σ_sc²/b_t² 即"免费隐私"额度；深衰落轮自动增大。
3. **DP 条件（实轴观测）**：(b_tΔ(k))² ≤ margin²·(σ_sc² + 2b_t²Nσ_a²)。
   σ_a 取上式时恰好取等，恢复后逐坐标 DP 噪声有信道无关闭式
   **σ_dp = Δ(k)/(√2·N·margin(ε))**。

## 3. 关键闭式与预期数字

- **宽松阈值（稀疏化换隐私的定量形态）**：σ_dp ≤ c_tx ⟺
  ε ≥ ε_loose(k) = (√(2k)/N + √ln(1/δ))² − ln(1/δ)。
  k=404 → ε_loose≈9.5；k=1000 → ε_loose≈16.8。**k\*∈[400,1000] 时
  "ε=10~15 即宽松"精确成立，环境参数零改动。**
- **本征免费区（诚实声明）**：σ_a=0 需要 margin·√F ≥ ρ；而 margin·√F 有
  与 ε 无关的下限 √(2d/N)≈201 ≪ ρ≈7×10⁴，故 ε≤30 时人工噪声恒开启。
  免费区不再作为卖点，只保留为协议完备性（代码报 `eps_free_intrinsic`）。
- **功率税**：√F(ε=5)≈346，√F(ε=30)≈82。学习侧零影响（σ_dp 与 b 无关；
  信道热噪声放大 √F 后仍 ~4×10⁻⁴ ≪ σ_dp）。
- **波形结构可见性**：逐坐标信号/人工噪声幅度比 = N·margin/(2√k)。
  k=404：ε=5 时 0.41（噪声主导、波形高斯化），ε=30 时 1.71（结构可见）。
  PAPR/削顶随 ε 的变化是新的诚实实验轴（Tegin&Duman 假设高斯输入，我们在
  大 ε 端保留稀疏结构、小 ε 端自然高斯化）。

## 4. 代码映射

| 内容 | 位置 |
|---|---|
| 缩放+补噪闭式、ε_loose、功率税、诊断字段 | `full_system_0810.py::scaling_limits`（新增必选参数 `d_model`） |
| 人工噪声注入（聚合等效 N(0,Nσ_a²)、独立种子流；结构线 r_sig 不含人工噪声，硬件线 r_rx 含） | `full_system_0810.py::OFDMAirCompChannel.transmit_round(sigma_a=...)` |
| 定理信道项改为 σ_eff²/b²（σ_eff²=σ_sc²+2b²Nσ_a²，目标函数显式含 ε） | `exp1_offline_ksearch.py::bound_terms` + `calibrate` 的 `noise_over_b2` |
| 逐轮 CSV 调试字段：`sigma_a_client / sigma_dp / sigma_dp_over_ctx / eps_loose_k / noise_tax_sqrt_f / free_intrinsic / regime(dp_noise|loose) / art_over_thermal_db` | `scaling_limits` 返回值 + `transmit_round` metrics，自动流入所有逐轮记录 |
| 免训练解析诊断（几秒出 ε×k 网格与建议） | `exp3_regime_diagnosis.py` |

## 5. 实验口径建议

- 实验三 ε 轴：{1, 2.5, 5, 10, 15, 20, 30}（对齐 R2 的 ε=20、R3 的 2.5–30、
  R4 的 25；δ=10⁻³ 对齐 R4）。
- 论文措辞：贡献 #2 由 "per-round intrinsic client-level privacy" 调整为
  "channel-noise-aware minimal artificial-noise per-round client-level privacy"；
  定理证明仅把 σ_sc² 换为 σ_tot,t²=σ_sc²+2b_t²Nσ_a²，其余不动。
- 跑任何训练前先执行 `python3 exp_0810/exp3_regime_diagnosis.py` 看
  cross-check 与 σ_dp/c_tx 网格。
