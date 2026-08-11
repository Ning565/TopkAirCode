# AirComp 场景下的 PAPR 与接收端削顶场景设置

本文档是主通信场景中 PAPR、AGC 与接收端有限动态范围部分的独立展开版。本文统一采用每轮全部 $N$ 个客户端成功聚合、逐轮客户端级 $(\epsilon,\delta)$-DP 和逐轮信道自适应闭式缩放，并保留已确定的物理信号链、径向限幅、参数设置、评价指标和收敛性对应关系。

# 1. 整体场景与建模边界

**主场景一句话定义**


> OFDM AirComp-FL 系统每个逻辑通信轮均由全部 $N$ 个客户端完成本地计算和无线聚合，不进行算法层面的客户端采样，也不把深衰落解释为学习客户端缺失。深衰落候选突发由物理层等待或重新调度处理，只影响通信时延。客户端上传实值模型更新，每个实坐标占用一个复 OFDM 子载波资源，复信道预均衡后在基站实轴对齐，FFT 后取实部并除以 $b_tN$ 恢复。公共 AirComp 缩放采用 $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\}$，只依赖当前公开无线状态、公开稀疏度和公开裁剪上界，不依赖实际私有更新范数；每个通信轮分别满足客户端级 $(\epsilon,\delta)$-DP。基站采用高分辨率但有限复包络动态范围的接收前端，忽略有限位量化；每轮通过理想 RMS-AGC 设置一次增益，轮内固定。结构性 PAPR 使用无噪聚合信号统计，实际径向削顶作用于含信号和噪声的接收波形。该径向算子是接收端有限复包络动态范围抽象，而不是精确 I/Q ADC 模型。


**纳入主模型的组件**


| 组件 | 口径 |
|---|---|
| 接收前端 | 高分辨率（量化噪声忽略）+ 有限复包络动态范围 |
| 削顶机制 | AGC 后复包络径向限幅（保相位），作用于含噪接收前端输入波形；不是精确I/Q ADC转移特性 |
| AGC | 理想逐轮 RMS-AGC，轮内 395 个 OFDM 符号共用同一增益 |
| PAPR 统计 | 双口径：结构线（无噪）+ 硬件线（含噪），见本文件第 3 节 |
| 过采样 | $L_{\mathrm{os}}=4$ 基准，$L_{\mathrm{os}}=8$ 仅少量波形实验验证 |
| 客户端参与 | 每个逻辑轮全部 $N$ 个客户端成功上传；深衰落只触发等待/重调度 |
| 模型映射 | 一个实模型坐标占用一个复子载波；FFT 后取实部恢复 |
| 公共缩放 | $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\}$；逐轮 DP、逐轮信道自适应 |


**明确排除的组件（建模边界）**


以下机制**不进入主模型**，仅在相关工作或未来扩展中一句话提及：

- **有限位量化 / Bussgang 量化模型**：低分辨率 ADC/DAC 是独立扩展方向（Tegin & Duman, IEEE TWC 2021 已专门研究），本文假设量化级数充分大、只保留削顶失真——与 Rietman & Linnartz（TWC 2008）"$L$ 足够大"的分析口径同款。注意：Bussgang 分解本身适用于高斯输入的任意无记忆非线性（含削顶），本文不用它不是因为不适用，而是因为实验直接对波形执行削顶算子、理论保留精确残差能量，**不需要高斯近似**——这是与 Tegin & Duman 的方法论区分点，可作为卖点写入正文。
- **重传机制**：Hellström 等（TWC 2023）的方向，与本文正交。
- **发射端功放非线性 / ACLR / ICF 削顶**：Bielefeld 等（arXiv:2512.23381，在投）的发射端方向，与本文接收端口径互补。
- **完整频率选择性多径**：主场景文档中的频率平坦块衰落部分 的客户端级频率平坦块衰落假设不变。
- **复杂削顶恢复/校正算法**：本文只评估被动削顶失真，不引入接收端补偿。

---

# 2. AirComp 信号链与过采样波形

**全客户端成功聚合与频域信号**

本文不进行算法层面的客户端采样。每个逻辑通信轮均由全部 $N$ 个客户端形成稀疏更新并完成上行聚合。候选突发出现深衰落时，物理层等待下一次可行机会或重新调度同一批客户端；只有全客户端成功对齐的突发才计为一次学习轮。因此全文统一采用固定客户端集合 $\{1,\ldots,N\}$ 和固定聚合分母 $N$；候选突发是否成功只属于物理层时延与可行性判定。 候选突发采用公开成功条件

$$
|h_i^t|\ge h_{\mathrm{cut}},\qquad i=1,\ldots,N,
$$

其中 $h_{\mathrm{cut}}=0.1$；不满足时不更新模型或误差反馈状态。

模型更新为实向量。每个实模型坐标占用一个复 OFDM 数据子载波资源，不使用 I/Q 两路分别承载两个模型坐标，也不施加 Hermitian 对称。令

$$
j(q,m)=(q-1)M+m+1,
$$

则

$$
s_{i,q,m}^t=\begin{cases}[\mathbf s_i^t]_{j(q,m)}, & j(q,m)\le d,\\0, & j(q,m)>d,\end{cases}\qquad s_{i,q,m}^t\in\mathbb R.
$$

实际发送符号因复信道预均衡而为复数，但理想信道对齐后的有效聚合信号位于实轴，基站 FFT 后取实部恢复模型坐标。令公开逐坐标门限为 $c_{\mathrm{tx}}=\eta\tau C$。客户端 $i$ 的整轮平均发射功率为

$$
P_i^t=\frac{b_t^2}{SM|g_i^t|^2}\|\mathbf s_i^t\|_2^2,
$$

由 $\|\mathbf s_i^t\|_0\le k$、$\|\mathbf s_i^t\|_\infty\le c_{\mathrm{tx}}$ 得到

$$
B_P^t(k)=\min_{1\le i\le N}\frac{|g_i^t|\sqrt{SM P_{\mathrm{cap}}}}{c_{\mathrm{tx}}\sqrt{k}}.
$$

主通信场景给出的逐轮隐私接口为

$$
\Delta(k)=2c_{\mathrm{tx}}\sqrt{k},\qquad 0<b_t\le B_{\epsilon,\mathrm{ex}}(k),
$$

并唯一采用

$$
b_t=b_t^\star(k)=\min\left\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\right\}.
$$

该规则保证第 $t$ 轮分别满足客户端级 $(\epsilon,\delta)$-DP 和全部客户端的平均功率约束，无需跨轮隐私预算、未来信道或剩余预算控制。第 $t$ 轮频域接收信号为

$$
y_{q,m}^t=b_t\sum_{i=1}^{N}s_{i,q,m}^t+z_{q,m}^t,\qquad z_{q,m}^t\sim\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2).
$$

**过采样时域波形**


对每个 OFDM 符号做零填充过采样 IFFT。令

$$
Q=L_{\mathrm{os}}M=4096,
$$

并用 $\mathcal Z_{\mathrm{os}}(\cdot)$ 表示居中零填充。为使过采样前后的时域平均功率口径一致，本文固定采用功率保持的频域缩放

$$
\mathbf y_{q,\mathrm{os}}^t
=
\sqrt{\frac QM}\,
\mathcal Z_{\mathrm{os}}(\mathbf y_q^t),
$$

随后通过酉 IFFT 得到

$$
r_q^t[n]
=
\mathrm{IFFT}_Q^{\mathrm u}
\!\left(\mathbf y_{q,\mathrm{os}}^t\right)[n],
\qquad
n=0,\dots,Q-1.
$$

关键约束是：噪声必须先在频域按

$$
z_{q,m}^t\sim\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2)
$$

生成，与聚合信号相加后共同执行同一套零填充、$\sqrt{Q/M}$ 缩放和酉 IFFT。不得在时域另行生成一份噪声，否则容易因 FFT 归一化不同而重复乘除 $M$ 或 $L_{\mathrm{os}}$。

本文所有信号、噪声、AGC、削顶门限、FFT 恢复和 NMSE 计算均使用这一统一归一化。PAPR 本身是峰值功率与平均功率之比，不受共同幅度尺度影响；功率保持缩放的作用是让 $P_{\mathrm{avg}}^t$、绝对门限 $A_{\max}^t$ 和理论附录中的波形能量具有统一物理口径。

时域峰值来自两级叠加：其一是 $N$ 个客户端在同一子载波上的 AirComp 叠加；其二是多个子载波在 IFFT 中的相干叠加。Top-k、Rand-k 和 Full 通过活跃子载波结构、客户端支持重叠以及非零更新值的幅度和符号关系共同影响最终波形。


**波形的两个版本**


定义两条共享同一组全部客户端、同一频域更新和同一过采样归一化的波形：

$$
r_{\mathrm{sig},q}^t[n]
=
\mathrm{IFFT}_Q^{\mathrm u}
\!\left(
\sqrt{\frac QM}\,
\mathcal Z_{\mathrm{os}}
\left(
 b_t\sum_{i=1}^{N}s_{i,q,m}^t
\right)
\right)[n],
$$

即无噪聚合信号；以及

$$
r_{\mathrm{rx},q}^t[n]
=
\mathrm{IFFT}_Q^{\mathrm u}
\!\left(
\sqrt{\frac QM}\,
\mathcal Z_{\mathrm{os}}(\mathbf y_q^t)
\right)[n]
=
r_{\mathrm{sig},q}^t[n]
+r_{\mathrm{noise},q}^t[n],
$$

即含噪接收前端输入。结构性 PAPR/PSR 使用第一条波形；实际 AGC、削顶、削顶率、失真能量和聚合 NMSE 使用第二条波形。

---

# 3. PAPR、AGC 与有限动态范围削顶

**核心原则：两种统计回答不同问题，不可互相替代，也不可混用。**


**结构线——无噪声 PAPR（回答"稀疏结构如何塑造峰值"）**


$$
\mathrm{PAPR}_{\mathrm{sig},q}^t
= \frac{\max_n |r_{\mathrm{sig},q}^t[n]|^2}
{\frac{1}{Q}\sum_n |r_{\mathrm{sig},q}^t[n]|^2}.
$$

- 统计对象是**无噪聚合信号**，与 SNR 解耦：无噪 PAPR 的差异由稀疏支持（support）、客户端支持重叠、子载波占用模式以及**非零更新值的幅度与相位关系**共同决定——不得写成"只由 support 结构决定"；
- 选择无噪口径的原因：低 SNR 档位（如 $P_{\max}=10$ dBm 或隐私支路压低 $b_t$ 时）下，复高斯噪声自身的高峰值统计会淹没机制间的结构差异，含噪 PAPR 无法作为结构证据；
- 报告形式：**CCDF** $\Pr\{\mathrm{PAPR}_{\mathrm{sig}} > \xi\}$，按 OFDM 符号为单位统计，读取超越概率 $10^{-3}$ 处的 PAPR 值作为机制间对比的标量指标。**不允许只报告平均 PAPR。**
- **全零符号规则**：若某 OFDM 符号上全部 $N$ 个客户端均在对应资源上发送零（该符号无噪聚合信号全零），PAPR 分母为零、无定义——该符号**不进入 PAPR CCDF 统计**，并**单独报告全零/静默 OFDM 符号比例**（高稀疏率下 Top-k/Rand-k 可能出现，且该比例本身就是稀疏结构的信息量）。

**轮归一化峰值压力（PSR）——衔接逐符号 PAPR 与逐轮门限**：标准 PAPR 按每个 OFDM 符号自身平均功率归一，而实际削顶门限 $A_{\max}^t=\gamma A_{\mathrm{rms}}^t$ 按**整轮**平均功率设置，二者归一化基准不一致：一个符号自身 PAPR 不高、但整体能量远高于本轮均值时仍可能频繁削顶；反之符号 PAPR 高但能量很小则未必越过逐轮门限。因此在标准 PAPR CCDF 之外，增加与逐轮 AGC 直接对应的指标

$$
\mathrm{PSR}_q^t
=\frac{\max_n|r_{\mathrm{sig},q}^t[n]|^2}
{\frac{1}{SQ}\sum_{q',n}|r_{\mathrm{sig},q'}^t[n]|^2},
\qquad
\mathrm{PSR}_{\mathrm{round}}^t=\max_q \mathrm{PSR}_q^t ,
$$

分别以 CCDF（逐符号）与轮标量（整轮最大值）形式报告。证据链相应修正为：**标准 PAPR + 轮归一化峰值压力 → 实际削顶率 → 削顶失真**——单说"PAPR 降低所以削顶减少"并不总是严格成立，必须由 PSR 补全归一化基准的衔接。


**硬件线——含噪削顶（回答"实际接收机受到多大失真"）**


削顶及其所有派生统计一律作用于含噪接收前端输入 $r_{\mathrm{rx},q}^t[n]$。物理依据是热噪声在接收前端即与信号共同出现，有限动态范围算子应作用于信号与噪声之和；该波形由含噪频域聚合向量经同一过采样酉 IFFT 得到，并与收敛性附录一致。



**定义**


基站在每个通信轮利用该轮**含噪**接收波形的全部有效OFDM样本估计平均功率。循环前缀只进入突发持续时间计算，不进入本文的PAPR、AGC和削顶残差统计：

$$
P_{\mathrm{avg}}^t
= \frac{1}{S\,Q}
\sum_{q=1}^{S}\sum_{n=0}^{Q-1}
\big|r_{\mathrm{rx},q}^t[n]\big|^2,
\qquad
A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}.
$$

**命名说明**：$P_{\mathrm{avg}}^t$ 是平均功率（早期版本误记为 $P_{\mathrm{rms}}$——该量纲是功率不是幅度）；RMS 幅度是其平方根 $A_{\mathrm{rms}}^t$。AGC 增益与归一后波形为

$$
a_t=\frac{1}{A_{\mathrm{rms}}^t},
\qquad
\bar r_q^t[n] = a_t\, r_{\mathrm{rx},q}^t[n].
$$


**四条口径声明（正文必须写明）**


1. **更新粒度**：每通信轮设置一次增益，轮内全部 $S=395$ 个 OFDM 符号共用。不逐采样点更新（否则削顶退化为自适应压缩器）、不逐符号更新（否则抹掉符号间能量差异，而这正是 Top-k support 结构的表现之一）。该粒度与"轮内块衰落 + $b_t$ 逐轮设定"的模型假设自洽，但**不宣称是唯一正确实现**——实际硬件 AGC 可按突发/帧/时隙或慢跟踪周期更新。
2. **非因果理想化**：用整轮全部样本计算 $P_{\mathrm{avg}}^t$ 再回头设置该轮增益是非因果的，正文命名为"**理想逐轮 RMS-AGC**"（ideal per-round RMS-AGC）。物理对应物：基站在轮初利用前导/参考信号估计接收 RMS 功率后锁定增益。不得暗示实际硬件能提前知道整轮波形。
3. **SNR 与隐私不变性**：AGC 对信号和噪声施加同一增益，是共同缩放，不改变削顶前的 SNR、不改变 $b_t$ 与隐私的任何比值关系。它不是"把噪声重新归一化成 1"，而是接收机增益控制的仿真抽象。
4. **逐轮隐私与后处理不变性**：第 $t$ 轮条件于相同的先前公开 transcript、当前公开信道状态 $\{g_i^t\}_{i=1}^{N}$ 和公开 $b_t$，查询敏感度不超过 $b_t\Delta(k)$。取 $b_t\le B_{\epsilon,\mathrm{ex}}(k)$ 时，该轮分别满足客户端级 $(\epsilon,\delta)$-DP。AGC、径向削顶、FFT、除以 $N$ 和模型更新均是含噪 AirComp 观察的后处理，不增加该轮隐私泄漏。本文不声称完整 $T$ 轮 transcript 满足相同的 $(\epsilon,\delta)$。


**理想 AGC 下的尺度不变性与 $b_t$ 的正确解释**


理想逐轮 RMS-AGC 会消除接收波形的公共绝对幅度尺度：忽略噪声时，对整个接收信号乘任意公共比例 $c>0$，

$$
\frac{c\,r_{\mathrm{sig}}}{\sqrt{\mathbb E|c\,r_{\mathrm{sig}}|^2}}
=\frac{r_{\mathrm{sig}}}{\sqrt{\mathbb E|r_{\mathrm{sig}}|^2}},
$$

归一化波形不变。由此在高 SNR 极限下有三条推论：

- 增大 $b_t$ **不会**因绝对接收幅度变大而直接增加削顶；
- 减小发射功率 **不会**因绝对幅度降低而直接减少削顶；
- Top-k 降低发送总能量的效果会被理想 AGC 的增益补偿掉——留下来的只有**归一化波形的峰值结构差异**。

因此本模型真正研究的是：**Top-k、Rand-k 和 Full 改变归一化波形的峰值结构，从而改变相同相对动态范围下的削顶失真**。$b_t$ 的主要作用仍是控制恢复后的等效信道噪声 $z_m^t/(b_tN)$ 并在功率与隐私约束间权衡；$b_t$ 只通过改变含噪波形中信号与噪声的**比例**间接影响削顶统计——高 SNR 时该间接影响弱（波形由信号结构主导），低 SNR 时噪声占比上升、削顶统计趋向高斯波形行为。

这里的 $b_t=b_t^\star(k)$ 由当前轮公开信道上限与单轮隐私上限共同确定。公开规则只使用 $\{g_i^t\}$、$P_{\mathrm{cap}}$、$S$、$M$、$k$ 和 $c_{\mathrm{tx}}$，不使用实际更新范数。这样既保留稀疏上传相对 Full 获得更大可行缩放的核心机制，也避免公共缩放本身成为当前私有更新的函数。相同 $k$ 下 Top-k 与 Rand-k 具有相同最坏情况功率/隐私缩放上界，其差异来自更新保留、支持重叠和实际波形。

**措辞禁令**：正文不得写"较大的 $b_t$ 直接导致接收前端输入幅度更大、从而更容易削顶"。正确表述："理想 RMS-AGC 消除了公共绝对幅度尺度，削顶主要由归一化波形的峰值结构决定；$b_t$ 通过改变信号与噪声比例间接影响含噪削顶统计。"这直接约束"隐私—功率—PAPR"三者关系的解释方式。

---

**削顶算子与命名声明**


AGC 后施加保持复相位的径向限幅：

$$
\mathcal C_\gamma(u) =
\begin{cases}
u, & |u|\le\gamma,\\[2mm]
\gamma\,\dfrac{u}{|u|}, & |u|>\gamma.
\end{cases}
$$

**命名声明（正文必须遵守）**：该算子是圆形复包络限幅器，是"接收前端有限**复包络**动态范围"的简化抽象，**不是典型 I/Q ADC 的精确饱和方式**。Rietman & Linnartz（TWC 2008）的实际模型是实部、虚部分别经过 ADC 的逐分量离散化与削顶（原文对 in-phase 与 quadrature 分量分别量化，削顶判据按 $|\operatorname{Re}(y_n)|$、$|\operatorname{Im}(y_n)|$ 与 $C$ 逐分量比较，即矩形 I/Q 削顶），与本文的径向（圆形）削顶几何不同。因此：

- 正文使用"**接收端有限动态范围削顶**"或"**复包络饱和削顶**"称呼本模型，**不得称为"精确 ADC 削顶模型"**；
- Rietman & Linnartz 用于支撑"接收端动态范围、headroom、backoff 与削顶问题的存在性、$P/C^2$ 尺度律与 3–11 dB backoff 扫描区间"，**不用于支撑径向算子的具体几何形式**；
- 建模声明模板：本文采用高分辨率、有限复包络动态范围的接收前端抽象；径向限幅保持复信号相位，用于刻画接收波形峰值越过动态范围产生的饱和失真，不声称复现具体 I/Q ADC 的逐分量转移特性。


**回退（backoff）定义**


AGC 后波形 RMS 为 1，因此相对门限与回退的换算为

$$
\gamma = 10^{B_{\mathrm{clip}}/20},
\qquad
B_{\mathrm{clip}} = 20\log_{10}\gamma\ \ [\mathrm{dB}].
$$

等价的绝对门限（供理论附录使用）为轮级随机量

$$
A_{\max}^t = \gamma A_{\mathrm{rms}}^t = \gamma\sqrt{P_{\mathrm{avg}}^t}.
$$

尺度不变性依据：Rietman & Linnartz（TWC 2008）证明削顶误差只通过比值 $P/C^2$（平均功率/削顶电平平方）起作用，与绝对幅度无关——这是门限相对 RMS 定义、整个削顶模型尺度不变的文献依据。


**削顶残差**


$$
e_{\mathrm{clip},q}^t[n] = \mathcal C_\gamma\big(\bar r_q^t[n]\big) - \bar r_q^t[n],
$$

削顶后的波形经酉 FFT 回到频域，得到受削顶失真影响的聚合结果；残差的频域投影（数据子载波部分）即等效削顶聚合误差。


**AGC 逆缩放与模型更新恢复（完整公式）**


令 AGC 增益 $a_t = 1/A_{\mathrm{rms}}^t$（$A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}$，见本节“理想逐轮 RMS-AGC”），完整接收链为

$$
\bar r_q^t[n] = a_t\, r_{\mathrm{rx},q}^t[n]
\;\xrightarrow{\ \text{削顶}\ }\;
\widetilde r_q^t[n] = \mathcal C_\gamma\!\big(a_t\, r_{\mathrm{rx},q}^t[n]\big)
\;\xrightarrow{\ \text{酉 FFT}\ }\;
\widetilde Y_{q,m}^t .
$$

恢复模型更新时**必须先除以 $a_t$ 还原物理接收幅度，再除以 AirComp 缩放**：

$$
\widehat s_{q,m}^t
=
\frac{\operatorname{Re}\{a_t^{-1}\widetilde Y_{q,m}^t\}}
{b_tN}.
$$

**实现警告**：若代码只除以 $b_tN$ 而不除以 $a_t$（即不乘 $A_{\mathrm{rms}}^t$），AGC 增益不会自动对消，模型更新会被额外缩放 $a_t$ 倍——"AGC 在恢复缩放中对消"指的是显式执行本节 $a_t^{-1}$ 逆缩放后的净效果，不是自动发生的。

---

**主实验档位**


$$
B_{\mathrm{clip}} \in \{3,\ 6,\ 9,\ \infty\}\ \mathrm{dB}.
$$

| 档位 | $\gamma$ | 逐样本越限概率（复高斯近似 $e^{-\gamma^2}$） | 物理分工 | 文献锚点 |
|---|---:|---:|---|---|
| 3 dB | 1.413 | $13.6\%$ | 强削顶压力测试 | Rietman & Linnartz（TWC 2008）Fig. 4 backoff 扫描下端；发射端主动削顶文献的削顶比区间（约 1.4–2 即 3–6 dB） |
| **6 dB** | **1.995** | **$1.87\%$** | **基准工作点**：失真可测量、但不淹没信道噪声，用于区分 Top-k/Rand-k/Full | 位于"主动削顶"与"工程 headroom"之间 |
| 9 dB | 2.818 | $3.54\times10^{-4}$ | 弱削顶 | ≈ Rietman & Linnartz 的 8–9 dB 工程 headroom 惯例 |
| $\infty$ | — | 0 | 理想不削顶接收机（性能上界参照） | — |

**扫描区间的直接文献出处（原文已核实）**：Rietman & Linnartz（TWC 2008）Fig. 4 在 802.11a 接收机仿真中扫描 **AGC back-off 3–11 dB**（原文："for various values of the AGC back-off from 3 to 11 dB"），且其 AGC 定义为把接收信号功率缩放到 $10^{-\mathrm{BackOff_{dB}}/10}$——与本文"相对 RMS 的回退"同一口径。本文主实验档 $\{3,6,9\}$ 与附录档 11 dB 全部落在该扫描区间内，扫描范围有直接期刊出处，**不需要 Bielefeld（在投）作参数依据**——3 dB 档直接说明为强削顶压力测试即可。

**11 dB 档降入附录的定量依据**

11 dB（$\gamma=3.548$）的逐样本越限概率为 $3.40\times10^{-6}$。每轮时域样本数为
$$
S \times Q = 395 \times 4096 \approx 1.62\times10^{6},
$$

在复高斯近似下期望每轮仅约 $1.62\times10^6 \times 3.4\times10^{-6} \approx 5.5$ 个样本越限，且每个越限样本超出幅度极小，**预期**其削顶残差能量远低于信道噪声、结果与不削顶接近。因此 11 dB **不进主实验四曲线**，仅保留在附录参数敏感性扫描中（其叙事价值：8–9 dB 工程惯例的"外一档"验证，同时是 Rietman Fig. 4 扫描区间的上端点）。




**档位定性声明（正文措辞约束）**


- 6 dB 基准必须写成"**文献支持范围内选择的基准值**"，**不得写成 3GPP 规定值或行业标准值**——不存在适用于所有接收机的统一 ADC 削顶回退标准，实际动态范围取决于 AGC 策略、ADC 满量程与硬件实现；
- 采用多档扫描本身就是对"无统一标准"的稳妥回应。


**削顶概率速查（含附录档）**


| $B_{\mathrm{clip}}$ | 3 dB | 4 dB | 6 dB | 8 dB | 9 dB | 11 dB |
|---|---:|---:|---:|---:|---:|---:|
| $\gamma$ | 1.413 | 1.585 | 1.995 | 2.512 | 2.818 | 3.548 |
| $e^{-\gamma^2}$ | $13.6\%$ | $8.1\%$ | $1.87\%$ | $0.18\%$ | $3.54\times10^{-4}$ | $3.40\times10^{-6}$ |

（勘误记录：早期审计报告曾把 4 dB 误算为 6.6%，正确值为 $e^{-2.512}\approx8.1\%$。）

**适用范围声明**：$\Pr(|u|>\gamma)=e^{-\gamma^2}$ 仅对 $u\sim\mathcal{CN}(0,1)$ 的独立样本严格成立。本文的 Top-k 稀疏 AirComp 波形不一定严格高斯（稀疏度高、活跃子载波少时偏离中心极限条件），且过采样样本之间存在相关性。因此本表**只用于数量级自检**（第 5 节“仿真实现要点” 自检项），不作为实际削顶率的理论真值；正文引用时必须带"复高斯近似下"限定语。

---

| 项 | 设置 | 依据 |
|---|---|---|
| 基准 $L_{\mathrm{os}}$ | 4 | Tellambura（IEEE Commun. Lett. 2001）：$\ge4$ 倍过采样足以逼近连续时间波形真实峰值；Bielefeld（在投）同款 $L_{\mathrm{os}}=4$ 作旁证 |
| 精度验证 | 8 | 仅在少量波形级实验中验证 $L_{\mathrm{os}}=4$ 的 PAPR/削顶统计已收敛，不进主实验 |

---



# 4. 评价指标、隐私与收敛性对应

实际恢复结果为

$$
\widehat{\mathbf s}_{\mathrm{clip}}^t=\frac1N\sum_{i=1}^{N}\mathbf s_i^t+\mathbf e_{\mathrm{ch}}^t+\mathbf e_{\mathrm{dr}}^t,
$$

其中 $\mathbf e_{\mathrm{ch}}^t=\operatorname{Re}\{\mathbf z^t\}/(b_tN)$，$\mathbf e_{\mathrm{dr}}^t$ 是径向削顶残差经过 AGC 逆缩放、FFT、实部提取和 $b_tN$ 归一化后的等效聚合误差。收敛界的通信部分为

$$
\Phi_{\mathrm{ch}}\propto\frac1T\sum_{t=0}^{T-1}\frac1{b_t^2N^2},
$$

以及

$$
\Phi_{\mathrm{dr}}\propto\frac1T\sum_{t=0}^{T-1}\mathbb E\left[\frac{E_{\mathrm{clip}}^t(k,b_t)}{b_t^2N^2}\right].
$$

由于每个逻辑轮固定聚合全部 $N$ 个客户端，收敛界不包含无线参与抽样误差。路径损耗和衰落只通过 $B_P^t(k)$、$b_t^\star(k)$ 和实际波形进入通信误差，不改变学习参与集合。

只要 $b_t=b_t^\star(k)=\min\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\}$，第 $t$ 轮分别满足客户端级 $(\epsilon,\delta)$-DP；收敛定理保留实际序列 $\{b_t^\star(k)\}$。AGC、径向削顶、FFT 和模型更新均为该轮含噪观察的后处理。

评价指标分为结构波形指标、硬件失真指标、总通信 NMSE 和学习指标。定义

$$
\mathrm{NMSE}_{\mathrm{clip}}^t=\frac{\|\widehat{\mathbf s}_{\mathrm{clip}}^t-\widehat{\mathbf s}_{\mathrm{lin}}^t\|_2^2}{\|\widehat{\mathbf s}_{\mathrm{lin}}^t\|_2^2},
$$

以及

$$
\mathrm{NMSE}_{\mathrm{total}}^t=\frac{\|\widehat{\mathbf s}_{\mathrm{clip}}^t-\mathbf s_{\mathrm{ideal}}^t\|_2^2}{\|\mathbf s_{\mathrm{ideal}}^t\|_2^2},\qquad \mathbf s_{\mathrm{ideal}}^t=\frac1N\sum_{i=1}^{N}\mathbf s_i^t.
$$

| 检查项 | 统一后的口径 | 结论 |
|---|---|---|
| 客户端参与 | 每个逻辑轮全部 $N$ 个客户端成功上传 | 固定学习参与集合 |
| 聚合分母 | 固定为 $N$ | 与全部客户端的目标函数一致 |
| 公共缩放 | $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\}$ | 每轮可随信道变化 |
| 隐私 | 每轮分别满足 replace-one 客户端级 $(\epsilon,\delta)$-DP | 无跨轮预算分配 |
| 理想逐轮 AGC | 显式定义并在恢复时逆缩放 | 与物理链一致 |
| 径向削顶 | $\mathcal C_\gamma(z)=z\min\{1,\gamma/|z|\}$ | 非精确 I/Q ADC |
| 收敛通信项 | $1/(b_t^2N^2)$ 与 $E_{\mathrm{clip}}^t/(b_t^2N^2)$ | 仅信道噪声与径向残差 |

# 5. 参数设置、文献依据与仿真实现

**三层引用结构**


| 层 | 文献 | 支撑的命题 | 状态 |
|---|---|---|---|
| 机制层 | Rietman & Linnartz, IEEE TWC 2008 | 有限动态范围、8–9 dB headroom 惯例、$P/C^2$ 尺度律、"量化级数足够大只留削顶"口径、**Fig. 4 的 3–11 dB AGC backoff 扫描区间**。注意：其 ADC 模型为 I/Q 逐分量离散化削顶（矩形），本文引用其问题定义与参数区间，**不引用其算子几何**（见本节“径向限幅与命名声明”） | 已发表 ✅ |
| 场景层 | Hellström, Fodor, Fischione 等, "Federated Learning Over-the-Air by Retransmissions", IEEE TWC, vol. 22, no. 12, 2023（DOI 10.1109/TWC.2023.3268742；会议版 GLOBECOM 2022） | AirComp 聚合误差影响学习性能、物理层处理可改善。**注意：该文研究截断反演功控错位与信道噪声引起的估计误差及重传补救，全文无 clipping/saturation/ADC 动态范围建模，严禁写成接收端削顶的直接相关工作** | 已发表 ✅ |
| 本文 | —— | 稀疏结构 → 接收端 PAPR → 削顶失真：场景层与机制层之间的空白交叉点 | —— |

**边界文献（相关工作中一句话定位）**


| 文献 | 定位 | 状态 |
|---|---|---|
| Tegin & Duman, IEEE TWC 2021 | 低分辨率 ADC/DAC + Bussgang 是**正交扩展方向**，本文暂不建模 | 已发表 ✅ |
| Bielefeld 等, arXiv:2512.23381 | 发射端 PA 峰值约束的**同期在投背景**，与本文接收端口径互补；**只作背景引用，不作参数来源**；引用标注必须为 "submitted / arXiv preprint" | 在投 ⚠️ |
| Tellambura, IEEE Commun. Lett. 2001 | 过采样倍数 $L_{\mathrm{os}}=4$ 的主来源 | 已发表 ✅ |
| Şahin & Yang, IEEE COMST 2023 | AirComp 综述，将 PAPR 列为实践挑战——领域定位引用 | 已发表 ✅ |
| Demir & Björnson, "The Bussgang Decomposition of Nonlinear Systems", IEEE SPM Lecture Notes（IEEE 文档 9307295） | Bussgang 适用于任意无记忆非线性的出处（若正文需要提及"本文不依赖 Bussgang 近似"时引用） | 已发表 ✅ |



---

| 参数 | 设置 | 用途 | 来源 |
|---|---:|---|---|
| 全局调度 | 每轮全部 $N$ 个客户端 | 无算法层客户端采样 | 系统模型选择 |
| 全客户端成功聚合 | 全部 $N$ 个客户端；候选深衰落突发等待或重调度 | 数据无关截断反演 | 与收敛/隐私统一 |
| 模型映射 | 一个实坐标占一个复子载波；FFT 后取实部 | 不使用 I/Q 双坐标打包 | 系统模型选择 |
| 公共缩放 | 唯一采用 $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{ex}}(k),B_P^t(k)\}$ | 单轮客户端级隐私、逐轮平均功率，且不依赖实际私有更新范数 | 功率、隐私与核心机制统一 |
| 过采样倍数 $L_{\mathrm{os}}$ | 4 | 正式实验默认 | Tellambura 2001 |
| 精度验证过采样 | 8 | 仅波形级验证 | 仿真自检 |
| 基准削顶回退 | 6 dB（$\gamma\approx1.995$，越限率 $\approx1.9\%$） | 主实验工作点 | Rietman & Linnartz 区间内的文献支持基准值（非标准值） |
| 主实验回退扫描 | $\{3,6,9,\infty\}$ dB | 强削顶 / 基准 / 弱削顶 / 理想上界 | Rietman & Linnartz Fig. 4 的 3–11 dB AGC backoff 扫描区间（原文已核实） |
| 附录敏感性档 | 11 dB | 8–9 dB 惯例外一档验证 | 复高斯近似下预期接近 $\infty$，由附录实际波形验证 |
| ADC 量化 | 不建模（高分辨率假设） | 只留饱和削顶 | Rietman & Linnartz "$L$ 足够大"口径 |
| AGC 更新周期 | 每轮一次，轮内固定 | 理想逐轮 RMS-AGC（非因果声明见 本节“AGC 口径声明”） | 与块衰落/逐轮 $b_t$ 假设自洽 |
| 削顶方式 | 复包络径向限幅（保相位） | 接收前端动态范围抽象，非 I/Q 逐分量特性（见本节“径向限幅与命名声明”） | 本文建模选择；Rietman 仅支撑问题与参数区间 |
| AGC 逆缩放 | 恢复时显式乘 $a_t^{-1}=A_{\mathrm{rms}}^t$ | 见本节“AGC 逆缩放与恢复” | 实现必需，防止模型更新被额外缩放 |
| 轮归一化峰值压力 PSR | $\mathrm{PSR}_q^t$ CCDF + $\mathrm{PSR}_{\mathrm{round}}^t$ | 衔接逐符号 PAPR 与逐轮门限（本节“结构线”） | 归一化基准匹配所需（本文设计） |
| 隐私后处理声明 | AGC/削顶为含噪输出后的确定性后处理 | 不削弱该通信轮已经建立的客户端级 DP（本节“AGC口径声明”第4条） | DP 后处理不变性 |
| $b_t$ 解释约束 | 禁写"$b_t$ 大→更易削顶"；削顶由归一化峰值结构决定 | 见本节“尺度不变性与 $b_t$” 尺度不变性 | 理想 AGC 推论 |
| 全零符号处理 | 不进 PAPR CCDF；单独报告静默符号比例 | 见本节“结构线” | PAPR 分母为零无定义 |
| PAPR 统计口径 | 结构线无噪：PAPR CCDF（读 $10^{-3}$ 点）+ PSR | 稀疏结构证据 | 见本节“结构线” |
| 削顶统计口径 | 含噪接收前端输入：$\rho_{\mathrm{clip}}$、$D_{\mathrm{clip}}$、$\mathrm{NMSE}_{\mathrm{clip}}$/$\mathrm{NMSE}_{\mathrm{total}}$ | 动态范围失真证据 | 见第3节“硬件线”和第4节 |
| 噪声生成 | 频域 $\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2)$ 与信号同 IFFT | 避免 FFT 归一化重复缩放 | 主场景文档的物理噪声部分 |
| 评价链 | {PAPR CCDF + PSR} → $\rho_{\mathrm{clip}}$ → $D_{\mathrm{clip}}$ → {$\mathrm{NMSE}_{\mathrm{clip}}$, $\mathrm{NMSE}_{\mathrm{total}}$} → 测试精度 | 完整证据链（七项） | 见第 4 节 |

---

1. **全客户端与波形双份**：每个计入学习递推的逻辑轮均由全部 $N$ 个客户端完成聚合。对每个 OFDM 符号同时生成 $r_{\mathrm{sig}}$（结构线）与 $r_{\mathrm{rx}}$（含噪动态范围线），两者使用同一组客户端、同一频域更新、同一当前信道、同一 $b_t^\star(k)$ 和同一个酉 IFFT，仅相差接收噪声项；
2. **AGC 实现**：整轮 $S\times Q$ 个含噪样本一次性算 $P_{\mathrm{avg}}^t$（$A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}$），归一后统一施加 $\mathcal C_\gamma$；禁止逐符号归一；
3. **削顶后链路（含 AGC 逆缩放）**：削顶波形经酉 FFT → 取数据子载波 → **先乘 $a_t^{-1}=A_{\mathrm{rms}}^t$ 还原物理幅度** → 再取实部并除以 $b_tN$ 恢复模型更新（见本节“AGC 逆缩放与恢复”；漏掉 $a_t^{-1}$ 是最容易犯的实现错误）；
4. **自检项**：仿真输出的 $\rho_{\mathrm{clip}}$ 应与本节“削顶概率近似”表在同档位下**同量级**（仅数量级校验：实际波形非严格高斯、过采样样本相关，允许偏离）；$L_{\mathrm{os}}=8$ 复跑少量轮验证 CCDF 曲线重合；全链路（信号/噪声/AGC/门限/恢复）确认使用同一 FFT 归一化约定（第 2 节“过采样时域波形”）；
5. **$\infty$ 档实现**：直接旁路削顶算子（不是设一个很大的 $\gamma$），保证参照曲线严格无失真；
6. **统计粒度**：PAPR CCDF 与 PSR CCDF 按 OFDM 符号聚合全轮全种子（**剔除全零符号**，静默符号比例单独输出），$\mathrm{PSR}_{\mathrm{round}}^t$ 按轮输出；$\rho_{\mathrm{clip}}$、$D_{\mathrm{clip}}$ 按轮记录时间序列（观察训练过程中稀疏结构演化对削顶压力的影响）；
7. **NMSE 双链实现**：同一轮内并行跑削顶链与旁路链，客户端更新、信道、$b_t$、**噪声实现完全相同**（固定 RNG 状态/复用同一噪声张量），唯一差别是是否经过 $\mathcal C_\gamma$，由此计算 $\mathrm{NMSE}_{\mathrm{clip}}$；另存无噪真实平均 $\mathbf s_{\mathrm{ideal}}^t$ 计算 $\mathrm{NMSE}_{\mathrm{total}}$（第 4 节的两个定义不得混用）。

---
