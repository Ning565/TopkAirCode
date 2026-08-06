# AirComp 场景下的 PAPR 与接收端削顶场景设置

本文档是《[无线联邦学习_AirComp_实验场景与通信参数设置](./无线联邦学习_AirComp_实验场景与通信参数设置.md)》中 PAPR、AGC 与接收端有限动态范围部分的独立展开版。本文保留已确定的物理信号链、公开缩放、无线活跃集合、径向限幅、参数设置、评价指标和收敛性对应关系，并采用较少的章节组织。

## 1. 整体场景与建模边界

**主场景一句话定义**


> OFDM AirComp-FL 系统每轮调度全部 $N$ 个客户端，不进行算法层面的客户端采样；数据无关的 Rayleigh 小尺度信道和截断反演产生实际无线活跃集合 $\mathcal A_t$。客户端上传实值模型更新，每个实坐标占用一个复 OFDM 子载波资源，复信道预均衡后在基站实轴对齐，FFT 后取实部恢复。公共 AirComp 缩放 $b_t$ 仅依赖公开信道、公开稀疏度和公开裁剪上界，不依赖实际私有更新范数。基站采用高分辨率但有限复包络动态范围的接收前端，忽略有限位量化；每轮通过理想 RMS-AGC 设置一次增益，轮内固定。结构性 PAPR 使用无噪聚合信号统计，实际径向削顶作用于含信号和噪声的接收波形。主实验扫描 3、6、9 dB 回退和不削顶四种情况，并通过标准 PAPR、轮归一化峰值压力（PSR）、削顶率、削顶失真、聚合 NMSE 与测试精度评价完整影响链。


**纳入主模型的组件**


| 组件 | 口径 |
|---|---|
| 接收机 ADC | 高分辨率（量化噪声忽略）+ 有限输入动态范围 |
| 削顶机制 | AGC 后复包络径向限幅（保相位，有限动态范围抽象），作用于含噪 ADC 输入波形 |
| AGC | 理想逐轮 RMS-AGC，轮内 395 个 OFDM 符号共用同一增益 |
| PAPR 统计 | 双口径：结构线（无噪）+ 硬件线（含噪），见本文件第 3 节 |
| 过采样 | $L_{\mathrm{os}}=4$ 基准，$L_{\mathrm{os}}=8$ 仅少量波形实验验证 |
| 客户端参与 | 全客户端调度；$\mathcal A_t$ 仅由数据无关信道截断产生 | 
| 模型映射 | 一个实模型坐标占用一个复子载波；FFT 后取实部恢复 | 
| 公共缩放 | $b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$，不依赖实际私有更新范数 |


**明确排除的组件（建模边界）**


以下机制**不进入主模型**，仅在相关工作或未来扩展中一句话提及：

- **有限位量化 / Bussgang 量化模型**：低分辨率 ADC/DAC 是独立扩展方向（Tegin & Duman, IEEE TWC 2021 已专门研究），本文假设量化级数充分大、只保留削顶失真——与 Rietman & Linnartz（TWC 2008）"$L$ 足够大"的分析口径同款。注意：Bussgang 分解本身适用于高斯输入的任意无记忆非线性（含削顶），本文不用它不是因为不适用，而是因为实验直接对波形执行削顶算子、理论保留精确残差能量，**不需要高斯近似**——这是与 Tegin & Duman 的方法论区分点，可作为卖点写入正文。
- **重传机制**：Hellström 等（TWC 2023）的方向，与本文正交。
- **发射端功放非线性 / ACLR / ICF 削顶**：Bielefeld 等（arXiv:2512.23381，在投）的发射端方向，与本文接收端口径互补。
- **完整频率选择性多径**：主场景文档的客户端级频率平坦块衰落假设不变。
- **复杂削顶恢复/校正算法**：本文只评估被动削顶失真，不引入接收端补偿。

---

## 2. AirComp 信号链与过采样波形

**全客户端调度、信道活跃集合与频域聚合**


本文**不进行算法层面的客户端采样**。每个通信轮均调度全部 $N$ 个客户端进行本地训练并形成待发送稀疏更新；随后由数据无关的小尺度信道状态决定哪些客户端能够执行截断信道反演。定义

$$
I_i^t=\boldsymbol{1}\{|h_i^t|\ge h_{\mathrm{cut}}\},
\qquad
\mathcal A_t=\{i:I_i^t=1\},
\qquad
N_t=|\mathcal A_t|.
$$

因此，**全调度** 与 **实际上行活跃** 是两个不同概念：所有客户端均参与本地训练，但只有 $i\in\mathcal A_t$ 的客户端进入本轮无线 AirComp 聚合。**$\mathcal A_t$** 仅由公开信道状态产生，与客户端数据、本地梯度和稀疏支持无关。由于所有客户端的归一化小尺度衰落均满足 $h_i^t\sim\mathcal{CN}(0,1)$，其激活概率相同；条件于 $N_t=n$，$\mathcal A_t$ 是大小为 $n$ 的等概率随机子集。若极小概率事件 $N_t=0$ 发生，则该轮跳过，不计为一次有效全局更新。

模型更新为实向量。每个实模型坐标占用一个复 OFDM 数据子载波资源，不使用 I/Q 两路分别承载两个模型坐标，也不施加 Hermitian 对称。令

$$
j(q,m)=(q-1)M+m+1,
$$

则

$$
s_{i,q,m}^t=
\begin{cases}
[\mathbf s_i^t]_{j(q,m)}, & j(q,m)\le d,\\
0, & j(q,m)>d,
\end{cases}
\qquad
s_{i,q,m}^t\in\mathbb R.
$$

实际发送符号因复信道预均衡而为复数，但理想信道对齐后的有效聚合信号位于实轴，基站 FFT 后取实部恢复模型坐标。

AirComp 缩放系数只依赖公开信道与公开裁剪上界，不依赖实际私有更新范数。令公开逐坐标门限为

$$
c_{\mathrm{tx}}=\eta\tau C.
$$

本文将 $P_{\mathrm{cap}}$ 定义为客户端在固定 $S$ 个 OFDM 符号上行突发内的**整轮平均发射功率预算**。采用酉 IFFT 时，客户端 $i$ 的整轮平均发射功率为

$$
P_i^t
=
\frac{1}{SM}
\sum_{q=1}^{S}\sum_{m=0}^{M-1}|x_{i,q,m}^t|^2
=
\frac{b_t^2}{SM|g_i^t|^2}\|\mathbf s_i^t\|_2^2.
$$

由于公开稀疏与逐坐标裁剪约束满足

$$
\|\mathbf s_i^t\|_0\le k,
\qquad
\|\mathbf s_i^t\|_\infty\le c_{\mathrm{tx}},
$$

从而 $\|\mathbf s_i^t\|_2\le c_{\mathrm{tx}}\sqrt{k}$，公开功率上界写为

$$
B_P^t(k)
=
\min_{i\in\mathcal A_t}
\frac{|g_i^t|\sqrt{SM P_{\mathrm{cap}}}}
{c_{\mathrm{tx}}\sqrt{k}}.
$$

该定义对完整固定资源栅格的平均功率进行约束，因此保留 $B_P^t(k)\propto1/\sqrt{k}$ 的稀疏度增益。短时发射峰值、功率放大器非线性和发射端 PAPR 不进入本文主模型。

在固定公开活跃集合下采用 replace-one 客户端级邻接，稀疏查询的最坏情况敏感度为

$$
\Delta(k)=2c_{\mathrm{tx}}\sqrt{k}.
$$

对于单轮 $(\epsilon,\delta)$ 隐私，公开隐私上界为

$$
B_{\epsilon,\delta}(k)
=
\frac{\sigma_{\mathrm R}}{\Delta(k)}
\left(
\sqrt{\epsilon+\ln(1/\delta)}
-
\sqrt{\ln(1/\delta)}
\right),
$$

其中

$$
\sigma_{\mathrm R}
=\sqrt{\frac{\sigma_{\mathrm{sc}}^2}{2}}
=\frac{\sigma_{\mathrm{sc}}}{\sqrt2},
$$

$\sigma_{\mathrm{sc}}^2$ 是复频域噪声功率，$\sigma_{\mathrm R}$ 是进入实模型恢复和隐私校准的实部噪声标准差。最终使用

$$
b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}.
$$

该规则保留 $k$ 减小时功率和隐私允许的 $b_t$ 增大这一核心机制，同时避免公共缩放系数随私有更新范数变化。

第 $t$ 轮第 $q$ 个 OFDM 符号（$q=1,\dots,S$，$S=395$）第 $m$ 个子载波（$m=0,\dots,M-1$，$M=1024$）上的基站接收频域信号为

$$
y_{q,m}^t
=
b_t\sum_{i\in\mathcal A_t}s_{i,q,m}^t+z_{q,m}^t,
\qquad
z_{q,m}^t\sim\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2),
$$

其中

$$
\sigma_{\mathrm{sc}}^2=N_0\Delta fF
\approx1.89\times10^{-16}\ \mathrm W
$$

（$-127.2$ dBm），不归一化为 1。



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

时域峰值来自两级叠加：其一是 $N_t$ 个活跃客户端在同一子载波上的 AirComp 叠加；其二是多个子载波在 IFFT 中的相干叠加。Top-k、Rand-k 和 Full 通过活跃子载波结构、客户端支持重叠以及非零更新值的幅度和符号关系共同影响最终波形。


**波形的两个版本**


定义两条共享同一活跃集合、同一频域更新和同一过采样归一化的波形：

$$
r_{\mathrm{sig},q}^t[n]
=
\mathrm{IFFT}_Q^{\mathrm u}
\!\left(
\sqrt{\frac QM}\,
\mathcal Z_{\mathrm{os}}
\left(
 b_t\sum_{i\in\mathcal A_t}s_{i,q,m}^t
\right)
\right)[n],
$$

即无噪聚合信号；以及

$$
r_{\mathrm{ADC},q}^t[n]
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

## 3. PAPR、AGC 与有限动态范围削顶

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
- **全零符号规则**：若某 OFDM 符号上所有活跃客户端均静默（该符号无噪聚合信号全零），PAPR 分母为零、无定义——该符号**不进入 PAPR CCDF 统计**，并**单独报告全零/静默 OFDM 符号比例**（高稀疏率下 Top-k/Rand-k 可能出现，且该比例本身就是稀疏结构的信息量）。

**轮归一化峰值压力（PSR）——衔接逐符号 PAPR 与逐轮门限**：标准 PAPR 按每个 OFDM 符号自身平均功率归一，而实际削顶门限 $A_{\max}^t=\gamma A_{\mathrm{rms}}^t$（$A_{\mathrm{rms}}^t$ 的正式定义见下文“定义”段）按**整轮**平均功率设置，二者归一化基准不一致：一个符号自身 PAPR 不高、但整体能量远高于本轮均值时仍可能频繁削顶；反之符号 PAPR 高但能量很小则未必越过逐轮门限。因此在标准 PAPR CCDF 之外，增加与逐轮 AGC 直接对应的指标

$$
\mathrm{PSR}_q^t
=\frac{\max_n|r_{\mathrm{sig},q}^t[n]|^2}
{\frac{1}{SQ}\sum_{q',n}|r_{\mathrm{sig},q'}^t[n]|^2},
\qquad
\mathrm{PSR}_{\mathrm{round}}^t=\max_q \mathrm{PSR}_q^t ,
$$

分别以 CCDF（逐符号）与轮标量（整轮最大值）形式报告。证据链相应修正为：**标准 PAPR + 轮归一化峰值压力 → 实际削顶率 → 削顶失真**——单说"PAPR 降低所以削顶减少"并不总是严格成立，必须由 PSR 补全归一化基准的衔接。


**硬件线——含噪削顶（回答"实际接收机受到多大失真"）**


削顶及其所有派生统计一律作用于含噪 ADC 输入 $r_{\mathrm{ADC},q}^t[n]$。物理依据：热噪声在 LNA 处即混入，ADC 看到的必然是信号+噪声之和；此口径同时与收敛性分析附录的 ADC 输入定义严格一致（附录中 ADC 输入波形由含噪频域聚合向量经酉 IFFT 得到）。



**定义**


基站在每个通信轮利用该轮**含噪**接收波形的全部样本估计平均功率：

$$
P_{\mathrm{avg}}^t
= \frac{1}{S\,Q}
\sum_{q=1}^{S}\sum_{n=0}^{Q-1}
\big|r_{\mathrm{ADC},q}^t[n]\big|^2,
\qquad
A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}.
$$

**命名说明**：$P_{\mathrm{avg}}^t$ 是平均功率（早期版本误记为 $P_{\mathrm{rms}}$——该量纲是功率不是幅度）；RMS 幅度是其平方根 $A_{\mathrm{rms}}^t$。AGC 增益与归一后波形为

$$
a_t=\frac{1}{A_{\mathrm{rms}}^t},
\qquad
\bar r_q^t[n] = a_t\, r_{\mathrm{ADC},q}^t[n].
$$


**四条口径声明（正文必须写明）**


1. **更新粒度**：每通信轮设置一次增益，轮内全部 $S=395$ 个 OFDM 符号共用。不逐采样点更新（否则削顶退化为自适应压缩器）、不逐符号更新（否则抹掉符号间能量差异，而这正是 Top-k support 结构的表现之一）。该粒度与"轮内块衰落 + $b_t$ 逐轮设定"的模型假设自洽，但**不宣称是唯一正确实现**——实际硬件 AGC 可按突发/帧/时隙或慢跟踪周期更新。
2. **非因果理想化**：用整轮全部样本计算 $P_{\mathrm{avg}}^t$ 再回头设置该轮增益是非因果的，正文命名为"**理想逐轮 RMS-AGC**"（ideal per-round RMS-AGC）。物理对应物：基站在轮初利用前导/参考信号估计接收 RMS 功率后锁定增益。不得暗示实际硬件能提前知道整轮波形。
3. **SNR 与隐私不变性**：AGC 对信号和噪声施加同一增益，是共同缩放，不改变削顶前的 SNR、不改变 $b_t$ 与隐私的任何比值关系。它不是"把噪声重新归一化成 1"，而是接收机增益控制的仿真抽象。
4. **活跃集合条件隐私与后处理不变性**：每轮公开状态为 $\mathcal H_t=(\mathcal A_t,N_t,\{g_i^t\}_{i\in\mathcal A_t},b_t)$。$\mathcal H_t$ 由信道和公开规则产生，与客户端数据独立；条件于该状态，采用固定活跃集合上的 replace-one 客户端级邻接。若改变的客户端不在 $\mathcal A_t$ 中，本轮敏感度为零；若其在 $\mathcal A_t$ 中，敏感度不超过 $b_t\Delta(k)$。由于对每个公开状态均满足相同的单轮 $(\epsilon,\delta)$ 上界，边缘化信道随机性后仍保持同一 DP 保证。AGC、径向削顶、FFT、除以 $N_t$ 和模型更新均是含噪 AirComp 观察后的确定性后处理，不会增加隐私泄漏。


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

因此本模型真正研究的是：**Top-k、Rand-k 和 Full 改变归一化波形的峰值结构，从而改变相同相对动态范围下的削顶失真**。$b_t$ 的主要作用仍是控制恢复后的等效信道噪声 $z_m^t/(b_tN_t)$ 并在功率与隐私约束间权衡；$b_t$ 只通过改变含噪波形中信号与噪声的**比例**间接影响削顶统计——高 SNR 时该间接影响弱（波形由信号结构主导），低 SNR 时噪声占比上升、削顶统计趋向高斯波形行为。

这里的 $b_t$ 由公开规则 $b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$ 决定。公开功率上界只使用 $\mathcal A_t$、$g_i^t$、$P_{\mathrm{cap}}$、$S$、$M$、$k$ 和 $c_{\mathrm{tx}}$，不使用实际更新范数。这样既保留稀疏上传相对 Full 获得更大可行 $b_t$ 的核心机制，也避免公共缩放本身成为私有数据函数。相同 $k$ 下 Top-k 与 Rand-k 具有相同最坏情况功率/隐私缩放上界，其差异来自更新保留、支持重叠和实际波形。

**措辞禁令**：正文不得写"较大的 $b_t$ 直接导致 ADC 输入幅度更大、从而更容易削顶"。正确表述："理想 RMS-AGC 消除了公共绝对幅度尺度，削顶主要由归一化波形的峰值结构决定；$b_t$ 通过改变信号与噪声比例间接影响含噪削顶统计。"这直接约束"隐私—功率—PAPR"三者关系的解释方式。

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


令 AGC 增益 $a_t = 1/A_{\mathrm{rms}}^t$（$A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}$，见本节“定义”段与“四条口径声明”），完整接收链为

$$
\bar r_q^t[n] = a_t\, r_{\mathrm{ADC},q}^t[n]
\;\xrightarrow{\ \text{削顶}\ }\;
\widetilde r_q^t[n] = \mathcal C_\gamma\!\big(a_t\, r_{\mathrm{ADC},q}^t[n]\big)
\;\xrightarrow{\ \text{酉 FFT}\ }\;
\widetilde Y_{q,m}^t .
$$

恢复模型更新时**必须先除以 $a_t$ 还原物理接收幅度，再除以 AirComp 缩放**：

$$
\widehat s_{q,m}^t
=
\frac{\operatorname{Re}\{a_t^{-1}\widetilde Y_{q,m}^t\}}
{b_tN_t}.
$$

**实现警告**：若代码只除以 $b_tN_t$ 而不除以 $a_t$（即不乘 $A_{\mathrm{rms}}^t$），AGC 增益不会自动对消，模型更新会被额外缩放 $a_t$ 倍——"AGC 在恢复缩放中对消"指的是显式执行本节 $a_t^{-1}$ 逆缩放后的净效果，不是自动发生的。

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

**适用范围声明**：$\Pr(|u|>\gamma)=e^{-\gamma^2}$ 仅对 $u\sim\mathcal{CN}(0,1)$ 的独立样本严格成立。本文的 Top-k 稀疏 AirComp 波形不一定严格高斯（稀疏度高、活跃子载波少时偏离中心极限条件），且过采样样本之间存在相关性。因此本表**只用于数量级自检**（第 5 节“仿真实现要点”自检项），不作为实际削顶率的理论真值；正文引用时必须带“复高斯近似下”限定语。

---

| 项 | 设置 | 依据 |
|---|---|---|
| 基准 $L_{\mathrm{os}}$ | 4 | Tellambura（IEEE Commun. Lett. 2001）：$\ge4$ 倍过采样足以逼近连续时间波形真实峰值；Bielefeld（在投）同款 $L_{\mathrm{os}}=4$ 作旁证 |
| 精度验证 | 8 | 仅在少量波形级实验中验证 $L_{\mathrm{os}}=4$ 的 PAPR/削顶统计已收敛，不进主实验 |

---



## 4. 评价指标、隐私与收敛性对应

**核心原则：PAPR 下降不自动等于学习性能提高，必须用完整证据链闭合。** 指标按传导链排列：

$$
\{\text{PAPR CCDF} + \mathrm{PSR}\}
\;\rightarrow\;
\rho_{\mathrm{clip}}
\;\rightarrow\;
D_{\mathrm{clip}}
\;\rightarrow\;
\{\mathrm{NMSE}_{\mathrm{clip}},\ \mathrm{NMSE}_{\mathrm{total}}\}
\;\rightarrow\;
\text{测试精度}.
$$

| # | 指标 | 定义 | 口径 |
|---|---|---|---|
| 1 | PAPR CCDF | $\Pr\{\mathrm{PAPR}_{\mathrm{sig}}>\xi\}$，读 $10^{-3}$ 超越概率点 | **结构线（无噪）** |
| 2 | 轮归一化峰值压力 PSR | $\mathrm{PSR}_q^t$ CCDF 与 $\mathrm{PSR}_{\mathrm{round}}^t$（第 3 节“结构线”），峰值相对整轮平均功率 | **结构线（无噪）** |
| 3 | 削顶比例 | $\rho_{\mathrm{clip}} = \dfrac{\#\{(q,n):\ \lvert\bar r_q^t[n]\rvert>\gamma\}}{S\,Q}$ | **硬件线（含噪）** |
| 4 | 归一化削顶失真 | $D_{\mathrm{clip}} = \dfrac{\sum_{q,n}\lvert\mathcal C_\gamma(\bar r_q^t[n])-\bar r_q^t[n]\rvert^2}{\sum_{q,n}\lvert\bar r_q^t[n]\rvert^2}$ | **硬件线（含噪）** |
| 5 | 削顶专属 NMSE | $\mathrm{NMSE}_{\mathrm{clip}}^t$（下方定义，同噪声实现对照） | 硬件线（隔离削顶贡献） |
| 6 | 总聚合 NMSE | $\mathrm{NMSE}_{\mathrm{total}}^t$（下方定义，相对无噪无削顶真值） | 硬件线（总链路） |
| 7 | 测试精度 | 200 轮训练后的全局模型测试集精度 | 端到端 |

**聚合 NMSE 的两个定义（不得混用）**：

削顶专属 NMSE——两条链路使用**相同客户端更新、相同信道、相同 $b_t$、相同噪声实现**，唯一差别是是否经过削顶：

$$
\mathrm{NMSE}_{\mathrm{clip}}^t
=\frac{\big\|\widehat{\mathbf s}_{\mathrm{clip}}^t-\widehat{\mathbf s}_{\mathrm{noclip}}^t\big\|_2^2}
{\big\|\widehat{\mathbf s}_{\mathrm{noclip}}^t\big\|_2^2}.
$$

它测的是**纯粹由有限动态范围造成的误差**。

总聚合 NMSE——相对不含信道噪声、不含削顶的真实平均更新：

$$
\mathrm{NMSE}_{\mathrm{total}}^t
=\frac{\big\|\widehat{\mathbf s}_{\mathrm{clip}}^t-\mathbf s_{\mathrm{ideal}}^t\big\|_2^2}
{\big\|\mathbf s_{\mathrm{ideal}}^t\big\|_2^2},
\qquad
\mathbf s_{\mathrm{ideal}}^t
=\frac{1}{N_t}\sum_{i\in\mathcal A_t}\mathbf s_i^t .
$$

前者回答"削顶本身造成了多少误差"，后者回答"实际通信链路总共造成了多少误差"，二者不要混在同一个 NMSE 中。

指标 3、4 的期望值可与第 3 节“削顶概率速查”表交叉验证（仿真自检项）。

---

技术附录**修改后应**与本场景采用同一系统模型，不再把实验中的随机活跃集合、AGC 或径向削顶留在定理之外。**状态说明（重要）**：下表描述的是统一后的**目标口径**，作为附录修订完成前的验收清单——当前 `convergence_analysis_revised.tex` 尚未落实该重组（tex 中尚无随机活跃集合 $N_t$、$\mathcal V_{\mathrm{act}}^t$ 与 $A_{\max}^t$ 框架；且 tex 现用 $\mathcal A_t$ 表示“对齐成功事件”，与本场景的“无线活跃集合”同名不同义，重组时须先把对齐事件改名，如 $\mathcal E_t$，以消除记号冲突）。附录修订完成并逐行对照通过后，方可将本段改写为事实陈述。

| 检查项 | 统一后的附录口径 | 结论 |
|---|---|---|
| 全客户端与活跃集合 | 每轮调度全部 $N$ 个客户端；$I_i^t=\boldsymbol 1\{|h_i^t|\ge h_{\mathrm{cut}}\}$，$\mathcal A_t=\{i:I_i^t=1\}$，$N_t=|\mathcal A_t|$ | ✅（目标口径）与物理场景一致 |
| 无客户端采样 | $\mathcal A_t$ 只由数据无关信道产生，不是学习算法的客户端采样 | ✅ 目标仍为 $N$ 客户端全局目标 |
| 活跃集合误差 | 定义 $e_{\mathrm{act}}^t=N_t^{-1}\sum_{i\in\mathcal A_t}s_i^t-N^{-1}\sum_i s_i^t$；条件于 $N_t$ 为零均值，并保留有限总体方差项 $\mathcal V_{\mathrm{act}}^t$ | ✅ 主定理显式包含 |
| 公共 $b_t$ | $b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$；仅依赖公开信道和公开裁剪上界 | ✅ 不引入数据相关公共增益 |
| 实复映射 | 实模型坐标映射到复子载波，预均衡符号为复数，对齐后信号在实轴，FFT 后取实部 | ✅ 与 $S=395$ 一致 |
| 含噪接收波形 | 频域信号与 $\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2)$ 噪声共同执行功率保持过采样 IFFT | ✅ 一致 |
| 理想逐轮 AGC | $P_{\mathrm{avg}}^t$、$A_{\mathrm{rms}}^t$、$a_t$ 和 $A_{\max}^t=\gamma A_{\mathrm{rms}}^t$ 均进入附录定义 | ✅ 不再是固定门限特例 |
| 径向动态范围削顶 | 附录使用 $\mathcal C_\gamma(z)=z\min\{1,\gamma/|z|\}$ 和精确残差 $E_{\mathrm{clip}}^t=\frac MQ\sum(|y|-A_{\max}^t)_+^2$ | ✅ 不依赖 Bussgang 近似 |
| 聚合分母 | 信道噪声与削顶误差均按 $b_tN_t$ 恢复 | ✅ 主定理使用 $1/(b_t^2N_t^2)$ |
| 隐私 | 条件于公开 $\mathcal H_t$ 的 replace-one 客户端级 DP；再利用 $\mathcal H_t\perp\mathcal D$ 得到无条件单轮 DP | ✅ 与场景一致 |

新主定理的结构为

$$
\frac1T\sum_{t=0}^{T-1}\mathbb E\|\nabla f(\theta^t)\|_2^2
\le
\Phi_A(k)
+\Phi_{\mathrm{act},A}
+\Phi_{\mathrm{ch},A}
+\Phi_{\mathrm{clip},A},
$$

其中

$$
\Phi_{\mathrm{act},A}
\propto
\frac1T\sum_t\mathbb E[\mathcal V_{\mathrm{act},A}^t(k,N_t)],
$$

$$
\Phi_{\mathrm{ch},A}
\propto
\frac1T\sum_t\mathbb E\left[\frac1{b_t^2N_t^2}\right],
$$

$$
\Phi_{\mathrm{clip},A}
\propto
\frac1T\sum_t\mathbb E\left[
\frac{E_{\mathrm{clip},A}^t(k,b_t)}{b_t^2N_t^2}
\right].
$$

当 $N_t=N$ 时，$\mathcal V_{\mathrm{act},A}^t=0$，定理自动退化为无信道中断的全无线参与特例；因此主定理与实验场景不再分离。

---

## 5. 参数设置、文献依据与仿真实现

**三层引用结构**


| 层 | 文献 | 支撑的命题 | 状态 |
|---|---|---|---|
| 机制层 | Rietman & Linnartz, IEEE TWC 2008 | 有限动态范围、8–9 dB headroom 惯例、$P/C^2$ 尺度律、"量化级数足够大只留削顶"口径、**Fig. 4 的 3–11 dB AGC backoff 扫描区间**。注意：其 ADC 模型为 I/Q 逐分量离散化削顶（矩形），本文引用其问题定义与参数区间，**不引用其算子几何**（见第 3 节“削顶算子与命名声明”） | 已发表 ✅ |
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
| 无线活跃集合 | $\mathcal A_t=\{i:|h_i^t|\ge h_{\mathrm{cut}}\}$，$N_t=|\mathcal A_t|$ | 数据无关截断反演 | 与收敛/隐私统一 |
| 模型映射 | 一个实坐标占一个复子载波；FFT 后取实部 | 不使用 I/Q 双坐标打包 | 系统模型选择 |
| 公共缩放 | $b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$ | $B_P^t(k)$ 为整轮平均功率上界；两支均按 $1/\sqrt{k}$ 缩放，且不依赖实际私有更新范数 | 功率、隐私与核心机制统一 |
| 过采样倍数 $L_{\mathrm{os}}$ | 4 | 正式实验默认 | Tellambura 2001 |
| 精度验证过采样 | 8 | 仅波形级验证 | 仿真自检 |
| 基准削顶回退 | 6 dB（$\gamma\approx1.995$，越限率 $\approx1.9\%$） | 主实验工作点 | Rietman & Linnartz 区间内的文献支持基准值（非标准值） |
| 主实验回退扫描 | $\{3,6,9,\infty\}$ dB | 强削顶 / 基准 / 弱削顶 / 理想上界 | Rietman & Linnartz Fig. 4 的 3–11 dB AGC backoff 扫描区间（原文已核实） |
| 附录敏感性档 | 11 dB | 8–9 dB 惯例外一档验证 | 复高斯近似下预期接近 $\infty$，由附录实际波形验证 |
| ADC 量化 | 不建模（高分辨率假设） | 只留饱和削顶 | Rietman & Linnartz "$L$ 足够大"口径 |
| AGC 更新周期 | 每轮一次，轮内固定 | 理想逐轮 RMS-AGC（非因果声明见第 3 节“四条口径声明”第 2 条） | 与块衰落/逐轮 $b_t$ 假设自洽 |
| 削顶方式 | 复包络径向限幅（保相位） | 接收前端动态范围抽象，非 I/Q 逐分量特性（见第 3 节“削顶算子与命名声明”） | 本文建模选择；Rietman 仅支撑问题与参数区间 |
| AGC 逆缩放 | 恢复时显式乘 $a_t^{-1}=A_{\mathrm{rms}}^t$ | 见第 3 节“AGC 逆缩放与模型更新恢复” | 实现必需，防止模型更新被额外缩放 |
| 轮归一化峰值压力 PSR | $\mathrm{PSR}_q^t$ CCDF + $\mathrm{PSR}_{\mathrm{round}}^t$ | 衔接逐符号 PAPR 与逐轮门限（第 3 节“结构线”） | 归一化基准匹配所需（本文设计） |
| 隐私后处理声明 | AGC/削顶为含噪输出后的确定性后处理 | 不削弱单轮 DP（第 3 节“四条口径声明”第 4 条） | DP 后处理不变性 |
| $b_t$ 解释约束 | 禁写“$b_t$ 大→更易削顶”；削顶由归一化峰值结构决定 | 见第 3 节“理想 AGC 下的尺度不变性” | 理想 AGC 推论 |
| 全零符号处理 | 不进 PAPR CCDF；单独报告静默符号比例 | 见第 3 节“结构线” | PAPR 分母为零无定义 |
| PAPR 统计口径 | 结构线无噪：PAPR CCDF（读 $10^{-3}$ 点）+ PSR | 稀疏结构证据 | 见第 3 节“结构线” |
| 削顶统计口径 | 含噪 ADC 输入：$\rho_{\mathrm{clip}}$、$D_{\mathrm{clip}}$、$\mathrm{NMSE}_{\mathrm{clip}}$/$\mathrm{NMSE}_{\mathrm{total}}$ | 硬件失真证据 | 见第 3 节“硬件线”和第 4 节 |
| 噪声生成 | 频域 $\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2)$ 与信号同 IFFT | 避免 FFT 归一化重复缩放 | 主场景文档的物理噪声部分 |
| 评价链 | {PAPR CCDF + PSR} → $\rho_{\mathrm{clip}}$ → $D_{\mathrm{clip}}$ → {$\mathrm{NMSE}_{\mathrm{clip}}$, $\mathrm{NMSE}_{\mathrm{total}}$} → 测试精度 | 完整证据链（七项） | 见第 4 节 |

---

**仿真实现要点**

1. **活跃集合与波形双份**：每轮先由公开小尺度信道生成 $\mathcal A_t$ 与 $N_t$；所有结构线和硬件线使用同一活跃集合。随后对每个 OFDM 符号同时生成 $r_{\mathrm{sig}}$（结构线）与 $r_{\mathrm{ADC}}$（硬件线），共用同一套频域数据与同一个酉 IFFT，仅噪声项之差；
2. **AGC 实现**：整轮 $S\times Q$ 个含噪样本一次性算 $P_{\mathrm{avg}}^t$（$A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}$），归一后统一施加 $\mathcal C_\gamma$；禁止逐符号归一；
3. **削顶后链路（含 AGC 逆缩放）**：削顶波形经酉 FFT → 取数据子载波 → **先乘 $a_t^{-1}=A_{\mathrm{rms}}^t$ 还原物理幅度** → 再取实部并除以 $b_tN_t$ 恢复模型更新（见第 3 节“AGC 逆缩放与模型更新恢复”；漏掉 $a_t^{-1}$ 是最容易犯的实现错误）；
4. **自检项**：仿真输出的 $\rho_{\mathrm{clip}}$ 应与第 3 节“削顶概率速查”表在同档位下**同量级**（仅数量级校验：实际波形非严格高斯、过采样样本相关，允许偏离）；$L_{\mathrm{os}}=8$ 复跑少量轮验证 CCDF 曲线重合；全链路（信号/噪声/AGC/门限/恢复）确认使用同一 FFT 归一化约定（第 2 节“过采样时域波形”）；
5. **$\infty$ 档实现**：直接旁路削顶算子（不是设一个很大的 $\gamma$），保证参照曲线严格无失真；
6. **统计粒度**：PAPR CCDF 与 PSR CCDF 按 OFDM 符号聚合全轮全种子（**剔除全零符号**，静默符号比例单独输出），$\mathrm{PSR}_{\mathrm{round}}^t$ 按轮输出；$\rho_{\mathrm{clip}}$、$D_{\mathrm{clip}}$ 按轮记录时间序列（观察训练过程中稀疏结构演化对削顶压力的影响）；
7. **NMSE 双链实现**：同一轮内并行跑削顶链与旁路链，客户端更新、信道、$b_t$、**噪声实现完全相同**（固定 RNG 状态/复用同一噪声张量），唯一差别是是否经过 $\mathcal C_\gamma$，由此计算 $\mathrm{NMSE}_{\mathrm{clip}}$；另存无噪真实平均 $\mathbf s_{\mathrm{ideal}}^t$ 计算 $\mathrm{NMSE}_{\mathrm{total}}$（第 4 节的两个定义不得混用）。

---
