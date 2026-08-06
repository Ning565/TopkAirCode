# 无线联邦学习 AirComp 实验场景与通信参数设置

## 1. 整体场景描述

本文考虑一个单小区、单基站、单天线客户端的无线联邦学习系统。基站位于小区中心并维护全局模型，$N=20$ 个客户端在距离基站 $r_{\min}=10\ \mathrm m$ 至 $R=250\ \mathrm m$ 的圆环区域内按面积均匀随机分布，每个随机种子生成一套在完整训练期间保持不变的用户拓扑。客户端与基站之间的物理信道由距离相关的大尺度路径损耗和轮间变化的小尺度 Rayleigh 块衰落共同构成：大尺度部分采用参考距离 $r_0=1\ \mathrm m$、参考路径损耗 $PL_0=30\ \mathrm{dB}$、路径损耗指数 $\alpha=3$ 的简化对数距离模型，小尺度部分采用 $h_i^t\sim\mathcal{CN}(0,1)$，有效信道为 $g_i^t=\sqrt{\beta_i}h_i^t$；一个通信轮内信道保持不变，不同轮重新生成。本文每轮调度全部 $N$ 个客户端，不进行算法层面的客户端采样，但采用阈值 $h_{\mathrm{cut}}=0.1$ 的截断信道反演，因此实际进入第 $t$ 轮 AirComp 聚合的是由数据无关信道决定的无线活跃集合 $\mathcal A_t=\{i:|h_i^t|\ge h_{\mathrm{cut}}\}$，其大小记为 $N_t=|\mathcal A_t|$。

上行采用基于 OFDM 的模拟空中计算（AirComp）。系统使用 $M=1024$ 个子载波和 $\Delta f=15\ \mathrm{kHz}$ 的子载波间隔，将实值模型更新固定映射到复基带 OFDM 资源栅格：一个实模型坐标占用一个复数据子载波资源，不使用 I/Q 双坐标打包，也不施加 Hermitian 对称。活跃客户端在相同时间和相同子载波上发送模拟更新，并利用上行 CSI 对复信道幅度和相位进行预均衡，使信号在基站处对齐到公共正实缩放系数 $b_t$；基站经 FFT 后取实部并除以 $b_tN_t$，直接恢复活跃客户端更新的平均值。公共缩放系数不依赖实际私有更新范数，而仅由公开信道、公开稀疏度 $k$、公开裁剪上界、整轮平均功率预算和单轮隐私预算共同确定：$b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$。较小的 $k$ 相对 Full 可放宽功率和隐私两条上界并允许更大的 $b_t$；相同 $k$ 下 Top-k 与 Rand-k 具有相同的最坏情况缩放上界，其差异来自重要更新保留、误差反馈和波形结构。

接收端噪声不再人为设置为无量纲的 $\sigma_0^2=1$，而是由热噪声功率谱密度 $N_0=-174\ \mathrm{dBm/Hz}$、基站噪声系数 $NF=5\ \mathrm{dB}$ 和实际带宽计算得到。单子载波复高斯噪声用于 AirComp 频域聚合，整带宽噪声用于物理链路预算。客户端功率由小区边缘路径损耗、接收噪声和目标平均接收 SNR 确定，基准整轮平均功率预算为 $P_{\mathrm{cap}}=20\ \mathrm{dBm}=100\ \mathrm{mW}$，在 $R=250\ \mathrm m$ 处对应约 $15.2\ \mathrm{dB}$ 的参考边缘平均接收 SNR，并扫描 $10$、$15$、$20$ 和 $23\ \mathrm{dBm}$。接收前端采用高分辨率但有限复包络动态范围的简化模型：结构性 PAPR 在无噪聚合波形上统计，实际削顶作用于含信号和噪声的接收波形；系统采用理想逐轮 RMS-AGC 和保相位径向限幅，主实验扫描 $\{3,6,9,\infty\}\ \mathrm{dB}$ 回退。系统暂不建模下行误差、阴影衰落、跨小区干扰、非理想 CSI、同步误差、有限位量化、发射端功放非线性和完整频率选择性多径，基准实验假设下行广播无误差，并具有完美上行 CSI、时间同步、载波频率同步和相位同步。

## 2. 损耗模型

**小区范围**
基站位于圆心，小区基准服务半径设为

$$
R=250\ \mathrm{m}.
$$

为避免客户端无限接近基站并使对数距离路径损耗模型失效，设置最小接入距离

$$
r_{\min}=10\ \mathrm{m}.
$$

因此，客户端分布在 $[r_{\min},R]$ 构成的圆环区域内，而不是从 $r=0$ 开始生成


**客户端随机分布**
在每次独立实验开始时，$N$ 个客户端的位置在以基站为圆心、内半径为 $r_{\min}$、外半径为 $R$ 的圆环区域内随机生成。为保证客户端在整个服务区域的面积上均匀随机分布，客户端到基站的距离 $r_i$ 的概率密度为

$$
f_r(r)=\frac{2r}{R^2-r_{\min}^2},
\qquad r_{\min}\le r\le R.
$$

具体地，首先分别生成两个独立的均匀随机变量

$$
U_i\sim\mathcal U(0,1),
\qquad
\varphi_i\sim\mathcal U(0,2\pi),
$$

然后根据

$$
r_i=
\sqrt{
r_{\min}^2+
U_i\left(R^2-r_{\min}^2\right)
}
$$

生成客户端到基站的距离，并通过

$$
x_i=r_i\cos\varphi_i,
\qquad
y_i=r_i\sin\varphi_i
$$

得到客户端的二维位置。

这里不能直接令 $r_i\sim\mathcal U(r_{\min},R)$。若距离在区间内均匀采样，靠近基站的单位面积中将分布更多客户端，无法保证客户端在整个圆环面积上均匀随机分布。

**每个独立实验或训练随机种子生成一套新的客户端位置。在一次完整训练过程中，客户端位置保持不变，因此由距离产生的路径损耗属于长期固定的大尺度信道特征；不同随机种子对应不同的客户端拓扑。正式实验结果至少使用 5 个独立随机种子进行统计。**

所有客户端采用相同的发射功率配置和功率约束，不再人为给不同客户端随机设置 SNR。不同客户端的平均接收 SNR 主要由其随机位置对应的距离路径损耗自然产生，小尺度衰落则在不同通信轮次中随机变化。

**简化对数距离模型**
采用简化的对数距离路径损耗模型：

$$
PL(r)\,[\mathrm{dB}]
=PL_0+10\alpha\log_{10}\!\left(\frac{r}{r_0}\right),
$$

其中

$$
PL_0=30\ \mathrm{dB},\qquad
r_0=1\ \mathrm{m},\qquad
\alpha=3.
$$

对应的线性大尺度信道功率增益为

$$
\beta_i
=10^{-PL(r_i)/10}
=\beta_0\left(\frac{r_i}{r_0}\right)^{-\alpha},
\qquad
\beta_0=10^{-3}.
$$

典型距离下的路径损耗为：

| 距离 | 50 m | 100 m | 150 m | 200 m | 250 m |
|---|---:|---:|---:|---:|---:|
| 路径损耗 | 81.0 dB | 90.0 dB | 95.3 dB | 99.0 dB | 101.9 dB |

该模型是一个便于理论分析和仿真实现的**简化路径损耗抽象**。参数量级参考城市微蜂窝环境，但本文并未实现完整的 3GPP TR 38.901 UMi 信道模型，因此论文中不应表述为“直接采用完整 3GPP UMi 模型”。完整 3GPP 模型还会包含 LOS/NLOS 状态、基站和终端高度、阴影衰落、断点距离、多径簇、时延和角度等因素。

基准实验取 $\alpha=3$

**小尺度信道系数的含义**
客户端 $i$ 在第 $t$ 个通信轮的小尺度信道系数记为

$$
h_i^t\sim\mathcal{CN}(0,1).
$$

这里 $h_i^t$ 是一个复数，可写为

$$
h_i^t=|h_i^t|e^{j\phi_i^t}.
$$

其中 $|h_i^t|$ 描述瞬时幅度变化，$\phi_i^t$ 描述无线传播造成的瞬时相位旋转。严格来说，$h_i^t$ 服从零均值单位方差复高斯分布，而其幅度 $|h_i^t|$ 服从 Rayleigh 分布，并满足

$$
\mathbb E|h_i^t|^2=1.
$$

**Rayleigh 模型主要对应没有稳定直射分量、由大量反射和散射路径构成的 NLOS 环境。它并不是所有通信场景唯一的小尺度衰落模型；存在明显直射路径时可采用 Rician 模型，3GPP 的完整宽带信道模型也比单一 Rayleigh 系数更复杂。本文选择 Rayleigh，是为了建立一个通信论文中常用、参数明确并且便于与 AirComp 功率控制结合的 NLOS 基准模型。**


**有效物理信道**
将大尺度路径损耗和小尺度衰落合并，客户端 $i$ 第 $t$ 轮的有效信道为

$$
g_i^t=\sqrt{\beta_i}\,h_i^t.
$$

因此

$$
\mathbb E|g_i^t|^2=\beta_i.
$$

距离决定客户端的长期平均信号强弱，小尺度 Rayleigh 衰落决定某一轮信道的瞬时波动。也就是说，远距离客户端长期较弱，而同一客户端在不同通信轮中仍会随机出现较好或较差的瞬时信道。

**块衰落假设**
当前模型采用**客户端级频率平坦块衰落**：

- 在一个通信轮内，客户端 $i$ 的 $g_i^t$ 保持不变；
- 同一客户端在该轮使用的所有 OFDM 子载波共享这个等效信道系数；
- 不同通信轮重新独立生成 $h_i^t$；
- 不同客户端之间的 $h_i^t$ 相互独立。

这一设置是对宽带信道的简化。OFDM 在本文中主要用于将高维模型更新映射到多个子载波并构造时域 PAPR，而不进一步模拟频率选择性多径。若以后扩展为频率选择性信道，可将 $g_i^t$ 改为每个子载波的 $g_{i,m}^t$，但功率控制、截断策略和活跃集合也需要同步重写。

载频基准取 $f_c=900$ MHz。以行人速度 3 km/h 估算，相干时间约为 169 ms；当前一轮上行约为 28.2 ms，因此“轮内信道不变、轮间重新生成”的块衰落假设在该移动速度下具有合理性。

**截断信道反演、全调度与无线活跃集合**
Rayleigh 信道可能出现 $|h_i^t|$ 非常接近零的深衰落。若仍进行完全信道反演，所需发射功率会趋于无穷，因此任何有限功率预算都无法保证所有瞬时信道下完成反演。

本文每轮调度全部 $N$ 个客户端，不进行学习算法层面的客户端采样。所有客户端均执行本地训练并形成待发送稀疏更新；随后仅由数据无关的小尺度信道决定本轮实际能够完成截断反演的客户端。定义

$$
I_i^t=\mathbf{1}\{|h_i^t|\ge h_{\mathrm{cut}}\},
\qquad
\mathcal A_t=\{i:I_i^t=1\},
\qquad
N_t=|\mathcal A_t|,
$$

其中

$$
h_{\mathrm{cut}}=0.1.
$$

因此，**全客户端调度**与**无线有效参与**是两个不同概念：全部客户端均参加本地训练，但只有 $i\in\mathcal A_t$ 的客户端进入本轮 AirComp 聚合。截断判据使用归一化小尺度衰落 $|h_i^t|$，而不是包含距离损耗的 $|g_i^t|$。这样所有客户端具有相同且与数据无关的激活概率，避免远距离客户端因路径损耗而被系统性排除；路径损耗仍通过 $g_i^t=\sqrt{\beta_i}h_i^t$ 进入信道反演和公共功率缩放。

Rayleigh 模型下，单客户端的激活概率和中断概率分别为

$$
q=\Pr(|h_i^t|\ge h_{\mathrm{cut}})
=e^{-h_{\mathrm{cut}}^2}
\approx0.99005,
$$

$$
1-q
=\Pr(|h_i^t|<0.1)
=1-e^{-0.1^2}
\approx0.995\%.
$$

因此，$N=20$ 时每轮期望活跃客户端数约为 $Nq\approx19.8$。由于各客户端的 $h_i^t$ 独立同分布，条件于 $N_t=n$，$\mathcal A_t$ 是从 $N$ 个客户端中形成的大小为 $n$ 的等概率随机子集。若极小概率事件 $N_t=0$ 发生，则该轮跳过，不计为一次有效全局更新。$\mathcal A_t$、$N_t$ 和信道状态均与客户端本地数据独立，并作为本轮公开无线状态进入功率控制、隐私分析和收敛性分析。实验中记录每轮 $N_t$ 和信道中断率。

## 3. 噪声与功率
### 噪声生成

**噪声功率谱密度 $N_0$**
基站接收端的热噪声功率谱密度取

$$
N_0=-174\ \mathrm{dBm/Hz}.
$$

在线性单位下

$$
N_0
=10^{(-174-30)/10}
\approx3.98\times10^{-21}\ \mathrm{W/Hz}.
$$

该数值对应约 290 K 环境温度下的标准热噪声基准。它描述每 1 Hz 带宽中的噪声功率，而不是高斯分布的方差已经等于 $-174$。

**接收机噪声系数 $NF$**
真实接收机的低噪声放大器、混频器等电路会额外引入噪声，因此引入基站接收机噪声系数

$$
NF=5\ \mathrm{dB}.
$$

对应的线性噪声因子为

$$
F=10^{NF/10}=3.162.
$$

**单子载波噪声**
每个子载波带宽为 $\Delta f=15$ kHz，因此单子载波噪声功率为

$$
\sigma_{\mathrm{sc}}^2
=N_0\Delta f F.
$$

代入数值得

$$
\sigma_{\mathrm{sc}}^2
\approx1.89\times10^{-16}\ \mathrm W,
$$

即

$$
\sigma_{\mathrm{sc}}^2
\approx-127.2\ \mathrm{dBm}.
$$

AirComp 频域聚合噪声直接建模为

$$
z_m^t\sim\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2).
$$

若模型更新只使用实部，则

$$
\operatorname{Re}\{z_m^t\}
\sim\mathcal N\!\left(0,\frac{\sigma_{\mathrm{sc}}^2}{2}\right).
$$

本文记

$$
\sigma_0^2\equiv\sigma_{\mathrm{sc}}^2,
\qquad
\sigma_0\approx1.37\times10^{-8}\ \sqrt{\mathrm W}.
$$

这里的 $\sigma_0^2$ 不再人为设置为 1，而是直接使用上述物理值。

**整带宽噪声**
整个 15.36 MHz 带宽内的总噪声功率为

$$
\sigma_{\mathrm{total}}^2
=N_0BF
=M\sigma_{\mathrm{sc}}^2
\approx1.93\times10^{-13}\ \mathrm W,
$$

即

$$
\sigma_{\mathrm{total}}^2
\approx-97.1\ \mathrm{dBm}.
$$

两个噪声口径的用途不同：

- AirComp 每个频域坐标的噪声使用 $\sigma_{\mathrm{sc}}^2$；
- 整带宽链路预算使用 $\sigma_{\mathrm{total}}^2$；
- PAPR 和削顶的时域仿真应先在频域生成 $z_m^t$，再与信号一起经过相同的 IFFT，避免因 FFT 归一化方式不同而重复乘或除 $M$。


### SNR/功率

**SNR 的统一定义**
本文报告的 SNR 统一定义为**平均接收 SNR**：包含距离路径损耗，并对小尺度 Rayleigh 衰落取平均。

客户端 $i$ 的平均接收 SNR 为

$$
\overline{\mathrm{SNR}}_i^{\mathrm{rx}}
=\frac{P_i\beta_i}{\sigma_{\mathrm{total}}^2}.
$$

若总发射功率均匀分到 $M$ 个子载波，也可写成

$$
\overline{\mathrm{SNR}}_i^{\mathrm{rx}}
=\frac{(P_i/M)\beta_i}{\sigma_{\mathrm{sc}}^2}.
$$

上述两个表达式在“总功率均匀分配到全部子载波”的参考链路预算中相等。实际 AirComp 聚合的有效噪声还会受到 $b_t$、模型更新幅度和活跃客户端数影响，因此平均接收 SNR 只是描述物理链路环境的参考量，不等于最终聚合结果的有效 SNR。

瞬时接收 SNR 为

$$
\mathrm{SNR}_{i,t}^{\mathrm{inst}}
=\overline{\mathrm{SNR}}_i^{\mathrm{rx}}|h_i^t|^2.
$$


**由边缘 SNR 反算功率**
小区边缘 $R=250$ m 的路径损耗约为

$$
PL(R)=101.9\ \mathrm{dB}.
$$

若要求边缘平均接收 SNR 至少为 5 dB，则所需总发射功率为

$$
P_{\min}[\mathrm{dBm}]
=5+PL(R)+\sigma_{\mathrm{total}}^2[\mathrm{dBm}]
$$

$$
=5+101.9-97.1
\approx9.8\ \mathrm{dBm}.
$$

圆整后取 10 dBm，即 10 mW。该功率对应约 5.2 dB 的边缘平均接收 SNR。


**基准功率和扫描范围**
实验采用以下功率档位：

| 功率档位 | 发射功率 | 250 m 边缘平均接收 SNR | 含义 |
|---|---:|---:|---|
| 低功率档 | 10 dBm = 10 mW | 约 5.2 dB | 由边缘 5 dB 下限反算 |
| 中间扫描点 | 15 dBm ≈ 31.6 mW | 约 10.2 dB | 低功率档与基准档之间的插值点，无独立物理含义 |
| **基准档** | **20 dBm = 100 mW** | **约 15.2 dB** | **主要实验工作点** |
| 上限档 | 23 dBm ≈ 200 mW | 约 18.2 dB | 常见终端功率上限量级 |

因此，基准实验采用整轮平均发射功率预算

$$
P_{\mathrm{cap}}=20\ \mathrm{dBm}=100\ \mathrm{mW},
$$

并在

$$
P_{\mathrm{cap}}\in\{10,15,20,23\}\ \mathrm{dBm}
$$

范围内进行敏感性扫描。另记当前考虑的设备硬件功率上限为

$$
P_{\mathrm{HW}}=23\ \mathrm{dBm}\approx200\ \mathrm{mW}.
$$

这里的链路预算 SNR 是客户端以给定总功率发送时的参考量。实际 AirComp 运行功率由公共缩放系数 $b_t$、有效信道和稀疏更新共同决定，不一定始终达到 $P_{\mathrm{cap}}$。有限功率也不能保证所有 Rayleigh 深衰落下完成信道反演；瞬时深衰落通过 $h_{\mathrm{cut}}$ 截断，而不是通过无限增加功率解决。

## 4. AirComp 部分

### OFDM 资源与实模型更新映射

系统采用 $M=1024$ 个 OFDM 子载波，子载波间隔为

$$
\Delta f=15\ \mathrm{kHz},
$$

对应的名义系统带宽为

$$
B=M\Delta f=15.36\ \mathrm{MHz}.
$$

当前模型更新维度为 $d=404222$，且发送更新为实向量

$$
\mathbf s_i^t\in\mathbb R^d.
$$

每个实模型坐标占用一个复 OFDM 数据子载波资源，不使用一个子载波的 I/Q 两路分别承载两个模型坐标，也不施加 Hermitian 对称。令

$$
j(q,m)=(q-1)M+m+1,
\qquad
q=1,\ldots,S,
\quad
m=0,\ldots,M-1,
$$

则第 $q$ 个 OFDM 符号、第 $m$ 个子载波承载

$$
s_{i,q,m}^t=
\begin{cases}
[\mathbf s_i^t]_{j(q,m)}, & j(q,m)\le d,\\
0, & j(q,m)>d,
\end{cases}
\qquad
s_{i,q,m}^t\in\mathbb R.
$$

因此每轮完整模型更新需要

$$
S=\left\lceil\frac dM\right\rceil=395
$$

个 OFDM 符号。不含循环前缀时，一个符号时长约为 $1/\Delta f=66.7\ \mu\mathrm{s}$，395 个符号约为 26.3 ms；计入普通循环前缀后约为 28.2 ms。所有客户端必须把同一模型坐标映射到同一资源位置，AirComp 才能完成逐坐标求和。Top-k 或 Rand-k 未选择的坐标在固定资源位置发送零，完整的 $395\times1024$ 资源栅格仍被保留，因此稀疏化本身不直接减少 OFDM 符号数或带宽；它主要影响学习误差、公共缩放系数、子载波占用结构和 PAPR。

### 复信道预均衡、活跃集合聚合与实部恢复

第 $t$ 轮所有客户端均被调度并形成本地更新，但只有 $i\in\mathcal A_t$ 的客户端完成截断反演并进入无线聚合。若不进行预均衡，基站在第 $q$ 个 OFDM 符号、第 $m$ 个子载波上收到

$$
y_{q,m}^t
=\sum_{i=1}^{N}g_i^t s_{i,q,m}^t+z_{q,m}^t,
$$

该结果包含不同的复信道权重，并不是模型更新之和。活跃客户端利用上行 CSI 发送

$$
x_{i,q,m}^t
=\frac{b_t}{g_i^t}s_{i,q,m}^t,
\qquad i\in\mathcal A_t,
$$

其中

$$
w_i^t=\frac{b_t}{g_i^t}
=b_t\frac{(g_i^t)^*}{|g_i^t|^2}\in\mathbb C,
\qquad b_t>0.
$$

虽然 $s_{i,q,m}^t$ 为实数，但预均衡系数为复数，因此实际发射符号为复数。由于 $g_i^tw_i^t=b_t$，基站频域观测为

$$
y_{q,m}^t
=b_t\sum_{i\in\mathcal A_t}s_{i,q,m}^t+z_{q,m}^t.
$$

理想对齐信号位于实轴。在线性未削顶链路中，基站取实部并除以公共缩放和实际活跃客户端数：

$$
\widehat s_{q,m}^t
=\frac{\operatorname{Re}\{y_{q,m}^t\}}{b_tN_t}
=\frac1{N_t}\sum_{i\in\mathcal A_t}s_{i,q,m}^t
+\frac{\operatorname{Re}\{z_{q,m}^t\}}{b_tN_t}.
$$

接收端存在 AGC 和径向限幅时，先对削顶后的时域波形执行 AGC 逆缩放和 FFT，再取实部并除以 $b_tN_t$。Top-k 未选择坐标而发送零与客户端因深衰落未进入 $\mathcal A_t$ 是两种不同情况：前者属于稀疏向量本身，后者改变无线聚合集合和分母 $N_t$。

定义全部客户端更新平均值与无线活跃客户端平均值为

$$
\bar{\mathbf s}^t=\frac1N\sum_{i=1}^{N}\mathbf s_i^t,
\qquad
\bar{\mathbf s}_{\mathcal A_t}^t
=\frac1{N_t}\sum_{i\in\mathcal A_t}\mathbf s_i^t,
$$

并定义信道活跃误差

$$
\mathbf e_{\mathrm{act}}^t
=\bar{\mathbf s}_{\mathcal A_t}^t-\bar{\mathbf s}^t.
$$

条件于 $N_t=n$，$\mathcal A_t$ 是大小为 $n$ 的等概率随机子集，因此

$$
\mathbb E[\mathbf e_{\mathrm{act}}^t\mid N_t=n,\{\mathbf s_i^t\}_{i=1}^{N}]=\mathbf0,
$$

且

$$
\mathbb E[\|\mathbf e_{\mathrm{act}}^t\|_2^2
\mid N_t=n,\{\mathbf s_i^t\}_{i=1}^{N}]
=
\frac{N-n}{n(N-1)}
\cdot
\frac1N\sum_{i=1}^{N}
\|\mathbf s_i^t-\bar{\mathbf s}^t\|_2^2.
$$

因此，本文仍优化由全部 $N$ 个客户端定义的全局目标；数据无关的无线中断不会引入系统性客户端偏置，但会增加一个显式聚合方差项。收敛性分析直接保留随机 $N_t$ 和该活跃集合误差，与实验场景保持一致。

### 公共缩放系数 $b_t$

$b_t$ 表示所有活跃客户端经过预均衡后在基站处对齐到的公共接收幅度，单位为 $\sqrt{\mathrm W}$/模型更新单位。恢复后的信道噪声为

$$
\frac{\operatorname{Re}\{z_{q,m}^t\}}{b_tN_t}.
$$

$b_t$ 越大，恢复后的等效噪声越小、聚合越准确，但所需发射功率越高且内生信道噪声提供的隐私越弱；$b_t$ 越小则相反。为避免公共缩放系数成为私有更新的函数，本文不根据实际 $\|\mathbf s_i^t\|_2$ 或实际 OFDM 块能量选择 $b_t$，而使用公开稀疏度和公开逐坐标裁剪阈值的确定性上界。发送更新满足

$$
\|\mathbf s_i^t\|_0\le k,
\qquad
\|\mathbf s_i^t\|_\infty\le c_{\mathrm{tx}},
$$

因此

$$
\|\mathbf s_i^t\|_2\le c_{\mathrm{tx}}\sqrt{k}.
$$

把 $P_{\mathrm{cap}}$ 定义为固定 $S$ 个 OFDM 符号上行突发内的整轮平均发射功率预算。采用酉 IFFT 时，客户端 $i\in\mathcal A_t$ 的整轮平均发射功率为

$$
P_i^t
=\frac1{SM}\sum_{q=1}^{S}\sum_{m=0}^{M-1}|x_{i,q,m}^t|^2
=\frac{b_t^2}{SM|g_i^t|^2}\|\mathbf s_i^t\|_2^2.
$$

由公开上界得到公共功率缩放上限

$$
B_P^t(k)
=\min_{i\in\mathcal A_t}
\frac{|g_i^t|\sqrt{SM P_{\mathrm{cap}}}}
{c_{\mathrm{tx}}\sqrt{k}}.
$$

该上限只依赖公开信道状态、公开活跃集合、固定资源参数、公开 $k$、公开裁剪阈值和公开功率预算，不依赖实际私有更新值，并满足 $B_P^t(k)\propto1/\sqrt{k}$。最终 $b_t$ 还要受到单轮隐私上限约束，完整定义见第 5 节。相同 $k$ 和裁剪规则下，Top-k 与 Rand-k 具有相同的最坏情况功率和隐私缩放上界；Top-k 的额外优势来自重要坐标保留、误差反馈和波形结构，而不是更小的同-$k$ 最坏情况敏感度。

AirComp 在本系统中的作用可以概括为：允许多客户端在相同无线资源上并发聚合；直接计算联邦学习所需的和或平均值；复用接收机已有的高斯噪声提供单轮客户端级隐私；并通过公共 $b_t$ 将稀疏度、发射功率、恢复噪声和隐私约束统一起来。

## 5. 其他场景

### 单轮客户端级隐私

本文采用**单轮、replace-one 客户端级差分隐私**。客户端总体始终为固定的 $N$ 个客户端；相邻数据集

$$
\mathcal D=(D_1,\ldots,D_N),
\qquad
\mathcal D'=(D_1',\ldots,D_N')
$$

仅在一个客户端 $l$ 的完整本地数据集上不同。每轮公开无线状态定义为

$$
\mathcal H_t=
\left(
\mathcal A_t,
N_t,
\{g_i^t\}_{i\in\mathcal A_t},
b_t
\right).
$$

无线活跃集合和信道状态由与数据独立的小尺度衰落产生，$b_t$ 由公开功率与隐私规则确定，因此

$$
\mathcal H_t\perp\mathcal D.
$$

条件于给定的 $\mathcal H_t$，加入噪声前的 AirComp 查询为

$$
Q_t(\mathcal D\mid\mathcal H_t)
=b_t\sum_{i\in\mathcal A_t}\mathbf s_i^t(D_i).
$$

发送更新满足

$$
\|\mathbf s_i^t\|_0\le k,
\qquad
\|\mathbf s_i^t\|_\infty\le c_{\mathrm{tx}},
$$

故

$$
\|\mathbf s_i^t\|_2\le c_{\mathrm{tx}}\sqrt{k}.
$$

若发生变化的客户端 $l\notin\mathcal A_t$，本轮条件敏感度为零；若 $l\in\mathcal A_t$，replace-one 敏感度满足

$$
\Delta_2(Q_t\mid\mathcal H_t)
\le b_t\Delta(k),
\qquad
\Delta(k)=2c_{\mathrm{tx}}\sqrt{k}.
$$

每个模型坐标最终使用频域复噪声的实部。若

$$
z_m^t\sim\mathcal{CN}(0,\sigma_0^2),
$$

则真正进入实值聚合机制的噪声标准差为

$$
\sigma_{\mathrm R}=\frac{\sigma_0}{\sqrt2},
\qquad
\operatorname{Re}\{z_m^t\}\sim\mathcal N(0,\sigma_{\mathrm R}^2).
$$

采用当前单轮高斯机制口径，公开隐私缩放上限写为

$$
B_{\epsilon,\delta}(k)
=\frac{\sigma_{\mathrm R}}{\Delta(k)}
\left(
\sqrt{\epsilon+\ln(1/\delta)}
-\sqrt{\ln(1/\delta)}
\right).
$$

最终公共缩放系数为

$$
b_t
=\min\left\{
B_P^t(k),
B_{\epsilon,\delta}(k)
\right\}.
$$

因为对每个可能的公开状态 $\mathcal H_t$ 都使用同一个 $(\epsilon,\delta)$ 上界，且 $\mathcal H_t$ 与数据独立，对信道随机性边缘化后，整个随机机制仍满足相同的单轮客户端级 DP。除以 $N_t$、AGC、径向削顶、FFT、实部提取和模型更新均属于含噪输出之后的后处理，不增加隐私泄漏。该保证不保护客户端是否发生信道中断；$\mathcal A_t$、$N_t$ 和信道状态作为公开无线侧信息处理。

基准隐私参数为

$$
\epsilon=5,
\qquad
\delta=10^{-3},
$$

并扫描

$$
\epsilon\in\{0.5,1,2,3,4,5,8,10\}.
$$

这里使用的是单轮隐私口径，不包含 200 轮训练全过程的隐私组合。功率和隐私两个公开上界均具有 $1/\sqrt{k}$ 的稀疏度缩放，因此 Top-k 或 Rand-k 相对 Full 可以获得更大的可行 $b_t$；相同 $k$ 下二者的最坏情况上界相同。

### PAPR、AGC 与接收端有限动态范围

OFDM 多个子载波在时域叠加后可能形成较高瞬时峰值。本文采用高分辨率但有限复包络动态范围的接收前端抽象，忽略有限位量化，仅研究接收波形越过动态范围后产生的饱和失真。削顶采用保持相位的复包络径向限幅，不声称复现具体 I/Q ADC 的逐分量转移特性。完整信号链、文献依据和评价方法见独立文档《[AirComp_PAPR与接收端削顶场景设置](./AirComp_PAPR与接收端削顶场景设置.md)》，此处保留与主场景直接相关的设置。

每轮全部客户端均被调度，结构线与硬件线使用同一无线活跃集合 $\mathcal A_t$ 和同一公共 $b_t$。一个实模型坐标占用一个复子载波资源，过采样采用 $L_{\mathrm{os}}=4$，并用 $L_{\mathrm{os}}=8$ 做少量精度验证。频域信号和噪声共同执行零填充、$\sqrt{Q/M}$ 功率保持缩放和酉 IFFT。结构性 PAPR 和轮归一化峰值压力在无噪聚合波形上统计；实际削顶、削顶率、削顶失真和聚合 NMSE 在含噪接收波形上统计。

接收端采用理想逐轮 RMS-AGC：第 $t$ 轮根据整轮含噪波形计算平均功率 $P_{\mathrm{avg}}^t$ 和 RMS 幅度 $A_{\mathrm{rms}}^t=\sqrt{P_{\mathrm{avg}}^t}$，增益 $a_t=1/A_{\mathrm{rms}}^t$ 在轮内保持不变。AGC 后使用门限

$$
\gamma=10^{B_{\mathrm{clip}}/20}
$$

执行径向限幅。主实验采用

$$
B_{\mathrm{clip}}\in\{3,6,9,\infty\}\ \mathrm{dB},
$$

其中 6 dB 为基准；另设 11 dB 档，不入主实验四曲线，仅用于附录敏感性扫描。

恢复时必须显式执行 AGC 逆缩放：
$$
\widehat s_{q,m}^t
=\frac{\operatorname{Re}\{a_t^{-1}\widetilde Y_{q,m}^t\}}
{b_tN_t}.
$$

理想 RMS-AGC 消除了公共绝对幅度尺度，因此 $b_t$ 不会仅因绝对接收幅度增大而直接提高削顶率；它主要通过改变含噪波形中的信号与噪声比例间接影响削顶统计。完整评价链为

$$
\{\mathrm{PAPR\ CCDF}+\mathrm{PSR}\}
\rightarrow\rho_{\mathrm{clip}}
\rightarrow D_{\mathrm{clip}}
\rightarrow
\{\mathrm{NMSE}_{\mathrm{clip}},\mathrm{NMSE}_{\mathrm{total}}\}
\rightarrow\text{测试精度}.
$$

主收敛性分析与该场景保持一致，显式包含信道活跃集合方差项、$1/(b_t^2N_t^2)$ 信道噪声项和 $E_{\mathrm{clip}}^t/(b_t^2N_t^2)$ 径向削顶项。



## **通信参数总表**

| 类别 | 参数 | 基准设置 | 说明 |
|---|---|---:|---|
| 拓扑 | 客户端数 $N$ | 20 | 单基站、单小区；每轮全部客户端调度 |
| 参与 | 无线活跃集合 | $\mathcal A_t=\{i:|h_i^t|\ge h_{\mathrm{cut}}\}$ | 无算法层客户端采样；$N_t=|\mathcal A_t|$ |
| 拓扑 | 最小距离 $r_{\min}$ | 10 m | 避免近场和对数路损奇点 |
| 拓扑 | 小区半径 $R$ | 250 m | 可扫描 50–250 m |
| 路损 | 参考距离 $r_0$ | 1 m | 对数距离模型 |
| 路损 | 参考路径损耗 $PL_0$ | 30 dB | 对应 $\beta_0=10^{-3}$ |
| 路损 | 路损指数 $\alpha$ | 3 | 扫描 $\{2.2,3,3.5\}$ |
| 载波 | 载频 $f_c$ | 900 MHz | 用于移动性和相干时间说明 |
| 小尺度 | $h_i^t$ | $\mathcal{CN}(0,1)$ | 客户端级频率平坦块 Rayleigh |
| 截断 | $h_{\mathrm{cut}}$ | 0.1 | 约 1% 客户端单轮无线中断 |
| OFDM | 子载波数 $M$ | 1024 | 每符号资源数 |
| OFDM | 子载波间隔 $\Delta f$ | 15 kHz | 名义带宽 15.36 MHz |
| OFDM | 模型维度 $d$ | 404222 | 每轮约 395 个 OFDM 符号 |
| OFDM | 实复映射 | 一个实坐标占一个复子载波 | 不使用 I/Q 双坐标打包；FFT 后取实部 |
| 噪声 | $N_0$ | −174 dBm/Hz | 290 K 热噪声基准 |
| 噪声 | 基站噪声系数 $NF$ | 5 dB | 线性噪声因子 3.162 |
| 噪声 | 单子载波噪声 $\sigma_{\mathrm{sc}}^2$ | $1.89\times10^{-16}$ W | AirComp 频域复噪声 |
| 噪声 | 实部标准差 $\sigma_{\mathrm R}$ | $\sqrt{\sigma_{\mathrm{sc}}^2/2}$ | 用于实值聚合和隐私校准 |
| 噪声 | 整带宽噪声 $\sigma_{\mathrm{total}}^2$ | $1.93\times10^{-13}$ W | 链路预算参考 |
| 功率 | 基准 $P_{\mathrm{cap}}$ | 20 dBm = 100 mW | 整轮平均发射功率预算；边缘参考 SNR 约 15.2 dB |
| 功率 | 扫描范围 | 10、15、20、23 dBm | $P_{\mathrm{HW}}=23$ dBm 为当前硬件上限 |
| 隐私 | DP 口径 | 单轮 replace-one 客户端级 | 条件于公开无线状态；边缘化信道后仍保持同一 DP |
| 隐私 | $(\epsilon,\delta)$ | $(5,10^{-3})$ | $\epsilon$ 扫描 0.5–10 |
| AirComp | 公共接收缩放 | $b_t=\min\{B_P^t(k),B_{\epsilon,\delta}(k)\}$ | 数据无关；两支均按 $1/\sqrt{k}$ 缩放 |
| CSI/同步 | 基准假设 | 完美 | 非理想情况不在当前基准中 |
| 天线 | 客户端/基站 | 单天线/单天线 | 上行 AirComp MAC |
| 下行 | 广播模型 | 无误差 | 不建立下行模型 |
| 硬件 | 接收前端 | 高分辨率、有限复包络动态范围 | 不建模有限位量化 |
| 硬件 | AGC 后回退 | 6 dB | 主实验 $\{3,6,9,\infty\}$ dB；11 dB 入附录 |
| 硬件 | AGC 更新周期 | 每轮一次 | 理想逐轮 RMS-AGC，轮内固定，恢复时逆缩放 |
| 硬件 | 过采样倍数 | 4 | 8 倍仅做少量验证 |
| 统计 | PAPR/削顶口径 | 双口径 | 无噪结构线；含噪硬件线 |
