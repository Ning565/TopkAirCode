# 修改方案：人工噪声补足 DP 机制在 convergence_analysis.tex 与 v8.tex 的落地

> 日期：0811。依据：`汇报/代码/exp_0810/DP_MECHANISM_0810.md`（§1–§6）+ 三篇文献机制实证
> （Liu TWC'24 式(9)-(11) 发射侧噪声与功率分账；Wei JSAC'22 式(6) 逐轮 LDP+组合另计；
> Liu&Simeone JSAC'21 free region）。
> 本方案只描述"改哪里、改成什么"，不执行修改。
> **0811 修订**：经四篇文献收敛性处理方式调研（证据见 §4 表），呈现方式定稿为 **min 形式逐字保留**
> （噪声增广双 ceiling：注噪抬高隐私顶、征税功率顶），机制数学与代码（commit a955484 起）完全不变。

---

## 0. 框架确认（回答"b 是否仍受 power 与 privacy 共同约束"）

**是。基本思路和框架不变**，b_t 仍是功率约束与隐私约束共同决定的公开逐轮缩放因子。
变化仅在"共同约束的实现方式"：

| | 旧（现 tex） | 新（本方案） |
|---|---|---|
| 隐私约束 | $b_t\le B_{\epsilon,\mathrm{ex}}(k)$（单边压 b） | $(b_t\Delta(k))^2\le m^2(\epsilon)\,\sigma_{\mathrm{eff},t}^2$（总噪声条件） |
| 功率约束 | $b_t\le B_P^t(k)$（只含信号能量） | $\frac{b_t^2}{SM\lvert g\rvert^2}\big(\lVert s\rVert^2+d\sigma_{a,t}^2\big)\le P_{\mathrm{cap}}$（信号+噪声分账） |
| 自由度 | 仅 $b_t$ → 取 $\min$ | $(b_t,\sigma_{a,t})$；呈现仍为 $\min$：注噪抬高隐私顶、征税功率顶 |
| 协议规则 | $b_t^\star=\min\{B_{\epsilon,\mathrm{ex}},B_P^t\}$ | $b_t^\star=\min\{B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t}),B_{P,\mathrm{art}}^t(k)\}$（**形式逐字不变**；数值 $=B_P^t(k)/\sqrt F$，Proposition 闭式求值） |
| 退化情形 | — | 信道足够差时 $\sigma_{a,t}=0$，两支退回 $B_{\epsilon,\mathrm{ex}},B_P^t/\sqrt F$，规则与旧 min 行为一致（Liu&Simeone free region） |

min 形式保留的代数依据（top-up 区两支**精确相等**）：标定式取等 $\iff b_t=B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t})$，
而 $b_t=B_{P,\mathrm{art}}^t(k)$ 由定义成立——注入的噪声把隐私顶恰好抬到功率顶的高度，min 两支重合；
free region 区隐私支严格松弛、功率支起作用，与旧规则的功率支路完全相同。

不变的全部内容：per-round client-level replace-one DP 口径（不声明 T 轮组合）、
敏感度引理 $\Delta(k)=2c_{\mathrm{tx}}\sqrt k$、$b_t$ 公开广播且不依赖当轮私有范数、
EF+Top-k 收敛界结构（引理 A.1–A.6 一个不动）、E_clip 硬件线全链路、学习率条件。

---

## 1. 统一记号与新公式清单（两个 tex 共用，进 notation table）

| 记号 | 定义 | 说明 |
|---|---|---|
| $m(\epsilon)$ | $\sqrt{\epsilon+\ln(1/\delta)}-\sqrt{\ln(1/\delta)}$ | 隐私裕度（现文 $B_{\epsilon,\mathrm{ex}}$ 括号项，显式命名） |
| $a_i^t$ | $\sim\mathcal N(\mathbf 0,\sigma_{a,t}^2\mathbf I_d)$，实值、铺满全部 $d$ 个栅格坐标 | 客户端人工噪声；全维注入（支持集私有，只铺自身支持集会泄露支持集——协议决定见 DP_MECHANISM §6.2） |
| $\sigma_{a,t}^2$ | $\dfrac{1}{2N}\max\!\Big\{0,\ \dfrac{\Delta^2(k)}{m^2(\epsilon)}-\dfrac{\sigma_{\mathrm{sc}}^2}{b_t^2}\Big\}$ | 补足标定；只含公开量 $(c_{\mathrm{tx}},k,N,\epsilon,\delta,\sigma_{\mathrm{sc}},b_t)$，零新增信令 |
| $\sigma_{\mathrm{eff},t}^2$ | $\sigma_{\mathrm{sc}}^2+2N b_t^2\sigma_{a,t}^2$ | 逐轮有效噪声（热噪声 + N 份聚合人工噪声，复等效口径） |
| $F$ | $1+\dfrac{2d}{N m^2(\epsilon)}$ | 噪声功率税（与 k 无关；由 $\Delta^2(k)=4c_{\mathrm{tx}}^2k$ 消去 k） |
| $b_t$ | $B_P^t(k)/\sqrt F$ | 新协议缩放规则 |
| $\sigma_{\mathrm{dp}}$ | $\dfrac{\Delta(k)}{\sqrt2\,N\,m(\epsilon)}=\dfrac{\sqrt2\,c_{\mathrm{tx}}\sqrt k}{N\,m(\epsilon)}$ | 补噪激活时恢复域逐坐标噪声标准差——**与信道、功率完全无关** |
| $\epsilon_{\mathrm{loose}}(k)$ | $\big(\tfrac{\sqrt{2k}}{N}+\sqrt{\ln(1/\delta)}\big)^2-\ln(1/\delta)$ | $\sigma_{\mathrm{dp}}=c_{\mathrm{tx}}$（DP 噪声降到逐坐标裁剪门限）的临界预算 |
| $B_{\epsilon,\mathrm{art}}(k;\sigma_a)$ | $\dfrac{m(\epsilon)\sigma_{\mathrm{sc}}}{\sqrt{(\Delta^2(k)-2Nm^2(\epsilon)\sigma_a^2)_+}}$ | 噪声增广隐私顶；$\sigma_a=0$ 时退回 $B_{\epsilon,\mathrm{ex}}$，随 $\sigma_a$ 递增，$\sigma_a^2\ge\Delta^2/(2Nm^2)$ 时 $=+\infty$ |
| $B_{P,\mathrm{art}}^t(k)$ | $B_P^t(k)/\sqrt F$ | 噪声征税功率顶（按最大注入 $\bar\sigma_a=\Delta/(\sqrt{2N}m)$ 预留功率预算） |

关键恒等式（附录装配的全部依据，已在代码中机器精度验证）：

1. **取等闭式**：DP 约束 $(b_t\Delta)^2\le m^2\sigma_{\mathrm{eff},t}^2$ 代入 $\sigma_{a,t}^2$ 标定式即取等（补噪激活时）；
2. **恢复噪声**：$e_{\mathrm{noise}}^t=\frac1N\sum_i a_i^t+\frac{\mathrm{Re}\{\mathbf N^t_{\mathcal D}\}}{b_tN}$，
   逐坐标方差 $=\frac{\sigma_{a,t}^2}{N}+\frac{\sigma_{\mathrm{sc}}^2}{2b_t^2N^2}=\dfrac{\sigma_{\mathrm{eff},t}^2}{2b_t^2N^2}$
   —— **与现文 $\mathbb E\lVert e_{\mathrm{ch}}\rVert^2=\frac{d\sigma_{\mathrm{sc}}^2}{2b_t^2N^2}$ 形式完全同构，字面替换即可**；
3. **确定化**：补噪激活时 $\sigma_{\mathrm{eff},t}^2/b_t^2=\Delta^2(k)/m^2(\epsilon)$，信道项从随机量变确定量；
4. **功率税**：取等时 $\lVert s\rVert^2+d\sigma_{a,t}^2\le kc_{\mathrm{tx}}^2 F$，故 $b_t=B_P^t/\sqrt F$ 保证功率可行；
5. **min 恒等式**：$\min\{B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t}),B_{P,\mathrm{art}}^t(k)\}=B_{P,\mathrm{art}}^t(k)=B_P^t(k)/\sqrt F$，
   且 top-up 区两支精确相等（标定式取等即 $b_t=B_{\epsilon,\mathrm{art}}$ 的代数恒等变形）。

### 三个已定的协议决策点（写方案时必须遵守，与代码 commit 一致）

- **D1 min 形式保留**：规则陈述为 $b_t^\star=\min\{B_{\epsilon,\mathrm{art}},B_{P,\mathrm{art}}^t\}$（与旧论文同形），
  Proposition 闭式求值 $=B_P^t/\sqrt F$；功率顶按最大注入水平 $\bar\sigma_a$ 预留（工程口径：为最坏注入预算功率），
  free region 内略保守（未用满功率），在 remark 中诚实声明；与代码 `b_star=b_power/sqrt(tax_f)` 逐字一致，零代码改动。
- **D2 论文用 $P_{\mathrm{cap}}$**：`p_operating_dbm` 只是实验对齐旋钮，不进论文。
- **D3 实轴全维注入**：$a_i$ 实值、铺满 d 个栅格坐标（复轴注入费一倍功率；只铺支持集泄露支持集）。

---

## 2. convergence_analysis.tex 修改（13 处，按行号从上到下）

### C1｜L113–124 发射信号与观测：加入人工噪声

现文（L114–118）：$X_{A,i,\ell}^t[m]=\frac{b_t}{g_i^t}S_{A,i,\ell}^t[m;k]$。

改为（$A$ 为 $a_i^t$ 按同一坐标映射铺到栅格）：

```latex
Each client draws an artificial-noise vector $a_i^t\sim\mathcal N(\mathbf 0,\sigma_{a,t}^2\mathbf I_d)$,
independent across clients and rounds, real-valued, and mapped to the same $d$ grid coordinates as the
model update. Let $A_{i,\ell}^t[m]$ denote its grid image. The client applies complex channel inversion
\begin{equation}
    X_{A,i,\ell}^t[m]
    =\frac{b_t}{g_i^t}\Big(S_{A,i,\ell}^t[m;k]+A_{i,\ell}^t[m]\Big),
    \qquad i=1,\ldots,N,
\end{equation}
```

观测式（L120–123）同步：
$Y=b_t\sum_i\big(S_{A,i,\ell}^t[m;k]+A_{i,\ell}^t[m]\big)+N_\ell^t[m]$。

L124 的口径句改为：

```latex
with $N_\ell^t[m]\sim\mathcal{CN}(0,\sigma_{\mathrm{sc}}^2)$. The Gaussian randomness used for the
per-round client-level privacy guarantee is the superposition of the real part of the thermal noise
and the $N$ aggregated artificial-noise components; we define the round-wise effective noise level
\begin{equation}
    \label{eqn:effective_noise}
    \sigma_{\mathrm{eff},t}^2\triangleq\sigma_{\mathrm{sc}}^2+2N b_t^2\sigma_{a,t}^2 .
\end{equation}
```

紧随其后给标定式（新 label，满足记忆 33eb7975 的文档化要求）：

```latex
The injection level is calibrated in closed form from public quantities only,
\begin{equation}
    \label{eqn:artificial_noise_calibration}
    \sigma_{a,t}^2=\frac{1}{2N}\max\!\left\{0,\;
    \frac{\Delta^2(k)}{m^2(\epsilon)}-\frac{\sigma_{\mathrm{sc}}^2}{b_t^2}\right\},
    \qquad
    m(\epsilon)\triangleq\sqrt{\epsilon+\ln(1/\delta)}-\sqrt{\ln(1/\delta)},
\end{equation}
so that no additional signaling beyond the existing broadcast of $b_t$ is required. When the channel
noise alone already meets the privacy requirement the injection vanishes and privacy is purely
intrinsic; artificial noise is added only to top up the deficit.
```

### C2｜L126–140 功率：信号+噪声分账，功率税

现文 $P_{A,i}^t=\frac{b_t^2}{SM|g_i^t|^2}\lVert s_{A,i}^t(k)\rVert_2^2$（L128–131）改为：

```latex
P_{A,i}^{t}
=\frac{b_t^2}{SM|g_i^t|^2}\big\|s_{A,i}^t(k)+a_i^t\big\|_2^2,
\qquad
\mathbb E\,P_{A,i}^{t}
=\frac{b_t^2}{SM|g_i^t|^2}\Big(\|s_{A,i}^t(k)\|_2^2+d\sigma_{a,t}^2\Big).
```

$B_P^t(k)$ 定义（L134–140）**保持原样**（仍是"纯信号"上界），其后加：

```latex
Substituting the calibration~\eqref{eqn:artificial_noise_calibration} at equality and
$\Delta^2(k)=4c_{\mathrm{tx}}^2k$ gives
$\|s\|_2^2+d\sigma_{a,t}^2\le kc_{\mathrm{tx}}^2F$ with the $k$-independent noise power tax
\begin{equation}
    \label{eqn:noise_power_tax}
    F\triangleq 1+\frac{2d}{N m^2(\epsilon)} .
\end{equation}
```

并加功率口径 remark（§6.4 已于 0812 用 Laurent–Massart 裕量解决；Liu TWC'24 式(10) 为期望口径先例）：

```latex
\begin{remark}[Expected versus Realized Noise Power]
The constraint is stated in expectation. Conditional on $s_{A,i}^t(k)$,
$\|s_{A,i}^t(k)+a_i^t\|_2^2/\sigma_{a,t}^2$ is noncentral chi-square with $d$ degrees of freedom
and noncentrality $\|s_{A,i}^t(k)\|_2^2/\sigma_{a,t}^2$. A specified high-probability per-burst
guarantee follows from the Laurent--Massart upper-tail bound: with
$\bar\lambda=kc_{\mathrm{tx}}^2/\sigma_{a,t}^2$ and $t=\ln(1/(1-\eta))$ for declared confidence
$\eta\in(0,1)$, the relative margin
$M=(2\sqrt{(d+2\bar\lambda)t}+2t)\sigma_{a,t}^2/(kc_{\mathrm{tx}}^2+d\sigma_{a,t}^2)$ satisfies
$\Pr[\|s+a\|^2\le\mathbb E\|s+a\|^2(1+M)]\ge\eta$ per burst, uniformly over
$\|s\|^2\le kc_{\mathrm{tx}}^2$; discounting the budget by $(1+M)^{-1}$ upgrades the constraint.
In the intrinsic regime $\sigma_{a,t}=0$ no margin is needed.
\end{remark}
```

### C3｜L141–154 隐私上限：$B_{\epsilon,\mathrm{ex}}$ 推广为噪声增广隐私顶 $B_{\epsilon,\mathrm{art}}$

$\Delta(k)$ 定义（L142–144）**不动**。$B_{\epsilon,\mathrm{ex}}$ 定义（L145–154）**原式原位保留**，
紧随其后新增噪声增广隐私顶（min 规则的隐私支记号，旧式是其 $\sigma_a=0$ 特例）：

```latex
Injecting artificial noise raises the privacy ceiling. Define the noise-augmented privacy ceiling
\begin{equation}
    \label{eqn:privacy_ceiling_art}
    B_{\epsilon,\mathrm{art}}(k;\sigma_a)
    \triangleq
    \frac{m(\epsilon)\,\sigma_{\mathrm{sc}}}
    {\sqrt{\big(\Delta^2(k)-2N m^2(\epsilon)\sigma_a^2\big)_+}},
\end{equation}
which reduces to $B_{\epsilon,\mathrm{ex}}(k)$ at $\sigma_a=0$, is increasing in $\sigma_a$, and equals
$+\infty$ once $\sigma_a^2\ge\Delta^2(k)/(2Nm^2(\epsilon))$: with sufficient injected noise the privacy
requirement no longer restricts the scaling. By the
calibration~\eqref{eqn:artificial_noise_calibration}, $\sigma_{a,t}=0$ if and only if
$b_t\le B_{\epsilon,\mathrm{ex}}(k)$, recovering the intrinsic-noise designs
of~\cite{liu2020privacy,koda2020differentially}.
```

### C4｜L155–162 缩放规则：min 形式逐字保留，两支换成噪声增广 ceiling

现文 `eqn:per_round_scaling`（$b_t=\min\{B_{\epsilon,\mathrm{ex}},B_P^t\}$）替换为（**形式不变，只换两支记号**）：

```latex
Provisioning the power budget for the maximal injection level
$\bar\sigma_a\triangleq\Delta(k)/(\sqrt{2N}\,m(\epsilon))$ defines the noise-taxed power ceiling
\begin{equation}
    \label{eqn:power_ceiling_art}
    B_{P,\mathrm{art}}^t(k)
    \triangleq
    \min_{1\le i\le N}
    \frac{|g_i^t|\sqrt{SM P_{\mathrm{cap}}}}
    {c_{\mathrm{tx}}\sqrt{k}\,\sqrt{F}}
    =\frac{B_P^t(k)}{\sqrt F}.
\end{equation}
The protocol keeps the min-form per-round scaling rule
\begin{equation}
    \label{eqn:per_round_scaling}
    b_t=b_t^\star(k)
    \triangleq
    \min\big\{B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t}),\,B_{P,\mathrm{art}}^t(k)\big\},
\end{equation}
with $\sigma_{a,t}$ given by~\eqref{eqn:artificial_noise_calibration} evaluated at
$b_t=B_{P,\mathrm{art}}^t(k)$. Thus $b_t$ may vary across rounds through the current public channel
state, but it does not depend on current private update norms and requires no cross-round
privacy-budget allocation.
\begin{proposition}[Closed-form evaluation of the min rule]
\label{prop:min_closed_form}
In every round, $b_t^\star(k)=B_{P,\mathrm{art}}^t(k)=B_P^t(k)/\sqrt F$. In the top-up regime the
calibration holds with equality, hence
$B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t})=B_{P,\mathrm{art}}^t(k)$ exactly: the injected noise
raises the privacy ceiling precisely to the power ceiling and both branches of the min coincide. In
the intrinsic regime $\sigma_{a,t}=0$ and
$B_{\epsilon,\mathrm{art}}=B_{\epsilon,\mathrm{ex}}(k)\ge B_{P,\mathrm{art}}^t(k)$, so the rule
coincides with the classical min rule with the power branch active.
\end{proposition}
```

Proposition 证明两行：标定式取等 $\iff b_t=B_{\epsilon,\mathrm{art}}(k;\sigma_{a,t})$（代数恒等变形）；
free region 分支由 $\max\{0,\cdot\}$ 定义直接得。**与代码逐字一致**（`b_star=b_power/sqrt(tax_f)`、
`sigma_a_sq` 补足式），零代码改动。

### C5｜L203–222 信道误差 → 有效噪声误差

现文 $e_{\mathrm{ch}}^t$ 定义与方差（L204–215）改为：

```latex
e_{\mathrm{noise}}^t
=\frac1N\sum_{i=1}^{N}a_i^t+\frac{\mathrm{Re}\{\mathbf N_{\mathcal D}^t\}}{b_tN},
\qquad
\mathbb E\|e_{\mathrm{noise}}^t\|_2^2
=\frac{d\,\sigma_{\mathrm{eff},t}^2}{2b_t^2N^2}.
```

物理更新式 `eqn:physical_update_dr`（L217–222）中 $e_{\mathrm{ch}}^t\to e_{\mathrm{noise}}^t$。
全文其余 $e_{\mathrm{ch}}$（L663/680/683/685/694/753/755）同步换名。
L224 过滤域段落补一句："Artificial noise is drawn independently of all signals conditioned on
$\mathcal F_t$ and the public $b_t$, hence $e_{\mathrm{noise}}^t$ is conditionally zero mean."
（L685 的零均值论证因此逐字成立。）

### C6｜L300–306 assum:public_scaling 微调

引用对象自动跟随新 `eqn:per_round_scaling`；加一句
"and the public calibration~\eqref{eqn:artificial_noise_calibration}"；$\mathbb E[1/b_t^2]<\infty$ 不动。

### C7｜L308–316 隐私定理与证明（唯一实质性证明改动，仍只换一个方差）

**注意条件化口径**：现文把 "data-independent randomness of unchanged clients" 条件化掉了——
旧机制下唯一的机制随机性是热噪声，这样写没问题；新机制下 **N 份人工噪声必须保留为机制随机性，
不能被条件化**。定理陈述改为：

```latex
\begin{theorem}[\bf Per-Round Client-Level Privacy]
    \label{thm:per_round_privacy}
    Consider replace-one client adjacency. Conditional on the same preceding public transcript,
    current public channel state, public $b_t$, and the data-dependent signals of unchanged clients,
    with the thermal noise and all $N$ artificial-noise vectors acting as mechanism randomness, the
    noisy AirComp observation in round $t$ satisfies client-level $(\epsilon,\delta)$-DP under the
    protocol scaling~\eqref{eqn:per_round_scaling} and calibration~\eqref{eqn:artificial_noise_calibration},
    in every round and in both regimes.
\end{theorem}
\begin{proof}
    The pre-noise query has sensitivity at most $b_t\Delta(k)$. The real-part Gaussian randomness of
    the observation has variance $\sigma_{\mathrm{sc}}^2/2+Nb_t^2\sigma_{a,t}^2=\sigma_{\mathrm{eff},t}^2/2$:
    the artificial noise of every client is data independent, identically distributed under adjacent
    executions, and superposed with the thermal noise at the receiver, so it is valid mechanism
    randomness~\cite{liu2024mimo}. The order-$\alpha$ Gaussian RDP cost is therefore at most
    $\alpha b_t^2\Delta^2(k)/\sigma_{\mathrm{eff},t}^2$. The
    calibration~\eqref{eqn:artificial_noise_calibration} enforces
    $b_t^2\Delta^2(k)\le m^2(\epsilon)\,\sigma_{\mathrm{eff},t}^2$; applying the standard RDP-to-DP
    conversion and optimizing over $\alpha>1$ yields $(\epsilon,\delta)$-DP. AGC, radial limiting,
    inverse-gain restoration, FFT recovery, division by $N$, and model updating are post-processing.
\end{proof}
```

L316 的"per-round，不声明组合"句**逐字保留**（Wei JSAC'22 同款口径，先例充分）。

### C8｜Auxiliary Lemmas 末尾（L318 节内，`cor:clipping_linkage` 之后）：唯一新增引理

符合"新内容只进装配层"的推导规范；A.1–A.6 不重推：

```latex
\begin{lemma}[\bf Effective-Noise Aggregation Error]
    \label{lemma:noise_error}
    Under Assumption~\ref{assum:public_scaling} and the
    calibration~\eqref{eqn:artificial_noise_calibration}, the aggregation noise error
    $e_{\mathrm{noise}}^t$ is conditionally zero mean given $\mathcal F_t$ and satisfies
    \begin{equation}
        \mathbb E_t\|e_{\mathrm{noise}}^t\|_2^2
        =\frac{d\,\sigma_{\mathrm{eff},t}^2}{2 b_t^2 N^2},
        \qquad
        \frac{\sigma_{\mathrm{eff},t}^2}{b_t^2}
        =\max\left\{\frac{\Delta^2(k)}{m^2(\epsilon)},\;\frac{\sigma_{\mathrm{sc}}^2}{b_t^2}\right\}.
    \end{equation}
\end{lemma}
\begin{proof}
    Zero mean and the variance follow from the independent Gaussian sum
    $\frac1N\sum_i a_i^t+\mathrm{Re}\{\mathbf N^t_{\mathcal D}\}/(b_tN)$ with per-coordinate variance
    $\sigma_{a,t}^2/N+\sigma_{\mathrm{sc}}^2/(2b_t^2N^2)=\sigma_{\mathrm{eff},t}^2/(2b_t^2N^2)$.
    The second identity is~\eqref{eqn:artificial_noise_calibration} evaluated in the two regimes.
\end{proof}
```

（第二个恒等式就是"确定化"性质，供主定理 remark 和正文 J_A 直接引用。）

### C9｜装配处字面替换（6 处，结构零改动）

| 行号 | 现文 | 改为 |
|---|---|---|
| L639–640（主定理） | $\frac{8\Gamma_A(k)L\sigma_{\mathrm{sc}}^2d}{T\eta\tau N^2}\sum_t\frac1{b_t^2}$ | $\frac{8\Gamma_A(k)Ld}{T\eta\tau N^2}\sum_t\frac{\sigma_{\mathrm{eff},t}^2}{b_t^2}$ |
| L791 | "the channel-noise variance" | "Lemma~\ref{lemma:noise_error}" |
| L802（T₂ 界） | $+\frac{Ld\sigma_{\mathrm{sc}}^2}{2b_t^2N^2}$ | $+\frac{Ld\sigma_{\mathrm{eff},t}^2}{2b_t^2N^2}$ |
| L851、L893（单步下降） | 同上 | 同上 |
| L918（$B_A$ 定义） | $\frac{4L\sigma_{\mathrm{sc}}^2d}{T\eta\tau N^2}\sum_t\frac1{b_t^2}$ | $\frac{4Ld}{T\eta\tau N^2}\sum_t\frac{\sigma_{\mathrm{eff},t}^2}{b_t^2}$ |
| L1045（最终装配） | 同 L639 | 同 L639 |

$\sigma_{\mathrm{eff},t}$ 是 $\mathcal F_t$-可测公开量（由 $b_t$ 与公开标定决定），
提出条件期望的合法性与原 $\sigma_{\mathrm{sc}}^2/b_t^2$ 完全相同——证明每一步的措辞都不用动。

### C10｜L1056–1062 corollary

"use $b_t=b_t^\star(k)$ from (per_round_scaling)" → "use the scaling and calibration
\eqref{eqn:per_round_scaling}, \eqref{eqn:artificial_noise_calibration}"。其余不动。

### C11｜L1064–1090 Interpretation：信道项确定化 remark + b 规则更新

L1066–1069 "recovered channel-noise term proportional to $\frac1T\sum 1/(b_t^2N^2)$" 改为
"$\frac1T\sum_t\sigma_{\mathrm{eff},t}^2/(b_t^2N^2)$"，并新增：

```latex
\begin{remark}[Channel-invariant privacy noise]
Whenever the top-up is active, Lemma~\ref{lemma:noise_error} gives
$\sigma_{\mathrm{eff},t}^2/b_t^2=\Delta^2(k)/m^2(\epsilon)$, so the recovered-noise term equals the
deterministic quantity $8\Gamma_A(k)L d\,\Delta^2(k)/\big(\eta\tau N^2 m^2(\epsilon)\big)$: path loss
and fading drop out of the privacy-noise term entirely, and the sparsity--privacy coupling
$k/m^2(\epsilon)$ enters the bound explicitly. The per-coordinate recovered noise level is
$\sigma_{\mathrm{dp}}=\sqrt2\,c_{\mathrm{tx}}\sqrt k/(N m(\epsilon))$; it falls below the coordinate
clipping threshold $c_{\mathrm{tx}}$ exactly when
$\epsilon\ge\epsilon_{\mathrm{loose}}(k)=(\sqrt{2k}/N+\sqrt{\ln(1/\delta)})^2-\ln(1/\delta)$.
In the intrinsic regime the term reduces to the familiar random quantity
$\sigma_{\mathrm{sc}}^2/b_t^2$.
\end{remark}
```

L1077 段（两条 ceiling 均 $\propto1/\sqrt k$）结论仍成立，改述为
"$B_{\epsilon,\mathrm{ex}}(k)$ and $B_P^t(k)$ both scale as $1/\sqrt k$, while the tax $F$ is
$k$-independent; hence sparse transmission increases the feasible $b_t$ and, through
$\Delta(k)\propto\sqrt k$, directly reduces $\sigma_{\mathrm{dp}}$."
L1088 的 $b_t^\star=\min\{B_{\epsilon,\mathrm{ex}},B_P^t\}$ → $\min\{B_{\epsilon,\mathrm{art}},B_{P,\mathrm{art}}^t\}$（数值 $=B_P^t(k)/\sqrt F$，形式不变）。

### C12｜不动清单（附录）

Assumptions smoothness/unbiased/variance/heterogeneity/clipping/contraction/dr_residual（后者仅按
C5 的换名微调一词）；Lemma topk / error_bound / local_div / dr_error / peak_stress / overlap /
cor:clipping_linkage；$\Delta(k)$ 敏感度式；`eqn:eclip_def` 及整条 E_clip 链
（削顶作用于含人工噪声的接收波形，物理正确，pathwise 界形式不变——在 assum:dr_residual 后加一句说明即可）；
学习率条件 `eqn:lr_condition_new`；$\Phi_A,\mu_A,\Gamma_A,\Lambda,\Psi,R_A,\mathcal E_A,C_{\mathrm{dr}}$ 全部定义。

### C13｜文头（若有 notation/系统概述段）

在 §Algorithm Flow 开头的机制综述句里加一短句声明混合机制；视排版加或不加。

---

## 3. v8.tex 修改（11 处，按行号）

### V1｜L48 Abstract

三处措辞：
1. "receiver noise can provide intrinsic differential privacy" →
   "receiver noise can supply part of the randomness required by differential privacy"；
2. min 句式主体保留，只把两支换成噪声增广版：
   "The public scaling is selected causally as
   $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{art}}(k),B_{P,\mathrm{art}}^t(k)\}$, combining the
   noise-augmented single-round privacy ceiling with the noise-taxed current-channel power ceiling;
   the minimal public artificial-noise level is calibrated in closed form so that the privacy ceiling
   never throttles the signal, the injection vanishes whenever the channel noise alone suffices, and
   no future CSI or cross-round privacy allocation is required."；
3. 末尾卖点句可加 "The privacy-induced recovered noise becomes channel invariant,
   $\sigma_{\mathrm{dp}}\propto\sqrt k/(N m(\epsilon))$, making the sparsity--privacy coupling explicit."

### V2｜L62–79 Introduction 与贡献

- L62 段（intrinsic privacy 动机段）尾部加两句诚实定位：
  "However, at physically realistic link budgets, relying on receiver noise alone forces either an
  astronomical privacy budget or a transmit power reduced far below the receiver hardware operating
  region: the power--privacy crossover scales with the squared burst-energy SNR of the weakest client.
  We therefore adopt a channel-noise-aware hybrid mechanism in which artificial noise only tops up
  the intrinsic deficit."（97dB 细账不进正文，指向附录 remark/引用 Liu&Simeone 的阈值结构。）
- L72 "with per-round intrinsic channel-noise privacy" →
  "with per-round channel-noise-aware minimal-artificial-noise privacy"。
- 贡献 #2（L78–79)：公式保留 min 形式，换成
  $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{art}}(k),B_{P,\mathrm{art}}^t(k)\}$ + $\sigma_a$ 标定一行；卖点措辞
  "closed-form minimal-noise calibration (no per-round numerical optimization), reducing verbatim to
  the intrinsic min rule of~\cite{liu2020privacy,koda2020differentially} when $\sigma_{a,t}=0$"。

### V3｜L93–97 Related Work II-B 重新定位

在 L95 段尾加：
"In our design the free-region structure of~\cite{liu2020privacy,koda2020differentially} is preserved
as a degenerate regime: artificial noise is injected only when the intrinsic noise is insufficient.
Compared with the transmit-side noise/power split of~\cite{liu2024mimo}, which optimizes the two
power fractions numerically per round, our split is closed form, uses a single receive antenna,
protects client-level (rather than sample-level) adjacency, and couples the noise scale to the
sparsity level through $\Delta(k)$."
L97 PFELS 段保留，其"complementary but distinct"清单中加一条
"and we do not rely on intrinsic noise alone at realistic link budgets"。

### V4｜L249–303 System Model：聚合模型与功率/缩放小节

- Over-the-Air Aggregation Model（L249 起）：发射信号式加 $(S+A)$（同附录 C1，含 $a_i^t$ 定义句与
  \eqref{eqn:effective_noise} 对应式）；
- Per-Round Power Constraint and Public Scaling（L280–303）：功率式换成分账口径（同 C2），
  依次给 $m(\epsilon)$、$\Delta(k)$（原有）、标定式（同 C1 的 calibration，正文版 label 如
  `eqn:an_calibration`）、$F$、新缩放规则 $b_t=B_P^t(k)/\sqrt F$（替换 L299）、
  free-region 等价判据 $\sigma_{a,t}=0\iff b_t\le B_{\epsilon,\mathrm{ex}}(k)$（$B_{\epsilon,\mathrm{ex}}$
  定义保留在 L293 原位）。L302 "depends only on current public CSI…" 句逐字保留（仍然成立）。

### V5｜L351–373 Threat Model

两处：
1. 条件化口径修正（同 C7）："data-independent randomness of unchanged clients are conditioned to be
   identical" → "the data-dependent signals of unchanged clients are conditioned to be identical,
   while the thermal noise and all $N$ artificial-noise vectors act as mechanism randomness"；
2. 加一句对抗鲁棒性："The honest-but-curious BS observes only the superposition
   $b_t\sum_i(s_i+a_i)+n$ and cannot separate any individual artificial-noise component from the
   thermal noise~\cite{liu2024mimo}."

### V6｜L374–395 Protocol Summary（算法框）

- L382 \STATE：句式不变，仅换 ceiling 记号并加 $\sigma_a$ 计算：
  "Compute $\sigma_{a,t}$ from~\eqref{eqn:an_calibration} and
  $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{art}}(k),B_{P,\mathrm{art}}^t(k)\}$ and broadcast
  $\boldsymbol\theta^t$, $b_t^\star(k)$, and the fixed resource map"；
- 客户端侧新增一个 \STATE（在裁剪/映射之后、发射之前）：
  "Each client computes $\sigma_{a,t}$ from~\eqref{eqn:an_calibration} using public quantities only,
  draws $a_i^t\sim\mathcal N(\mathbf 0,\sigma_{a,t}^2\mathbf I_d)$, and transmits the pre-equalized
  superposition of $s_{A,i}^t(k)$ and $a_i^t$."

### V7｜L400–435 Privacy Analysis

- 敏感度引理（L404–411）**逐字不动**；
- 隐私定理（L419–425）与证明（L428）按附录 C7 同步（$\sigma_{\mathrm{sc}}^2\to\sigma_{\mathrm{eff},t}^2$，
  条件改为"under the protocol scaling and calibration, in every round and both regimes"）；
- L431 per-round remark 保留；"Since $B_{\epsilon,\mathrm{ex}}\propto1/\sqrt k$, sparse upload permits a
  larger private scaling" 改为等价的新表述："Since $\Delta(k)\propto\sqrt k$, sparse upload directly
  reduces the required noise level: $\sigma_{\mathrm{dp}}=\sqrt2c_{\mathrm{tx}}\sqrt k/(Nm(\epsilon))$,
  and the budget at which $\sigma_{\mathrm{dp}}$ falls below the clipping threshold is the closed form
  $\epsilon_{\mathrm{loose}}(k)$."（附 $\epsilon_{\mathrm{loose}}$ 公式；这是"稀疏化换隐私"的新定量卖点。）
- L434 Privacy–Power Coupling remark 重写为分账视角：
  "The pair $(b_t,\sigma_{a,t})$ plays the role of the two power fractions in~\cite{liu2024mimo};
  here both are closed form. The privacy requirement no longer throttles the signal scaling directly:
  it enters only through the $k$-independent tax $F$, and the recovered privacy noise is channel
  invariant whenever the top-up is active."

### V8｜L965–1004 Convergence-Guided Joint Design

- 约束组（L979）：$b_t\le B_{\epsilon,\mathrm{ex}}(k)$ 替换为
  "the calibration~\eqref{eqn:an_calibration} (privacy holds with equality by construction)" +
  功率约束含 $d\sigma_{a,t}^2$；
- L982 "no cross-round privacy constraint, separable across rounds" 逐字保留；
- L988 $b^\star$ 公式 → $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{art}}(k),B_{P,\mathrm{art}}^t(k)\}$（引 Proposition 求值 $=B_P^t/\sqrt F$）；
- $J_A(k)$ 的信道项：由附录 remark，补噪激活时为确定量
  $\frac{8\Gamma_A(k)Ld}{\eta\tau N^2}\cdot\frac{4c_{\mathrm{tx}}^2k}{m^2(\epsilon)}$ ——
  目标函数首次显式含 $\epsilon$；加一句
  "the objective now contains the explicit sparsity--privacy trade $k/m^2(\epsilon)$: decreasing $k$
  simultaneously lowers the retained-energy loss ceiling and the privacy noise."
- L1004 结尾句"applies $B_P^t$ causally"保留。

### V9｜L1022–1066 Simulation

- Legacy 三图段（L1031）与旧参数表（L1050 $\epsilon=2.0$）**本次不动**（已标 legacy 口径），
  待 exp_0810 新结果出来后整节按新机制重写（ε 轴 {1,2.5,5,10,15,20,30}、$c_{\mathrm{tx}}$ 独立门限、
  报告 $\sigma_{a,t}$/$\sigma_{\mathrm{dp}}$/regime 转变）；
- 唯一必须现在改的：L1065 尾句 "a DP-to-energy-limited regime transition can be identified only by
  verifying when $B_{\epsilon,\mathrm{ex}}(k)$ exceeds $B_P^t(k)$…" →
  "the operational regime transition is characterized by the closed form
  $\epsilon_{\mathrm{loose}}(k)$ rather than a ceiling crossover"（避免与新机制矛盾）。

### V10｜L1071 Conclusion

同步三处：unique protocol rule 公式 → $b_t^\star(k)=\min\{B_{\epsilon,\mathrm{art}}(k),B_{P,\mathrm{art}}^t(k)\}$ + 闭式标定；
"per-round intrinsic client-level privacy" → "per-round channel-noise-aware minimal-artificial-noise
client-level privacy (intrinsic in the free region)"；其余（组合、量化为扩展）不动。

### V11｜Notation table（L143 附近的 tab:notation）

新增行：$m(\epsilon)$、$a_i^t,\sigma_{a,t}$、$\sigma_{\mathrm{eff},t}$、$F$、
$\sigma_{\mathrm{dp}}$、$\epsilon_{\mathrm{loose}}(k)$（定义同 §1 表）。
同时同步 $c_{\mathrm{tx}}$ 行的说明：独立公开裁剪门限（与 $\eta\tau C$ 解耦，代码 0811 口径）。

---

## 4. 文献引用支撑（正文引用点 → 文献证据）

| 论证点 | 文献证据（已从 PDF 核实原文） |
|---|---|
| 发射侧噪声+功率显式分账 | Liu TWC'24 式(9)–(11)：$x=\frac{s_1}{L}g+s_2 n$，$|s_1|^2+|s_2|^2\le P_{\max}$ |
| N 份噪声叠加共同保护 | Liu TWC'24："artificial noises from all local users are superposed at the receiver" |
| 逐轮保证+组合另计 | Wei JSAC'22 式(6)：$\bar\epsilon_i=\sqrt{E_i\ln(1/\delta)/\ln(1/\delta)}\,\epsilon_i$ |
| 噪声二阶矩直接进收敛界 | Wei JSAC'22 Theorem 1 的 $4\eta Cq_i\sqrt{2\tau\ln(1/\delta_i)}/\epsilon_i$ 加性项 |
| free region 与"加噪次优"的适用边界 | Liu&Simeone JSAC'21："privacy for free below an SNR threshold"；其"加噪次优"结论不含接收端硬件线约束，正是我们必须偏离之处 |
| 收敛界只带一个噪声方差项（无特殊处理，0811 核实） | Liu TWC'24 Thm.1：辅助函数 $A(\cdot)$ 含 $\sum_m\lvert f_0^Hh_m\rvert^2\lvert s_{m,2}\rvert^2+\sigma_z^2$ 一个聚合误差项；PFELS(TDSC) Thm.4：$Lk\sigma_0^2/(2r^3(\beta^t)^2)$ 单项；Wei JSAC'22 Thm.1：噪声 STD 加性项 |

---

## 5. 一致性检查清单（改完后逐项过）

1. `eqn:per_round_scaling` 的所有 \eqref 引用点（附录 L302/L310/L1058/L1088；正文 L299/L382/L988）语义仍通；
2. min 规则在两个 tex 中逐字保留（形式与旧论文相同），两支统一为 $B_{\epsilon,\mathrm{art}}/B_{P,\mathrm{art}}$，$B_{\epsilon,\mathrm{ex}}$ 以 $\sigma_a=0$ 特例出现；
3. $e_{\mathrm{ch}}$ 全部换名 $e_{\mathrm{noise}}$（附录 8 处），无残留；
4. 公式-代码对照：标定式 = `full_system_0810.scaling_limits` 的 `sigma_a_sq`；税式 = `tax_f`；
   规则 = `b_star`；$\epsilon_{\mathrm{loose}}$ = `eps_loose_k`（均已机器精度验证，commit a955484）；
5. 与 DP_MECHANISM_0810.md §1–§6、通信场景文档隐私节三方口径一致（场景文档隐私节待本方案落地后同步）；
6. Abstract / Intro 贡献 / Conclusion 三处的机制描述互相一致；
7. 编译检查：新增 label 无重名（eqn:effective_noise / eqn:artificial_noise_calibration /
   eqn:noise_power_tax / eqn:privacy_ceiling_art / eqn:power_ceiling_art / prop:min_closed_form /
   lemma:noise_error / eqn:an_calibration）。

## 6. 执行顺序

1. **convergence_analysis.tex**（C1→C13，理论口径先锚定；预计一次提交）；
2. **v8.tex**（V1→V11；预计一次提交）；
3. 编译两文（latexmk），核对 §5 清单；
4. 同步场景文档隐私节（汇报8.4/0807 版，以 DP_MECHANISM_0810.md 为准）；
5. exp_0810 新结果产出后重写 v8 §VI Simulation（本方案 V9 只做最小防矛盾修改）。

---

## 7. 0812 追加：BS 端后处理去噪协议与影响文档遗留问题的主流化处理

> 依据：`人工噪声补足机制对隐私功率与收敛性的影响.md`（0811 用户审阅文档）+
> 五篇文献机制核实（PFELS TDSC'24 / Amiri&Gündüz TSP'20 / Jeon TWC'21 /
> Liu TWC'24 / Wei JSAC'22）。§1–§6 的全部内容（min 规则、标定、功率税、
> Laurent–Massart 裕量、隐私定理、引理 A.1–A.7）**零改动**。

### 7.1 不收敛的机理定位（为什么全维加噪本身没错，我们却收敛困难）

- 人工噪声派（Wei JSAC'22 / Liu TWC'24）全 d 维注噪且不做接收端去噪，但其**信号
  也是全 d 维的**：噪声/信号能量比 = 逐坐标 SNR，没有维度放大；
- 我们是稀疏传输（Top-k/Rand-k），有效信号只占 k 维、聚合并集 ≤ Nk 维，而噪声
  铺满 d 维：能量比被放大 d/k 倍（k/d=0.001 时即 1000 倍），χ_A 有 k 无关下限
  2d/(N²m²)，0811 实测噪声/信号范数比 59–88——**"k 维信号 vs d 维噪声"的失配是
  稀疏化+人工噪声组合独有的问题**，两派单独都不会遇到；
- PFELS 的噪声只有 k 维是因为 rand-k 投影矩阵由**服务器生成并公开广播**
  （Algorithm 2 line 3–4），输出天然落在公开 k 维子空间；私有 Top-k 支持集
  不可复制此路（只铺支持集泄露支持集，见影响文档 §2）；
- 关键账目：即使做到理想 k 维噪声，ε=15、N=20 时噪声/信号范数比也只降
  √(d/k)≈31.6 倍、剩 ≈1.9–2.8（与 PFELS 正常收敛的"噪声≈信号"量级一致）。
  所以主要矛盾是 d−k 个纯噪声坐标全部进模型，去噪解决"60 倍→2 倍"这一段。

### 7.2 协议决定（D4）：发射侧不动 + BS 端公开 top-m 幅度截断（条件启用）

主流先例：Amiri&Gündüz TSP'20（"The PS tries to reconstruct the sum of the actual
sparse gradient vectors from its noisy observation. We use approximate message
passing (AMP)"）与 Jeon TWC'21（服务器端压缩感知重构，利用梯度稀疏先验）——
**接收端稀疏重构/去噪是 AirComp 稀疏传输的通信主流路线**。DP 合法性：BS 收到的
含噪观测已完成隐私机制，其后任何只依赖公开量的映射均为 post-processing，
(ε,δ) 一字不动（隐私定理证明末句的 post-processing 枚举加一词即可）。

**协议**：BS 恢复 d 维向量后，可选保留幅度最大的 m=min(d, N·k) 个坐标、其余
归零（真实聚合信号支持集并集 ≤ Nk，N、k 公开 ⇒ m 公开，零新增信令）。

**0812 实测修正（落地后的诚实结论，详见 DP_MECHANISM §6.5）**：合成链路审计
符合理论（截断后噪声能量 14.4%，次序统计量预测 17.6%；无噪时无损），但
ε=15、N=20 基线上学习侧是**负结果**：逐坐标聚合信号尺度 ≈c_tx/N=5e-4 而
σ_dp≈5.3e-3，幅度排序被噪声主导，截断误杀 ~98% 信号漂移（MNIST 24 轮：不
截断 9.4%→29.9% 持续上升，截断卡死 ~10%）；未截断时噪声零均值，SGD 逐轮
平均掉它。**故截断作为条件选项落地（默认关闭），适用条件：σ_dp 低于逐坐标
聚合信号尺度（逐坐标 SNR≳1，即大 ε/大 N 工作点）**。升级阶梯（不在本次
范围）：MMSE 逐坐标收缩（无硬判决无误杀，但整体衰减更新需配合服务器侧
学习率补偿）→ AMP 重构（期刊卖点，Amiri/Jeon 同款）。基线工作点的收敛依据
回到旧实验三经验（全 d 维零均值噪声 σ=0.0055 仍训到 81%）+ σ_dp 操作性
指标 + χ_eff 绝对量纲门槛。noisy-vector-后-Top-k 备选维持 §6.2 判定：不采纳。

**收敛分析收编方式**：主定理**一字不动**（在截断关闭时逐字适用；截断为可选
增强）；新增 remark（rem:denoise）：若 BS 施加只依赖公开量的截断算子 T_m，
则噪声项有效维度由 d 降为 m·(1+2ln(d/m)) 量级，另加截断偏差项，并**显式写明
适用条件**（σ_dp 低于 c_tx/N 量级时才启用，否则排序被噪声主导）；与影响文档
§3 "若能严格证明输出只位于公开子空间则 d 可换 k" 兼容——我们不声称 k（支持集
私有），声称的是公开的 m=Nk 量级且附适用条件。

### 7.3 其余各项的主流化处理（对照影响文档问题清单）

| 问题 | 主流做法 | 处理 |
|---|---|---|
| 功率税 √F、噪声主导发射功率 | Liu TWC'24 功率分账 \|s₁\|²+\|s₂\|²≤Pmax，强隐私下噪声占大头，诚实呈现不消除 | 已对齐（√F 闭式 + N 分摊），不再动 |
| 功率高概率口径 | 期望口径为主流（Liu 式(10)）；我们更强 | 已解决（Laurent–Massart，§6.4），不再动 |
| c_tx 两难 | 固定公开裁剪常数（Wei JSAC'22 的 C），不做自适应（另耗预算且违反 b_t 不依赖私有范数口径） | 保持 c_tx=0.01；饱和后 SNR 不改善的根因是 d 维噪声，去噪落地后用同一审计重扫再定 |
| exp1 归一化消除绝对量级 | 主流离线目标用绝对 MSE/SNR 量纲，无方法内归一化 | χ_eff(k)=min(d,Nk)·σ_dp²/E‖s̄‖² 作可行性门槛（CLI 可调），归一化分数只在可行集内选 k*；逐候选照常报告 chi 审计 |
| 论文同步清单五条 | — | 前四条已落地（c1a170f/94a3b98）；第五条（随机压缩算子重定义）不触发：本协议不改压缩算子 |

### 7.4 落地清单（代码 + tex）

代码（exp_0810）：
1. `full_system_0810.py`：`transmit_round` 增 `denoise_m` 参数（top-m 截断 +
   `nmse_total_denoised`/`denoise_keep_energy` 审计）；`LearnConfig.bs_denoise_mode`
   （"topm"/"off"，**默认 off**，0812 实测负结果后修正）；`run_training` 按协议传
   m=min(d,Nk)；
2. `exp1_offline_ksearch.py`：校准链路同步传 denoise_m；新增 `chi_eff` 列与
   `--chi-max` 可行性门槛（默认 100，全不可行时回退选 min χ_eff 并告警）；
3. `exp2_online_ksearch.py`：透传 `--bs-denoise` CLI（默认 off）；
4. `DP_MECHANISM_0810.md` 增 §6.5 记录机制、实测负结果与适用条件。

tex（增量极小，主定理/min 规则/隐私证明主体零改动）：
- **C14**（convergence_analysis.tex）：lemma:noise_error 后或 Interpretation 节新增
  remark（Server-Side Post-Processing Denoising：T_m 定义、有效维度、偏差项、
  post-processing 不损 DP、引 Amiri/Jeon）；
- **V12**（v8.tex）：系统模型加一句 BS top-m 截断协议；隐私证明 post-processing
  枚举加 "server-side magnitude truncation"；joint design 节 χ_eff 门槛一句。

### 7.5 不动清单（0812 版）

发射链与 d 维注噪、σ_a 标定、σ_eff、min 规则与 Proposition、√F、Laurent–Massart
裕量、隐私定理主体、引理 A.1–A.7、E_clip 链、学习率条件、c_tx=0.01、
exp_0810 全部已有结果（去噪默认关闭，旧口径逐字复现；`--bs-denoise topm`
仅在逐坐标 SNR≳1 的工作点启用）。
