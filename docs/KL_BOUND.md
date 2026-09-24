# Centered-radius KL sandwich：独立证明与来源核查

日期：2026-09-24。状态：解析证明完成；数值检查仅作补充；不声称新概率不等式。

## 1. 结论与论文定位

候选式 **正确**。设有限离散分布 \(p_i>0,\sum_i p_i=1\)，
\[
q_i=\frac{p_i e^{\delta_i}}{\sum_jp_j e^{\delta_j}},\qquad
X_i=\delta_i-\mathbb E_p\delta,
\quad b=\max_i|X_i|,
\quad v=\mathbb E_pX^2>0.
\]
令 \(h(z)=2(e^z-1-z)/z^2\)，并按连续延拓定义 \(h(0)=1\)。则
\[
\boxed{h(-b)\le \frac{2D_{\rm KL}(p\Vert q)}{v}\le h(b).}
\]
上下常数在仅给定 centered radius 上界 b、允许方差变化的类中均最优；极限由稀有的单侧极端点达到。对于固定非零方差，经典两点极值式更强。

**来源核查改变新颖性判定：** Seneta & Weber (1982), Theorem 1 和 Corollary 1(iii) 已给更强的、固定均值/方差/单侧界的两点 MGF 上下界。上界也属于 Bennett (1962)。因此不能把本 sandwich 当作原创概率不等式、独立 safe-LR 新证书，或文章的主要理论创新。可作为明确署名来源的基础命题，服务于有限参数步迁移问题。

## 2. KL 与 centered log-MGF 的恒等式

由 \(\log(p_i/q_i)=\log\mathbb E_pe^\delta-\delta_i\)，
\[
D_{\rm KL}(p\Vert q)=\log\mathbb E_pe^\delta-\mathbb E_p\delta
=\log\mathbb E_pe^X.
\]
这是固定 \(p\) 下的正向 KL（基准分布到扰动分布）。反向 KL 是 \(K'(1)-K(1)\)，其中 \(K(t)=\log\mathbb E_pe^{tX}\)，不能混用。

## 3. 上界：Bennett 型点态余项

对全部实数 \(x\)，
\[
\frac{e^x-1-x}{x^2}=\int_0^1(1-t)e^{tx}\,dt
\]
按连续方式定义 \(x=0\) 的值。因此该函数关于 \(x\) 单调递增。若 \(X\le b\)，则
\[
e^X\le1+X+\frac{e^b-1-b}{b^2}X^2.
\]
取期望并用 \(\log(1+u)\le u\)，得到
\[
\log\mathbb E e^X
\le\log\left(1+\frac{v}{b^2}(e^b-1-b)\right)
\le\frac{v}{b^2}(e^b-1-b).
\]
这已经证明所需上界。注意：这部分仅需 \(X\le b\)，无须下界。

## 4. 下界：先取得可达到的两点极值

只需假定 \(X\ge-b\)、\(\mathbb EX=0\)、\(\mathbb EX^2=v>0\)。令
\[
s=v/b^2,\qquad a=v/b=sb.
\]
构造二次 Hermite 插值多项式 \(P\)，满足
\[
P(-b)=e^{-b},\quad P(a)=e^a,\quad P'(a)=e^a.
\]
标准 Hermite 余项给出：对 \(x\ge-b\)，存在相应 \(\xi\)，使
\[
e^x-P(x)=\frac{e^\xi}{6}(x+b)(x-a)^2\ge0.
\]
在插值节点处等式由构造成立。因此 \(\mathbb Ee^X\ge\mathbb EP(X)\)。

取两点随机变量 \(Y\)：
\[
\Pr(Y=-b)=\frac{s}{1+s},\qquad
\Pr(Y=sb)=\frac1{1+s}.
\]
可直接核对 \(\mathbb EY=0\)、\(\mathbb EY^2=sb^2=v\)。由于 \(P\) 是二次多项式，\(\mathbb EP(X)=\mathbb EP(Y)=\mathbb Ee^Y\)。故
\[
\boxed{\mathbb Ee^X\ge\frac{e^{sb}+s e^{-b}}{1+s}.}
\]
这正是 Seneta–Weber (1982), Theorem 1(ii)(a) 取 \(\tau(x)=e^x\) 的特例。对固定 \(b,v\)，等号由上述两点分布达到。若同时要求 \(|X|\le b\)，则 \(s\le1\)，该两点分布仍在允许区间内。

## 5. 从两点极值到所需的线性方差下界

固定 \(b>0\)，定义
\[
F_b(s)=\log\frac{e^{bs}+s e^{-b}}{1+s},\qquad s\ge0.
\]
有
\[
F_b(0)=0,\qquad F_b'(0)=b-1+e^{-b}.
\]
下面证明 \(F_b\) 凸。写
\[
t=b(1+s),\qquad r=e^{-t}.
\]
直接两次微分可得
\[
F_b''(s)=\frac{1}{(1+s)^2}
+\frac{r(b^2s-2b)-r^2}{(1+sr)^2}
\]
以及等价式
\[
F_b''(s)=\frac{r}{(1+s)^2(1+sr)^2}
\left[2(\sinh t-t)+s\{t^2-2t+2-2e^{-t}\}\right].
\]
对 \(t\ge0\)，\(\sinh t-t\ge0\)。再令
\(G(t)=t^2-2t+2-2e^{-t}\)。则
\[
G(0)=0,\qquad G'(t)=2(t-1+e^{-t})\ge0,
\]
最后一步由 \(e^{-t}\ge1-t\) 得到。因此 \(G(t)\ge0\)，进而 \(F_b''(s)\ge0\)。

凸函数高于其在零点的切线，故
\[
F_b(s)\ge s(b-1+e^{-b}).
\]
结合上一节，
\[
\log\mathbb Ee^X\ge\frac{v}{b^2}(e^{-b}-1+b),
\]
下界证毕。该证明实际上对任意 \(s\ge0\) 都成立，并不依赖 \(s\le1\)。

## 6. Sharpness 与更强的经典式

取 \(s\downarrow0\)。对下界使用第4节两点分布；对上界使用其符号相反的分布。两者均满足 centered radius 恰为 \(b\)、方差 \(sb^2\)。
\[
\lim_{s\downarrow0}\frac{2}{sb^2}
\log\frac{e^{sb}+s e^{-b}}{1+s}=h(-b),
\]
\[
\lim_{s\downarrow0}\frac{2}{sb^2}
\log\frac{e^{-sb}+s e^{b}}{1+s}=h(b).
\]
所以仅依赖 \(b\) 的系数不能改进。但若 \(s=v/b^2\) 已知，可直接使用更强的经典式：
\[
\boxed{
\log\frac{e^{sb}+s e^{-b}}{1+s}
\le D_{\rm KL}(p\Vert q)
\le\log\frac{e^{-sb}+s e^{b}}{1+s}.}
\]
上界可由 Hermite 插值在 \(b\) 取单节点、在 \(-sb\) 取双节点证明；余项中 \((x-b)(x+sb)^2\le0\)。固定 \(b,v\) 的两端均可达到。这正是 Seneta–Weber Corollary 1(iii) 的对数形式。

还可分别使用 \(a=-\min X\)、\(c=\max X\)，得到更紧的非对称系数
\[
h(-a)\le2D_{\rm KL}/v\le h(c),
\]
以及分别代入 \(a\)、\(c\) 的两点极值。该改善也是同一经典结果的推论。

## 7. centered radius b 与 range Δ 不能混为一谈

\[
b=\max_i|\delta_i-\mathbb E_p\delta|,
\qquad \Delta=\max_i\delta_i-\min_i\delta_i,
\quad b\le\Delta\le2b.
\]
在指数倾斜分布 \(p_t\propto p e^{tX}\) 下，
\[
K''(t)=\operatorname{Var}_{p_t}(X),\qquad
K'''(t)=\mathbb E_{p_t}(X-\mathbb E_{p_t}X)^3.
\]
因为倾斜后的均值仍位于原区间内，正确的一般界是
\[
|K'''(t)|\le\Delta K''(t).
\]
积分得到 \(v e^{-\Delta t}\le K''(t)\le v e^{\Delta t}\)，再用
\(K(1)=\int_0^1(1-t)K''(t)dt\)，得到经典 generalized-self-concordance 形式
\[
h(-\Delta)\le2K(1)/v\le h(\Delta).
\]

**不能把上式证明中的 Δ 直接换成 b。** 反例：\(X=\pm b\) 各概率 \(1/2\)，则
\[
K(t)=\log\cosh(bt),\qquad
|K'''(t)|/K''(t)=2b\tanh(bt),
\]
当 \(bt>\operatorname{arctanh}(1/2)\) 时大于 \(b\)。因此 centered-radius sandwich 虽然正确，靠这个错误三阶导数界得到的旧证明必须撤回并替换。

## 8. 对 attention 实验的适用边界

1. 恒等式和上述界对**实际前后 logits 的差值**完全有效（在同一可见 attention mask 支持集内）。
2. 若用 \(\delta=\eta J_\theta\ell\,u\) 代替真正的 \(\ell(\theta+\eta u)-\ell(\theta)\)，需要另外控制 Taylor 余项；上述命题本身不能保证局部线性化迁移到实际参数步。
3. 它约束单行分布 KL，不等于 loss 不增加、训练不发散、没有 entropy collapse，不能直接当作 safe learning-rate 定理。
4. \(v=0\) 时 \(X=0\)，KL也为0，但归一化比值 \(2KL/v\) 未定义；实现必须明确处理退化行。
5. 极小方差但固定 b 的稀有事件极限，归一化比值并不一定趋向1；只有扰动幅度整体趋零的局部极限才恢复二次近似。

## 9. 参考来源与核对状态

- Seneta, E.; Weber, N. C. (1982). *Attainable bounds for expectations*. Journal of the Australian Mathematical Society (Series A), 33, 411–420. DOI: https://doi.org/10.1017/S144678870001884X 。已核对原文 Theorem 1（p.414）及 Corollary 1（p.415）。PDF: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D9D5BB90840A4C10A044C66D48877F/S144678870001884Xa.pdf/attainable-bounds-for-expectations.pdf
- Bennett, G. (1962). *Probability Inequalities for the Sum of Independent Random Variables*. Journal of the American Statistical Association, 57(297), 33–45. DOI: https://doi.org/10.1080/01621459.1962.10482149 。出版信息已核对；本报告直接核对的上下 MGF 表达式来自 Seneta–Weber 对 Bennett 的整理。
- Bach, F. (2010). *Self-concordant analysis for logistic regression*. Electronic Journal of Statistics, 4, 384–414. arXiv: https://arxiv.org/abs/0910.4627 。Generalized self-concordance 的指数型二阶余项上/下界相关来源；本报告对有限 log-MGF 的 range 版本给出自含推导。原作者PDF: https://www.di.ens.fr/~fbach/bach_ejs_self_concordance.pdf

数值审计脚本：`check_sharp_bound.py`。输出：`numerical_audit.json`。数值检查不替代本报告的解析证明，也不构成新颖性依据。


## 10. 实际数值审计结果与浮点告警

固定随机种子20260924，尝试50,000组分布；排除方差小于1e-18的42组后，检查49,958组。

- 最初直接用 double-precision log-sum-exp 得到14个异常，集中于极小方差，甚至出现负的KL估计。这些是真实记录的浮点异常，不能删除后声称原始检查“零违反”。原输出保存在 `raw_float_audit_before_stability_fix.json`。
- 全部14个异常都使用150位 Decimal、重新精确归一化概率、重新计算中心后复核：**0个真实反例**。高精度下最小下界余量约9.93e-19，最小上界余量约3.40e-18，均为正。
- 将主计算改为稳定余项形式 `log1p(E[expm1(X)-X])` 后，49,958组中没有超过预设浮点容差的违反；仍有约1e-13量级的舍入负余量，输出中完整保留。
- 稀有事件 sharpness 序列另用200位 Decimal计算，包括s=1e-60。不要误把任意有限精度末位余差当成数学证据；sharpness由第6节解析极限证明。对于b=100，即使s=1e-60，s*exp(b)仍约2.69e-17，因此上界相对差约1.34e-17是有限s效应，并不要求已经达到1e-60精度的极限。

这组检查也提示实现上的实质风险：以极小方差归一化KL会放大绝对舍入误差。实验代码应使用稳定余项实现并单独报告退化/低方差行，不能把负KL或异常比值解释为模型机制。
