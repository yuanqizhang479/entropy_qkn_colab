# QK normalization：数学复核、可完成理论与方向修正

日期：2026-09-24。本文是独立解析推导，不冒充对原始实验包的复跑。正文的「已完成」指在所列假设下给出证明；不表示新颖性、实证解释力或论文录用已成立。经典恒等式、已有球面优化思想的专门化与本次统一整理均明确区分。

## 0. 核心修正

旧稿的「query-temperature JVP 恒等式」正确，但不能直接解释成 standard 与 Q-detach 训练臂之间的实际温度干预。二者中间隔着 VJP、参数共享、优化器状态和真实有限参数步。

建议问题改为：**QK normalization 的径向反传项，何时只改变有限步有效学习率，何时通过参数共享、预条件和上游网络改变注意力形状？局部公式能否预测实际 optimizer step？**

本文给出四个层次：独立 query；冻结输入、仅更新共享 W_Q；固定预条件；真实全模型一步更新。后一个层次不能凭前一个层次的成功自动通过。

## 1. 定义与 RMS Jacobian

对非零 q∈R^D 定义

\[
s(q)=\sqrt{\|q\|^2/D+\varepsilon},\qquad N(q)=q/s(q).
\]

直接微分 ds=q^T dq/(Ds)，得

\[
J_{\rm std}(q)=\frac1sI-\frac{qq^T}{Ds^3},\qquad
J_{\rm det}(q)=\frac1sI.
\tag{1}
\]

第二式是把分母在当前点冻结后指定的替代 Jacobian。因此

\[
J_{\rm det}-J_{\rm std}=\frac{qq^T}{Ds^3}.
\tag{2}
\]

令 u=q/||q||，κ=||q||²/(Ds²)，则

\[
J_{\rm std}=s^{-1}(I-\kappa uu^T).
\tag{3}
\]

ε=0 时 κ=1，是切空间投影乘以 1/s；ε>0 时不是精确投影。q=0、ε>0 时应使用式(1)，不使用未定义的 u；q=0、ε=0 时归一化本身未定义。

对固定扰动 d，令 α=q^T d/||q||²，则

\[
(J_{\rm det}-J_{\rm std})d=\kappa\alpha N(q).
\tag{4}
\]

若 content logits 为 x_j=a N(q)^T N(k_j)，则相应差为

\[
\delta x_j^{\rm sub}=\kappa\alpha x_j.
\tag{5}
\]

这就是正确的 query-wise content-temperature tangent 恒等式。

### 1.1 恒等式边界

1. 它比较两个 Jacobian，并不比较两个训练器产生的真实参数更新。
2. detached 与 standard 的数值前向相同。若每次有限差分都重新计算分母，两者的真实有限差分相同；detach 自动微分给出的是人为指定的替代微分。若要赋予 J_det 一个通常意义的导数对象，应明确使用当前点分母冻结的替代函数 q'↦q'/s(q_0)。
3. z_j=x_j+b_j 时，缩放的是 content x_j。只有 b_j 在该行恒定时，才等价于总 logits 的温度方向（加常数不影响 softmax）。
4. 固定线性 RoPE、固定 gain 保持径向齐次性。必须按实现注明 normalization 轴和变换顺序。若加了 softcap，如 z=C tanh(x/C)，则 δz=cx sech²(x/C)，一般不再是 cz。
5. K 侧逐 key 径向系数 β_j 导致 δx_ij=β_jx_ij；除非 β_j 对当前行可见 keys 恒定，它不是该行的统一温度。

来源地位：RMS Jacobian、齐次性、softmax 微分是基础微积分和既有 normalization 理论，不能称为本文创新。

## 2. 独立 query：精确有限步等价与一阶零空间

假设 ε=0，独立优化 q，固定其他变量，采用无动量、无预条件的 SGD。令

\[
q=ru,\quad \|u\|=1,\quad
h=\partial L/\partial N(q),\quad
h_r=u^Th,\quad h_T=h-h_ru.
\]

同一前向点的两个反传梯度为

\[
g_S=h_T/s,\qquad g_D=(h_T+h_ru)/s.
\tag{6}
\]

令 a=η/(sr)，则一步更新精确为

\[
q_S^+=r(u-ah_T),\qquad
q_D^+=r\{(1-ah_r)u-ah_T\}.
\tag{7}
\]

若 1−ah_r>0，利用 N(cq)=N(q) 对 c>0 成立，得到

\[
N(q_D^+(\eta))
=N\left(q_S^+\left(\frac{\eta}{1-ah_r}\right)\right).
\tag{8}
\]

即 detached 的归一化结果精确等于 standard 在同一初始点、固定同一梯度下，采用

\[
\eta_{\rm eff}=\frac{\eta}{1-ah_r}
\tag{9}
\]

的结果。

**这是一点一步等价，不是多步训练轨迹等价。** 半径轨迹、后续梯度和 optimizer state 都可能变化。1−ah_r≤0 时不能解释成正学习率重标；跨过原点或方向翻转需要单独处理。

### 2.1 二阶差异

由于 s=r/√D，

\[
N(q_S^+)=\sqrt D\frac{u-ah_T}{\sqrt{1+a^2\|h_T\|^2}}.
\]

detached 等价于将 a 替换成 a/(1−ah_r)=a+a²h_r+O(a³)。展开得

\[
N(q_D^+)-N(q_S^+)
=-\sqrt D\,a^2h_rh_T+O(a^3).
\tag{10}
\]

因此一阶变化相同，差异通常从二阶出现，领先项沿切向 h_T。若 h_r=0 或 h_T=0，二阶系数也消失；后者在未越过原点时可完全没有归一化输出差。

更短的零空间证明是

\[
DN(q)q=0,\qquad \varepsilon=0.
\tag{11}
\]

所以 raw-query 径向梯度差不能直接视为实际归一化输出的一阶温度差。

### 2.2 已有研究关系

RMSNorm 已讨论重缩放不变性和隐式学习率适应；AdamP 已讨论移除径向更新与有效步长；Spherical Perspective 已把 SGD、momentum、Adam 放到球面有效优化框架下。式(8)–(10)是这些思想在本对照上的精确专门化，不能未经新颖性复核包装成全新原理。

## 3. epsilon 修正与阶数交叉

由式(1)直接得到

\[
DN(q)q=\frac{\varepsilon}{s^3}q.
\tag{12}
\]

两臂梯度差为 r_g=q(q^Th)/(Ds³)，故独立 query 一步输出差满足

\[
N(q_D^+)-N(q_S^+)
=-\eta\frac{\varepsilon q(q^Th)}{Ds^6}+O(\eta^2).
\tag{13}
\]

ε>0 时有一个被 ε/s² 抑制的一阶项。足够小的 η 下，它可能超过二阶有限步项。不能要求实际默认 epsilon 的所有小步长均呈二阶输出、四阶两臂 KL。

因此理论单元检验应先设 ε=0 并远离 q=0，再单独测 epsilon 的交叉区域。有限精度下，还须避开舍入误差主导的步长。

## 4. 共享 W_Q：跨 token Gram 耦合

令 q_i=W_Qx_i，冻结全部 x_i，只更新 W_Q，采用 SGD。对于第 i 个 query，定义

\[
c_i=\frac{q_i^Th_i}{Ds_i^3},\qquad
\Delta g_{q_i}=c_iq_i.
\tag{14}
\]

链式法则给出参数梯度差

\[
\Delta G_W=\sum_i c_iq_ix_i^T.
\tag{15}
\]

所以对目标 token j，两臂 raw query 之差精确为

\[
\Delta q_j=-\eta\sum_i c_iq_i(x_i^Tx_j).
\tag{16}
\]

ε=0 时令 P_j=I−u_ju_j^T，使用 DN(q_j)=P_j/s_j，得到

\[
\boxed{
\Delta N(q_j)
=-\frac{\eta}{s_j}\sum_{i\ne j}
c_i(x_i^Tx_j)P_jq_i+O(\eta^2).
}
\tag{17}
\]

自身项 i=j 被投影消去，其他 token 项一般不消失。因此每个 token 内的径向反传差，在共享参数聚合后对另一个 token 可以成为切向变化。

一个直接上界是

\[
\|\Delta N(q_j)\|
\le\frac{\eta}{s_j}\sum_{i\ne j}
|c_i|\,|x_i^Tx_j|\,\|P_jq_i\|+O(\eta^2).
\tag{18}
\]

决定量为输入 Gram 相关、query 方向差异、径向系数。输入正交、全部相关 query 共线或系数抵消时，领先项仍可为零，不能写成「共享必然产生效应」。

范围：这是冻结输入、只更新 W_Q 的受控问题。全模型 Q-detach 还会通过 W_Q^T Δg_q 改变上游 hidden states 的反传与上游参数梯度。

## 5. 统一公式：径向到切向耦合算子 PMR

考虑一个明确 normalization 站点，堆叠全部 queries。令

\[
B=\partial q/\partial\theta,\qquad
S=\operatorname{blockdiag}(s_iI),
\]

\[
R=\operatorname{blockdiag}(u_iu_i^T),\qquad P=I-R.
\]

设 ε=0，采用在两臂之间固定的线性预条件算子 A。与该站点有关的参数梯度差为

\[
\Delta g=B^TS^{-1}Rh.
\tag{19}
\]

更新差为 −ηAΔg，而 dy=S^−1PBdθ。P、R 与 S 逐块交换，因此

\[
\boxed{
\Delta y=-\eta PMRh+O(\eta^2),\qquad
M=S^{-1}BAB^TS^{-1}.
}
\tag{20}
\]

令 L=PMR。它度量 raw-query 径向反传差经参数化与预条件后进入切空间的程度。

- 独立 query、A 为每个 query 上的标量倍数时，L=0。
- 共享 W_Q 时，B B^T 含跨 token Gram 项，恢复式(17)。
- 各向异性 A 一般使 L 非零，即使 B=I。
- 如果有效输出映射进一步消去了 Lh，attention 仍可能无一阶差；不能只凭 ||Lh||>0 宣称注意力效应。

### 5.1 严格边界

1. 式(19)–(20)用于单站点干预，h 与 B 的定义必须固定清楚。
2. 若全部层都 Q-detach，较早站点的反传 h 会受到下游干预影响；不能直接把所有原始 baseline 站点项相加。应测量实际全梯度差，或用逐站点 hybrid computational graphs 做 telescoping。
3. 即使跟踪的是某层 normalized query，全模型更新还可能改变该层 K、V、位置参数、gain 等旁路。完整 attention logit 变化必须使用总 Jacobian，不能只用 query 这一路。
4. 该统一形式具有经典参数诱导度量/球面投影结构。当前把它称为「本次统一整理」，而不是未经文献比对的原创定理。

## 6. 真实 AdamW 与 momentum 的边界

投影与预条件通常不交换：若 P=I−uu^T，A 为非标量对角矩阵，通常 P A u≠0。即使原梯度差是径向的，预条件之后也可能进入切向。

真实 Adam 更不等于「当前梯度乘固定对角矩阵」：

\[
m^+=\beta_1m+(1-\beta_1)g,\qquad
v^+=\beta_2v+(1-\beta_2)g^2.
\]

偏差校正后更新 U(g;m,v) 的分子与分母都依赖 g。正确的实际一步对照为

\[
\boxed{
\Delta\theta=-\eta\{U(g_D;m,v)-U(g_S;m,v)\}.
}
\tag{21}
\]

应复制同一个模型与 optimizer state，分别真实计算两臂梯度及下一步；不可用临时几个 batch 的 EMA 或 surrogate direction 代替。

在相同起点、固定各向同性 momentum 下，当前梯度的即时差仍可为径向，故 momentum 不是每次都会破坏独立 query 的一阶零空间。然而多步之后，两臂动量历史不同；旧 query 的径向方向通常不对齐当前 query，历史项可产生切向效应。

相同参数起点的 decoupled weight decay 在即时两臂差中可抵消，但它改变共同的半径和后续有效步长。若两臂 gradient clipping 系数不同，局部梯度改变还会缩放全模型更新；必须记录并分析。

## 7. 熵方向导数

令 p=softmax(z)，F=diag(p)−pp^T。对任意方向 v，

\[
\dot p=Fv.
\]

H(p)=−Σp log p，且 F1=0、log p=z−log Z，因此

\[
\boxed{\dot H=-z^TFv=-\operatorname{Cov}_p(z,v).}
\tag{22}
\]

若 z=x+b，v=cx，则

\[
\dot H=-c\{\operatorname{Var}_p(x)+\operatorname{Cov}_p(b,x)\}.
\tag{23}
\]

只有 b 在该行恒定时才简化为 −c Var_p(x)。实际系数 c 可以正也可以负，因此「存在温度方向」不自动表示 entropy 下降。

沿有限 logit 线段 z+t v，精确有

\[
H(p_1)-H(p_0)
=-\int_0^1\operatorname{Cov}_{p_t}(z+tv,v)\,dt.
\]

塌缩后 p 接近 one-hot，F 接近退化，局部导数普遍变小。因此晚期局部分量小不等于其在塌缩形成阶段无作用。

如果把「temperature component」定义成关于 Fisher 内积对 x 的正交投影，则其与熵导数的关系可能部分由定义强制成立。此时不能再把这种相关性作为独立机制证据。

### 7.1 Fisher temperature–shape 分解与退化边界

在固定 p 下，设总 logit 方向为 v，且 Var_p(z)>0。定义

\[
\beta=\frac{\operatorname{Cov}_p(z,v)}{\operatorname{Var}_p(z)},\qquad
v_{\rm shape}=v-E_pv-\beta(z-E_pz).
\tag{22a}
\]

由定义 E_p[v_shape]=0 且 Cov_p(z,v_shape)=0。因此关于 softmax Fisher 半内积 \(\langle a,b\rangle_F=a^TFb=\operatorname{Cov}_p(a,b)\)，温度分量与 shape 分量正交，并有

\[
\operatorname{Var}_p(v)=\beta^2\operatorname{Var}_p(z)+\operatorname{Var}_p(v_{\rm shape}),
\qquad
\dot H=-\beta\operatorname{Var}_p(z).
\tag{22b}
\]

证明只是将去均值 v 展成两项、使用协方差为零，再代入式(22)。这是一个经典内积空间的正交分解，并非 temperature 因果机制的新证据；熵公式由投影定义必然成立。

若研究的是 content x 而 z=x+b，应改用 \(\beta_x=\operatorname{Cov}_p(x,v)/\operatorname{Var}_p(x)\)。这能得到关于 x 的方差正交分解，但 shape 与 z 的协方差一般不为零，故不能再无条件写成 \(\dot H=-\beta_x\operatorname{Var}_p(x)\)。非恒定 b 的项必须保留。

若 Var_p(z)=0，总 logits 在支持集内相等，temperature 系数不可识别；此时所有方向的一阶 entropy 导数都为零，虽然后续变化未必为零。若 Var_p(x)=0，content-temperature 同样不可识别。若 Var_p(v)=0，所谓温度占比也无定义。实验应预先规定阈值、单列退化行及其权重；不能删除后把条件样本的比值当总体结果。

## 8. 正向 KL 的精确式与 range 界

令实际参数更新为 d，定义真实 logit 增量

\[
u=z(\theta+d)-z(\theta),\qquad p=\operatorname{softmax}(z).
\]

则 p_i^+=p_i e^{u_i}/E_p e^u，由定义直接得到

\[
\boxed{D_{\rm KL}(p\Vert p^+)=\log E_p e^u-E_pu.}
\tag{24}
\]

令 ψ(t)=log E_p e^{tu}，p_t∝p e^{tu}。ψ''(t)=Var_{p_t}(u)，因此 Taylor 积分式给出

\[
D_{\rm KL}(p\Vert p^+)
=\int_0^1(1-t)\operatorname{Var}_{p_t}(u)dt.
\tag{25}
\]

这是经典 exponential-family / log-partition / Bregman 结构，不是本文创新。

令 R=max u−min u，V=Var_p(u)。由于

\[
e^{-tR}\le p_{t,i}/p_i\le e^{tR},
\]

结合方差表征 Var(X)=inf_a E(X−a)²，得到

\[
e^{-tR}V\le\operatorname{Var}_{p_t}(u)\le e^{tR}V.
\]

代入积分即得

\[
\boxed{c_-(R)V\le D_{\rm KL}(p\Vert p^+)\le c_+(R)V,}
\tag{26}
\]

\[
c_-(R)=\frac{e^{-R}-1+R}{R^2},\qquad
c_+(R)=\frac{e^R-1-R}{R^2},\qquad c_\pm(0)=1/2.
\]

### 8.1 三阶误差界及证明

ψ'''(t)=E_{p_t}[(u−E_{p_t}u)^3]，因为均值始终位于原区间内，

\[
|\psi'''(t)|\le R\operatorname{Var}_{p_t}(u)\le Re^{tR}V.
\]

使用三阶 Taylor 积分余项，

\[
\begin{aligned}
\left|D_{\rm KL}(p\Vert p^+)-V/2\right|
&\le\frac12\int_0^1(1-t)^2|\psi'''(t)|dt\\
&\le\frac{Re^R}{6}V.
\end{aligned}
\tag{27}
\]

这里的 R 是 range。不能在上述三阶导数证明中直接换成基准分布下的 centered radius b。centered-radius 更强界及其经典 Seneta–Weber 来源见同目录 sharp_bound/sharp_bound_proof.md；那份第4–6节 Hermite、凸性与 sharpness 证明已另行独立复核。

## 9. 从 full-model JVP 到真实 endpoint 的可控余项

设 d=ηv，a=J_z(θ)v，并定义

\[
u=\eta a+r.
\tag{28}
\]

a 必须包含全部实际更新参数；r 是完整模型有限步余项，不只是固定输入 QK 双线性余项。令

\[
\sigma_a=\sqrt{\operatorname{Var}_p(a)},\qquad
\sigma_r=\sqrt{\operatorname{Var}_p(r)}.
\]

在去均值的 L²(p) 空间，三角不等式给出

\[
(|\eta|\sigma_a-\sigma_r)_+
\le\sqrt{\operatorname{Var}_p(u)}
\le |\eta|\sigma_a+\sigma_r.
\]

结合式(26)，得到

\[
\boxed{
c_-(R)(|\eta|\sigma_a-\sigma_r)_+^2
\le D_{\rm KL}(p\Vert p^+)
\le c_+(R)(|\eta|\sigma_a+\sigma_r)^2.
}
\tag{29}
\]

又由 Var(ηa+r)=η²Var(a)+2ηCov(a,r)+Var(r) 和 Cauchy–Schwarz，

\[
\boxed{
\left|D_{\rm KL}-\tfrac12\eta^2\sigma_a^2\right|
\le\frac{Re^R}{6}\operatorname{Var}_p(u)
+|\eta|\sigma_a\sigma_r+\tfrac12\sigma_r^2.
}
\tag{30}
\]

如果每个可见 logit 在参数更新路径上的 Hessian operator norm 都至多 M，则逐坐标 Taylor 余项满足

\[
\|r\|_\infty\le\tfrac12M\eta^2\|v\|^2.
\]

由于方差至多 range²/4，

\[
\sigma_r\le\tfrac12\operatorname{osc}(r)
\le\tfrac12M\eta^2\|v\|^2.
\tag{31}
\]

当 R=O(η)、σ_r=O(η²) 时，式(30)给出通常的三阶误差。

### 9.1 不能越过的解释边界

- 未取得或证明 M 时，不能称为先验认证。用实测 r 和 R 得到的是事后 endpoint 诊断。
- 如果 Var_p(a)≈0，比值会退化，应报告绝对误差和低方差行。
- dropout、数据、mask 和随机状态必须匹配；mask 支持改变或不光滑路径需要单独分析。
- Q/K 都移动时已有双线性交叉项；上游层、gain、归一化分母等移动会引入更多项，不能用 fixed-input 结果代表 full model。
- 单行 attention KL 的界不等于训练 loss 单调、不发散、没有 entropy collapse 或 safe learning-rate 保证。
- 模型和梯度可以用任意固定精度验证，但推导中的小步长极限不能靠浮点取消误差确认。

## 10. 精确能量匹配 placebo

考虑实际被扣除的项 r_true=κuu^T h。若只减去等范数随机向量，||h−r||² 中的交叉项会改变，因而不匹配残余能量。

设 h≠0、D≥2，令

\[
e=h/\|h\|,\qquad c=|u^Th|/\|h\|.
\]

在 e 的正交补中抽单位向量 w，定义

\[
v=ce+\sqrt{1-c^2}\,w,\qquad
r_{\rm pl}=\kappa vv^Th.
\tag{32}
\]

由于 ||v||=1、v^T h=c||h||=|u^Th|，

\[
\|r_{\rm pl}\|=\|r_{\rm true}\|,
\]

\[
h^Tr_{\rm pl}=h^Tr_{\rm true}=\kappa(u^Th)^2,
\]

所以

\[
\boxed{
\|h-r_{\rm pl}\|^2=\|h-r_{\rm true}\|^2
=\|h\|^2-(2\kappa-\kappa^2)(u^Th)^2.
}
\tag{33}
\]

h=0 时全部项设零。c=0 时删除项为零；c=1 时有效随机方向退化，不能假装仍存在充分随机 placebo。

### 10.1 修正旧三臂判据

「standard≠placebo 且 placebo≈detach」不是方向特异性的普适充要条件。精确能量匹配 placebo 本身会改变方向，没有一般定理要求它等价 detached。训练中它也未必对应任何固定前向损失的真梯度，应称人为反传干预。

匹配层级必须明确：上述构造匹配 activation-gradient 的删除项和残余项，不自动匹配参数梯度、clipping、动量或实际 optimizer displacement。要主张机制，应在干预前规定哪一个中介量应如何变化，并核对预测符号和大小。

### 10.2 固定线性 projector 的不可能性与条件依赖构造的性质

记 \(P_u=I-\kappa uu^T\)、\(P_v=I-\kappa vv^T\)。对于固定单位 u,v 以及 0<κ≤1，

\[
P_v^2-P_u^2=-\kappa(2-\kappa)(vv^T-uu^T).
\tag{33a}
\]

如果要求对所有 h 同时有 ||P_vh||²=||P_uh||²，则对称矩阵 P_v²−P_u² 的二次型恒为零，故该矩阵为零，进而 vv^T=uu^T。也就是说：**一个与真实方向不同的固定线性随机 rank-one projector，不可能对所有 h 同时精确匹配残余能量。**

式(32)并不违反这个结论，因为 v 依赖当前 h。它是逐梯度条件化的、一般非线性的 gradient surgery，可作为人为反传对照；不是一个对所有 h 固定的线性 normalization Jacobian。不能因其非线性就一概禁止，但必须如实标注，并在具体干预层级解释结果。

## 11. 可证伪的定量预测

### P1：独立 query 的零空间预测

在 ε=0、固定 K、独立 query、同起点 SGD 且不存在退化抵消时，式(10)给出归一化 query、logits、概率的两臂差为 O(η²)。固定支持且起始概率为正时，KL 在两分布相等处的领先项是 Fisher 二次型，因此

\[
\|p_D^+-p_S^+\|=O(\eta^2),\qquad
D_{\rm KL}(p_S^+\Vert p_D^+)=O(\eta^4).
\tag{34}
\]

非零领先系数时可观察对应斜率；若系数为零可能更高阶，不能据此否定定理。

### P2：共享/预条件耦合预测

若 PMRh 经实际 logits 与 softmax Jacobian 后仍非零，则概率两臂差有非零 O(η) 项，通常

\[
\|p_D^+-p_S^+\|=O(\eta),\qquad
D_{\rm KL}(p_S^+\Vert p_D^+)=O(\eta^2).
\tag{35}
\]

必须检验式(17)/(20)的系数与剩余 O(η²) 误差，而不仅仅拟合一个斜率。非零 epsilon、小步长舍入误差、softmax 常量方向都可能改变读数。

### P3：实际 optimizer-state 的有限步预测

完整模型复制同一个 checkpoint 与 optimizer state 做一步分叉，使用实际 Δθ 计算 full-model JVP，再测真实 endpoint 和式(29)/(30)的余项。若研究范围内真实步长的误差持续过大，应报告局部解释的失败区域；不能自动把失败改称「刻画了边界」而继续保留原强主张。

这些是可先用 CPU 双精度验证、随后在真实 checkpoint 测量的预测，不要求先做大模型六臂长训练。

## 12. 理论项目状态与下一步

| 项目 | 当前数学状态 | 新颖性与用途 |
|---|---|---|
| RMS Jacobian、content-temperature JVP | 已证明 | 经典基础，不声称创新 |
| 独立 query 一步有效 LR 等价 | 已证明 | 既有球面优化思想的专门化 |
| 二阶差异及 epsilon 修正 | 已证明 | 支持阶数预测 |
| 共享 W_Q 的 Gram 耦合项 | 已证明，固定输入限定 | 候选核心机制，需要实证与文献比对 |
| 单站点 PMR 统一算子 | 已证明 | 本次整理，不声称原创原理 |
| 真实 AdamW 更新差 | 精确对象已定义 | 必须保存/复制真实 optimizer state 核验 |
| entropy/KL 恒等式与余项 | 已证明 | 经典结构的明确应用 |
| 精确 activation-energy placebo | 已构造并证明 | 对照工具，不是机制结论 |
| 多层多步训练的长期因果机制 | 尚未建立 | 不在当前理论中承诺 |
| collapse onset、泛化或稳定 LR 改善 | 尚未建立 | 需要独立预注册实证 |

建议候选题目：*When Does the Radial Backward Term of Query–Key Normalization Affect Attention? A Finite-Step and Optimizer-State Analysis*。

当前最小可信贡献应是：清楚区分替代 JVP 与训练干预；给出共享/优化器耦合的可计算预测；在真实一步更新中验证解释的范围；再决定是否值得用短程训练研究长程效应。经典等式的数量不能替代研究新颖性。

## 13. 已核对的一手来源与归属

1. Zhang, B.; Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS. https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html ; https://arxiv.org/abs/1910.07467 。已核对官方摘要：rescaling invariance 和 implicit learning-rate adaptation 已在原作中明确讨论。
2. Xu, J.; Sun, X.; Zhang, Z.; Zhao, G.; Lin, J. (2019). *Understanding and Improving Layer Normalization*. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2019/hash/2f4fe03d77724a7217006e5d16728874-Abstract.html ; https://arxiv.org/abs/1911.07013 。已核对官方摘要：normalization 的 backward derivatives 本身已是既有研究对象，不能把 forward-identical backward ablation 作为新思想。
3. Heo, B. et al. (2021; preprint 2020). *AdamP: Slowing Down the Slowdown for Momentum Optimizers on Scale-invariant Weights*. ICLR. https://arxiv.org/abs/2006.08217 ; https://clovaai.github.io/AdamP/ 。已核对作者页面及摘要：移除径向更新、有效学习率与 momentum 的关系已有明确结果。
4. Roburin, S. et al. (2022; preprint 2020). *Spherical Perspective on Learning with Normalization Layers*. Neurocomputing. https://arxiv.org/abs/2006.13382 ; https://www.sciencedirect.com/science/article/pii/S092523122200159X 。已核对论文摘要/出版页：球面有效学习率、方向及 Adam 的分析已有先例。
5. Bach, F. (2010). *Self-concordant analysis for logistic regression*. Electronic Journal of Statistics, 4, 384–414. https://arxiv.org/abs/0910.4627 。本稿 range 型 log-MGF 余项结构的经典相关来源；本文给出自含证明。
6. Seneta, E.; Weber, N. C. (1982). *Attainable bounds for expectations*. Journal of the Australian Mathematical Society (Series A), 33, 411–420. https://doi.org/10.1017/S144678870001884X 。centered-radius 两点极值更强先例，由 sharp_bound 独立审查核对原文；本文独立复核了该审查的自含下界与 sharpness 证明。

除上述明确的一手来源外，本文没有把尚未全文核对的文献或原实验结果当作证明依据。已有数学推导与随后 numerical audit 应分别报告，数值一致性不能证明新颖性。
