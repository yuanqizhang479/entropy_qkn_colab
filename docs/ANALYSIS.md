# 统计单位、预先指定的终点和结果解释

本包研究 **QK-RMS 归一化的反向传播如何改变真实的有限参数更新**。标准归一化与 Q-detach 的数值前向相同，反向不同；因此需要区分归一化层的代数性质、模型中真实优化器更新的效果、以及训练后的语言建模表现。

本文件规定分析口径，不预言结果。代码生成的图表不是录用保证，也不把负结果改写成性能优势。

## 1. 哪个量是一个独立重复

独立重复是预先指定的**训练随机种子**。同一种子下 standard 与 q_detach 使用相同初始化、数据顺序、训练预算和评估文本，构成一对。

- 同一种子中的 attention head、layer、query row、token、minibatch 和 checkpoint 都不是新的独立重复。
- 单个诊断文件先在一个种子内部汇总。各层、各 head、各 batch 的有效 causal query row 等权；第一行只有一个可见 key，其注意力恒为 1，故排除。这里没有用长序列的更多 key 数量重新加权。
- 跨种子时每个种子等权。不能因为某一个种子生成更多 checkpoint，就给它更大的统计权重。
- 不把 pilot 与 confirmation 合并后声称结果已经独立确认。默认 pilot 为 `11,22,33`；confirmation 使用另一组固定新种子，详见协议及配置。

## 2. 预先指定的实际更新终点

在 **standard 训练轨迹**的同一个 checkpoint，从完全相同的模型参数、AdamW 动量、二阶矩、下一训练批次及优化器设置出发，分别执行一次 standard 与 q_detach 更新。测量批次来自独立 validation 文本，且不得用它选择更新。

每个种子的主机制描述量为

\[
D_s=\frac{1}{|\mathcal R|}\sum_{r\in\mathcal R}
D_{\mathrm{KL}}\!\left(p^{+}_{S,r}\Vert p^{+}_{QD,r}\right).
\]

其中 \(\mathcal R\) 为上文定义的行集合。字段名为 `endpoint_kl_mean`。方向固定为 **standard 更新后的 attention 到 Q-detach 更新后的 attention**。这是两次实际更新的注意力差异，不是语言模型输出词表分布的 KL，也不是性能改善指标。

主分析使用 `alpha=1`，即真实优化器更新。其他 alpha 仅用于有限步长和 Taylor 余项诊断；不能在看到结果后挑选最有利的 alpha，也不能用足够小 alpha 下的一阶近似正确替代实际步长下的证据。

`D_s > 0` 只能说明干预改变了注意力。它本身不能说明径向梯度一定有害、模型一定更好，或差异全部来自一个指定层。首先还需与同臂重放的数值误差尺度比较；实际影响很小，即使方向一致也应报告很小。

## 3. 机制支持量及其边界

共同起点 logits 为 \(z\)，实际两臂 logits 差为
\(u=z(\theta+\Delta\theta_{QD})-z(\theta+\Delta\theta_S)\)。
全参数一阶预测是

\[
v=J_z(\theta)(\Delta\theta_{QD}-\Delta\theta_S).
\]

这里的 Jacobian 是**数值前向的导数**。不能对 detached RMS 的 surrogate backward 再求一个所谓前向 JVP，作为实际函数变化的预测。

以共同起点注意力 \(p=\operatorname{softmax}(z)\) 为权重，逐行移除常数方向，并定义

\[
\beta=\frac{\operatorname{Cov}_p(z,v)}{\operatorname{Var}_p(z)},\qquad
v_{\rm temp}=\beta(z-\mathbb E_pz),\qquad
v_{\rm shape}=v-\mathbb E_pv-v_{\rm temp}.
\]

temperature comparator 只投影**预测增量** \(v\)，不使用观测到的终点来拟合，因此不存在用真实结果反拟合 comparator 的信息泄漏。`shape_energy_fraction` 为各行形状分量 Fisher 能量总和与各行预测 Fisher 能量总和之比，不是逐行分数的简单平均。

`jvp_fisher_relative_rmse` 与 `temperature_fisher_relative_rmse` 使用相同的观测增量 Fisher 能量作分母。汇总器另报告

\[
I_s=1-\frac{\mathrm{RMSE}_{\rm full,s}}{\mathrm{RMSE}_{\rm temp,s}}.
\]

正值表示完整一阶预测在这个诊断中比 temperature-only comparator 误差小。若分母为零或接近数值退化，原始程序返回 `null`；汇总器不会用零填补，也不只选择未退化种子计算一个貌似完整的置信区间。

需要同时报告以下限制：

- Fisher 正交分解是一条代数恒等式；它本身不是新颖性的实验证据。
- 熵导数 \(-\operatorname{Cov}_p(z,v)\) 与温度投影的关系按定义成立，二者高度相关不能再作为独立验证。
- 所测余项和 KL 界可以检验有限步变化，但没有预先有效的曲率界时，是事后诊断，不是已证明的安全学习率。
- 单层、固定线性预条件器下的推导不能直接写成对多层 AdamW 的一般定理。实验使用真实一步 AdamW 更新，其中一阶矩、二阶矩、梯度裁剪以及参数共享都可能改变结果。
- 熵低不等于模型差；attention sink 或某层尖锐注意力也不自动构成训练失败。

## 4. 性能终点与配对置信区间

训练表现单独用固定预算结束时、相同文本上得到的 token 平均 NLL（nats/token）评估。配对效应定义为

\[
d_s=\mathrm{NLL}_{QD,s}-\mathrm{NLL}_{S,s}.
\]

正值表示 Q-detach 更差，负值表示更好。汇总程序对 \(d_s\) 计算均值和基于独立种子的 Student t 近似 95% 区间：

\[
\bar d\ \pm\ t_{0.975,n-1}\frac{s_d}{\sqrt n}.
\]

这不是分布无关保证。三个 pilot 种子的区间可能很宽，对分布偏离也敏感；六个 confirmation 种子也不是自动足够的功效保证。一个种子只能给描述值，不给区间。程序不把有界机制指标的区间强行截到 \([0,1]\)，避免隐藏近似区间本身的局限。

不能因 NLL 差异“不显著”就声称两方法等价或非劣。若需要非劣性结论，必须在确认实验前规定有实际意义的非劣界及对应检验；本包默认不作该结论。validation 用于方案选择，锁定方案后应保留 test 与外部文本作为最终泛化检查，不能反复查看 test 调参。

## 5. 如何运行汇总

```bash
python -m qknlab.summarize \
  --runs-root /content/drive/MyDrive/qkn_entropy/runs/pilot \
  --out /content/drive/MyDrive/qkn_entropy/analysis/pilot \
  --phase pilot --expected-seeds 11,22,33 --step 1000
```

`--step` 必须换成协议中预先指定、实际完成的诊断步数；同一命令也选择该步的训练 validation NLL。如果以最终训练步作为主分析，务必使用配置的终点步数。跨多个 checkpoint 的诊断可以逐步另外运行，用作带明确标签的探索图，不能将它们合成更多独立种子。

confirmation 命令对应改成独立目录、`--phase confirm` 及预先固定的新种子列表。`--runs-root` 应只包含一个模型/数据/预算条件；外部数据诊断放在独立目录并分别汇总，不能混池。

稳健性实验保持**探索性**标签，每个独立条件分别汇总。例如实际完成 2500 步、种子为 `11,22,33` 的一个条件：

```bash
python -m qknlab.summarize \
  --runs-root /content/drive/MyDrive/qkn_entropy/robustness/condition_name \
  --out /content/drive/MyDrive/qkn_entropy/analysis/robustness_condition_name \
  --phase robustness --expected-seeds 11,22,33 --step 2500
```

`condition_name` 换成该条件的实际输出目录；不能把不同精度、裁剪或学习率条件合并成更多种子。这些区间是探索性描述，未作多个稳健性条件之间的多重比较校正。`smoke` 仅用于工程验证，不提供研究推断汇总入口。

`qknlab.evaluate` 生成的最终 test / 外部文本评估是独立 JSON 文件，不混入训练 validation 日志。每个数据集放一个目录，然后增加 `--evaluations-root`：

```bash
python -m qknlab.summarize \
  --runs-root /content/drive/MyDrive/qkn_entropy/runs/confirm \
  --evaluations-root /content/drive/MyDrive/qkn_entropy/test/wikitext103 \
  --out /content/drive/MyDrive/qkn_entropy/analysis/confirm_wikitext103 \
  --phase confirm --expected-seeds 101,202,303,404,505,606 --step 2500
```

同样换成独立的 `test/pg19` 目录和输出目录，得到外部文本结果。两个数据集不要放到同一 `--evaluations-root` 下。该功能还检查相同 token 文件哈希、评分 token 数、上下文长度、评估协议、来源训练数据及配对初始化；主 held-out 评估只接受达到完整训练预算的 checkpoint。预先指定的文本前缀可以评估，但会明确显示 `full_split=false`，不能写成整个数据集结果。

生成：

| 文件 | 内容 |
|---|---|
| `summary.json` | 分析状态、逐种子值、区间、输入文件哈希及所有不匹配原因 |
| `training_seeds.csv` | 每个种子每个训练臂在所选步的 NLL |
| `diagnostic_seeds.csv` | 每个种子的实际一步诊断，不展开成伪独立 head 样本 |
| `paired_nll.png` / `.pdf` | 同一种子的两个训练臂用线连接 |
| `diagnostic_seeds.png` / `.pdf` | 每种子的机制描述量；完整组才绘制均值和近似区间 |
| `evaluation_seeds.csv` | 指定 `--evaluations-root` 后生成的最终 test / 外部文本配对值 |
| `paired_evaluation_nll.png` / `.pdf` | 指定 `--evaluations-root` 后生成的 held-out 配对图 |

缺配对种子、重复 run、不同数据 manifest、不同模型/训练配置、不同精度、配对初始化不一致、训练未到预算、未明确选择多个诊断 checkpoint 中的一个，都会使程序返回非零状态并写出审计原因。逐种子 CSV 仍可用于查错，但不会悄悄删掉失败种子后给出“完整”的推断。

训练运行和诊断运行可以分别汇总；若一个完全缺失，报告会明确标记 `not_available`，不能把它解释为该项已通过。额外未请求种子会被列为排除项。分析者仍应检查排除日志，不能用缩小 `--expected-seeds` 的方法在看过结果后挑选种子。

## 6. Pilot 到 confirmation 的锁定

在查看确认种子结果前，应固定并保存：研究假设、主要终点、诊断 checkpoint、alpha=1、模型大小、数据 manifest、训练预算、评估文本及批次、优化器/裁剪/精度、种子列表、失败与重跑规则、分析脚本提交号。

软件不会验证研究者是否真的提前锁定协议，也不会凭 `--phase confirm` 将一次探索分析变成预注册确认。pilot 的用途是估计实际效应、数值误差和算力预算，决定是否值得完成确认；如果修改主要终点或实验条件，需明确记录修改理由，并保持确认种子未被用于这次选择。
