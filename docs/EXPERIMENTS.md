# 实验协议与 GPU 运行清单

协议版本：2026-09-24。这里规定要检验的问题，不预填结果。探索实验允许据实修订方案；确认实验开始前，用 `scripts/run_suite.py --freeze-only` 固定代码、数据清单、配置、种子、比较组和诊断定义。

## 1. 本论文检验什么

`standard` 与 `q_detach` 在同一参数下具有相同的数值前向。后者仅对 query RMS 分母停止梯度，因此改动的是反向传播。我们检验这种改动如何通过共享参数、真实 AdamW 一阶/二阶矩、梯度裁剪和有限步长，产生不同的注意力分布。

**禁止混淆**：detach 的自动微分规则不是数值前向函数的真实导数。预测有限参数更新时，JVP 必须用正常数值前向的导数和实际参数增量；不得用 detached surrogate JVP 代替。注意力熵变小也不自动代表性能变坏。

## 2. 运行顺序和实验预算

| 阶段 | 数据与模型 | 种子/比较 | 产物与用途 |
|---|---|---|---|
| CPU 理论与软件校验 | 合成输入、小模型 | 固定测试种子 | 雅可比、前向相等、梯度差异、数值阶数、恢复状态等；不当作自然语言结论 |
| GPU smoke | 合成 token、小模型、4 步 | 11；S/QD | 验证当前 Colab 的 CUDA、训练、保存、诊断链路；不是论文结果 |
| GPU pilot | WikiText-103 raw 固定前缀子集；6 层、宽 256、4 头；500 步 | 11、22、33；S/QD | 检查数值误差、机制大小、吞吐量和显存；明确标注探索性 |
| GPU confirmation | 相同语料处理和架构；2500 步 | 101、202、303、404、505、606；S/QD | 主实验；从头使用新初始化种子，不能把 pilot 种子混入主置信区间 |
| 稳健性 1 | 学习率 `1e-4`；其余同确认配置 | 11、22、33；S/QD | 探索性学习率依赖；使用独立结果目录 |
| 稳健性 2 | 关闭梯度裁剪；其余同确认配置 | 11、22、33；S/QD | 判断主差异是否主要由全局裁剪造成 |
| 稳健性 3 | 宽 384；其余见 `robustness_width384.json` | 11、22、33；S/QD | 探索性规模外推；显存和耗时先试运行 |
| 可选 K 控制 | `k_detach`，与 S/QD 同配置 | 与对应阶段配对 | 研究 Q/K 非对称性；无结果前不能声称 Q 特异性 |
| 最终功能评估 | 锁定的 WikiText 测试集与 PG-19 测试前缀 | 确认种子的最终检查点 | 记录 token NLL/困惑度及语料迁移；测试集不用于调参 |

S=`standard`，QD=`q_detach`。3 个探索种子和 6 个确认种子是可执行的初始预算，**不是功效分析证明充分的样本量**。完整稳健性网格不是默认一次全部启动。不要根据哪个种子显著来决定保留哪些结果。

主配置有效批量为 `4 × 4 × 256 = 4096` 个预测 token/优化器步；500 步约为 2.048M token，2500 步约为 10.24M token。分词语料默认保留 20M 训练 token；采用完整官方 validation/test 时，评估环节仍按其明确声明的计入 token 预算运行。数据采样可能重复读取 token，因此这些是训练处理量，不是唯一见过的 token 数。训练语料子集不能写成“用完整 WikiText-103 训练”。实际参数量、token 数和吞吐量以运行日志为准。

没有预先承诺 GPU 小时数。Colab GPU 型号、配额和会话会变化；先运行 smoke 和首个 pilot，并记录每步耗时、峰值显存及数据准备时间。结果写入 Drive，重连后用原协议重跑命令即可恢复。主模型连同 Adam 状态每份检查点约 0.21 GB；保留计划检查点和 latest 后，pilot 六条训练轨迹约需 6 GB，确认十二条轨迹约需 12 GB，另有数据、日志和临时文件。三组稳健性实验需要更多空间；免费 15 GB Drive 不能同时容纳完整实验包的所有结果。开始确认阶段前，把 pilot 检查点完整备份到自己的其他存储并核验，或增加存储配额；不要删除未备份的检查点。不同 GPU/库版本间恢复不保证逐 bit 一致，应保留每段运行环境记录。

## 3. 真实优化器状态的一步反事实

Pilot 的标准轨迹在步骤 `0, 50, 200, 500` 保存诊断检查点；确认轨迹在 `0, 250, 1000, 2500` 保存。每个检查点：

1. 从完全相同的参数、AdamW 矩、随机数和下一个训练批次创建 S/QD 分支；实际执行相同学习率、裁剪、权重衰减与精度策略的一步更新。
2. 记录两支真实参数增量、损失、梯度范数、裁剪和优化器状态。刚初始化的步骤 0 与有成熟矩状态的后续步骤分开解释。
3. 在固定 validation 测量批次上，计算更新前后的注意力分布、KL 和熵。诊断 `batch_size=1`、`seq_len=128`；smoke 为 16。训练梯度仍使用检查点保存的训练批量和长度，不被测量批量替换。
4. 用真实前向 Jacobian 和实际参数增量，计算 full-model JVP、温度/形状分解与非线性余项。
5. 取增量倍率 `α ∈ {0.125, 0.25, 0.5, 1}`。`α=1` 是真实下一步；倍率是在已得到的参数增量上插值，而不是用缩放学习率重跑 Adam。最后训练检查点的诊断是按设定的学习率下限额外做的一步反事实，不计入训练曲线。小倍率用于局部阶数和数值诊断，不能替代真实一步的机制证据。

诊断只在标准训练轨迹进行是主方案。对 QD 轨迹再做反事实是单独的次要分析，不混入主样本。关闭裁剪配置是解释性稳健性检查；不能悄悄用其结果替换裁剪主配置。

## 4. 主指标和统计单位

预定主指标：**确认阶段最终标准轨迹检查点、`α=1` 时，两支真实更新后注意力分布的 `KL(p_S⁺ || p_QD⁺)` 的平均值**，对应诊断字段 `endpoint_kl_mean`。先排除只有一个允许 key、KL 恒为零的第一条 query 行；其余因果注意力行按相同权重聚合为一个 seed 级结果。具体公式与权重见 `docs/METRICS.md`。

每个独立初始化种子产生一个主观测值。注意力头、层、行和检查点都是该种子内重复测量，不能当作独立样本扩充 n。报告全部 seed 值、均值和 seed 级不确定性；六个 seed 的置信区间仍可能较宽。

主指标是非负分布距离。显著大于零本身并不证明发现了新的、有用的机制：需要同时量化重复运行数值误差、绝对效应、Fisher 形状分量和真实步长下的预测余项。Full-model JVP 本身已有相关先例；贡献应来自可证伪的结构解释与实际证据，不来自“JVP 比没有预测好”的必然结论。

支持性指标：真实更新的 KL、熵变化、Fisher 温度/形状能量、full-JVP 与纯温度模型的误差、实际参数增量、剪裁率、验证 NLL。训练曲线中的同一步 S/QD NLL 差为配对比较。温度分量与一阶熵变化存在代数关系，不能把它们的相关性再当作独立验证。

不预设“熵越大越好”或“QD 必须等于随机投影对照”。不使用旧的删除范数匹配随机投影作为严格匹配控制。若将来加入自适应能量匹配投影，需单独声明它依赖当前梯度，且不保证参数更新能量匹配。

## 5. 数据与评估约定

详见 `docs/DATA.md`。WikiText 是已有语言建模基准；使用 Salesforce 发布的原始版本固定 revision，不自行抓取最新 Wikipedia。默认 tokenization 采用 GPT-2 BPE，validation/test 保留官方划分；已发现的跨划分完全重复文章从训练侧移除，保留官方评估侧。这个经过明确去重的训练变体及排除清单写入 `manifest.json`，不能称为未改动的完整官方基准。文件哈希和配置也写入清单。前缀截取是确定的预算子集，不是随机抽样或完整基准。

PG-19 来自原始研究者维护的数据集，使用其测试集和锁定的文件版本，作为独立语料的功能评估。它不证明对网页、代码、多语言或大模型的普适性。

报告的是本实现的 **GPT-2 token 级 NLL/困惑度**，不是 WikiText 论文中使用词级词表和不同上下文协议的官方困惑度。最终评估按上下文块重置，每个目标 token 只计一次；明确说明上下文长度、计入 token 数和丢弃尾部情况。禁止直接用数字宣称优于不同分词和协议的公开论文。

测试集仅在选择、配置和代码冻结后打开；通过 `qknlab.evaluate --allow-test` 明确执行。`--allow-test` 是记录研究流程的开关，不构成统计学上的外部审计。所有曾查看过的测试结果都应计入研究记录。

## 6. 命令清单

在仓库根目录运行；路径可替换为自己的 Drive 路径。

```bash
python -m qknlab.data prepare --dataset wikitext103 --out data/wikitext103 --max-train-tokens 20000000
python -m qknlab.data verify --data-dir data/wikitext103
python scripts/run_suite.py --stage pilot --data-dir data/wikitext103 --out results/pilot --device cuda
```

仅看将执行的命令：增加 `--dry-run`。分次跑相同计划的种子：增加 `--run-seeds 11`，随后 `--run-seeds 22 33`；不要用 `--seeds` 改变已开始的计划。暂时只做前 50 步：增加 `--max-steps 50`，之后去掉该选项恢复完整日程。只执行某类任务：`--tasks train` 或 `--tasks diagnostics`。

确认阶段先审查 pilot，然后固定协议。两条命令的计划参数必须相同：

```bash
python scripts/run_suite.py --stage confirm --data-dir data/wikitext103 --out results/confirmation --device cuda --protocol results/protocol_confirmation_v1.json --freeze-only
python scripts/run_suite.py --stage confirm --data-dir data/wikitext103 --out results/confirmation --device cuda --protocol results/protocol_confirmation_v1.json
```

代码、数据或配置变动会使冻结校验失败。需要改动时保留旧结果和协议，使用新版本协议文件和新的输出目录，并解释变更原因；不要修改旧文件来隐藏修订。

三个稳健性配置分别运行，种子显式固定为探索种子：

```bash
python scripts/run_suite.py --stage robustness --config configs/robustness_lr.json --seeds 11 22 33 --data-dir data/wikitext103 --out results/robustness_lr --device cuda
python scripts/run_suite.py --stage robustness --config configs/robustness_no_clip.json --seeds 11 22 33 --data-dir data/wikitext103 --out results/robustness_no_clip --device cuda
python scripts/run_suite.py --stage robustness --config configs/robustness_width384.json --seeds 11 22 33 --data-dir data/wikitext103 --out results/robustness_width384 --device cuda
```

K 控制可在**新的独立目录**执行：

```bash
python scripts/run_suite.py --stage pilot --arms standard q_detach k_detach --diagnostic-arms standard q_detach k_detach --data-dir data/wikitext103 --out results/pilot_k_control --device cuda
```

最终测试示例：

```bash
python -m qknlab.evaluate --checkpoint results/confirmation/runs/seed_101/standard/checkpoints/step_002500.pt --data-dir data/wikitext103 --split test --out results/test/wikitext103/seed101_standard.json --device cuda --max-tokens 500000 --allow-test
python -m qknlab.data prepare --dataset pg19 --out data/pg19 --max-eval-tokens 1000000
python -m qknlab.evaluate --checkpoint results/confirmation/runs/seed_101/standard/checkpoints/step_002500.pt --data-dir data/pg19 --split test --out results/test/pg19/seed101_standard.json --device cuda --max-tokens 500000 --allow-test
```

以上评估须对所有确认种子和两支重复执行，不能只挑一个种子。Notebook 已提供循环。

## 7. 停止、失败和负面结果

出现 NaN、状态恢复不一致、数据校验失败或主信号接近数值误差时，先定位问题，不扩大算力。若发现主要效应由裁剪或极端学习率驱动，论文必须据此收缩结论。若 full-model 真实步长余项过大，只能说局部线性解释不足；不得用很小的 α 替代 α=1 以制造支持。

如果机制证据可复现但 NLL 没有改善，可写解释性论文，不能宣称新训练方法提高性能。如果所有确认结果都很弱，应重新界定问题并报告限制，而不是因为已经选择 Entropy 就保留错误方向。
