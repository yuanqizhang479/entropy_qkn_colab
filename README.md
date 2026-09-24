# Entropy QK-Norm 实验包

目标：研究 **QK 归一化的径向反传项，怎样经过共享参数、AdamW 状态和有限参数更新，改变注意力分布**。这是机制研究；不预设 Q-detach 更好，不把数学恒等式或已有 KL 界包装成新发现。

本包包含可上传 GitHub 的源码、Colab notebook、官方数据下载及校验、配对训练、真实优化器状态的一步分叉、全参数真实前向 JVP、种子级统计与作图。**验证范围与实际运行记录见 `validation/VALIDATION_REPORT.md`。GPU 长程结果尚需运行，不能把 CPU 自检视为 GPU 实验结论。**

## 直接在 Colab 开始

1. 解压，将本目录内的文件上传到你的 GitHub 仓库根目录。根目录应直接看到 `qknlab/`、`configs/`、`requirements.txt` 和 `notebooks/`，不要只上传 ZIP。
2. 在 [Colab](https://colab.research.google.com/) 选择“打开笔记本 → GitHub”，输入仓库地址，打开 `notebooks/Entropy_QKN_Colab.ipynb`。也可直接把此 notebook 上传到 Colab。
3. GitHub 路线设置 `SOURCE_MODE="github"`，将 `REPO_URL` 改成自己的仓库地址；选定一次代码提交并记录。若暂不公开项目，保留默认 `SOURCE_MODE="upload_zip"`，在提示时上传完整实验 ZIP 即可。免费环境的 GPU 类型与可用时间不保证固定，先选择“运行时 → 更改运行时类型 → GPU”。
4. 顺序执行环境检查、自动测试、smoke、数据准备和 pilot。日志与检查点写入你指定的 Google Drive 目录。Colab 重连后用相同代码、数据和输出目录续跑。
5. 先拿到 pilot 的完整输出再启动确认实验。确认阶段运行前用 notebook 中的命令冻结配置、源码、数据清单和种子，之后不得一边看确认结果一边改协议。

无需预先手工下载全部语料，也不需要付费数据账号。默认从官方地址自动下载；访问失败时的文件直链和离线导入方法见 [docs/DATA.md](docs/DATA.md)。

## 实验清单

| 阶段 | 任务 | 默认规模 | 输出用途 |
|---|---|---|---|
| G0 | 代数、梯度、掩码、JVP、续跑自检 | CPU/小模型 | 防止代码把假设“验证”为真 |
| G1 | `standard` 与 `q_detach` 配对探索训练 | 3 个种子 × 2 臂，500 步，6 层/256 维 | 检查信号、吞吐量、数值误差与预算 |
| G2 | 从 standard 检查点复制真实 AdamW 状态再分叉 | 多检查点；α=0.125/0.25/0.5/**1** | 测真实注意力改变及 JVP 余项 |
| G3 | 冻结协议后的确认训练与分叉 | 6 个新种子 × 2 臂，2500 步 | 种子级效应和置信区间 |
| G4 | 官方 WikiText test 与 PG-19 test 评估 | 冻结后评估最终检查点 | 语言建模功能影响、域外限制 |
| R1 | K-detach 反向传播对照 | 同一检查点，`--diagnostic-arms standard q_detach k_detach` | 检查 Q 侧温度解释的适用边界 |
| R2 | 较低学习率、关闭裁剪、较宽模型 | 3 份独立配置 | 判断观察是否依赖单一训练条件 |

R1/R2 的完整命令在 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。它们应由探索结果决定优先级并在确认前约定；**不能只报告支持假设的组合**。六个确认种子是可执行的起始预算，不是预先声称已经达到统计功效。

主模型采用 MHA、学习式位置嵌入、参数无关 Q/K RMSNorm、固定注意力缩放、无 dropout。它是可审计的小型机制模型，不能直接外推到 RoPE/GQA/MLA、大型预训练模型或所有 QKN 实现。训练使用 GPT-2 BPE；NLL 以 nats/BPE token 报告，BPE perplexity 不能直接与论文的 word-level perplexity 排名比较。

## 本地命令

需要 Python ≥3.11；本次 CPU 参考环境为 Python 3.12、PyTorch 2.6.0。Colab 保留其预装 CUDA PyTorch，运行自动测试验证兼容性，不把 CPU wheel 安装到 GPU 环境。

```bash
# CPU 主机；Colab 用 notebook 安装 requirements.txt 即可
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python scripts/verify_geometry.py

# 仅用于工程自检的随机整数数据，不可作为论文证据
python -m qknlab.data prepare --synthetic --out data/smoke --max-train-tokens 8192 --max-eval-tokens 2048
python scripts/run_suite.py --stage smoke --data-dir data/smoke --out runs/smoke --device cpu

# 官方数据：默认固定前 2000 万 BPE 训练 tokens；验证/测试完整保留
python -m qknlab.data prepare --dataset wikitext103 --out data/wikitext103 --max-train-tokens 20000000
python -m qknlab.data verify --data-dir data/wikitext103
python scripts/run_suite.py --stage pilot --data-dir data/wikitext103 --out runs/pilot --device cuda
python -m qknlab.summarize --runs-root runs/pilot --out outputs/pilot --phase pilot --expected-seeds 11,22,33 --step 500
```

本次真实下载审计发现训练集与测试集有 1 篇完整文章精确重复（The Hustler (film)）。脚本保留官方验证/测试集，从训练集排除该文章，再选择训练前缀；处理记录和哈希写入数据清单。因此应称为 WikiText-103 的精确去重训练变体。使用数据前缀是明确的计算预算选择，**不是完整 WikiText-103 训练，也不是随机代表性抽样**。清单记录边界与全文档重复审计。要使用完整训练集，传 `--max-train-tokens 0`，并在确认前固定这个选择。

## 为什么这样设计

- 两臂数值前向相同，改变的是反向传播规则。训练产生真实参数增量后，始终用 **standard 数值前向的导数** 做 JVP，不能把 detach 的替代导数当真实函数的导数。
- 一步分叉保留 AdamW 的一阶、二阶矩、裁剪、混合精度状态和同一组微批次。α 是实际更新完成后的参数插值；α=1 才是真实更新，α 不是另跑一次学习率。
- 主机制终点为最终 standard 检查点上 `KL(attention_S+ || attention_QD+)` 的有效行平均。它必须结合重复执行误差、绝对量级、形状分量和训练结果解释。KL 非负或局部 JVP 准确本身不构成创新。
- 温度/形状分解使用 Fisher 度量；熵导数与温度系数的相关关系部分由定义直接成立，不作为独立发现。
- 每个随机种子是一个重复单位。层、头、token、检查点用来观察结构，不当成额外独立样本。确认缺臂或数据不一致时汇总器会拒绝生成完整推断。
- 官方 test 不参与调参；PG-19 只作冻结后的域外评估。自动检查的是完整文档精确重复，不能声称已经排除了所有近似重复或跨语料语义重合。

## 时间、显存、存储与恢复

默认单 GPU 顺序运行，不同时启动全部种子。先用真实 pilot 的 `tokens_per_second`、`update_seconds` 和 GPU 显存记录估算总时长；本包不提供未经测量的 A100/T4 小时承诺。中断后同目录重跑 suite，训练自动读取最近完整检查点。

按约 1800 万参数、FP32 权重与 Adam 状态估算，单个带优化器检查点约 210 MB；显式里程碑、滚动恢复副本会累加。pilot 约数 GB，确认阶段约十余 GB，二者及消融全保留可能超过免费 Drive 的可用空间。确认前把 pilot 检查点备份到本地或扩充可用空间；保留日志、数据清单、配置和必要检查点。程序不会自动删除你的研究结果。

显存不足时，**先在探索阶段**减小 microbatch 并相应增加 `grad_accum`，保持每步 token 数；修改后生成新配置和输出目录，配对两臂都重跑。不要只改某一臂，也不要把不同精度或硬件造成的差异包装成机制效应。精确续跑通过的是本环境内恢复测试，不保证跨硬件逐比特一致。

需要发回继续分析的文件：每个运行的 `run_manifest.json`、`metrics.jsonl`、`summary.json`，全部 diagnostics JSON、统计输出、冻结协议、数据 `manifest.json`、环境记录；保留最终及关键里程碑 `.pt`，以便补实验。

## 文档索引

- [docs/DATA.md](docs/DATA.md)：官方来源、论文引用、版本、下载和许可。
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)：运行矩阵、确认冻结、续跑和消融。
- [docs/METRICS.md](docs/METRICS.md)：KL/JVP/温度/形状指标的精确定义。
- [docs/ANALYSIS.md](docs/ANALYSIS.md)：统计单位、配对比较与缺失处理。
- [docs/THEORY.md](docs/THEORY.md)：理论推导、假设边界及已知先例。
- [docs/ENTROPY_SUBMISSION.md](docs/ENTROPY_SUBMISSION.md)：写作结构、可复现材料与 AI 辅助披露。

本包完成的是实验基础设施和核验；研究结果、论文创新性和 11 月中旬前接收仍需真实实验和同行评审支持。
