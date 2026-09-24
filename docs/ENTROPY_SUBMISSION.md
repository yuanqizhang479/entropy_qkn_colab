# 面向 Entropy 的写作与投稿安排

核查日期：2026-09-24。本文件是写作计划，不是已完成实验结果或接收保证。

## 研究定位

建议论文主线：**Query–Key 归一化的反向传播几何，如何经过共享参数和真实优化器状态，改变有限步长下的注意力分布。**

信息论对象必须承担解释作用：关注真实更新引起的注意力 KL、Fisher 度量下的温度与形状分量、熵的一阶变化和有限步长余项。语言模型验证集 NLL 用来衡量功能影响。不能只在一般训练曲线旁边附一条熵曲线，就声称是信息论贡献。

Entropy 的官方范围包含熵与信息论的理论、新见解和应用，也覆盖机器学习；官方说明计算研究应提供足够细节以便复现。这为本研究提供范围上的匹配，但不意味着编辑或审稿人一定认可创新性。[1]

建议文章类型为常规 **Article**。具体栏目和编辑选择以投稿系统当日选项为准。不能将期刊显示的首次决定时间当作接收时间，也不应为了 11 月期限夸大贡献。

## 建议结构

1. **Introduction**：明确未解问题；区别“归一化改变前向尺度”和“相同前向下改变反向传播”的研究问题；交代对真实注意力更新的意义。
2. **Related Work**：QK-Norm、DetachNorm/RMSNorm、尺度不变优化、注意力熵塌缩、函数空间更新。说明哪些等式或界限已有相关先例。
3. **Theory**：完整列出假设，区分独立激活、共享投影矩阵、固定预条件器和真实 AdamW。证明与反例一并呈现。
4. **Experimental Protocol**：公开版本、语料子集、分词、初始化、优化器状态、随机种子、预算、统计单位与探索/确认分离。
5. **Results**：先给校验与反事实机制结果，再给短程训练和独立语料的功能结果；同时报告无效或相反结果。
6. **Discussion and Limitations**：有限模型规模、有限步数、语料范围、训练精度、未完成的大规模验证。不要外推到现代大语言模型或所有注意力结构。
7. **Back Matter**：作者贡献、资金、数据可用性、代码与补充材料、利益冲突、AI 使用说明，以及参考文献。

## 必须区分的要求

| 项目 | 来源与落实 |
|---|---|
| 数据可用性声明 | MDPI/Entropy 作者指南要求；给出原始来源、具体版本和派生实验数据位置。[2] |
| 可复现细节 | Entropy 官方范围说明强调；提供代码、运行参数、环境、状态检查点和日志。[1] |
| AI 辅助披露 | MDPI 对实质性生成、设计、分析等使用要求披露；本项目涉及设计和代码，不能仅称为语法润色。[3] |
| 预先冻结协议、配对随机种子、负面结果 | 本项目提高证据质量的设计标准，**不是声称期刊规定必须使用这些数量或方法**。 |
| 存档 DOI | 建议在投稿前将代码版本与派生结果存档到合适的长期仓库；不要提前填入不存在的 DOI。 |

本实验包不重新分发完整 WikiText/PG-19 原文。数据可用性声明需指向原发布者，并提供下载脚本、锁定版本、清单与处理规则。公开模型检查点、分词文件或语料片段之前，还应遵循各自许可。

## 可填写的英文声明草稿

以下占位内容必须在真实运行、人工复核后填写，不可原样当作已完成事实提交。

**Data and code availability:**

> The source corpora are publicly available from the repositories cited in the Methods section. The exact dataset revisions, source-file identifiers, preprocessing settings, and checksums are recorded in the accompanying data manifests. Experiment code, configurations, and derived results are available at [repository URL and immutable commit or archive DOI]. The repository does not redistribute the complete source corpora. [Describe any access or license limitations.]

**AI-assisted research disclosure:**

> During this study, the authors used [tool name, version if available, and access dates] to assist with literature search, experimental design, mathematical analysis, code development, and manuscript drafting. Human authors checked the cited sources, reviewed the mathematical arguments, inspected and tested the implementation, and verified the reported experiments. The authors reviewed and edited the outputs and take full responsibility for the final work.

这段是针对本研究的自拟表述，并非逐字引用期刊模板。只有实际完成了所述人工核查才保留对应句子。Methods 应进一步描述 AI 参与哪些环节，Acknowledgments 提供工具信息，投稿流程也要按当日要求申报。[3] AI 工具不得列为作者。[2]

## 截止日期倒排

目标是尽早提交完整、可信且范围明确的稿件。建议按真实吞吐量调整：

| 时间窗口 | 决策或交付 |
|---|---|
| 9 月 24–27 日 | Colab 环境检查、试运行、记录吞吐量和显存；执行探索种子 |
| 9 月 28–30 日 | 审查机制信号与数值误差；确定是否继续主线；冻结确认协议 |
| 10 月 1–6 日 | 完成确认种子、独立语料评估、所有预先约定的稳健性检查 |
| 10 月 7–10 日 | 完成全文、引用和代码审计，争取提交 |
| 此后 | 及时处理编辑询问和审稿意见，保留补实验能力 |

这是一份执行目标。11 月中旬前是否接收取决于编辑和审稿流程，不能由代码包或期刊平均时长保证。如果探索实验没有支持原假设，应缩小结论、报告限制，必要时调整论文问题，而不是反复筛选随机种子。

## 官方链接

1. Entropy Aims & Scope: https://www.mdpi.com/journal/entropy/about
2. Entropy Instructions for Authors: https://www.mdpi.com/journal/entropy/instructions
3. MDPI Research and Publication Ethics, Artificial Intelligence: https://www.mdpi.com/ethics
4. MDPI LaTeX templates and author resources: https://www.mdpi.com/authors

2026-09-24 检索获得官方范围、作者指南索引和 AI 政策内容；部分官方页面直接访问返回 HTTP 429。提交前应在浏览器重新查看实时要求。
