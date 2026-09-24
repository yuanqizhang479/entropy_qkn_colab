# 数据来源、下载、校验与解释边界

本包选择 **WikiText-103 raw** 作为主语料，**PG-19 官方 test** 作为可选跨域评价。它们都是公开语言建模基准：WikiText 源于 Merity 等的 *Pointer Sentinel Mixture Models*；Transformer-XL 和 Compressive Transformer 使用了 WikiText-103；PG-19 由 Compressive Transformer 论文提出。这里保证的是来源可追踪、处理可复现，不声称所有 QK-Norm 论文都使用这两个数据集。

这是机制实验：相同模型、初始化、数据流、训练预算下比较标准 RMS 反传与分母 detach；不是挑战这些语料上已有的最优困惑度。

## 1. 主数据：WikiText-103

- 原始论文：[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)。
- 作者所属机构维护的数据仓库：[Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)。
- 实验使用的固定版本：[b08601e04326c79dfdd32d625aee71d232d685c3 / wikitext-103-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)。
- 同类语言模型论文依据：[Transformer-XL，ACL 2019](https://aclanthology.org/P19-1285/)。
- 官方卡片声明 CC BY-SA 3.0 / GFDL；使用和再分发须保留原始语料的许可与署名。本包不打包语料全文。

### 已发现并修正的真实数据问题

2026-09-24 对固定版本四个 Parquet 文件逐一校验 SHA-256，重建整篇文章后，发现 **1 篇完全相同的文章同时出现在官方 train 和 test**：

| 字段 | 值 |
|---|---|
| 文章标题 | The Hustler (film) |
| train 文档单元索引（本包重建，0 起） | 3821 |
| test 文档单元索引（本包重建，0 起） | 59 |
| 原文字符数 | 17868 |
| UTF-8 `text.strip()` SHA-256 | `79e23ba5454fce579d81088db666e12869463168a985ace6895aa1974b4c77fd` |

处理规则已写进代码：**保持官方 validation/test 内容和归属不变，先将与任一完整 held-out 文章完全相同的 train 文章排除，再选取训练 token 前缀。** 每次准备数据都重新执行完整文章审计，不能跳过小预算之外的文章；排除项写入 manifest 和文章账本。

文档边界采用“两侧空行包围的一级标题”规则。额外检查排除了 `= 1 rad / s ... =` 这类公式续行被误认为标题的错误，并加入回归测试。固定 raw 快照重建得到 train/validation/test **28472/60/60 个文档单元**；原始论文 Table 1 报告的是28475/60/60篇。后来的 [ACL 2019 W-LDA 论文 Table 1](https://aclanthology.org/P19-1640.pdf) 也记录了28472个 WikiText-103 训练文档，但这不能证明其预处理与本包相同。本包保留并披露这个计数差异，不声称逐篇恢复了原始爬虫内部文章ID；标题规则定义的是本实验的 EOS 与精确去重单元，不删去其中正文。**禁止为凑齐原论文计数而插入、拆分或编造文章。**

因此论文应写明“WikiText-103 raw，GPT-2 BPE，对训练集执行完整文章级 held-out 精确去重”，不能写成“未修改的标准 WikiText-103 benchmark”。`full_source_train_heldout_exact_duplicate_articles` 与 `after_filter_cross_split_exact_document_duplicates` 分别记录原始来源重复数和过滤后的重复数。审计只排除**完全相同文章**；不保证无段落级、近似或语义重合。上述检查不是新的学术发现主张。

### 自动准备

```bash
# 20M GPT-2 训练 token 前缀；完整 validation/test；需要约315MB原始Parquet下载。
python -m qknlab.data prepare --dataset wikitext103 --out data/wikitext103 \
  --max-train-tokens 20000000 --max-eval-tokens 0
python -m qknlab.data verify --data-dir data/wikitext103

# 如计划使用全部去重后的训练文本，将 --max-train-tokens 改为0。
# 这改变训练数据配置，必须重新冻结协议；不可覆盖正在确认实验的数据目录。
```

默认源缓存为 `.cache/qknlab-sources`，可指定 `--cache-dir /content/drive/MyDrive/qkn_sources` 避免 Colab 重连后重下载。输出目录已存在时拒绝覆盖；如只是继续实验，不要重建数据。只有明确需要重新生成时使用 `--force`。

### 手动下载备用

若 Colab 的网络不能访问自动下载地址，下载下面四个文件放在同一文件夹：

- [train-00000-of-00002.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00000-of-00002.parquet)
- [train-00001-of-00002.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00001-of-00002.parquet)
- [validation-00000-of-00001.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/validation-00000-of-00001.parquet)
- [test-00000-of-00001.parquet](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/test-00000-of-00001.parquet)

文件的完整固定 SHA-256 在 `qknlab/data.py:WIKITEXT_FILES`，自动和离线模式都校验。任一字节改变即拒绝使用。

同一文件夹再放入以下两个 tokenizer 文件（保留链接中的文件名）：

- [vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe)
- [encoder.json](https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json)

然后：

```bash
python -m qknlab.data prepare --dataset wikitext103 --out data/wikitext103 \
  --offline-dir /content/drive/MyDrive/qkn_raw --max-train-tokens 20000000
```

这是无需联网的处理模式；缺失文件或校验失败会停止，不会悄悄换数据源。下载端点访问无需 Hugging Face token、账号或接受 gated 条款。

## 2. 跨域数据：PG-19 test

- 官方仓库：[google-deepmind/pg19](https://github.com/google-deepmind/pg19)。
- 原始论文：[Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)。
- 官方数据桶：[deepmind-gutenberg](https://console.cloud.google.com/storage/browser/deepmind-gutenberg)。
- 本包直接读取官方桶的 test 文本，不执行 Hugging Face 的远程 Python 数据脚本。官方 test 共100本书；`configs/pg19_test_sources.json` 固定每本书的对象 generation、原始大小和 MD5，准备时另外记录 SHA-256。generation URL 锁定具体对象版本，不跟随可变最新版。
- 官方仓库将该基准标为 Apache-2.0；历史书籍仍应保留来源信息并按所处地区的原始文本许可使用。书籍为历史英语，不代表一般网页、现代对话或所有领域。

```bash
python -m qknlab.data prepare --dataset pg19 --out data/pg19 --max-eval-tokens 1000000
python -m qknlab.data verify --data-dir data/pg19
```

本命令产生 `test.bin`，不产生 train，也不从 PG-19 调参。按已固定列表顺序选取最多1M BPE token；为审计和数据固定，会读取全部100本官方测试书。需要完整测试集时使用 `--max-eval-tokens 0`。不得把1M前缀结果写成完整 PG-19 benchmark 结果。

手动下载时，`configs/pg19_test_sources.json` 的每条记录对应 URL：

```text
https://storage.googleapis.com/deepmind-gutenberg/{name}?generation={generation}
```

保存为 `qkn_raw/test/书籍编号.txt`，再添加上述 tokenizer 文件，执行带 `--offline-dir qkn_raw` 的命令。不要下载整个近20亿词的 PG-19 train；本包不需要它。

PG-19 是独立来源的书籍域测试，不意味着已经证明它与 Wikipedia 没有引用、片段或近似重合。本包不进行跨 WikiText/PG-19 的片段去重，不以它声称“严格零污染外部测试”。

## 3. 分词、打包和数值解释

使用固定依赖 `tiktoken==0.9.0` 的 `gpt2` 编码，50257 个 token、EOS 为50256。两份 OpenAI tokenizer 原始文件 SHA-256 固定在程序中，自动下载与离线文件均验证。原文中的特殊 token 字面串通过 `encode_ordinary` 当普通文本编码，只有程序插入的 EOS 才是文档边界。

WikiText 数据表的一行通常是段落/标题，不是一篇文档。程序依据上述孤立一级标题规则重建文档单元，保留其原始文本内容，**每个文档单元结束才插入 EOS**。PG-19 每本书一个文档。token 数预算按固定来源顺序截取；最终文档可能被截断，账本明确标记。训练使用可跨 EOS 的连续窗口，始终满足 `y[t]=x[t+1]`；不混合 canonical split。

输出为小端 `uint16` 的 `.bin`（50256不溢出）和 `manifest.json`；manifest 记录源版本、文件校验和、分词器、token 数、文档账本哈希、去重项、依赖版本。随机批次使用独立 NumPy Generator；checkpoint 保存其状态，恢复后精确重放下一批次。

本包报告的是 **GPT-2 BPE token NLL（自然对数）及其指数 BPE perplexity**。WikiText 旧论文通常采用另一套词级分词，PG-19 官方也提出按原始词数归一化的指标。**不能将本包 BPE perplexity 直接与这些原始表格的 word-level perplexity 比大小。** 跨方法比较必须使用完全相同的数据 manifest 与 tokenizer。

## 4. 本地验证和合成数据

`--synthetic` 创建小规模整数 token，仅用于安装、梯度、checkpoint 和运行流程检查；manifest 明确写 `synthetic=true` 和 `paper_eligible=false`。它不能进入论文主结果，也不能代替真实语料实验。

```bash
python -m qknlab.data prepare --synthetic --out data/smoke \
  --max-train-tokens 8192 --max-eval-tokens 2048 --synthetic-vocab-size 128
```

本交付的真实语料 smoke 使用4096/1024/1024个 WikiText token，目的仅为验证官方文件、处理与模型输入链条。完整 GPU 实验必须依据冻结的正式协议准备足够训练数据。`validation/` 中的审计 JSON 是校验记录，不是模型性能证据。
