# Entropy QK-Norm：先读这份运行说明

日期：2026-09-24。完整代码在 `Entropy_QKN_Colab_Package_2026-09-24.zip`。

## 现在怎样启动

1. 解压，把 `entropy_qkn_colab/` 内的文件放到 GitHub 仓库根目录。应直接看到 `qknlab/`、`configs/`、`notebooks/` 和 `requirements.txt`。
2. 在 Colab 打开 `notebooks/Entropy_QKN_Colab.ipynb`，选择 GPU 运行时。
3. 使用 GitHub 路径时设置 `SOURCE_MODE="github"` 并填写 `REPO_URL`。如暂不公开项目，可用默认 `SOURCE_MODE="upload_zip"` 直接上传完整实验 ZIP，无需公开仓库或把凭据写进 notebook。
4. 顺序执行环境检查、自动测试、smoke 和官方数据准备。
5. 首次可以把 `PILOT_RUN_SEEDS=[11]`、`PILOT_MAX_STEPS=50`，先测显存和速度。此时尚未完成 pilot，先跳过完整统计单元。
6. 正常后设置 `PILOT_RUN_SEEDS=[11,22,33]`、`PILOT_MAX_STEPS=None`，同目录续跑完整 pilot，再执行统计。不要修改已经开始运行的模型、数据或训练配置。
7. 检查真实步长效应、重复执行误差、NLL 和裁剪后，再冻结确认协议，运行六个新种子。notebook 中确认、消融、测试集评估分别有明确开关。

重连 Colab 后重新执行准备单元，使用相同 `RUN_TAG`；源码版本、数据缓存与训练状态会被核对。中断后重复执行最终评估单元时，只有身份和协议一致的完整输出才会跳过。

## 数据从哪里来

- **WikiText-103**：[Salesforce 官方固定版本](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1)。Transformer-XL 等语言模型论文使用的基准；本包使用 GPT-2 BPE、去除一篇与测试集完全重复的训练文章，再选择固定训练前缀。
- **PG-19**：[Google DeepMind 官方仓库](https://github.com/google-deepmind/pg19)。Compressive Transformer 提出的书籍语言建模基准；本包只下载官方 test，用于冻结后的域外评估。

默认自动下载，不需要先手动下载整个语料库。WikiText 原始文件约315MB；PG-19 只需100本测试书，无需下载庞大的训练集。网络不通时，包内 `docs/DATA.md` 提供四个 WikiText 文件、两个分词文件的直接链接、PG-19 对象清单及离线导入命令。

## 核验边界

软件已进行 CPU 自动测试、真实来源文件下载与哈希验证、小规模真实文本训练/分叉/汇总测试，详细记录在 `validation/VALIDATION_REPORT.md`。**这不等于已经完成 GPU 长程实验。** 三项 CUDA 专用测试会在你的 Colab GPU 上执行，通过后才继续训练。

这次审计发现并修正了官方 train/test 的整篇重复文章，以及文本中数学公式可能被误识别为文章标题的处理问题。原始来源、修正规则、文章清单与排除记录均可追踪。没有声称已排除所有近似重复。

## 资源与回传

主模型17,653,248参数。pilot 为3个种子×2臂×500步；确认实验为6个新种子×2臂×2500步。默认顺序执行，先测实际速度再安排总预算。两阶段检查点全部保留合计可能超过免费 Drive 的15GB可用空间；确认前备份 pilot 检查点或准备额外存储。

完成后发回：`summary_*`、`diagnostics/`、每条运行的 `run_manifest.json`、`metrics.jsonl`、`summary.json`，以及冻结协议、环境记录、数据 `manifest.json`。保留关键 `.pt` 检查点供补实验。

写作将按 Entropy 的信息论主线组织：注意力 KL、Fisher 温度/形状分解、有限步余项及真实训练影响。不得预设 detach 有益；不能把自检图或随机数据测试当作论文结果。
