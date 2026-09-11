# FEMBA 自监督预训练有效性消融

这套实验独立于 v27：它只比较 FEMBA 的初始化与 encoder 是否训练，不训练 KAN 或优化 v27。状态分类和高低眩晕分类分别训练，每次仅有一个 `Linear(525,1,bias=True)`。

## 固定的 2×2 模型

| 组别 | 初始化 | encoder | 线性头 |
|---|---|---|---|
| A1 Random-Frozen | 原生随机初始化 | 冻结，始终 eval | 训练 |
| A2 Random-Scratch | 与 A1 相同 | 全参数监督训练 | 训练 |
| A3 Pretrained-Frozen | 正式 checkpoint | 冻结，始终 eval | 训练 |
| A4 Pretrained-Finetune | 与 A3 相同 | 全参数 fine-tuning | 训练 |

四组都是同一个 `TemporalEncoder`：4 层双向 Mamba、原生 `mamba-ssm`，28,540,155 个 encoder 参数。`[30,1280] → [80,525] → token mean → Linear(525,1)`；线性头参数量为 526。encoder、head、shuffle 分别使用独立随机数流，四组共享头初始化及每个 epoch 的样本顺序。Random 两组不读取 checkpoint，也不要求配置中存在 `pretrain_checkpoint`。

Pretrained 两组必须加载 protocol manifest 指定的文件，SHA-256 为 `0e2ab9109d87a32c6b25f0c307fb8b1102ef5e0e83e86b9b07f7dee166daaa27`，完整加载全部 83 个 encoder 张量。正式评估从自己的完整模型 checkpoint 恢复，不重新加载预训练权重。

## 输入、划分与训练

完整设置以 [femba_ssl_ablation.json](../reproducibility/protocols/femba_ssl_ablation.json) 为准，执行时写入 checkpoint 并保存协议、数据与初始化哈希。原 v27 的协议文件和参考结果保持不变。

- 复用三数据集 MAT 版本、通道、标签、5 秒窗口、split seed=42 的身份隔离五折，以及已有 source-train/source-val 划分。
- 保留原来的逐窗归一化和逐被试无标签 EA。EA 使用该被试所有无标签窗口，是离线 transductive 预处理。
- 状态分类输入单个 5 秒窗口。训练、验证和测试均排除原协议 U3–U6 的四个索引，仅保持计分集合一致；不使用其信号做 anchor 校准。
- 严重度分类对原协议每个 subject/task-session 均匀选取固定 11 窗，每窗取 token mean，再对 11 窗求均值，最后接同一个线性头。city 的计分单位仍为路径，而非将 11 个窗口当作 11 个独立标签。
- 不使用三窗上下文、jitter、token dynamics、额外 LayerNorm/Dropout、辅助损失、KMeans、预测平滑或阈值校准。
- 四组统一在线读取原始窗口并编码，不使用 v27 pooled cache。冻结组仅 encoder 前向处于 `no_grad()`；可训练组保留全部计算图。

训练默认 seed=2001。AdamW 的 head LR=`1e-3`，A2/A4 encoder LR=`1e-4`，weight decay=`1e-2`，梯度裁剪=`1.0`。使用 CUDA bfloat16 autocast，pooling 与 BCE 在 float32 中计算；无需 fp16 GradScaler。状态 batch=32，严重度 batch=4，完整 epoch 遍历全部训练样本一次，最后不足整 batch 的样本仍参与训练。

训练使用 BCEWithLogitsLoss，`pos_weight=当前训练分区负样本数/正样本数`，没有 label smoothing。验证使用未加权 BCE，避免把训练分区的类别权重当成验证分布。最多训练 60 epoch，从第 20 epoch 起按验证 BCE 最小选轮，平局选较早 epoch，patience=15。

选轮结束后重新构造该组初始 encoder 和线性头，校验初始哈希完全一致，再用全部 source 被试训练选定轮数；此时只按 source 重算 pos_weight。outer-test 不参与采样、选轮、权重计算或校准。评分函数先完成所有预测，再附加标签计算指标。最终阈值固定为 sigmoid ≥ 0.5。

## 运行命令

在本仓库根目录、已通过预检的 Linux/WSL Python 环境运行。此机器可用 `/opt/miniconda3/envs/pytorch/bin/python`；脚本会自行加入 `src`，不需要额外设置 PYTHONPATH。默认配置为 `configs/paths.local.json`。

先查看完整 120 个 fold 任务，不读取 EEG、不初始化 GPU、也不写输出：

```bash
python scripts/run_femba_ssl_ablation.py --dry-run
```

只预检环境、固定协议、数据和必要的权重：

```bash
python scripts/run_femba_ssl_ablation.py --stage preflight
```

运行 24 个 smoke 检查（三数据集 × 两任务 × 四组，仅 fold_1）：

```bash
python scripts/run_femba_ssl_ablation.py --stage all --smoke --folds 1
```

Smoke 每组至少做两个 optimizer step、完整 source-val 推理和 checkpoint 保存重载检查。如果一个 epoch 不足两步，则进入下一 epoch；不会启动 20–60 epoch 选轮，不做 outer-test 评估。通过只代表训练链路可运行，不能据此判断 representation 是否有效。

完整训练、测试和汇总（本次实现交付**不执行**此命令）：

```bash
python scripts/run_femba_ssl_ablation.py --stage all
```

分开执行单组或阶段：

```bash
python scripts/run_femba_ssl_ablation.py --stage train --variants A2 --tasks state --datasets monifeixing --folds 1
python scripts/run_femba_ssl_ablation.py --stage evaluate --variants A2 --tasks state --datasets monifeixing --folds 1
# 完成四组、三数据集、五折的预测后执行：
python scripts/run_femba_ssl_ablation.py --stage summarize
```

`--config`、`--output-root` 可指定其他配置和输出目录；`--seed` 可显式设置其他 seed，每个 seed 单独存放、单独汇总。`--tasks state` 可仅汇总状态任务，但仍要求该任务的四组、三数据集和全部五折齐全。原 v27 CLI 的默认行为不变。

`all` 会在完整默认 120 个任务完成后自动汇总；对子集执行 `all` 只训练和评估所选任务。`summarize` 不启动训练，但当前仍执行本地环境/数据预检。默认拒绝覆盖对应的非空输出目录；再次 smoke 验收请使用新的 `--output-root`。

用户中断后，可显式恢复完整矩阵：

```bash
python scripts/run_femba_ssl_ablation.py --stage all --resume
```

`--resume` 校验已完成 fold 的协议、checkpoint/预测哈希及重算指标后保留原结果；中断的 fold 先整体归档到 `outputs/femba_ssl_ablation/interrupted_attempts/`，再按原 seed 从头重做该折。当前不保存 epoch 级优化器状态，因此不是从被打断的 optimizer step 接续。默认不隐式跳过或覆盖任何旧结果。

## 输出与审计

```text
outputs/femba_ssl_ablation/
  seed_2001/{state,severity}/{A1,A2,A3,A4}/{dataset}/fold_{1..5}/
    training/
      initialization.json
      selection_history.json
      refit_history.json
      checkpoint.pt
      report.json
    evaluation/
      predictions.csv
      report.json
  seed_2001/summary/
    metrics.csv
    aggregate_report.json
  smoke/seed_2001/
    {task}/{variant}/{dataset}/fold_1/training/
      initialization.json
      refit_history.json
      checkpoint.pt
      smoke_report.json
    validation/smoke_summary.json
```

每个 checkpoint 包含完整 encoder/head、四组标识、任务、数据集、fold、seed、超参数、source/val/test 身份、样本集合哈希、预训练来源、初始化/最终权重哈希、encoder/head 参数变化与梯度审计。A1/A3 必须没有 encoder 梯度，所有 encoder 参数和 buffer 保持不变；A2/A4 必须全部 encoder 参数进入优化器并收到非零梯度。训练结束后检查 encoder 实际变化和线性头更新，并要求保存重载后的校验 logits 完全一致。

Smoke 与正式 checkpoint 带不同标记且目录隔离。评估入口和汇总不能把 smoke、缺折、混 seed、混协议或混组文件当成完整结果。汇总核验逐样本集合、四组头初始化、成对 encoder 初始化、采样顺序、checkpoint/CSV 哈希，并从 CSV 重算指标。

单折和五折合并指标包括 ACC/BACC/AUROC；JSON 同时保留每折指标、折间均值及样本标准差（ddof=1）和三个数据集的等权宏平均。指标原值为 0–1，四个差值以百分点给出：

| 对比 | 解释 |
|---|---|
| A3−A1 | 冻结 representation 的预训练收益 |
| A4−A2 | 相同端到端训练预算下的预训练初始化收益 |
| A4−A3 | 预训练后 full fine-tuning 相对冻结的收益 |
| A2−A1 | 随机 FEMBA 架构通过监督训练获得的收益 |

折间标准差不是多 seed 方差，也不应将重叠 source 集合的五折当作五个完全独立试验。报告不会预设哪组获胜，不会将探针结果与使用额外模块和不同决策方法的 v27 直接归因为单一组件收益。

## 结论边界

本实验回答的是当前数据、预处理、单 seed 和固定训练预算下的差异，不能宣称各策略已经达到最优。四组均使用 FEMBA，无法单独证明 Mamba 架构优于其他架构。

本地 `run_pretrain_aux_largeblocks.py` 与 `build_aux_pretrain_hdf5.py` 指向 guoshanche + monifeixing 自监督预训练。正式权重与仓库发布 SHA-256 一致，但它是恢复出的 encoder-only 文件，完整预训练样本 manifest 已缺失。因而 monifeixing 的预训练样本/被试重叠风险未排除，跨数据集身份也未独立核实；不能将其直接称作严格未见数据迁移。保留这些限制不会改变这次实验使用的正式 checkpoint。

## 本机实施验收（2026-09-07）

在 `Ubuntu-22.04-Bio`、Python 3.10.20、PyTorch 2.11.0+cu128、mamba-ssm 2.3.1、RTX 5060 上完成：

- 194 项协议及数据资产检查通过，正式 checkpoint SHA-256 一致。
- 完整测试：`67 passed`，包含原有 47 项和新增 20 项测试。选轮、source refit、外层评估和汇总在合成数据上进行了端到端验证。
- 真实数据 smoke：24/24 通过，每组恰好两个 optimizer step，均完成完整 source-val 推理与保存重载检查。
- 12 个冻结组的 encoder 参数与 buffer 变化均为 0，encoder 梯度张量数为 0。
- 12 个可训练组的 83/83 encoder 参数张量均收到非零梯度，encoder 权重 L2 变化范围为 0.650867–0.830129；所有组线性头均更新。
- 24 组保存重载后的校验 logits 最大绝对误差均为 0；六个 dataset/task 组合的四组采样顺序一致，初始化配对一致。
- `scripts/verify_reproduction.py` 通过，原 v27/MLP 参考指标最大差异为 0.0 个百分点。

本地证据保存在 `outputs/femba_ssl_ablation/smoke/seed_2001/validation/smoke_summary.json` 和各组 `training/smoke_report.json`。这些文件及完整模型 checkpoint 不提交到 Git。

**完整 120 个 fold 的消融实验尚未运行，目前没有可用于判断预训练收益的正式结果。**
