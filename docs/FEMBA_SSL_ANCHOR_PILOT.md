# FEMBA 预训练 × 四锚点验证小试

本入口只训练 source-train、验证 source-val，不进行 source refit 或 outer-test 评分。A1–A4 分别运行 C0（原始 pooled 特征）和 C1（减去个人 U3–U6 四锚点的平均 pooled 特征）。所有模型均为原生四层双向 FEMBA + Linear(525,1)，任务分开训练，头为 526 个参数。没有 KAN、尺度除法、上下文扩展或聚类。

默认 seed=2001，三数据集、两个任务、八条件，仅 fold_1，共 48 个拟合。真实效果以完成后的 pilot 汇总为准，smoke 通过不代表准确率提高。

## 数据和梯度

沿用既有通道、切窗、标签和身份划分，保留逐窗归一化与逐被试无标签 EA。EA 是离线 transductive 预处理。C1 的参考段由协议指定，取均匀八位置中的 U3–U6；两种条件均从状态目标样本排除这些窗口。严重度仍为固定十一任务窗均值，city 为路径级标签。

任务和锚点共用 encoder。冻结组始终 eval/no_grad；可训练组的两条分支均反向传播，每次更新后重新编码，不 detach 或缓存中心。C0 不编码锚点。所有组共享头初始化和目标采样顺序，随机/预训练家族的 encoder 分别配对，预训练严格核验官方 SHA-256 和 83 个张量。

有效 batch：状态 32、严重度 4；microbatch 分别为 4、1，按当前有效 batch 的实际目标样本数归一化 BCE。锚点按 microbatch 内被试去重，不计入类别权重。AdamW head LR=1e-3，encoder LR=1e-5，weight decay=1e-2，clip=1.0，CUDA bfloat16，pooling 和 BCE 使用 float32。

## 选择与恢复

最多 60 epoch；每 epoch N 步，第一步、每 ceil(N/4) 个全局步骤以及 epoch 末验证。最低 source-val 未加权 BCE 的真实 checkpoint 入选，平局取较早步骤；从最佳步骤起 10N 步无改善时早停。不存在二十 epoch 的入选门槛。

保存 best.pt、last.pt 与完整 epoch 边界 boundary.pt；boundary 内包含 optimizer、随机数、当时的最佳模型、历史、梯度审计和累计统计。恢复从最近完整 epoch 重做未完成部分，同时回退历史和最佳模型。被撤销且无边界最佳模型可替代的 partial best 会被改名保留，不能用于评分。已完成结果只在核验协议、样本和文件哈希后保留。

## 命令

在已核验的 WSL 环境使用 `/opt/miniconda3/envs/pytorch/bin/python`：

```bash
python scripts/run_femba_ssl_anchor_pilot.py --dry-run
python scripts/run_femba_ssl_anchor_pilot.py --stage preflight
python scripts/run_femba_ssl_anchor_pilot.py --stage all --smoke
python scripts/run_femba_ssl_anchor_pilot.py --stage all
python scripts/run_femba_ssl_anchor_pilot.py --stage all --resume
```

支持 `--conditions`、`--variants`、`--tasks`、`--datasets`、`--folds 1`、`--seed`、`--config`、`--output-root`。`evaluate` 只重载最佳模型并验证 source-val，`summarize` 必须具有全部 48 个非 smoke 结果。旧入口、v1 协议和 v27 默认行为保持原样。

默认输出为 `outputs/femba_ssl_anchor_v2/{smoke,pilot}/seed_2001/{task}/{condition}/{variant}/{dataset}/fold_1/`。每个 job 的 training 子目录保存模型、初始化、history 和 report；evaluation 子目录保存逐样本预测与重载评估报告。

本机 D 盘剩余空间不足以稳妥同时容纳 smoke 与完整 pilot 权重，因此正式 pilot 使用：

```bash
python scripts/run_femba_ssl_anchor_pilot.py --stage all \
  --output-root /mnt/c/Users/Administrator/femba_ssl_anchor_v2
```

即 Windows `C:\Users\Administrator\femba_ssl_anchor_v2\pilot\seed_2001`。项目目录的 `outputs/femba_ssl_anchor_v2/launches/` 保存本机启动记录、日志及进程状态。中断恢复时必须指定同一个 output-root。

## 汇总与判断

输出每条件 ACC/BACC/AUROC、验证 BCE、类别比例、多数类 ACC、最佳步骤、实际更新次数、训练时间和显存。三数据集按等权宏平均，两个任务分别报告。主要指标为 BACC，ACC/AUROC 同时完整呈现。

- 锚点自身收益：Ai(C1)−Ai(C0)。
- 冻结预训练收益：A3−A1；可训练预训练收益：A4−A2。
- 冻结交互：(A3−A1)C1−(A3−A1)C0。
- 可训练交互：(A4−A2)C1−(A4−A2)C0。
- 另报各条件 A4−A3、A2−A1。

各差值为百分点。只有预训练组自身改善且交互为正，才记录为同时改善与放大的描述性信号；还需查看 C1 下预训练是否实际优于随机。不能因随机组下降就声称预训练模型准确率提高，不能只报告有利数据集。

这是使用验证集选模的单 seed、单折探索性小试；不提供独立测试结论或统计显著性宣称。静息参考段需在应用时可获取，EA 的 transductive 假设与 monifeixing 预训练重叠未排除的限制必须披露。完整有/无锚点五折矩阵将为 240 个 fold，本入口不启动该矩阵。

## 本机验收（2026-09-07）

- `Ubuntu-22.04-Bio`，PyTorch 2.11.0+cu128、mamba-ssm 2.3.1、RTX 5060；194 项协议与资产预检通过。
- 全套测试 `86 passed`，包含新增 17 项测试；覆盖 48 条件合成数据端到端汇总、双分支梯度、梯度累积、早期选步、中断回退、checkpoint 提交中断、样本隔离、混组和文件篡改拒绝。
- 真实 GPU smoke 48/48，通过各两次 optimizer 更新、完整验证推理和保存重载检查。24 个冻结 encoder 参数/buffer 均未变化；24 个可训练 encoder 的 83/83 参数张量均收到非零梯度。
- 12 个 C1 可训练项目均确认目标与锚点分支有梯度；48 项重载 logit 最大绝对误差为 0，初始化、锚点、目标样本和 shuffle 配对审计通过。
- 原 v27 和 MLP 参考结果核验通过，所有参考指标最大差异为 0.0 个百分点。

证据：`outputs/femba_ssl_anchor_v2/smoke/seed_2001/validation/integrity_audit.json`、`outputs/femba_ssl_anchor_v2/validation/full_tests.xml`、`outputs/femba_ssl_anchor_v2/validation/v27_reference.json`。以上是实现验收；正式 pilot 另行运行，不将 smoke 指标当成小试结论。
