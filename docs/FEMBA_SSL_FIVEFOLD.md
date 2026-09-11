# FEMBA 无锚点与四锚点完整五折

运行次序固定为 C0 无锚点 120 项完成并核验汇总后，再运行 C1 四锚点 120 项。每条件为 A1–A4 × state/severity × monifeixing/vrq/city × 固定五折，seed=2001。不会把旧 pilot 的验证指标作为 outer-test 结果，也不直接沿用旧 pilot checkpoint；每折重新执行选步与 source refit。

## 模型与预算

沿用已验证的小试结构：原生四层双向 FEMBA、525 维 token mean、526 参数线性头。严重度均匀十一窗；C1 减去同一被试参考段 U3–U6 的四窗特征均值，任务与锚点共享 encoder，可训练组两条分支都保留梯度。冻结 encoder 始终 eval。既有逐窗归一化、逐被试无标签 EA、状态锚点窗口排除与 city 路径标签不变。

AdamW head LR=1e-3、encoder LR=1e-5、weight decay=1e-2、clip=1.0，CUDA bfloat16，pooling/BCE 为 float32。有效 batch 为状态32/严重度4，microbatch 为4/1，训练权重仅按本阶段训练目标样本计算。

选步采用第一步起的 source-val 未加权 BCE，最小者入选、平局取较早步骤；最多60 epoch、无改善10个epoch等价步数早停。随后重置到同一原始 encoder/head 初始化，训练全部 source 被试，并重算 source 的 pos_weight。

设选择阶段最佳步骤为 b，每 epoch 步数为 N_select，全部 source 每 epoch 步数为 N_source：

```text
refit_steps = ceil(b * N_source / N_select)
```

保留选中 epoch 的比例，不简单向上取整为完整 epoch。refit 不再选模型、不查看 outer-test 指标，完成锁定步数后才用该模型评分 outer-test，阈值固定0.5。选择阶段和 refit 都保存完整 epoch 边界以恢复未完成作业。

## 全部结果在 D 盘

本机目录：`D:\桌面\GitHub\v2_github_reproduction_55657888\outputs\femba_ssl_fivefold_v3`。

```text
full/seed_2001/
  C0|C1/{task}/{variant}/{dataset}/fold_1..fold_5/
    selection/     # 验证选步曲线、初始化与报告
    refit/         # final.pt、报告与实际重训曲线
    evaluation/    # outer-test predictions.csv 与报告
    report.json    # 完整单折完成标识及文件哈希
    retention.json
  shared_encoders/
  summary/C0/      # 先交付无锚点完整结果
  summary/C1/
  summary/paired_comparison.json
smoke/seed_2001/
launches/          # 本机进程状态与完整日志
```

冻结组编码器按完整张量哈希共享存储，每折 final.pt 独立保存线性头及相对引用；可训练组 final.pt 独立包含全部 encoder/head。共享是无损存储，不改变模型精度或计算。重载时通过 `training.ssl_fivefold.restore_final(checkpoint, experiment_root, identity, device)` 解析并严格校验完整模型。移动实验时须同时保留 `shared_encoders/`，不能只复制某个冻结组的 head 文件。

预计240项最终权重约14 GB，另需运行中临时空间。每折最终模型与预测验证通过后，清理**本次新建**的 selection best/last/boundary 和 refit boundary 冗余权重，保留全部历史、初始化、哈希与报告。旧实验文件不在清理范围。剩余空间不足2 GB时，在启动下一折前安全停止，不删除其他文件。

## 命令与恢复

在已验证的 WSL/PyTorch 环境，从本仓库根目录执行：

```bash
python scripts/run_femba_ssl_fivefold.py --dry-run
python scripts/run_femba_ssl_fivefold.py --stage preflight
python scripts/run_femba_ssl_fivefold.py --stage all --conditions C0 C1
# 若被中断，按原队列继续；完成的折只核验并保留。
python scripts/run_femba_ssl_fivefold.py --stage all --conditions C0 C1 --resume
```

默认根目录始终位于本仓库 outputs 下；本机即 D 盘，不向 C 盘写入新结果。正式 C1 开始前必须存在同协议、同数据绑定的 C0 完整120项汇总。恢复还会核验实现源码指纹，避免混入修改后的代码。默认拒绝覆盖非空作业目录。

小规模新流程 GPU 验证命令：

```bash
python scripts/run_femba_ssl_fivefold.py --smoke --datasets monifeixing --folds 2 --variants A1 A4
```

该命令检查两个任务、两条件、冻结/可训练分支的选择→原初始化重训→重载，共8项；不调用 outer-test 评分，也不能算正式五折结果。

## 汇总与结论边界

每条件保存120折指标表、按数据集五折合并的 ACC/BACC/AUROC、折间均值和样本标准差，以及三数据集等权宏平均。检查 outer-test 被试跨折不重复，四组计分样本、头初始化和家族 encoder 初始化一致。某折仅有一个类别时，其 AUROC/BACC 及对应五折均值/标准差记为空；五折合并结果仍按全部预测计算。

两条件完成后核验计分集合一致，输出 A3−A1、A4−A2、A4−A3、A2−A1、各组锚点收益与预训练×锚点交互；不选择性省略负结果。仍保留单 seed、已知参考段、离线 transductive EA、预训练 monifeixing 样本重叠未排除，以及受先前结果启发的探索性设计等限制。

实现验收：全套100项测试通过；8项原生 CUDA 新流程 smoke 通过；原 v27 与 MLP 参考结果核验差异均为0.0个百分点。实现通过不表示240项正式训练已经完成，实时状态以 launches 和各条件汇总为准。
