# FEMBA C0 无锚点 DoG / MLP 下游对照

协议 `femba_c0_head_comparison_v1` 固定 seed=2001；VRQ、城市巡航；状态和严重度独立训练；A3 预训练冻结、A4 全参数微调。两个下游、两个任务、两组、两个数据集、五折共 **80 个新实验**，配对 **40 个已有 C0 线性基线**。实现、smoke 和正式训练完成是不同状态，分别以测试日志、smoke_summary.json 和最终 aggregate_report.json 为准。

## 模型

共享原生四层双向 FEMBA，输入 `[30,1280]`，tokens `[80,525]`。状态单窗取 token mean；严重度固定十一窗先各自 token mean、再对十一窗平均，**最后才进入非线性映射**。城市巡航仍使用路径级标签。

|下游|结构|参数量|
|---|---|---:|
|Linear（已有）|Linear(525,1)|526|
|DoG|LN525 → FractionalDoGPolynomialKAN(525,160,degree=2) → LN160 → Linear(160,1)|170749|
|MLP|LN525 → Linear(525,246) → SiLU → Linear(246,160) → LN160 → Linear(160,1)|170447|

DoG 和 MLP 参数量差约0.18%。全部 LN、映射和输出参数置于 model.head。映射随机流 seed+101，输出 seed+211；不同映射的最终线性层初始化一致，A3/A4 完整下游初始化配对一致。官方 checkpoint 的 SHA-256 和全部83个encoder张量严格核验。encoder 初始化与旧 C0 相同。

无参考锚点输入、中心减法、方向校准、KMeans、Dropout、辅助损失或阈值搜索。沿用 C0 原目标集合，状态排除既有U3–U6；EA仍由被试全部无标签窗口离线估计。这不是严格归纳式 EEG-only 协议。

## 训练和比较

AdamW：head=1e-3，A4 encoder=1e-5，weight decay=1e-2，clip=1；状态batch/microbatch=32/4，严重度=4/1。CUDA bfloat16，pooling、基函数和BCE为float32。只在当前训练分区计算pos_weight。最大60epoch，固定阈值0.5，无调参。

首次更新即验证；后续每ceil(N/4)步及epoch末验证。未加权source-val BCE最小者入选，平局取较早步；10N步无改善早停。选择后从同一原始初始化在全部source上重训 `ceil(best_step*N_source/N_select)` 步，最后评估outer-test。恢复回到完整epoch边界，同时恢复optimizer、RNG、历史、最佳模型和梯度审计。

线性基线只读核验全部40折：artifact/checkpoint哈希、划分、样本、标签、优化参数、原始encoder及shuffle。不同下游允许已声明的结构与实现版本差异，不要求525→1与160→1线性层同权重。DoG−Linear、MLP−Linear、DoG−MLP及各下游A4−A3均完整报告。BACC为主要描述指标，同时报告ACC/AUROC/BCE、五折合并结果、折间均值/样本标准差、类别比例、多数类ACC和两数据集等权宏平均。不按表现删除负结果或选择seed。

## 运行

在本仓库的WSL原生CUDA环境执行，默认配置来自configs/paths.local.json：

```bash
python scripts/run_femba_c0_head_comparison.py --dry-run
python scripts/run_femba_c0_head_comparison.py --stage preflight
python scripts/run_femba_c0_head_comparison.py --smoke
python -m pytest -q
python scripts/verify_reproduction.py
python scripts/run_femba_c0_head_comparison.py --stage all
# 中断后使用相同协议、实现和输出路径恢复
python scripts/run_femba_c0_head_comparison.py --stage all --resume
```

smoke固定fold_1共16项，选择和refit均至少两次更新，执行完整source-val推理和重载核验，不评分outer-test。`train`只执行选择和refit；`evaluate`仅重载最终模型；`summarize`必须有全部80个新结果及40个核验后的线性结果。任何子集或smoke不能标记完整。

输出固定在D盘，本机默认 `D:\桌面\GitHub\v2_github_reproduction_55657888\outputs\femba_c0_head_comparison_v1`：

```text
full|smoke/seed_2001/{head}/{task}/{variant}/{dataset}/fold_N/
  selection/  # 初始化、选步曲线与最佳/最后验证指标
  refit/      # 最终模型、历史和重训审计
  evaluation/ # 正式实验outer-test逐样本预测
  report.json # 正式单折完成及全部artifact哈希
full/seed_2001/shared_encoders/ # 无损共享冻结encoder
full/seed_2001/summary/         # CSV、JSON与RESULTS.md
launches/                     # 进程状态和各阶段完整日志
audit/baseline_source_snapshot.json # 修改前与原C0一致的源码指纹
```

默认拒绝覆盖。冻结encoder无损共享、完整下游独立保存；必须连同shared_encoders一起移动。仅在本次单折验证通过后删除本次冗余中间权重，保留最终模型、全部文本曲线及审计。旧结果不在清理范围。D盘剩余不足3GB时在下一折前停止。

## 结论边界

不设提升门槛、不保证DoG优于MLP。只有预训练A3/A4，不能证明预训练相对随机的收益。DoG对MLP比较整个映射，不能拆分分数阶和DoG各自贡献。单seed、固定预算、历史测试结果启发、transductive EA和缺失完整预训练样本manifest的限制必须随结果保留。

## 本机实现验收记录（2026-09-10）

- 16/16 原生 CUDA smoke 通过，覆盖两个数据集、两个任务、两种下游及 A3/A4；未评分 outer-test。
- 完整测试 120 项通过，包括池化、初始化配对、冻结/微调梯度、完整模型重载、阶段隔离、恢复回退、缺折/混组拒绝及负结果汇总。
- 原 v27/MLP 参考结果重新核验通过，全部指标差异为 0.0 个百分点。
- 40 个原 C0 线性基线的模型、预测、划分和协议完整性核验通过。
- 正式80项训练已在验收通过后启动。最终完成状态以 `full/seed_2001/summary/aggregate_report.json` 为准，以上实现验收不代表性能提升或正式训练已全部完成。
