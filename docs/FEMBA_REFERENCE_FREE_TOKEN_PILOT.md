# FEMBA 无静息参考逐 token 映射小试（允许 EA）

独立协议 `femba_reference_free_token_pilot_v1`，按用户修订保留逐被试无标签 EA。
**无静息参考校准不等于严格 inductive**：EA 使用每名被试协议内全部无标签窗口，包含其中的静息记录；
不根据静息身份挑选参考、不读取状态或严重度标签拟合 EA、不建立锚点中心或尺度。
不同窗口可以通过被试 EA 相互影响，这是明确允许的 transductive 处理。

只使用 VRQ、城市巡航 fold_1 的 source-train/source-val；seed=2001，预训练 encoder 冻结并保持 eval。
不加载 outer-test EEG，不调用 outer-test 评分，不做 source 重训。原 v27、C0 入口和产物不变。
MAT 和协议哈希、正式 checkpoint SHA-256 与 83 个张量严格校验。
数据标签延续旧协议（包括 VRQ 的阶段定义）；这些标签不是独立逐窗口症状测量。

## 模型和 52 项矩阵

状态输入 `[B,30,1280]`，严重度为目标 task-session/城市路径的固定均匀十一窗 `[B,11,30,1280]`。
FEMBA 输出每窗 `[80,525]`，80 对应有序时间位置。所有状态窗口均纳入，不再排除 U3–U6。
窗口内去均值、标准化及无标签 EA 复用现有数学定义，新数据以 float32 保存。
中性 `segment_NN` 仅用于追踪样本；阶段名称、被试 ID 和标签不进入预测接口。

映射 φ：LN(525) → 映射(525,160) → LN(160)。三个映射为原生 Fractional-DoG PolynomialKAN、
Linear(525,246)→SiLU→Linear(246,160) 的 MLP、degree=2 普通 PolynomialKAN。

| 表示 | 状态 | 严重度 |
|---|---|---|
| R0 | token mean→φ→Linear160 | token mean→十一窗mean→φ→Linear160 |
| R1 | 逐token φ→mean→Linear160 | 逐token φ→token mean→十一窗mean→Linear160 |
| R2 | 逐token φ→mean/std/相邻绝对差均值→Linear480 | 每窗R2统计480维→十一窗mean/std/后3窗减前3窗→Linear1440 |

std 为总体标准差 `sqrt(mean((x-mean)^2)+1e-6)`。时间变化不跨越不连续窗边界。
前后差仅描述当前目标任务，不把前段定义为正常参考。
另有纯 Linear525 基线，以及 R2 下分数阶固定1、关闭DoG、二者同时固定三组消融。
13配置 × 2任务 × 2数据集 = 52项；smoke另52项，不能混入研究结果。
禁用分支保持其余参数化，剔除其不可训练参数；与普通 PolynomialKAN 不等同。

所有下游 FP32，encoder CUDA BF16；映射seed+101、输出seed+211，采样独立随机数流。
AdamW lr=1e-3、wd=1e-2、clip=1，状态有效batch/micro=32/4，严重度4/1。
训练加权BCE的pos_weight只按source-train计算；source-val选未加权BCE最小。
第一步、每ceil(N/4)步及epoch末验证，平局保留较早；60轮上限，连续10N步未改善早停；阈值0.5。

## 运行及恢复

在原 WSL/CUDA 环境、仓库根目录执行：

```bash
python scripts/run_femba_reference_free_token_pilot.py --dry-run
python scripts/run_femba_reference_free_token_pilot.py --stage preflight
python scripts/run_femba_reference_free_token_workflow.py
```

workflow依次执行预检、52项GPU smoke、完整测试、原v27参考文件核验、52项小试并停止。
每项smoke至少两次更新，完整验证、GPU在线编码对照、冻结审计和保存重载。
直接入口也支持 `--stage train|evaluate|all|summarize --configs ... --tasks ... --datasets ... --smoke --resume`。
evaluate仅重载最佳模型验证source-val。复跑默认拒绝覆盖；恢复核验代码、协议、数据、缓存和模型身份。
恢复回到完整epoch边界并回退之后历史和最佳模型。部分矩阵只能标记partial。

产物均位于 D 盘仓库 `outputs/femba_reference_free_token_pilot_v1`，含：

- `shared/encoder.pt`：完整冻结encoder及来源校验，所有head检查点共同引用。
- `cache/`：新协议源分区tokens，FP32 memmap；哈希绑定数据/预处理/encoder，绝不复用旧缓存。
- `smoke|pilot/seed_2001/{config}/{task}/{dataset}/fold_1/`：完整下游best/last、epoch恢复点、初始化、曲线、验证预测、梯度和基函数诊断。
- `summary/`：每数据集指标、两数据集宏平均、配对差值、解释边界；`launches/`记录执行状态与日志。

缓存消除了重复encoder计算，不宣称各配置计算成本相同。记录编码时间、实际窗口数、缓存读取量、下游更新时间及峰值显存。
冻结encoder与完整head可组装成 `FrozenTokenProbe`。空间预检保留5GiB余量；不删除旧结果、不转存C盘。

## 解释边界

主要指标BACC，完整保留ACC、AUROC、BCE、类别比例、样本与被试数。
比较R1−R0、R2−R1、同结构DoG−MLP/普通PolynomialKAN及成分消融。
最佳模型直接关闭分支是推理诊断，与重新训练的消融分开报告。
只有同结构比较和成分消融都支持时，才讨论DoG的独立收益。
本次仅单seed小验证集，选模型与报告使用同一验证集，不构成独立测试或统计显著证据。
预训练样本manifest缺失、上游MAT原始处理来源不完整及受此前测试结果启发的限制均保留。
