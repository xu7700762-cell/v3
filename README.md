# VestibularFusion v3

本仓库保存 FEMBA 在两个前庭不适 EEG 数据集上的无锚点下游实验：城市巡航高低眩晕、VRQ 高低眩晕，以及 VRQ 静息/任务状态二分类。当前结果归档不包含模拟飞行数据集。

这里的“无锚点”有明确含义：模型不读取个人静息参考段，不选择 U3–U6，不计算静息中心或尺度，不用静息标签确定方向，也不执行 KMeans、阈值搜索或测试后校准。所有预测均由 EEG 直接产生一个 logit，决策阈值固定为 `sigmoid(logit) >= 0.5`。

## 审核结论

对当前三个入口、三个锁定协议、数据构造、模型前向、训练循环、检查点和汇总代码逐层核验后，得到以下结论：

- 训练与验证按被试隔离，同一被试不会同时出现在两个分区。
- 五名 outer-test 被试的 EEG 和标签均未载入，也未参与训练、选步或计分。
- 模型前向接口只接收 FEMBA token，不接收被试编号、session 名、静息参考或锚点张量；训练代码在前向之外单独构造标签并计算损失。
- source-validation 推理先计算全部 logit，随后才把标签附加到预测行中，用于 BCE 选步和指标计算。
- 正式预训练 checkpoint 按 SHA-256 校验，并要求 83 个 encoder 张量完整加载；encoder 全程冻结并保持 `eval()`。
- 每个实验保存数据、样本、代码、初始化、检查点和预测哈希；最佳检查点重载后必须逐项复现原预测。
- EA 不读取标签，但会按被试使用其全部协议窗口估计协方差，因此本协议属于离线 transductive 预处理，不能表述为纯 inductive 或在线盲测。
- 当前结果是 source-validation 开发结果。验证标签参与最佳步选择，所以不能把这些数值称为独立 outer-test 性能。

状态二分类中的静息/任务信息只用于定义监督目标 `0/1`。静息窗口与普通任务窗口一样独立送入模型，不作为个人校准参考，也不会与同一被试的其他窗口共同构造锚点特征。

## 数据与预处理

两套数据均使用 30 通道、每窗 1280 点的五秒 EEG。MAT 文件及协议文件在载入前核验 SHA-256。

每个被试独立执行无标签 Euclidean Alignment（EA）。设该被试的全部协议窗口为 `X_n`，代码先在时间轴去均值并按窗口标准化，然后计算平均通道协方差：

```text
R = mean_n(X_n X_n^T / (T - 1))
X_EA = (R + ridge I)^(-1/2) X
```

`R` 的估计不使用状态或严重度标签。由于验证被试自身的无标签窗口也参与该被试的 `R`，EA 被明确记录为 `offline_transductive_subject_EA=true`。

### 城市巡航

- 固定 outer fold 为 `fold_1`；只使用其中 20 名 source 被试开发，5 名 outer-test 被试保持封存。
- 每条巡航路径是一个严重度样本，完整路径切成连续、无重叠五秒窗；少于 11 个完整窗口的路径报错。
- 路径二分类标签和连续 `path_score` 均来自固定审计 manifest。当前完整 manifest 中低/高类别各 77 条路径，低类分数为 0–27，高类分数为 30–100。
- 三个内部验证组各含 5 名被试；每次用其余 15 名 source 被试训练。三个验证组没有覆盖的 5 名 source 被试始终只参与训练。

### VRQ

- 固定 outer fold 为 `fold_1`；18 名 source 被试用于开发，5 名 outer-test 被试保持封存。
- 三个内部验证组互不重叠，每组 6 名被试，并各含 3 名低严重度和 3 名高严重度被试；每次使用其余 12 名 source 被试训练。
- 状态任务以单个五秒窗为样本：协议中的 `rest01/rest02` 为 `0`，其余目标 session 为 `1`。session 名只在数据构造阶段确定监督标签，不进入模型前向。
- 被试级严重度任务从最终任务 session 均匀选取 11 个互不重复窗口，构成一个有序样本。
- 片段级严重度任务使用最终任务 session 的每个完整五秒窗，并把该被试的最终 SSQ 高低标签赋给这些窗口。这是弱监督：窗口共享被试标签，不能把窗口数量当作独立被试数量。

## FEMBA 编码器

三个实验共享同一套正式预训练 FEMBA：

```text
EEG [B,30,1280]
  -> Conv2d PatchEmbed(kernel=2x16, stride=2x16, embed=35)
  -> positional embedding
  -> 4 x [forward Mamba + reverse Mamba + residual + LayerNorm]
  -> tokens [B,80,525]
```

双向分支采用逐元素相加。正式 checkpoint 为 `pretrained_femba_v27.ckpt`，协议记录的 SHA-256 为：

```text
0e2ab9109d87a32c6b25f0c307fb8b1102ef5e0e83e86b9b07f7dee166daaa27
```

加载时要求 83 个 encoder 张量全部匹配，不允许缺失、额外或跳过张量。编码器采用 CUDA bfloat16 前向，输出转为 float32 token 后写入带哈希的只读缓存。下游训练期间 encoder 不再执行训练前向，参数和 buffer 均不更新。

## 两阶段共享基线残差框架

每个 seed、任务和内部划分先训练一次 `Base`，按 source-validation BCE 选出最佳 Base。随后 MLP、PolynomialKAN、FractionalKAN 和 Fractional-DoG 四个分支加载完全相同的 Base 权重并冻结，只训练各自的映射和残差分类器：

```text
final_logit = frozen_base_logit + trainable_residual_logit
```

这种两阶段设计保证所有非线性模块从同一个已训练基线出发。代码会校验 Base tensor hash；残差训练后若 Base 发生任何改变，实验直接失败。

### Base 分支

```text
FEMBA tokens
  -> LayerNorm(525)
  -> Linear(525,64)
  -> task-specific summary [192]
  -> LayerNorm(192)
  -> Linear(192,1)
```

### 参数匹配残差分支

```text
FEMBA tokens
  -> LayerNorm(525)
  -> MLP / PolynomialKAN / FractionalKAN / Fractional-DoG (525 -> 64)
  -> LayerNorm(64)
  -> task-specific summary [192]
  -> LayerNorm(192)
  -> Linear(192,1)
  -> add frozen Base logit
```

MLP 使用 `Linear(525,172) -> SiLU -> Linear(172,64)`。它与 Fractional-DoG 的总参数量近似一致：城市巡航分别为 139,744 和 139,667；VRQ 分别为 138,590 和 138,513，均只相差 77 个参数。

## Fractional-DoG PolynomialKAN

本实验中的 DoG 指高斯一阶导数基，不是 Difference of Gaussians，也不是额外的时域或频域滤波器。它在 525 维 FEMBA 表征的特征坐标上工作。

对 LayerNorm 后的输入 `x`，首先学习逐维尺度并限制坐标：

```text
u = tanh(x * exp(0.5 * tanh(log_scale)))
```

525 维被分为 15 组，每组学习一个位于 `[0.7,1.5]` 的分数阶 `q`：

```text
q = 0.7 + 0.8 * sigmoid(q_logit)
f_q(u) = sign(u) * ((abs(u) + eps)^q - eps^q)
```

随后构造三类基函数：

```text
B1 = f_q(u)
B2 = 2 * f_q(u)^2 - 1
B3_s = -v_s * exp(-0.5 * v_s^2),  s in {0.5,1.0,2.0}
```

其中 `v_s` 包含可学习的组内平移和三个尺度的 softmax 混合。各基函数分别进行 RMS 归一化，再通过可学习 gate 加权、拼接并线性投影到 64 维。对照关系为：

- `poly`：固定一阶和二阶多项式基。
- `fractional`：分数阶一阶/二阶基，不含高斯一阶导数基。
- `dog`：分数阶一阶/二阶基，再加入三尺度高斯一阶导数基。
- `mlp`：参数量匹配的普通两层非线性映射。

## 三个任务的汇聚方式

### VRQ 状态二分类

单个五秒窗产生 `[80,525]` token。Base 和残差分支分别映射 token，然后沿 80 个 token 计算：

```text
token mean + population standard deviation + mean absolute first difference
```

三部分拼接为 192 维，再产生一个窗口级 logit。

### VRQ 被试级高低眩晕

每名被试的最终任务均匀选择 11 个窗口。每窗先对 80 个 token 求均值，再沿 11 个有序窗口计算：

```text
window mean + population standard deviation + least-squares temporal slope
```

三部分拼接为 192 维，产生一个被试级高低严重度 logit。

### VRQ 片段级高低眩晕

模型结构与 VRQ 状态二分类完全相同，只改变监督目标：最终任务的每个五秒窗继承所属被试的高低严重度标签。主要指标按片段计算，同时报告两项被试审计：

- 先计算每名被试的片段准确率，再对被试等权平均。
- 对每名被试的片段概率求均值，再计算被试级 ACC、BACC 和 AUROC。

### 城市巡航路径级高低眩晕

每条路径使用全部完整五秒窗口。每窗对 80 个 token 求均值，沿完整路径计算均值、总体标准差和最小二乘时间斜率，形成 192 维路径特征。

城市巡航同时输出二分类 logit 和连续路径分数。训练目标为：

```text
L = BCE + 0.2 * Huber(path_score_z) + 0.1 * pairwise_rank_softplus
```

`path_score` 的均值和标准差只在当前 source-train 被试上计算。验证时先独立预测，再恢复原始量纲，用 MAE、RMSE 和排序相关系数审计连续分数。

## 训练、选模与完整性控制

所有配置统一使用：

| 设置 | 数值 |
|---|---:|
| optimizer | AdamW |
| 下游学习率 | `1e-3` |
| weight decay | `1e-2` |
| 梯度裁剪 | `1.0` |
| 状态/片段有效 batch | `32` |
| 状态/片段 microbatch | `4` |
| 被试/路径严重度有效 batch | `4` |
| 被试/路径严重度 microbatch | `1` |
| 最大 epoch | `60` |
| patience | `10` 个 epoch 的更新数 |
| 分类阈值 | `0.5` |

`pos_weight = 训练负样本数 / 训练正样本数`，只由当前 source-train 计算。每个 epoch 使用由 seed 和 epoch 决定的固定 shuffle。第一次参数更新后即验证，此后每约四分之一 epoch 和 epoch 末验证；按未加权 source-validation BCE 最小选择最佳检查点，完全相同时保留较早步骤。

训练会保存 `best.pt`、`last.pt`、完整 epoch 边界、optimizer、随机数状态、训练历史、梯度审计和逐样本预测。恢复只能回到完整 epoch 边界，并回退其后的历史与最佳记录。最终汇总还会核验：

- 协议、代码、数据和样本哈希一致；
- 配对配置使用相同训练/验证样本、标签和首次 shuffle；
- 残差配置加载同一 Base 且 Base 未变化；
- 保存后的最佳检查点重载预测完全一致；
- 结果矩阵完整，不能把缺失任务标为 complete。

## 开发实验规模

| 实验 | seed | 内部划分 | 配置 | 总任务数 | 评估分区 |
|---|---:|---:|---:|---:|---|
| 城市巡航路径级严重度 | 3 | 3 | 5 | 45 | source-validation |
| VRQ 被试级严重度 + 状态 | 3 | 3 | 5 x 2 tasks | 90 | source-validation |
| VRQ 片段级严重度 | 3 | 3 | 5 | 45 | source-validation windows |

三套实验共 180 个开发任务。所有 outer-test 评分路径均保持关闭。

## 结果

表中为 3 个 seed × 3 个内部划分的均值；括号内为九次运行的总体标准差。不同任务的计分单位不同，不能把城市路径、VRQ 被试和 VRQ 片段直接合并成一个宏平均。

### 城市巡航：路径级高低眩晕

| 配置 | ACC % | BACC % | AUROC % | subject-macro BACC % | path-score MAE |
|---|---:|---:|---:|---:|---:|
| Base | 75.62 | 73.98 (18.92) | 75.98 | 66.52 | 20.79 |
| MLP | 72.55 | 71.63 (18.55) | 75.77 | 67.31 | 19.15 |
| PolynomialKAN | 74.77 | 73.54 (19.80) | 76.89 | 68.53 | 19.02 |
| FractionalKAN | 74.38 | 73.08 (19.46) | **77.25** | 66.30 | 19.04 |
| Fractional-DoG | **75.97** | **75.00 (18.34)** | 76.50 | **69.42** | **18.54** |

DoG 相对 MLP 的平均 BACC 为 `+3.37 pp`，九次配对为 5 胜、4 平、0 负；相对关闭 DoG 基的反事实为 `+0.81 pp`。预注册晋级要求至少 6/9 次严格胜出，因此城市巡航的自动晋级为 `passed=false`，不能隐去这一负结论。

### VRQ：状态二分类

| 配置 | ACC % | BACC % | AUROC % | BCE |
|---|---:|---:|---:|---:|
| Base | **83.49** | **83.64 (1.72)** | 89.16 | 0.416 |
| MLP | 83.11 | 83.26 (1.97) | **89.44** | 0.411 |
| PolynomialKAN | 83.24 | 83.34 (1.83) | 89.24 | 0.410 |
| FractionalKAN | 83.25 | 83.36 (1.88) | 89.23 | 0.410 |
| Fractional-DoG | 83.37 | 83.53 (2.02) | 89.17 | **0.409** |

DoG 相对 MLP 的平均 BACC 为 `+0.27 pp`，九次配对中 6 次为正；相对关闭 DoG 基为 `+0.20 pp`，因此通过当前方向与稳定性规则。但 DoG 仍比 Base 低 `0.11 pp` BACC，不能表述为全面优于所有对照。

### VRQ：被试级高低眩晕

| 配置 | ACC % | BACC % | AUROC % | BCE |
|---|---:|---:|---:|---:|
| Base | 83.33 | 83.33 (20.79) | 86.42 | 0.384 |
| MLP | **85.19** | **85.19 (18.33)** | 86.42 | **0.362** |
| PolynomialKAN | 83.33 | 83.33 (20.79) | **87.65** | 0.364 |
| FractionalKAN | 83.33 | 83.33 (20.79) | **87.65** | 0.364 |
| Fractional-DoG | 77.78 | 77.78 (24.85) | 85.19 | 0.366 |

这一任务中 DoG 相对 MLP 的 BACC 为 `-7.41 pp`，0/9 次严格胜出，相对关闭 DoG 基为 `-1.85 pp`，晋级失败。这说明当前 DoG 优势没有出现在 VRQ 被试级严重度判断中。

### VRQ：片段级高低眩晕

| 配置 | 片段 ACC % | 片段 BACC % | 片段 AUROC % | BCE | 被试聚合 BACC % |
|---|---:|---:|---:|---:|---:|
| Base | 64.84 | 62.94 (8.32) | 71.08 | 0.597 | 70.37 |
| MLP | 64.47 | 63.78 (9.11) | 71.26 | 0.601 | **77.78** |
| PolynomialKAN | 64.95 | 63.59 (7.42) | 71.13 | 0.598 | 70.37 |
| FractionalKAN | 65.05 | 63.69 (7.32) | 71.14 | 0.598 | 70.37 |
| Fractional-DoG | **65.71** | **64.24 (8.12)** | **72.00** | **0.592** | 75.93 |

DoG 相对 MLP 的片段 BACC 为 `+0.45 pp`，九次配对中 6 次为正；相对关闭 DoG 基为 `+0.41 pp`，通过片段级晋级规则。被试概率聚合后，DoG 比 MLP 低 `1.85 pp` BACC，因此证据只支持片段级小幅改善，不支持被试级明显提升。

完整 CSV、JSON 和逐运行汇总见 [`results/`](results/)。

## 运行方法

复制并填写本地路径配置：

```bash
cp configs/paths.example.json configs/paths.local.json
python scripts/download_pretrained.py
```

预检：

```bash
python scripts/run_femba_severity_boosted_development.py --stage preflight
python scripts/run_femba_vrq_boosted_development.py --stage preflight
python scripts/run_femba_vrq_segment_severity_development.py --stage preflight
```

运行三套开发实验：

```bash
python scripts/run_femba_severity_boosted_development.py --stage all
python scripts/run_femba_vrq_boosted_development.py --stage all
python scripts/run_femba_vrq_segment_severity_development.py --stage all
```

只重载最佳检查点并重新核验预测：

```bash
python scripts/run_femba_severity_boosted_development.py --stage evaluate
python scripts/run_femba_vrq_boosted_development.py --stage evaluate
python scripts/run_femba_vrq_segment_severity_development.py --stage evaluate
```

入口会拒绝把输出和 token 缓存写到非 D 盘路径。默认输出在 `outputs/`，该目录不进入 Git；GitHub 只保存可审计的汇总文件。

## 结果解释边界

- 当前设计只评估预训练并冻结的 FEMBA，不能由这些结果单独证明预训练优于随机初始化，也不能证明 Mamba 优于其他架构。
- 完整预训练样本 manifest 缺失，无法排除 FEMBA 预训练数据与当前 VRQ 样本的潜在重叠。
- 设计过程参考了先前同数据集结果，属于探索性模型开发。
- 九次运行由三个 seed 和三个内部划分组成，不等于九个独立数据集。
- 城市巡航的三个验证组未覆盖全部 20 名 source 被试；5 名 source 被试始终处于训练侧。
- VRQ 片段共享被试标签且相互相关；片段级样本量不能代替被试样本量。
- 在 outer-test 解封并按同一锁定协议独立评估前，不应将当前数值写成最终泛化性能或统计显著结果。

## 仓库结构

```text
src/vestibular_fusion/model/       FEMBA、共享 Base 和非线性残差头
src/vestibular_fusion/training/    数据构造、EA、训练、恢复与完整性检查
src/vestibular_fusion/evaluation/  指标、配对比较和汇总
reproducibility/protocols/         锁定协议、划分和数据哈希
scripts/                           当前实验入口
results/                           当前三套开发实验汇总
tests/                             单元与协议测试
```

原始 EEG、问卷、完整 checkpoint、token 缓存和逐样本训练产物受数据授权或文件体积限制，不在 GitHub 中。移除旧锚点专项入口后，当前测试结果为 `162 passed`。
