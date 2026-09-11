# VestibularFusion v3

本仓库实现以预训练 FEMBA 为编码器、Fractional-DoG PolynomialKAN 为非线性下游的前庭不适 EEG 分类方法。当前版本集中报告三项任务：

- VRQ 五秒片段级高低眩晕分类；
- 城市巡航路径级高低眩晕分类；
- VRQ 五秒窗口状态二分类。

三个任务均采用被试隔离的数据划分，以固定的三组随机种子和三组内部验证划分运行。仓库提供模型、训练与评估代码、固定协议以及对应结果汇总。

## 方法概览

```text
30 通道五秒 EEG [B,30,1280]
  -> 无标签 Euclidean Alignment
  -> 预训练 FEMBA
  -> token 表征 [B,80,525]
  -> Base 分支 + 非线性残差分支
  -> 任务汇聚
  -> 二分类 logit
```

FEMBA 提取时空 token；Base 分支提供稳定的线性投影，非线性残差分支比较 MLP、PolynomialKAN、FractionalKAN 和 Fractional-DoG。最终预测为：

```text
final_logit = frozen_base_logit + trainable_residual_logit
```

## 数据与预处理

两套数据均使用 30 通道 EEG，并切分为连续、无重叠的五秒窗口，每窗包含 1280 个采样点。MAT 文件、样本清单与协议文件在载入前核验 SHA-256。

每名被试独立执行无标签 Euclidean Alignment（EA）。设预处理后的窗口为 `X_n`：

```text
R = mean_n(X_n X_n^T / (T - 1))
X_EA = (R + ridge I)^(-1/2) X
```

EA 先对窗口做时间均值中心化和全局尺度归一化，再用该被试窗口的平均通道协方差完成空间对齐。该步骤不读取分类标签，属于离线、逐被试的 transductive 预处理。

### VRQ

- 18 名 source 被试用于开发，另有 5 名 outer-test 被试未参与本次训练与结果统计。
- 三个内部验证组互不重叠，每组 6 名被试；每次使用其余 12 名被试训练。
- 状态任务以单个五秒窗为样本，判断静息状态与任务状态。
- 片段级高低眩晕任务使用最终任务阶段的每个完整五秒窗，片段继承所属被试的 SSQ 高低分组标签。

### 城市巡航

- 20 名 source 被试用于开发，另有 5 名 outer-test 被试未参与本次训练与结果统计。
- 每条巡航路径构成一个样本，使用路径内全部完整五秒窗。
- 三个内部验证组各含 5 名被试；每次用其余 15 名 source 被试训练。
- 二分类标签与连续路径分数来自固定数据清单。

## FEMBA 编码器

```text
EEG [B,30,1280]
  -> Conv2d PatchEmbed(kernel=2x16, stride=2x16, embed=35)
  -> positional embedding
  -> 4 x [forward Mamba + reverse Mamba + residual + LayerNorm]
  -> tokens [B,80,525]
```

双向 Mamba 分支采用逐元素相加。模型加载 `pretrained_femba_v27.ckpt`，协议记录的 SHA-256 为：

```text
0e2ab9109d87a32c6b25f0c307fb8b1102ef5e0e83e86b9b07f7dee166daaa27
```

加载时要求 83 个 encoder 张量完整匹配。编码器以 CUDA bfloat16 前向，输出转换为 float32 token 并保存到带哈希的缓存。下游实验固定 FEMBA 参数，使不同映射模块共享完全相同的编码表示。

## 两阶段下游

每个 seed、任务和内部划分先训练一个 Base，再加载并固定其最佳权重，训练非线性残差分支。

### Base

```text
FEMBA tokens [80,525]
  -> LayerNorm(525)
  -> Linear(525,64)
  -> task summary [192]
  -> LayerNorm(192)
  -> Linear(192,1)
```

### 非线性残差分支

```text
FEMBA tokens [80,525]
  -> LayerNorm(525)
  -> feature mapper (525 -> 64)
  -> LayerNorm(64)
  -> task summary [192]
  -> LayerNorm(192)
  -> Linear(192,1)
  -> add frozen Base logit
```

比较的映射模块包括：

| 配置 | 525 → 64 映射 |
|---|---|
| MLP | `Linear(525,172) -> SiLU -> Linear(172,64)` |
| PolynomialKAN | 一阶与二阶多项式基映射 |
| FractionalKAN | 可学习分数阶一阶与二阶基映射 |
| Fractional-DoG | 分数阶多项式基与多尺度高斯一阶导数基联合映射 |

MLP 与 Fractional-DoG 经过参数量匹配，两者只相差 77 个参数。

## Fractional-DoG PolynomialKAN

该模块在 FEMBA 的 525 维特征坐标上学习分数阶响应和局部变化响应。对归一化输入 `x`，先学习逐维尺度：

```text
u = tanh(x * exp(0.5 * tanh(log_scale)))
```

525 个特征坐标分为 15 组，每组学习一个 `q in [0.7,1.5]`：

```text
q = 0.7 + 0.8 * sigmoid(q_logit)
f_q(u) = sign(u) * ((abs(u) + eps)^q - eps^q)
```

映射使用分数阶一阶基、二阶基和三尺度高斯一阶导数基：

```text
B1 = f_q(u)
B2 = 2 * f_q(u)^2 - 1
B3_s = -v_s * exp(-0.5 * v_s^2),  s in {0.5,1.0,2.0}
```

三个尺度通过 softmax 学习混合比例。各基函数经过 RMS 归一化和可学习门控后拼接，再投影为 64 维表示。FractionalKAN 用于检验分数阶映射本身，Fractional-DoG 则进一步加入局部导数基。

## 三项任务的汇聚

### VRQ 状态二分类

每个五秒窗得到 80 个 token。映射后沿 token 维计算均值、总体标准差和平均绝对一阶差分，拼接成 192 维摘要并输出一个窗口级 logit。

### VRQ 片段级高低眩晕

模型与状态任务保持相同结构。每个最终任务五秒片段独立输出高低眩晕 logit。主要指标按片段统计，同时将同一被试的片段概率取平均，报告被试聚合指标。

### 城市巡航路径级高低眩晕

每个路径窗口先对 80 个 token 求均值，再沿完整路径计算均值、总体标准差和最小二乘时间斜率，拼接为 192 维路径摘要。模型同时预测高低眩晕类别与连续路径分数：

```text
L = BCE + 0.2 * Huber(path_score_z) + 0.1 * pairwise_rank_softplus
```

连续分数的均值和标准差只由当前训练分区计算。

## 训练设置

| 设置 | 数值 |
|---|---:|
| optimizer | AdamW |
| 下游学习率 | `1e-3` |
| weight decay | `1e-2` |
| 梯度裁剪 | `1.0` |
| 状态/片段有效 batch | `32` |
| 状态/片段 microbatch | `4` |
| 路径有效 batch | `4` |
| 路径 microbatch | `1` |
| 最大 epoch | `60` |
| patience | `10` 个 epoch 的更新数 |
| 分类阈值 | `0.5` |

`pos_weight` 仅按当前训练分区的正负样本数计算。模型从第一次更新后开始验证，此后在约四分之一 epoch 和 epoch 末验证，按 source-validation BCE 选择最佳检查点。每项配置使用 seed `2001/3001/4001` 和三个内部划分，共 9 次运行。

## 结果

表中 ACC、BACC 和 AUROC 为 9 次运行的均值，BACC 括号内为运行间标准差。

### VRQ：片段级高低眩晕

| 配置 | 片段 ACC % | 片段 BACC % | 片段 AUROC % | BCE | 被试聚合 BACC % |
|---|---:|---:|---:|---:|---:|
| Base | 64.84 | 62.94 (8.32) | 71.08 | 0.597 | 70.37 |
| MLP | 64.47 | 63.78 (9.11) | 71.26 | 0.601 | **77.78** |
| PolynomialKAN | 64.95 | 63.59 (7.42) | 71.13 | 0.598 | 70.37 |
| FractionalKAN | 65.05 | 63.69 (7.32) | 71.14 | 0.598 | 70.37 |
| Fractional-DoG | **65.71** | **64.24 (8.12)** | **72.00** | **0.592** | 75.93 |

Fractional-DoG 相对 MLP 的片段 BACC 提高 `0.45` 个百分点，9 次配对中 6 次取得更高 BACC；同时获得最高片段 ACC、BACC、AUROC 和最低 BCE。

### 城市巡航：路径级高低眩晕

| 配置 | ACC % | BACC % | AUROC % | subject-macro BACC % | path-score MAE |
|---|---:|---:|---:|---:|---:|
| Base | 75.62 | 73.98 (18.92) | 75.98 | 66.52 | 20.79 |
| MLP | 72.55 | 71.63 (18.55) | 75.77 | 67.31 | 19.15 |
| PolynomialKAN | 74.77 | 73.54 (19.80) | 76.89 | 68.53 | 19.02 |
| FractionalKAN | 74.38 | 73.08 (19.46) | **77.25** | 66.30 | 19.04 |
| Fractional-DoG | **75.97** | **75.00 (18.34)** | 76.50 | **69.42** | **18.54** |

Fractional-DoG 相对 MLP 的路径 BACC 提高 `3.37` 个百分点，9 次配对为 5 次更高、4 次持平，并取得最高 ACC、BACC、subject-macro BACC 和最低路径分数 MAE。

### VRQ：状态二分类

| 配置 | ACC % | BACC % | AUROC % | BCE |
|---|---:|---:|---:|---:|
| Base | **83.49** | **83.64 (1.72)** | 89.16 | 0.416 |
| MLP | 83.11 | 83.26 (1.97) | **89.44** | 0.411 |
| PolynomialKAN | 83.24 | 83.34 (1.83) | 89.24 | 0.410 |
| FractionalKAN | 83.25 | 83.36 (1.88) | 89.23 | 0.410 |
| Fractional-DoG | 83.37 | 83.53 (2.02) | 89.17 | **0.409** |

Fractional-DoG 相对 MLP 的 BACC 提高 `0.27` 个百分点，9 次配对中 6 次取得更高 BACC，并获得最低 BCE。

完整 CSV、JSON 和逐运行汇总见 [`results/`](results/)。

## 运行方法

先填写本地数据与权重路径：

```bash
cp configs/paths.example.json configs/paths.local.json
python scripts/download_pretrained.py
```

执行预检：

```bash
python scripts/run_femba_vrq_segment_severity_development.py --stage preflight
python scripts/run_femba_severity_boosted_development.py --stage preflight
python scripts/run_femba_vrq_boosted_development.py --stage preflight --tasks state
```

运行三个实验：

```bash
python scripts/run_femba_vrq_segment_severity_development.py --stage all
python scripts/run_femba_severity_boosted_development.py --stage all
python scripts/run_femba_vrq_boosted_development.py --stage all --tasks state
```

使用 `--stage evaluate` 可重载最佳检查点并复核预测，使用 `--resume` 可从完整 epoch 边界继续训练。默认训练产物保存在 D 盘 `outputs/`，适合提交的汇总文件保存在 `results/`。

## 结果范围

当前数值来自 source-validation 开发实验，尚未使用预留的 outer-test 被试。VRQ 片段共享被试标签，因此片段结果同时给出被试聚合指标。三项任务的计分单位不同，结果分别报告。

## 仓库结构

```text
src/vestibular_fusion/model/       FEMBA、Base 与非线性残差头
src/vestibular_fusion/training/    数据构造、EA、训练与恢复
src/vestibular_fusion/evaluation/  指标、配对比较与汇总
reproducibility/protocols/         固定协议、划分与数据哈希
scripts/                           三项实验入口
results/                           三项实验汇总
tests/                             单元测试与协议测试
```

原始 EEG、问卷、完整 checkpoint、token 缓存和逐样本训练产物不纳入 Git 仓库。当前测试套件共 `162 passed`。
