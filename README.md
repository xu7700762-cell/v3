# VestibularFusion v27

新增独立的 [无静息参考逐 token 映射小试](docs/FEMBA_REFERENCE_FREE_TOKEN_PILOT.md)：冻结 FEMBA，比较 R0/R1/R2 下的 Fractional-DoG、MLP、普通 PolynomialKAN 及成分消融。按用户修订允许逐被试无标签 EA（离线 transductive）；52 项只评估 fold_1 source-val，不使用静息参考校准，不评分 outer-test。

独立的 [FEMBA C0 DoG/MLP 下游对照](docs/FEMBA_C0_HEAD_COMPARISON.md) 使用 VRQ、城市巡航，比较预训练冻结/微调、两个任务及固定五折；80个新实验配对40个原C0线性结果。无锚点，保留原C0离线无标签EA，不预设性能提升。

本仓库的正式参考结果保留 `v27 PolynomialKAN`、`seed=2001`、三数据集固定五折协议，以及参数匹配的 MLP 基线。旧版本、其他 seed、训练缓存和原始数据均未提交。

另提供独立的 [FEMBA 自监督预训练有效性 2×2 消融](docs/FEMBA_SSL_ABLATION.md)：Random/Pretrained × Frozen/Trainable，两个下游任务分别使用纯 `Linear(525,1)`。该入口不改变 v27 模型及复现协议；实现与 smoke 验证不等于完整消融实验结果。

新增独立的 [FEMBA 预训练 × 四锚点配对验证小试](docs/FEMBA_SSL_ANCHOR_PILOT.md)：四组各比较无锚点/四锚点中心减法，三数据集 fold_1、两个任务共 48 项。采用第一步起选最佳模型和完整 epoch 边界恢复，仅评估 source-val，不执行 outer-test 评分。

[完整五折实验](docs/FEMBA_SSL_FIVEFOLD.md) 在每折选步后从原始初始化重训全部 source，再评估 outer-test；按无锚点120项→四锚点120项的顺序运行，所有新结果在本机 D 盘，共享存储相同的冻结 encoder 以节省空间。

> 纠错说明：此前截图中的数值来自 `v24`，不能标为 v27，因此没有写入本仓库。下表和 `reproducibility/reference/` 均来自重新核验后的真实 `femba_kan_mtl_v27` 产物。

## 模型与固定协议

```text
Raw EEG window [30,1280]
        │
        ▼
Frozen pretrained FEMBA
        │
        ▼
Token mean pooling [525]
        │
        ▼
LayerNorm → Fractional-DoG PolynomialKAN(525,160,degree=2) → LayerNorm
        │
        ├── U3-U6 anchors → state classification
        └── FEMBA token dynamics + bounded KAN residual → severity classification
```

- 数据集：`monifeixing`、`VRQ`、`city`；
- identity-disjoint 五折，split seed=`42`；
- training seed=`2001`；
- checkpoint schema=`femba_kan_mtl_v27`；
- 正式投影=`fractional_dog_polykan`；
- FEMBA encoder 全程冻结并保持 `eval()`；
- 每名被试使用 U3-U6 四个 reference anchors；
- 状态任务使用 anchor-oriented subject KMeans；
- 严重度任务使用 11 个均匀 task windows。

## v27 seed=2001 真实结果

所有数值均为五折合并后的 `ACC / BACC / AUROC`，单位为百分比。

### 状态分类

| 数据集 | v27 PolynomialKAN | 参数匹配 MLP | KAN−MLP ACC |
|---|---:|---:|---:|
| monifeixing | 80.69 / 80.89 / 85.95 | 80.59 / 80.68 / 86.08 | +0.10 pp |
| VRQ | 80.61 / 81.15 / 85.99 | 78.52 / 79.32 / 85.00 | +2.10 pp |
| city | 80.41 / 81.10 / 86.09 | 79.75 / 80.07 / 85.85 | +0.66 pp |
| 三数据集宏平均 | 80.57 / 81.05 / 86.01 | 79.62 / 80.02 / 85.64 | +0.96 pp |

### 高低眩晕分类

| 数据集 | v27 PolynomialKAN | 参数匹配 MLP | KAN−MLP ACC |
|---|---:|---:|---:|
| monifeixing | 66.67 / 66.67 / 62.96 | 66.67 / 66.67 / 69.14 | 0.00 pp |
| VRQ | 82.61 / 82.95 / 85.61 | 78.26 / 78.41 / 85.61 | +4.35 pp |
| city | 74.03 / 74.03 / 78.53 | 72.08 / 72.08 / 77.08 | +1.95 pp |
| 三数据集宏平均 | 74.43 / 74.55 / 75.70 | 72.34 / 72.38 / 77.27 | +2.09 pp |

严重度 ACC/BACC 的宏平均由 KAN 提升，但 AUROC 宏平均低于 MLP `1.57 pp`；仓库不隐藏这一结果。

## 无数据先验证仓库结果

只验证已发布结果不需要 EEG 数据或 GPU：

```bash
python -m pip install -e .
python scripts/verify_reproduction.py
```

该命令会核验参考文件 SHA-256，从 CSV 重新计算 ACC/BACC/AUROC，并与 `aggregate_report.json` 和 `expected_metrics.json` 交叉检查。

## 有数据时端到端复现

推荐使用 Linux/WSL2、Python 3.10、CUDA 12.8 和支持 bfloat16 的 NVIDIA GPU。

### 1. 创建环境

Conda：

```bash
conda env create -f environment.yml
conda activate vestibular-v27
```

或使用 Docker：

```bash
docker build -t vestibular-v27 .
```

### 2. 下载并校验 FEMBA 权重

```bash
python scripts/download_pretrained.py
```

下载文件：`local_assets/pretrained_femba_v27.ckpt`
SHA-256：`0e2ab9109d87a32c6b25f0c307fb8b1102ef5e0e83e86b9b07f7dee166daaa27`

### 3. 配置三套数据

```bash
cp configs/paths.example.json configs/paths.local.json
```

只需填写三个预处理数据目录。准确文件名、MAT 格式和校验规则见 [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)。问卷和旧工程目录不再是运行依赖。

### 4. 预检

```bash
python -m vestibular_fusion preflight --config configs/paths.local.json
```

预检会检查：固定环境、协议文件、五折身份隔离、全部 MAT 文件 SHA-256，以及 FEMBA checkpoint SHA-256。任何缺失或不一致都会显式失败。

### 5. 完整训练、评估和验收

只复现 v27 PolynomialKAN：

```bash
python scripts/reproduce_v27_seed2001.py \
  --config configs/paths.local.json
```

同时重跑参数匹配 MLP 基线：

```bash
python scripts/reproduce_v27_seed2001.py \
  --config configs/paths.local.json \
  --include-mlp
```

流程会依次运行三数据集五折训练、正式评估，并把新结果与发布结果逐项比较。默认验收容差为 `1.0 percentage point`。输出位于：

```text
outputs/reproduction_v27_seed2001/
  fractional_dog_polykan/
    training/{monifeixing,vrq,city}/fold_1..fold_5/
    evaluation/{monifeixing,vrq,city}/
```

训练代码启用固定随机种子、确定性 cuDNN、`CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 PyTorch deterministic algorithms。不同 GPU/驱动组合不承诺 checkpoint 位级一致，因此以公开指标容差作为复现验收标准。

## 仓库内容

```text
src/                         v27 模型、训练和评估代码
configs/paths.example.json   最小本地路径配置
reproducibility/protocols/   脱敏标签、固定划分、数据哈希
reproducibility/reference/   六组真实 v27/MLP 参考结果
scripts/download_pretrained.py
scripts/reproduce_v27_seed2001.py
scripts/verify_reproduction.py
environment.yml
Dockerfile
```

原始 EEG 和问卷受数据授权限制，不能由本仓库公开分发。拥有相同数据版本的使用者可以通过 manifest 哈希确认数据一致，并完整复现训练与评估。
新增独立的 [无锚点严重度动态残差小试](docs/FEMBA_SEVERITY_DYNAMICS_PILOT.md)：冻结预训练 FEMBA，使用完整任务路径，比较稳定基线、参数匹配 MLP、PolynomialKAN、FractionalKAN 与独立归一化的多尺度 Fractional-DoG；只评估 fold_1 source-val，不评分 outer-test。
