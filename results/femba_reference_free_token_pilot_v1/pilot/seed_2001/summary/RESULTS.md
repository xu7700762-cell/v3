# FEMBA 无锚点逐 token 映射小试（保留无标签 EA）

状态：complete；52/52 项。seed=2001，fold_1 source-val；不是独立测试结果。

EA 使用每名被试协议内全部无标签窗口，属于离线 transductive 处理。静息身份不进入预测或校准。

## vrq / state

| 配置 | ACC % | BACC % | AUROC % | BCE | 样本数 | 最佳步 |
|---|---:|---:|---:|---:|---:|---:|
| R0_dog | 80.64 | 80.46 | 85.16 | 0.47 | 532 | 198 |
| R0_mlp | 80.08 | 79.88 | 84.35 | 0.47 | 532 | 198 |
| R0_poly | 80.26 | 80.14 | 84.41 | 0.48 | 532 | 121 |
| R1_dog | 81.58 | 81.40 | 87.04 | 0.45 | 532 | 176 |
| R1_mlp | 79.70 | 79.43 | 83.33 | 0.49 | 532 | 84 |
| R1_poly | 81.95 | 81.77 | 86.28 | 0.48 | 532 | 440 |
| R2_dog | 81.95 | 81.84 | 86.94 | 0.45 | 532 | 121 |
| R2_mlp | 81.58 | 81.41 | 86.36 | 0.46 | 532 | 121 |
| R2_poly | 81.58 | 81.52 | 85.42 | 0.46 | 532 | 121 |
| linear | 80.08 | 79.83 | 84.24 | 0.48 | 532 | 495 |
| R2_order1 | 81.95 | 81.84 | 86.95 | 0.45 | 532 | 121 |
| R2_no_dog | 81.95 | 81.84 | 86.96 | 0.45 | 532 | 121 |
| R2_order1_no_dog | 81.95 | 81.84 | 86.95 | 0.45 | 532 | 121 |

## city / state

| 配置 | ACC % | BACC % | AUROC % | BCE | 样本数 | 最佳步 |
|---|---:|---:|---:|---:|---:|---:|
| R0_dog | 86.69 | 87.00 | 92.89 | 0.31 | 1691 | 2220 |
| R0_mlp | 85.63 | 86.41 | 92.04 | 0.33 | 1691 | 282 |
| R0_poly | 86.58 | 87.10 | 92.87 | 0.32 | 1691 | 2585 |
| R1_dog | 84.92 | 85.27 | 91.68 | 0.35 | 1691 | 555 |
| R1_mlp | 86.58 | 87.14 | 92.40 | 0.32 | 1691 | 185 |
| R1_poly | 85.27 | 85.43 | 92.08 | 0.34 | 1691 | 987 |
| R2_dog | 87.94 | 87.98 | 93.93 | 0.29 | 1691 | 470 |
| R2_mlp | 88.47 | 88.89 | 93.54 | 0.30 | 1691 | 282 |
| R2_poly | 88.11 | 88.22 | 94.00 | 0.29 | 1691 | 470 |
| linear | 84.86 | 84.77 | 92.10 | 0.34 | 1691 | 5735 |
| R2_order1 | 87.94 | 87.98 | 93.94 | 0.29 | 1691 | 470 |
| R2_no_dog | 87.94 | 87.98 | 93.94 | 0.29 | 1691 | 470 |
| R2_order1_no_dog | 87.94 | 87.98 | 93.94 | 0.29 | 1691 | 470 |

## vrq / severity

| 配置 | ACC % | BACC % | AUROC % | BCE | 样本数 | 最佳步 |
|---|---:|---:|---:|---:|---:|---:|
| R0_dog | 100.00 | 100.00 | 100.00 | 0.01 | 5 | 125 |
| R0_mlp | 100.00 | 100.00 | 100.00 | 0.00 | 5 | 164 |
| R0_poly | 100.00 | 100.00 | 100.00 | 0.03 | 5 | 113 |
| R1_dog | 60.00 | 58.33 | 83.33 | 0.42 | 5 | 88 |
| R1_mlp | 80.00 | 75.00 | 83.33 | 0.38 | 5 | 83 |
| R1_poly | 80.00 | 83.33 | 83.33 | 0.38 | 5 | 111 |
| R2_dog | 100.00 | 100.00 | 100.00 | 0.39 | 5 | 41 |
| R2_mlp | 100.00 | 100.00 | 100.00 | 0.32 | 5 | 42 |
| R2_poly | 100.00 | 100.00 | 100.00 | 0.37 | 5 | 74 |
| linear | 100.00 | 100.00 | 100.00 | 0.47 | 5 | 165 |
| R2_order1 | 100.00 | 100.00 | 100.00 | 0.39 | 5 | 41 |
| R2_no_dog | 100.00 | 100.00 | 100.00 | 0.39 | 5 | 41 |
| R2_order1_no_dog | 100.00 | 100.00 | 100.00 | 0.39 | 5 | 41 |

## city / severity

| 配置 | ACC % | BACC % | AUROC % | BCE | 样本数 | 最佳步 |
|---|---:|---:|---:|---:|---:|---:|
| R0_dog | 82.76 | 82.62 | 95.24 | 0.30 | 29 | 732 |
| R0_mlp | 86.21 | 86.19 | 92.38 | 0.36 | 29 | 282 |
| R0_poly | 86.21 | 86.19 | 94.29 | 0.32 | 29 | 858 |
| R1_dog | 89.66 | 89.76 | 95.71 | 0.27 | 29 | 330 |
| R1_mlp | 86.21 | 86.43 | 94.29 | 0.31 | 29 | 210 |
| R1_poly | 86.21 | 86.43 | 98.10 | 0.21 | 29 | 1434 |
| R2_dog | 82.76 | 82.38 | 96.67 | 0.36 | 29 | 186 |
| R2_mlp | 89.66 | 89.52 | 93.33 | 0.36 | 29 | 108 |
| R2_poly | 82.76 | 82.62 | 94.76 | 0.36 | 29 | 186 |
| linear | 82.76 | 82.86 | 94.29 | 0.35 | 29 | 1398 |
| R2_order1 | 82.76 | 82.38 | 96.67 | 0.36 | 29 | 186 |
| R2_no_dog | 82.76 | 82.38 | 96.67 | 0.36 | 29 | 186 |
| R2_order1_no_dog | 82.76 | 82.38 | 96.67 | 0.36 | 29 | 186 |

## 解释边界

不自动将正差值解释为统计显著；只改善汇总而未超过同结构 MLP，不能归因于 DoG。
关闭分支的推理诊断与重新训练的成分消融分别保存在 diagnostics.json 和配对表。

- No known-rest calibration or anchors; offline subject-wide unlabeled EA is explicitly allowed and transductive.
- State labels retain the existing dataset protocol semantics, not independent symptom measurements.
- Single seed and fold_1 validation used for selection; exploratory, not independent test performance.
- Pretraining sample manifest is incomplete; overlap has not been excluded.
- Upstream MAT processing is hash-locked but not a complete raw-acquisition provenance audit.
- Design informed by previous test results; all positive and negative results retained.
- Frozen pretrained encoder only; no claim about pretraining versus random initialization.
- Equal sample exposure rules and budget ceilings, not equal FLOPs or actual selected steps.
