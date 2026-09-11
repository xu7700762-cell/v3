# FEMBA VRQ 共享基线残差开发实验

状态：complete；90/90 项。outer-test 未评分。

## severity

| 配置 | ACC % | BACC % | AUROC % | BCE | 运行数 |
|---|---:|---:|---:|---:|---:|
| base | 83.33 | 83.33 | 86.42 | 0.384 | 9 |
| mlp | 85.19 | 85.19 | 86.42 | 0.362 | 9 |
| poly | 83.33 | 83.33 | 87.65 | 0.364 | 9 |
| fractional | 83.33 | 83.33 | 87.65 | 0.364 | 9 |
| dog | 77.78 | 77.78 | 85.19 | 0.366 | 9 |

DoG 晋级：`passed=False`

## state

| 配置 | ACC % | BACC % | AUROC % | BCE | 运行数 |
|---|---:|---:|---:|---:|---:|
| base | 83.49 | 83.64 | 89.16 | 0.416 | 9 |
| mlp | 83.11 | 83.26 | 89.44 | 0.411 | 9 |
| poly | 83.24 | 83.34 | 89.24 | 0.410 | 9 |
| fractional | 83.25 | 83.36 | 89.23 | 0.410 | 9 |
| dog | 83.37 | 83.53 | 89.17 | 0.409 | 9 |

DoG 晋级：`passed=True`

## 解释边界

- Only the eighteen fold_1 source subjects are used for development; five outer-test subjects remain excluded.
- The three six-subject validation groups are disjoint and each contains three low and three high severity subjects.
- All configurations use identical source examples, labels, shuffles and validation rules.
- The base selected for each seed, task and split is byte-identical and frozen for all residual comparisons.
- No known-rest calibration or anchors; subject-wide unlabeled EA is retained and explicitly transductive.
- Severity uses one eleven-window task-session sample per subject; state uses individual five-second windows.
- This three-split three-seed matrix is exploratory model development, not independent test evidence.
- No outer-test evaluation is automatically triggered by this experiment.
