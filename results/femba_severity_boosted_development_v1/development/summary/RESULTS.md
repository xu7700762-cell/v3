# FEMBA 城市严重度共享基线残差开发实验

状态：complete；45/45 项。outer-test 未评分。

| 配置 | BACC % | subject-macro BACC % | AUROC % | BCE | path-score MAE | 运行数 |
|---|---:|---:|---:|---:|---:|---:|
| base | 73.98 | 66.52 | 75.98 | 0.495 | 20.79 | 9 |
| mlp | 71.63 | 67.31 | 75.77 | 0.499 | 19.15 | 9 |
| poly | 73.54 | 68.53 | 76.89 | 0.491 | 19.02 | 9 |
| fractional | 73.08 | 66.30 | 77.25 | 0.490 | 19.04 | 9 |
| dog | 75.00 | 69.42 | 76.50 | 0.493 | 18.54 | 9 |

## DoG 晋级判定

`passed=False`

## 解释边界

- Only the twenty fold_1 source subjects are used for development; five outer-test subjects remain excluded.
- All configurations use identical source paths, continuous targets, loss weights, shuffles and validation rules.
- The base selected for a seed and split is byte-identical and frozen for all four residual comparisons.
- Path sub-sequences are not counted as independent samples; each recorded path remains one target.
- This three-split three-seed matrix is exploratory model development, not independent test evidence.
- A failed promotion rule is retained and stops automatic outer-test evaluation.
