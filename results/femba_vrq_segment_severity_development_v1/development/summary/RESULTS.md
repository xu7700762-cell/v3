# FEMBA VRQ 片段级严重度开发实验

状态：complete；45/45 项。outer-test 未评分。

| 配置 | 片段 ACC % | 片段 BACC % | 片段 AUROC % | BCE | 被试宏片段 ACC % | 被试聚合 BACC % | 运行数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 64.84 | 62.94 | 71.08 | 0.597 | 63.08 | 70.37 | 9 |
| mlp | 64.47 | 63.78 | 71.26 | 0.601 | 64.17 | 77.78 | 9 |
| poly | 64.95 | 63.59 | 71.13 | 0.598 | 63.74 | 70.37 | 9 |
| fractional | 65.05 | 63.69 | 71.14 | 0.598 | 63.85 | 70.37 | 9 |
| dog | 65.71 | 64.24 | 72.00 | 0.592 | 64.40 | 75.93 | 9 |

DoG 晋级：`passed=True`

## 解释边界

- City data, code paths, checkpoints and result artifacts are not read or modified by this protocol.
- Windows inherit a subject-level post-task severity label and are correlated weakly supervised samples, not independent subjects.
- Train and validation are strictly subject-disjoint; no subject contributes windows to both partitions.
- Only final-task windows enter severity classification; no rest window or rest identity is a model input.
- Subject-wide unlabeled EA is retained and explicitly transductive.
- Five outer-test subjects remain excluded and no outer-test evaluation is automatically triggered.
- Primary window metrics must be accompanied by subject-clustered or subject-aggregated audit metrics in any paper report.
