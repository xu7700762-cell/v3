# FEMBA 无锚点严重度动态残差小试

状态：complete；10/10 项。seed=2001，fold_1 source-val。

结果用于结构开发，不是独立测试结果。EA 为逐被试无标签离线 transductive 预处理。

## vrq / severity

| 配置 | ACC % | BACC % | subject-macro BACC % | AUROC % | BCE | N | 被试 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 80.00 | 83.33 | 80.00 | 100.00 | 0.374 | 5 | 5 |
| mlp | 80.00 | 83.33 | 80.00 | 100.00 | 0.331 | 5 | 5 |
| poly | 80.00 | 83.33 | 80.00 | 100.00 | 0.360 | 5 | 5 |
| fractional | 80.00 | 83.33 | 80.00 | 100.00 | 0.360 | 5 | 5 |
| dog | 80.00 | 83.33 | 80.00 | 100.00 | 0.358 | 5 | 5 |

## city / severity

| 配置 | ACC % | BACC % | subject-macro BACC % | AUROC % | BCE | N | 被试 |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 86.21 | 85.95 | 67.50 | 91.90 | 0.340 | 29 | 5 |
| mlp | 93.10 | 93.10 | 87.50 | 92.38 | 0.328 | 29 | 5 |
| poly | 89.66 | 89.29 | 70.00 | 91.90 | 0.345 | 29 | 5 |
| fractional | 89.66 | 89.29 | 70.00 | 91.90 | 0.346 | 29 | 5 |
| dog | 86.21 | 85.95 | 67.50 | 91.90 | 0.338 | 29 | 5 |

## 解释边界

- No known-rest calibration, anchor input, rest identity or target-subject label adaptation is used.
- Subject-wide unlabeled EA is retained and is an offline transductive preprocessing step.
- The same inputs, temporal summary, loss, split and optimization budget are used by all downstream controls.
- Single seed and fold_1 source-validation are used for architecture development; these are not independent test results.
- VRQ source-validation has one severity sample per subject and cannot support a stable performance claim.
- Outer-test EEG and labels are not loaded or scored by this pilot.
- Pretraining sample manifest is incomplete, so pretraining overlap has not been excluded.
- A positive validation difference is not called statistically significant or guaranteed to reproduce.
