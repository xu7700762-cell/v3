# FEMBA 无锚点严重度动态残差实验

该实验检验独立、归一化的 Fractional-DoG 基函数能否改善路径级严重度二分类。实验保留逐被试无标签 EA，但不读取静息参考身份、锚点、测试被试标签或 outer-test EEG。

## 模型

每个严重度样本使用当前有标签任务路径的全部互不重复有序窗口，最短路径必须具有 11 个完整窗口。预训练 FEMBA encoder 固定为 A3 并始终处于 `eval()`。稳定分支对 token mean 做 525→64 投影；可选残差分支逐 token 使用 MLP、PolynomialKAN、FractionalKAN 或 Fractional-DoG KAN。两个分支均以时间均值、总体标准差和最小二乘斜率汇总，最终输出为 `base_logit + sigmoid(gate) * residual_logit`。

完整 DoG 层将分数阶一次项、二次项和三尺度 DoG 项作为独立基函数。每个基函数做逐样本 RMS 归一化，使用独立权重；DoG 尺度固定为 0.5、1.0、2.0，并在 15 个特征组内学习尺度混合。MLP 与 DoG 映射参数量差异小于 1%。

## 运行

```bash
python scripts/run_femba_severity_dynamics_pilot.py --dry-run
python scripts/run_femba_severity_dynamics_pilot.py --stage preflight
python scripts/run_femba_severity_dynamics_pilot.py --stage all --smoke
python scripts/run_femba_severity_dynamics_pilot.py --stage all
```

输出固定保存在 D 盘的 `outputs/femba_severity_dynamics_pilot_v1`。正式小试为 5 个下游配置 × 2 个数据集，共 10 项 fold_1 source-validation 实验。小试不会加载或评分 outer-test，不能作为独立测试结论。

## 解释

主要比较为 `dog-base`、`dog-mlp`、`dog-poly` 和 `dog-fractional`。除总体 ACC、BACC、AUROC 和 BCE 外，同时报告 subject-macro BACC、DoG 基函数和输出贡献、分数阶范围、残差门控及移除 DoG 后的反事实预测。正差值不自动解释为统计显著，也不保证在未见数据复现。
