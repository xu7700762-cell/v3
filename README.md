# VestibularFusion v3

本仓库只汇总当前 FEMBA 下游实验的三类结果：城市巡航高低眩晕、VRQ 高低眩晕和 VRQ 状态二分类。

## 当前实验协议

当前结果严格不使用静息参考段、U3–U6 锚点、锚点中心减法、已知静息标签、KMeans 校准或任何参考段分类损失。模型输入只有原始 EEG 窗口和训练分区提供的监督标签。训练与验证按被试隔离；EA 只使用无标签 EEG，并明确属于离线 transductive 预处理。

统一数据流为：

```text
EEG window [30,1280]
    ↓
FEMBA encoder
    ↓
token mean pooling [525]
    ↓
Base / MLP / PolynomialKAN / Fractional-DoG head
    ↓
one binary logit
```

城市巡航结果按路径级高低眩晕计分。VRQ 被试级结果同时包含高低眩晕严重度和状态二分类；VRQ 片段级结果把最终任务的每个五秒窗口作为计分单位，并保留被试聚合审计。VRQ 片段继承被试标签，因此片段不是独立被试样本。

## 结果

结果汇总位于 [`results/`](results/)，当前只保留：

- [城市巡航高低眩晕](results/femba_severity_boosted_development_v1/development/summary/RESULTS.md)
- [VRQ 被试级高低眩晕和状态二分类](results/femba_vrq_boosted_development_v1/development/summary/RESULTS.md)
- [VRQ 五秒片段级高低眩晕](results/femba_vrq_segment_severity_development_v1/development/summary/RESULTS.md)

这些结果是固定 seed、内部 source-validation 开发实验；没有自动调用 outer-test。不要把窗口数当作独立被试数，也不要将单 seed 的开发差异表述为统计显著。

## 代码和复现

- `src/vestibular_fusion/model/`：FEMBA 与下游头
- `src/vestibular_fusion/training/`：训练与数据分区
- `src/vestibular_fusion/evaluation/`：指标和重载评估
- `scripts/run_femba_severity_boosted_development.py`：城市巡航严重度
- `scripts/run_femba_vrq_boosted_development.py`：VRQ 被试级严重度和状态
- `scripts/run_femba_vrq_segment_severity_development.py`：VRQ 片段级严重度
- `reproducibility/protocols/`：固定协议和哈希记录

原始 EEG、问卷、checkpoint 和训练缓存不在 GitHub 中。当前代码测试结果为 `179 passed`。

仓库保留少量原 v27 兼容代码以支持历史接口；这些兼容路径不参与上述三类当前结果，也不改变当前无锚点协议。
