# FEMBA/DoG 实验结果归档

本目录只保存可进入版本控制的实验汇总：`RESULTS.md`、CSV 指标表和 JSON 协议汇总。训练 checkpoint、逐样本缓存、预测中间文件和原始数据继续保存在本地 D 盘输出目录，不上传 GitHub。

## 当前保留结果

- `femba_severity_boosted_development_v1/development/summary/`：城市巡航高低眩晕严重度路径级结果。
- `femba_vrq_boosted_development_v1/development/summary/`：VRQ 被试级高低眩晕严重度和状态二分类结果。
- `femba_vrq_segment_severity_development_v1/development/summary/`：VRQ 五秒片段级高低眩晕严重度结果。

本目录不再归档模拟飞行（monifeixing）或其它历史试验的结果汇总。

所有结果均保留原协议中的验证集、seed、数据划分和结论边界；汇总文件不能替代完整 checkpoint 或 outer-test 产物。
