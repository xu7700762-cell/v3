# FEMBA/DoG 实验结果归档

本目录只保存可进入版本控制的实验汇总：`RESULTS.md`、CSV 指标表和 JSON 协议汇总。训练 checkpoint、逐样本缓存、预测中间文件和原始数据继续保存在本地 D 盘输出目录，不上传 GitHub。

## 主要结果

- `femba_severity_boosted_development_v1/development/summary/`：城市巡航严重度路径级开发实验。
- `femba_vrq_segment_severity_development_v1/development/summary/`：VRQ 五秒片段级严重度开发实验。
- `femba_c0_head_comparison_v1/full/seed_2001/summary/`：VRQ、城市巡航 C0 下游头比较。
- `femba_vrq_boosted_development_v1/development/summary/`：VRQ 被试级开发对照。
- `femba_reference_free_token_pilot_v1/pilot/seed_2001/summary/`：无参考 token 下游小试。
- `femba_severity_dynamics_pilot_v1/pilot/seed_2001/summary/`：严重度动态小试。
- `femba_ssl_fivefold_v3/full/seed_2001/summary/`：FEMBA 自监督五折配对汇总。

所有结果均保留原协议中的验证集、seed、数据划分和结论边界；汇总文件不能替代完整 checkpoint 或 outer-test 产物。
