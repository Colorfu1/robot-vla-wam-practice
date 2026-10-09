# 报告索引

`reports/` 按文档用途组织，实验操作指南与按日期记录的进展日志不再混放。

## 全局文档

- [实验运行记录](experiment_log.md)：所有实际实验的最小摘要与结果入口
- [指标说明](metrics.md)：通用指标的含义、单位和解释边界
- [写作规则](AGENTS.md)：本目录及其子目录必须遵守的文档规范

## 实验指南

- [FastWAM Square原生双视角与32步预测训练](guides/rl/fastwam_square_dual.md)

- [DM0.5 本地 LIBERO 推理与闭环验证](guides/dm05/README.md)
  · [模型结构与张量维度](guides/dm05/01_dm05_model_architecture.md)

- [实验指南总览](guides/README.md)
- [ACT 实验指南](guides/act/README.md)
- [PushT Diffusion Pilot](guides/pusht_diffusion/README.md)
- [SmolVLA LIBERO LoRA 与语言增强（目标协议已完成；C3 background retention 待补）](guides/smolvla_libero/README.md)
- [ManiSkill M0 PickCube、M1-W、M1-R 与 Peg/MS2 诊断指南](guides/maniskill/README.md)
- [文献模型复现实验指南](guides/literature/README.md)
- [RL 实验指南](guides/rl/README.md)

实验指南负责说明实验目标、实验设计、完整命令、结果落点和判断标准。实际运行后，
还需要在 [实验运行记录](experiment_log.md) 中补充对应条目。

## 每日进展

- [2026-10-09 FastWAM抓取后重规划诊断结论与复现步骤](daily/2026-10-09-FastWAM抓取后重规划诊断结论与复现步骤.md)
- [2026-10-08 FastWAM历史结果复核与执行窗口分析](daily/2026-10-08-FastWAM历史结果复核与执行窗口分析.md)

- [2026-09-23 Square图像转换EGL退出清理修复](daily/2026-09-23-Square图像转换EGL退出清理修复.md)

- [2026-09-21 DM0.5在Square上的零样本推理](daily/2026-09-21-DM0.5在Square上的零样本推理.md)

- [2026-09-21 DM0.5本地推理与LIBERO闭环验证](daily/2026-09-21-DM0.5本地推理与LIBERO闭环验证.md)

[每日进展索引](daily/README.md) 按日期记录阶段性结论、实现进展、验证结果、限制和下一步。
每日进展不能替代实验指南，也不能把尚未运行的计划写成实验结论。

- [2026-07-27 PushT BCChunk 与 Diffusion 模型规划和训练](daily/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练.md)
- [2026-07-29 PushT Benchmark 阶段性结论](daily/2026-07-29-PushT-Benchmark阶段性结论.md)
- [2026-07-29 PushT 实验进展总结](daily/2026-07-29-PushT实验进展总结.md)
- [2026-07-29 PushT E2 闭环采样结果](daily/2026-07-29-PushT-E2闭环采样结果.md)
- [2026-07-30 PushT E5 原型条件实验阶段性成果](daily/2026-07-30-PushT-E5原型条件实验实现.md)
- [2026-07-31 PushT 路线条件与 SmolVLA LIBERO 训练基建](daily/2026-07-31-PushT路线条件与SmolVLA-LIBERO训练基建.md)

- [2026-08-06 Prompt Swap 修复后合并结果](2026-08-06-PromptSwap修复后合并结果.md)
- [2026-08-06 C1 学习率缩放遗忘权衡诊断结论](2026-08-06-C1学习率缩放遗忘权衡诊断结论.md)
- [2026-08-06 C1 遗忘诊断与背景数据回放结论](2026-08-06-C1遗忘诊断与背景数据回放结论.md)
- [2026-08-07 C3 改写增强 LoRA 正式评测结论](daily/2026-08-07-C3改写增强LoRA正式评测结论.md)
- [2026-08-10 ManiSkill M0 收尾与 PickCube 论文参照](daily/2026-08-10-ManiSkill-M0收尾与PickCube论文参照.md)

## Square训练、评测与诊断记录

- [2026-09-18-DPPO历史版本与训练数据核验](2026-09-18-DPPO历史版本与训练数据核验.md)
- [2026-09-18-DPPO官方权重低成功率初步诊断](2026-09-18-DPPO官方权重低成功率初步诊断.md)
- [2026-09-20-DP与LightWAM训练场景600步对照](2026-09-20-DP与LightWAM训练场景600步对照.md)
- [2026-09-20-FastWAM同数据训练效率基准](2026-09-20-FastWAM同数据训练效率基准.md)
- [2026-09-20-LightWAM未来帧预测预览](2026-09-20-LightWAM未来帧预测预览.md)
- [2026-09-20-LightWAM训练场景配对评测入口](2026-09-20-LightWAM训练场景配对评测入口.md)
- [2026-09-20-Square多卡统一评测接入](2026-09-20-Square多卡统一评测接入.md)
- [2026-09-20-Square随机初始化600步统一评测](2026-09-20-Square随机初始化600步统一评测.md)
- [2026-09-21-FastWAM八卡吞吐与十二小时训练](2026-09-21-FastWAM八卡吞吐与十二小时训练.md)
- [2026-09-21-FastWAM十二小时训练结果与评测接入](2026-09-21-FastWAM十二小时训练结果与评测接入.md)
- [2026-09-21-FastWAM开环误差与未来视频诊断接入](2026-09-21-FastWAM开环误差与未来视频诊断接入.md)
- [2026-09-21-FastWAM成功失败全程预测可视化](2026-09-21-FastWAM成功失败全程预测可视化.md)
- [2026-09-21-FastWAM抓取关键帧定位](2026-09-21-FastWAM抓取关键帧定位.md)
- [2026-09-22-FastWAM三路未来预测对照](2026-09-22-FastWAM三路未来预测对照.md)
- [2026-09-22-FastWAM失败轨迹配对未来预测](2026-09-22-FastWAM失败轨迹配对未来预测.md)
- [2026-09-22-FastWAM示教中间状态接管实验](2026-09-22-FastWAM示教中间状态接管实验.md)
- [2026-09-22-FastWAM训练场景多卡失败选样](2026-09-22-FastWAM训练场景多卡失败选样.md)
- [2026-09-22-FastWAM训练场景逐步恢复证据](2026-09-22-FastWAM训练场景逐步恢复证据.md)
- [2026-09-22-Square分辨率与腕相机诊断](2026-09-22-Square分辨率与腕相机诊断.md)
