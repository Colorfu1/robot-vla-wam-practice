# 报告索引

`reports/` 按文档用途组织，实验操作指南与按日期记录的进展日志不再混放。

## 全局文档

- [实验运行记录](experiment_log.md)：所有实际实验的最小摘要与结果入口
- [指标说明](metrics.md)：通用指标的含义、单位和解释边界
- [写作规则](AGENTS.md)：本目录及其子目录必须遵守的文档规范

## 实验指南

- [实验指南总览](guides/README.md)
- [ACT 实验指南](guides/act/README.md)
- [PushT Diffusion Pilot](guides/pusht_diffusion/README.md)

实验指南负责说明实验目标、实验设计、完整命令、结果落点和判断标准。实际运行后，
还需要在 [实验运行记录](experiment_log.md) 中补充对应条目。

## 每日进展

[每日进展索引](daily/README.md) 按日期记录阶段性结论、实现进展、验证结果、限制和下一步。
每日进展不能替代实验指南，也不能把尚未运行的计划写成实验结论。

- [2026-07-27 PushT BCChunk 与 Diffusion 模型规划和训练](daily/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练.md)
- [2026-07-29 PushT Benchmark 阶段性结论](daily/2026-07-29-PushT-Benchmark阶段性结论.md)
- [2026-07-29 PushT 实验进展总结](daily/2026-07-29-PushT实验进展总结.md)
