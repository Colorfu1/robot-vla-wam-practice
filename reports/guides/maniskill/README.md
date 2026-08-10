# ManiSkill 实验指南

当前实验族从阶段零的 M0 开始。M0 只负责验证 PickCube-v1 的官方演示、replay、LeRobotDataset、
Diffusion 训练和原生 ManiSkill 闭环评测管路；它不是 Mini-WAM 或 Residual RL 实验。

## 指南导航

1. [M0 PickCube 端到端探针](00_m0_pickcube.md)：RL 原生 7D 主线、replay/闭环 rollout 可视化、Full/Trim 数据和三个 system 闭环参照
2. [PickCube 4D 论文参照基线](01_paper_aligned_pickcube_4d.md)：motion-planning、CPU、horizon 100 与官方 Table III 口径

实际运行结果在远端产生后，统一登记到 [实验运行记录](../../experiment_log.md)。M0 的上游路线见
[Mini-WAM 与残差 RL 三阶段实验计划](../../../plans/2026-08-06-Mini-WAM与残差RL三阶段实验计划.md)。
