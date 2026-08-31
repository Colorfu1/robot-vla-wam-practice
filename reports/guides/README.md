# 实验指南索引

实验指南按实验族组织：

- [ACT](act/README.md)：A-Sanity、Action Chunking、动作差分损失、Temporal Ensemble、
  observation 等权和 VAE
- [PushT Diffusion Pilot](pusht_diffusion/README.md)：Smoke、正式 benchmark、固定输入采样、
  多 seed 闭环 rollout、E3 采样速度、E4 纯旋转场景和 E5 Prototype-ID 条件控制
- [SmolVLA LIBERO（目标协议已完成；C3 background retention 待补）](smolvla_libero/README.md)：LoRA 任务适配、
  paraphrase 语言增强和 Prompt Swap 语言服从评测
- [ManiSkill（M0 已完成，M1-W StackCube GPU 协议已冻结，M2-W 已实现）](maniskill/README.md)：PickCube-v1 的 RL 7D/GPU 诊断线、
  4D/CPU 论文参照，以及 StackCube 的 GPU 数据、state-bank、Action-only Flow、闭环 rollout 和 Rerun 可视化入口
- [文献模型复现](literature/README.md)：W0 Fast-WAM LIBERO artifact、LeRobot integration 与 `lambda_video=1/0` 因果锚点
- [RL 三阶段验证](rl/README.md)：Policy Decorator / ManiSkill2、Light-WAM residual SAC 与 RLinf GRPO/PPO

每份指南均应提供可直接复制的命令。实际结果统一登记到
[实验运行记录](../experiment_log.md)。

长时间训练、评测和 rollout 的命令还必须说明如何观察运行进度：前台显示进度条，或在
`outputs/` 下持续写入带 completed/total 计数的日志；不能只在结束时生成结果 JSON。
