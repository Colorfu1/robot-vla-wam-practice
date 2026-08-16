# ManiSkill 实验指南

阶段零的 M0 已完成：它验证了 PickCube-v1 的官方演示、replay、LeRobotDataset、Diffusion 训练和原生
ManiSkill 闭环评测管路；它不是 Mini-WAM 或 Residual RL 实验。当前 M1-W 只保留 ManiSkill3
StackCube。CPU 和 GPU 都可独立运行，但不能跨平台混用状态/动作；M1-W 的 replay、数据生成和
rollout 统一走 GPU，因为 GPU 并行效率更高。PlugCharger 因 full replay survival 不足已排除。
M1-W 已关闭；Policy Decorator MS2/v2 M1-R 已完成 smoke，但按用户要求暂缓，待切换
`maniskill2-pidec` 环境后恢复。

## 指南导航

1. [M0 PickCube 端到端探针](00_m0_pickcube.md)：已完成的 RL 原生 7D/GPU 诊断线、数据问题、replay/闭环可视化与三个 system 参照
2. [PickCube 4D 论文参照基线](01_paper_aligned_pickcube_4d.md)：已完成的 motion-planning、CPU、horizon 100 与官方 Table III 口径复现
3. [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md)：ManiSkill3 StackCube 的 GPU live RGB 数据链；baseline、failed-only EE repair、final replay、state banks 与 frozen protocol 已完成
4. [M1-R Policy Decorator MS2/v2 协议预检](03_m1_r_protocol_audit.md)：独立 MS2/v2 identity、environment 与 50 条 smoke 已完成；full replay/state-bank 暂缓，执行计划见 [M1-R 计划](../../../plans/2026-08-13-M1-R-PolicyDecorator-MS2v2协议预检.md)
5. [M2-W StackCube Action-only Flow](04_m2_w_action_only_flow.md)：M1-W frozen protocol 上的 Action-only Flow smoke、50k 正式训练、development 选模与 baseline freeze 执行指南
6. [StackCube 训练样本可视化](05_stackcube_visualization.md)：复用 Rerun 组件查看双相机、Panda skeleton、动作/状态曲线与诊断场景 overlay，并随机抽样 10 个 train case
7. [M2 StackCube Diffusion Policy 一键训练](06_m2_diffusion_policy.md)：生成独立 RGB LeRobot 数据并调用原生 Diffusion Policy pipeline，避免改动既有数据和训练管路
8. [Peg P1–P3 历史诊断记录](../../experiment_log.md)：conversion、MS2 exact replay、controller contract 与精密接触分叉证据；后续作为 M1-R 输入
9. [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md)：官方 100-demo RGB Diffusion Policy 复现与 `success_once` 验证
10. [官方 StackCube RGB Golden Anchor 执行命令](08_official_stackcube_rgb_anchor_commands.md)：源数据同步、分阶段 replay/训练/验证与日志观察命令
11. [M2 base Action-only Flow 4D/7D 适配](09_m2_base_action_flow.md)：官方 RGB H5 适配、4D/7D head、懒读 H5 与 4 卡训练命令
12. [官方 Diffusion Policy 接入 M1-W 7D 数据](10_official_dp_m1w_7d.md)：官方 DP、M1-W live RGB/18D state、懒读 H5 与 torchrun 多卡训练入口
13. [官方 Diffusion Policy 18D/4D 对照实验](11_official_dp_18d4_ablation.md)：官方数据 25D→18D 与 M1-W 7D→4D 的同协议四卡对照
14. [M1-W 原生 4D 数据重生成](12_m1_w_native_4d_data.md)：从原始 joint action 官方转换到 4D，再 GPU replay、修复并实时采集 RGB

实际运行结果在远端产生后，统一登记到 [实验运行记录](../../experiment_log.md)。M0 的上游路线见
[Mini-WAM 与 Residual RL 完整实验总计划](../../../plans/2026-08-11-Mini-WAM与Residual-RL-完整实验总计划.md)。
