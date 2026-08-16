# 2026-08-10 ManiSkill M0 收尾与 PickCube 论文参照

## 重要进展

- M0 已完成：RL 原生 7D/GPU 与 motionplanning 4D/CPU 两条独立链路均通过 replay、LeRobot 转换、Diffusion 训练和闭环评测。
- RL 7D 演示 replay 为 1013/1013；训练/评测实际首帧在 `1e-4` 容差下无 exact duplicate，管路没有损坏。
- RL 数据存在 73.0% 成功后尾帧和显著动作超界，需要作为后续数据准入项处理，而不是继续默认它是唯一主线。
- 4D/CPU/100-step 的 1000-demo 基线在固定 holdout 为 397/400（99.25%），官方协议五次重复均值为 99.8%，达到论文量级。
- PickCube 已不适合作为 WAM 主实验；后续应先为 StackCube、PegInsertionSide 等任务建立任务专属基座和候选空间审计。

## 完成范围

结果类别：`Offline / Smoke / Formal / Closed-loop`。

M0 原目标是确认 ManiSkill 的官方演示能够经过 RGB replay、LeRobotDataset 和 LeRobot diffusion，最终在
ManiSkill 原生 API 上完成可靠闭环评测。该目标已经满足。后续的 4D 参照不是与 7D 数字直接比较的消融，
而是为区分“管路错误”与“数据/协议差异”而做的独立复现。

## RL 原生 7D/GPU 结果

| 项目 | 结果 |
| --- | --- |
| 数据 | 1013 demonstrations、50,360 帧；RGB + 21D `qpos+qvel+goal_pos`；7D `pd_ee_delta_pose`；20 Hz；horizon 50。 |
| Offline 数据审计 | 首次成功 median 14 steps；成功后尾帧 36,772/50,360（73.0%）；首次成功前仅 13,588 帧。 |
| Replay / 隔离 | 1013/1013 replay 写入；1008/1008 原生成功 episode 转为训练数据；600 个评测状态对 1013 个训练首帧为 0 条 exact duplicate。 |
| Smoke | 2k-step Full 为 0/50，Trim 为 1/50；差异 2 个百分点，不用该 smoke 作性能结论。 |
| Formal 20k | development winner `020000` 为 120/200（60.0%）；独立 test 为 visual 234/400（58.5%）、random 0/400、privileged PPO 396/400（99.0%）。 |
| 50k 诊断 | 同一 test block 的十 checkpoint sweep 峰值为 `045000` 的 340/400（85.0%），`050000` 为 339/400（84.75%）。 |

Train loss：训练 stdout 显示正常下降，但未保留跨 run 可比较的完整 loss series；M0 选模只使用闭环成功率。
Open-loop action error：未测量。Latency：20k development winner 的 policy forward p50/p95 为约
2179.6/2367.7 ms，environment step p50 为约 190.5 ms。失败主要是 timeout/truncation，20k visual
test 中 166/400 未成功，stage signal 中 grasp 为 72.0%、placed 为 62.75%。

### RL 数据限制

- 成功后尾帧占比高，Full 训练会大量学习已完成后的保持动作。
- 相对目标控制的 `[-1,1]` 范围，gripper 有 98.8%、`translation_z` 有 11.07% 的动作超界；今后复用 7D
  数据必须明确动作单位、归一化与限幅。
- PPO 教师是 42D 特权 state，视觉学生不含物体位姿和教师提供的相对向量；PPO 只能作参照，不是纯感知上限。
- RL `episode_seed` 不能独立复原每个 episode 的初始状态；GPU 路径应继续使用实际状态指纹做泄漏审计。
- 7D 旋转通道并未失活：旋转/平移动作绝对值中位数比为 7.73，超过预注册的 10% 门槛。

50k sweep 已反复读取原 `test` 状态，因此它是训练曲线诊断，不再是未触碰的最终泛化评测。后续若需要新的
7D 正式结论，必须创建新的 state block，并只在 development 选模后运行一次。

## motionplanning 4D/CPU 论文参照

| 项目 | 1000 demos | 100 demos |
| --- | ---: | ---: |
| 数据与协议 | motionplanning、4D `pd_ee_delta_pos`、`physx_cpu`、horizon 100、21D student state | 同左，按 episode ID 前 100 条训练 |
| Replay / 转换 | 1000/1000 成功；4D 动作越界 0%；成功前帧 93.14%；首次成功 p90 85 steps | 复用同一转换数据的前 100 条 |
| 官方 `official_best` | 100/100（100.0%） | 73/100（73.0%） |
| 五次官方重复 | 均值 99.8%，范围 99.0%–100.0% | 均值 57.8%，范围 48.0%–67.0% |
| 固定 CPU holdout | 397/400（99.25%） | 206/400（51.5%） |

同一固定 400 条 CPU holdout 的 random 为 0/400，privileged PPO 为 300/400（75.0%）。1000-demo
visual policy 的 policy-forward p50 为约 588.6 ms，environment-step p50 为约 6.2 ms；100-demo
并行 shard 合并后仅保留均值。该协议成功后不停机、始终运行 100 steps，因此 `success_at_end` 分别为
78.25% 与 23.0%，不能与 7D 成功即停结果直接比较。

官方论文 Table III 的相应数字为 1000 demos 1.00、100 demos 0.76。我们使用的是 LeRobot diffusion
而非官方 DP 实现，仍在 1000-demo 侧达到相同量级。`official_best` 与论文同为无固定 reset seed 的
max-over-checkpoints 口径；仓库内部应以固定 CPU holdout 作为隔离数值。

## 结论与后续边界

M0 可以标记为 `done`。它证明了管路、数据转换、闭环判定和可视化均可用，也证明此前 7D 低分不能归因为
LeRobot diffusion 管路根本错误。

不能把 4D 的 99.25% 与 7D 的 85.0% 解读为动作维度的因果比较：两者同时改变了专家来源、有效帧比例、
控制模式、后端、horizon 和终止语义。4D PickCube 已超过原计划的 95% 停止条件，不应进入 WAM reranking
主实验。下一步应先重写任务预检与候选空间审计：为 StackCube 和 PegInsertionSide 等候选建立各自的
数据/后端/状态划分合同及冻结视觉基座，再测其 `Oracle@K - first sample` headroom。

结果入口：

- `outputs/maniskill/m0_probe/`
- `outputs/maniskill/paper_aligned_4d/`
- `reports/guides/maniskill/00_m0_pickcube.md`
- `reports/guides/maniskill/01_paper_aligned_pickcube_4d.md`
