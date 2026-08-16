# Robot VLA WAM Practice

这是一个以 LeRobot 为工作流层的机器人操作实验仓库。当前 README 总结
`LIBERO`、`PushT`、`SmolVLA`、`ManiSkill` 四条项目线中每个子任务的目的与结论。

完整实验命令、协议和产物位置见 [`reports/guides/`](reports/guides/README.md)，实际运行记录见
[`reports/experiment_log.md`](reports/experiment_log.md)。本页不把计划、Smoke 或诊断结果写成
Formal 结论。

> 说明：LIBERO 是 benchmark。下面的“LIBERO”章节指 ACT 在 LIBERO 上的实验；“SmolVLA”章节
> 是另一条在 LIBERO-Goal 上进行语言条件适配的 VLA 实验线，两者的结果不能互相替代。

## 四条线的当前结论

| 项目 | 目的 | 当前结论 |
| --- | --- | --- |
| LIBERO / ACT | 验证 ACT 的训练、动作 chunk、平滑控制和 LIBERO 闭环能力 | A1-VAE 达到 19/20（95%），A2 + Temporal Ensemble 达到 18/20（90%）；当前证据只覆盖单任务、单 seed |
| PushT | 研究 Diffusion 的随机闭环行为、采样效率和条件控制 | 完整 DDPM sampling 能产生多个成功闭环 mode；DDIM-5 是当前最佳速度配置，但固定 observation 多峰性、Prototype-ID 和 Route Field 语义控制均未建立 |
| SmolVLA | 验证 LIBERO 域适配、LoRA、paraphrase 和语言目标服从 | C3 selected `015000` 达到 canonical 88/90、held-out paraphrase 67/90、Prompt Swap 首次达成 30/30；background regression 为 0/210，而 B0 基线为 183/210，当前配置仍有严重 background forgetting |
| ManiSkill | 验证 PickCube/StackCube 的数据、控制协议、Action-only Flow 和官方 Diffusion Policy 参照 | M0 PickCube 的 4D/CPU 固定 holdout 为 397/400（99.25%）；StackCube 主表按数据来源和 rollout contract 分组记录：官方 4D 默认 57D state 的 DP 在 100k 为 52/100 once、47/100 end；M2-W 50k 为 43/200、43/200；官方 DP + M1-W 7D 50k 为 27/200、0/200。不同组不组成单一排行榜 |

## 1. LIBERO：ACT 子任务总结

固定闭环对照主要使用 `libero_object/task 8`、20 个初始状态、10 Hz；除特别注明外，结果为
单个 training seed。ACT 的完整协议见 [ACT 实验指南](reports/guides/act/README.md)。

| 子任务 | 目的 | 结论 |
| --- | --- | --- |
| [A-Sanity](reports/guides/act/act_a_sanity.md) | 确认 ACT 能在少量 demonstration 上完成训练、离线预测和 LIBERO 闭环执行 | **已完成。** 20k 模型达到 18/20（90%）闭环成功；训练管路已打通，但没有保留 zero-update checkpoint 和完整 loss 日志，不能作为严格 B0→20k 因果对照 |
| [A0/A1 Action Chunking](reports/guides/act/act_action_chunking.md) | 在相同数据和训练预算下，比较单步预测与 action chunk 是否能保持成功率并改善动作连续性 | **已完成正式对照，但结果不支持预设假设。** 同数据、同 seed、同 20k updates、同 `n_action_steps=1` 重规划下，A0 为 9/20（45%），A1 为 4/20（20%）；A1 的 offline first-action MAE 降低 17.2%，但闭环 Action Jerk 上升 26.7%。A1 默认 `n_action_steps=5` 时为 3/20（15%）。单任务、单 seed，且未保留 step-0 checkpoint 和完整 loss 曲线 |
| [A2 Delta-action Loss](reports/guides/act/act_delta_action_loss.md) | 用相邻动作差分损失改善动作拟合、动作平滑性和闭环稳定性 | **支持该变体有效。** 在当前任务和 seed 上，success 从 A1 的 15% 提升到 70%，Action Jerk 降低 20.3%；没有 shared step-0 checkpoint 且只有一个 seed，不能外推为模型家族结论 |
| [A3 Temporal Ensemble](reports/guides/act/act_temporal_ensemble.md) | 用历史 action chunk 集成缓解每步重规划抖动并提高闭环成功率 | **支持组合配置有效。** A1 replan 从 4/20 提升到 18/20，A2 standard 从 14/20 提升到 18/20，Jerk 显著下降；A2 的无 Ensemble replan control 尚未补跑，不能隔离 Ensemble 的纯因果贡献 |
| A1-ObsEqual（[指南](reports/guides/act/act_observation_equal_and_vae.md)） | 让 episode 尾部 observation 获得与其他 observation 相同的监督权重 | **支持末尾监督不足是假设来源。** success 从 15% 提升到 45%，动作连续性没有退化；证据仍限于单任务、单 seed 和 20 个初始状态 |
| A1-VAE（[指南](reports/guides/act/act_observation_equal_and_vae.md)） | 用 ACT VAE 建模 action chunk 的未来不确定性 | **当前 A1 家族最佳 clean 闭环配置。** success 达到 19/20（95%），TV、Jerk 和 gripper toggles 均优于 A1；尚未完成 offline evaluation，不能补写 open-loop 误差 |

## 2. PushT：Diffusion 子任务总结

固定数据为 206 条 episode：165 train、20 development、21 held-out audit；模型使用两帧
observation、16 步内部 horizon，实际每次执行 8 步。完整协议见
[PushT Diffusion Guide](reports/guides/pusht_diffusion/README.md)。

| 子任务 | 目的 | 结论 |
| --- | --- | --- |
| [Baseline / checkpoint selection](reports/guides/pusht_diffusion/benchmark.md) | 公平训练 BCChunk 与 Diffusion，并只用 development seed 选择后续实验使用的 checkpoint | 选中 BCChunk-150k（development 11/20）和 Diffusion-175k（16/20）；Diffusion 在该 selection workload 上更强，但单 training seed 且评测参与选模，不能作为独立 final 泛化结论 |
| [E0 Demonstration 审查](reports/guides/pusht_diffusion/e0_data_inspection.md) | 审查 206 条轨迹的几何、接近方向、旋转和重新接触行为，为固定 observation 实验选候选 | 识别出 33 个全局策略组，证明数据存在全局轨迹多样性；但全局分组不等于同一 observation 下的 action 多模态，E0 不能单独证明这一点 |
| [E1 Fixed-observation Sampling](reports/guides/pusht_diffusion/e1_fixed_observation.md) | 在完全相同的 observation 上分离 initial noise 与完整 DDPM reverse sampling 对 action chunk 的影响 | 完整 reverse sampling stream 会改变 action chunk，但 20 samples 的 GMM/BIC 与聚类判据均为 0/10 通过；未检测到固定 observation 下的多 mode，不能声称已克服 MSE mode averaging |
| [E2 Multi-seed Rollout](reports/guides/pusht_diffusion/e2_multi_seed_rollout.md) | 检查 sampling seed 是否能在相同初始环境中产生不同的成功闭环轨迹 | E2b 中 16/20 个环境出现至少两个成功 mode，`any@10=90%`；支持“Diffusion 有多个成功闭环行为 mode”，但不能证明固定 `p(a|o)` 本身是多峰分布 |
| [E3 Sampling Speed](reports/guides/pusht_diffusion/e3_sampling_speed.md) | 比较 DDIM/DDPM 去噪步数、闭环成功率和 end-to-end latency | **DDIM-5 是当前最佳部署配置。** success 为 29/50（58%）、end-to-end p95 为 47.7 ms；相对 DDPM-100 的 60% 和 600.1 ms 加速 12.57×，但不能宣称严格等价或完成 non-inferiority 证明 |
| [E4 Rotation / Contact OOD](reports/guides/pusht_diffusion/e4_rotation_only.md) | 检查未适配的 Diffusion policy 是否能处理同中心纯旋转和不同初始接触几何 | 纯旋转 0/5，三种 contact offset 各 0/5；结果支持接触几何敏感，不支持固定 observation 下已有可用的左右旋转多模态；没有训练 E4 数据，不能评价 E4 训练能力 |
| [E5 Prototype-ID](reports/guides/pusht_diffusion/e5_mode_conditioned.md) | 用 trajectory prototype 作为显式 condition，验证模型能否按请求选择不同完整行为 | condition ID 确实改变 action chunk，但 requested→achieved prototype 矩阵不是对角，Mode-Diffusion P0/P1/P2 adherence 仅 14.3%/28.6%/19.0%；任务能力保留，但没有建立可靠 semantic prototype control |
| [E6 / RF7 Route Field](reports/guides/pusht_diffusion/e6_route_field_7x7.md) | 用 7×7 route field 显式控制路线的占用、进度和方向 | condition 被模型读到，但没有被当作可执行顺序指令；GT route oracle 仅 1/21，reversed/permuted 服从度为 0。无同状态配对监督、route family 稀疏是主要根因候选，负结论的可判定样本仍偏少 |

## 3. SmolVLA：LIBERO-Goal 子任务总结

SmolVLA 线固定使用 LIBERO-Goal 目标任务 1/4/8、两路 256×256 RGB、8 维 state、7 维相对
末端动作、10 Hz；目标训练集为 113 条 demonstration，正式目标协议为 90 条 canonical、
90 条 held-out paraphrase 和 30 条 Prompt Swap。

| 子任务 | 目的 | 结论 |
| --- | --- | --- |
| [准备与 Smoke](reports/guides/smolvla_libero/00_preparation.md) | 验证 LeRobot、LIBERO、SmolVLA、PEFT、双相机、state/action processor 和闭环 evaluator 能连通 | 环境、CUDA/BF16、C0 完整 episode Smoke、本地 PEFT 加载和首动作检查通过；instrumented LoRA Smoke 尚未闭合，因此 Phase 0 不能写成全部完成 |
| [数据、provenance 与 split](reports/guides/smolvla_libero/01_data_preparation.md) | 冻结 B0 背景 split、目标 Train/Validation/Test split 和 paraphrase 语料，并验证数据不泄漏 | 143/143 个目标 episode 唯一映射到 raw demonstration，action 误差全为 0，目标 Train/Validation 和 closed-loop split 互斥；数据契约成立，但这不等于模型结果成立 |
| [B0 Domain Adaptation](reports/guides/smolvla_libero/02_b0_domain_adaptation.md) | 用 37 个非目标任务把 SmolVLA 适配到 LIBERO 域，并保留背景任务能力 | selected `025000` 达到 background validation 121/140、held-out test 183/210（87.1%）；B0 域适配有效，但 held-out 只针对未见 init state，不是跨任务泛化 |
| Z0 Target Zero-shot（[统一评测](reports/guides/smolvla_libero/06_evaluation.md)） | 直接测试 B0 对三个目标任务的零样本迁移能力 | canonical 0/90、held-out paraphrase 2/90、Prompt Swap 0/30；当前设置下目标任务 adaptation 是必要的 |
| [C1 Default Non-PEFT](reports/guides/smolvla_libero/03_c1_non_peft.md) | 从同一个 B0 出发用目标 canonical demonstration 做普通参数适配 | canonical 87/90、held-out paraphrase 30/90、Prompt Swap 29/30，但 background 0/210；C1 学会了目标任务，却出现严重语言泛化缺口和 catastrophic forgetting |
| C1 scaled-LR diagnostic（[报告](reports/2026-08-06-C1学习率缩放遗忘权衡诊断结论.md)） | 检查按 global batch 缩放学习率能否缓解 C1 遗忘 | 最佳 step 10k 为目标 58/60、background 55/210；只能部分缓解，远低于 B0 的 183/210，且不替换正式 C1 |
| C1 background replay diagnostic（[报告](reports/2026-08-06-C1遗忘诊断与背景数据回放结论.md)） | 检查在目标训练中加入 background replay 是否比单独降 LR 更能保留旧能力 | `k=4/8` selected step 20k 为目标 59/60、background 149/210；replay 比降 LR 有效，但训练已见 background，不能等同纯 retention，`k=2/8` 和 `k=6/8` 尚未执行 |
| [C2 Canonical LoRA](reports/guides/smolvla_libero/04_c2_canonical_lora.md) | 在同一 B0 和 canonical 数据上，用 rank-32 LoRA 替代 C1 的 Non-PEFT 适配 | canonical 87/90 与 C1 打平，held-out paraphrase 降到 19/90，background 仅 8/210；LoRA 没有改善语言泛化，也没有防住遗忘 |
| [C3 LoRA + Paraphrase](reports/guides/smolvla_libero/05_c3_paraphrase_lora.md) | 用 25% canonical + 75% train paraphrase 改善未见语言表达下的目标执行 | selected `015000` 达到 canonical 88/90、held-out paraphrase 67/90、Prompt Swap 首次 30/30；background regression 为 0/210（B0 为 183/210），目标语言泛化改善但 background forgetting 仍未解决 |
| [Prompt Swap 口径修复](reports/2026-08-06-PromptSwap修复后合并结果.md) | 修正目标首次达成后 reset 导致的错误判定，区分首次达成和 terminal hold | 修复后应只引用 first-goal：Z0 0/30、C1 29/30、C2 28/30、C3 30/30；C3 terminal hold 29/30，说明达成后的保持仍有一条失败 |

## 4. ManiSkill：按阶段的实测结果

ManiSkill 线使用独立的 data provenance、controller、sim backend 和 rollout contract。完整命令、
产物位置和设备边界见 [ManiSkill 实验指南](reports/guides/maniskill/README.md)。下面的结果只引用
已经落盘的训练或评测产物；`Smoke`、协议 replay 和训练集 probe 不冒充独立泛化结果。

StackCube 的 GPU state-bank 结果统一是 development 200 states、最多 500 steps、`physx_cuda`；
官方数据的 CPU 结果是 `physx_cpu`，除 RGB anchor 特别注明外为 200 episodes、最多 200 steps。
官方 DP 的 native-4D/7D GPU exact-state 评测中 200 条轨迹均跑到 truncation，因此
`success_at_end=0`；主比较指标使用 `success_once`，不能把它误读成“从未完成过任务”。M2-W
使用自己的 evaluator，当前最佳 development candidate 为 step 50k、`n_action_steps=8`，
`success_once/success_at_end=43/200`。

### 4.1 数据与协议阶段

| 子任务 | 实测结果 | 结论 |
| --- | --- | --- |
| [M0 PickCube RL 7D](reports/guides/maniskill/00_m0_pickcube.md) | 1,013 demos、50,360 frames；20k checkpoint development 为 120/200（60.0%），当时独立 test 为 234/400（58.5%）；50k 在同一 test block 上 sweep 的峰值为 340/400（85.0%） | 7D/GPU 学生管路确实能学会，但 50k test sweep 已污染该 test block，只能作诊断，不能再用于选模或正式泛化结论 |
| [M0 PickCube 4D 论文参照](reports/guides/maniskill/01_paper_aligned_pickcube_4d.md) | motion-planning、4D `pd_ee_delta_pos`、CPU、horizon 100；1,000-demo 固定 holdout 为 397/400（99.25%），100-demo 为 206/400（51.5%） | LeRobot diffusion、CPU rollout 和官方 success 语义已达到可用基线；不能与 M0 的 7D/GPU 结果直接相减 |
| [M1-W StackCube 数据协议](reports/guides/maniskill/02_m1_w_protocol_audit.md) | GPU-native 7D source 1,000 条；baseline replay 805/1,000，failed-only repair 恢复 182/195；final live-RGB `D_core` 为 987 episodes、106,528 action frames，最终 replay 987/987 success-at-end；state bank train/development/test 为 2,000/200/400，split leakage 为 0 | **数据链已完成并冻结。** RGB、18D proprioception、7D action、env state、fingerprint 和 frozen protocol gate 均通过；M1-W 本身是数据/协议结果，不是 learned-policy 成绩 |
| [M1-W 原生 4D 数据](reports/guides/maniskill/12_m1_w_native_4d_data.md) | 官方 CPU joint→4D conversion 1,000/1,000；GPU native-4D baseline 760/1,000；repair 的 success-at-end 为 186/240（success-once 213/240）；accepted merge 946 条；最终实时 RGB replay 946/946 success-at-end | 4D 数据是重新转换并在 GPU controller 下验证过的独立数据链，不是从 7D 标签切片；它足以进入模型对照，但不能单独证明“动作维度”是唯一因果因素 |
| M1-R / Plug 历史诊断 | Policy Decorator MS2/v2 exact-action smoke 为 50/50，但 full replay 尚未执行；Plug actual-EEF retarget 在 416 条失败集上仅恢复 21/416（5.05%） | M1-R 仍是 smoke，Plug 已从 M1-W 正式数据线排除；这些结果不进入 StackCube learned-policy 对照 |

### 4.2 StackCube 模型对照子表

这是位于 M1-W 数据协议之后的总表；`official 模型复现` 作为其中一个阶段分组出现，不再单独
重复一张表。完整 official 协议、命令和 legacy 100-demo runner 说明见
[官方 StackCube RGB Golden Anchor 指南](reports/guides/maniskill/07_official_stackcube_rgb_anchor.md)。
表格采用“协议/数据阶段 → 模型 + 数据 → 训练与闭环结果”的层级记录，第一列按阶段合并，避免
把不同数据来源和 rollout contract 看成一条无条件排行榜。

<table>
<thead>
<tr>
<th>协议 / 数据阶段</th>
<th>模型 + 数据</th>
<th>训练与闭环结果</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="3">M1-W StackCube<br>数据协议</th>
<td><a href="reports/guides/maniskill/04_m2_w_action_only_flow.md">M2-W Action-only Flow + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>success_once / success_at_end = 43/200 / 43/200</code>（21.5% / 21.5%）。candidate 为 step 50k、<code>n_action_steps=8</code>；test 未访问。</td>
</tr>
<tr>
<td><a href="reports/guides/maniskill/06_m2_diffusion_policy.md">LeRobot Diffusion Policy + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>success_once / success_at_end = 19/200 / 19/200</code>（9.5% / 9.5%）。</td>
</tr>
<tr>
<td><a href="reports/guides/maniskill/10_official_dp_m1w_7d.md">official Diffusion Policy + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>success_once / success_at_end = 27/200 / 0/200</code>（13.5% / 0%）。train loss <code>0.0021</code>。</td>
</tr>
<tr>
<th rowspan="3">official 模型复现</th>
<td><a href="reports/guides/maniskill/07_official_stackcube_rgb_anchor.md">official Diffusion Policy + 官方 motion-planning 4D replay（单卡主 anchor）</a></td>
<td><strong>50k / 100k steps</strong>；官方默认 <code>agent + extra</code>，实际 57D state；CPU verification 100 episodes / horizon 200：50k 为 <code>34/100 / 33/100</code>（34.0% / 33.0%），100k 为 <code>52/100 / 47/100</code>（52.0% / 47.0%）。</td>
</tr>
<tr>
<td><a href="reports/guides/maniskill/07_official_stackcube_rgb_anchor.md">official Diffusion Policy + 同一官方 motion-planning 4D replay（4-GPU DDP）</a></td>
<td><strong>同一 990-demo 数据、100k steps / GPU 4,5,6,7</strong>；同一默认 57D state、4D action 和 CPU verification contract；global batch <code>256</code>；<code>success_once / success_at_end = 48/100 / 45/100</code>（48.0% / 45.0%）。仅训练并行方式不同，不替代单卡 100k anchor。</td>
</tr>
<tr>
<td><a href="reports/guides/maniskill/11_official_dp_18d4_ablation.md">official Diffusion Policy + 同一官方 motion-planning 4D replay + 显式 18D state</a></td>
<td><strong>同一官方 motion-planning 4D replay 来源；state 改为显式 18D qpos+qvel；本 run 50k steps、787 demos</strong>；CPU development 200 states / horizon 200；<code>success_once / success_at_end = 67/200 / 54/200</code>（33.5% / 27.0%）。train loss <code>0.00109</code>。数据来源相同，但 demos、state extractor 和 eval contract 不同，因此不是严格单变量 ablation。</td>
</tr>
<tr>
<th rowspan="2">M1-W 原生 4D<br>数据</th>
<td><a href="reports/guides/maniskill/12_m1_w_native_4d_data.md">official Diffusion Policy + M1-W 原生 4D 数据</a></td>
<td><strong>50k / 100k steps</strong>；GPU development 200 states / horizon 500：50k 为 <code>100/200 / 0/200</code>（50.0% / 0%），100k 为 <code>103/200 / 0/200</code>（51.5% / 0%）。它不是官方 motion-planning 4D replay。</td>
</tr>
<tr>
<td><a href="reports/guides/maniskill/09_m2_base_action_flow.md">M2-W Action-only Flow + M1-W 原生 4D 数据</a></td>
<td><strong>100k steps / 4-GPU</strong>；global batch <code>256</code>；step 100k、<code>n_action_steps=4</code> 被选中；train mean tail loss <code>0.02532</code>。GPU development：<code>83/95 / 83/95</code>（87.4% / 87.4%）；post-selection test：<code>82/96 / 82/96</code>（85.4% / 85.4%）。<a href="outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu/selected_checkpoint.json">selected checkpoint</a> · <a href="outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/test_gpu/metrics.json">test metrics</a></td>
</tr>
</tbody>
</table>

### 4.2.1 尚未形成正式结果的适配

| 实验 | 训练 budget | 数据 / eval | success_once / success_at_end | 为什么单列 |
| --- | ---: | --- | ---: | --- |
| [M2 base 官方 RGB 4D 适配](reports/guides/maniskill/09_m2_base_action_flow.md) | — | 4D/7D head、H5 lazy loader 和 config/smoke gate；不含已完成的 M1-W native-4D 100k route | N/A | 官方 RGB 4D 这条通用适配入口尚未形成正式训练+选模结果；M1-W native-4D 结果已在上方主表单列 |

### 4.2.2 表格的比较规则

- 先在 `M1-W StackCube 数据协议` 组内比较同一 GPU 7D 数据上的 M2-W、LeRobot DP 和 official DP。
- 在 `official 模型复现` 组内比较官方 4D/57D 单卡 50k→100k；4-GPU DDP 只作为同一 100k
  配置的工程变体，不与单卡预算曲线合并。
- `M1-W 原生 4D 数据` 组用于区分数据 provenance；它不能直接替代 official motion-planning 4D
  replay，也不能把 native-4D 的结果解释成动作维度的单变量结论。
- 每个结果单元格必须同时能追溯到“训练 steps + eval backend + split + episodes + horizon”；
  缺任一关键字段时写 `N/A` 或“未完成”，不从训练内 monitor 推断独立泛化结果。

### 4.3 ManiSkill 当前结论

- M0 已证明数据读取、Diffusion 训练、可视化和闭环评测管路可用；PickCube 的 4D/CPU 参照达到
  99.25%，所以后续 StackCube 的低分不是“整个 LeRobot diffusion 管路坏了”。
- M1-W 的 `D_core` 和 GPU live-RGB 数据协议已经通过；在同一 GPU 7D 数据上，M2-W 50k 为
  21.5%，LeRobot DP 50k 为 9.5%，官方 DP 7D 50k 为 13.5%。这说明模型/训练入口和泛化都需要
  继续追查，不能把差值归因给单一动作维度。
- 官方 4D CPU anchor 从 34%（50k）提升到 52%（100k），说明 StackCube 与官方 DP 都有继续
  训练的 headroom；M2-W 50k 的 21.5% 则主要暴露 train→development 泛化问题，而不是优化器没有下降。
- M2-W 在 M1-W 原生 4D 数据上完成的 100k route 为 development `83/95`（87.4%），post-selection
  test `82/96`（85.4%）；它明显高于当前 M2-W 7D 的 50k 结果，但训练预算、数据 provenance、
  action/control contract 和 rollout split 同时变化，不能把提升单独归因于 action 从 7D 变成 4D。
- 官方 18D/4D 50k 为 33.5% once、27.0% end；它与官方 57D/4D anchor 接近，但当前仍受
  demo 数、eval 数量和选模流程差异影响，不能视作已经完成的 57D→18D 单变量消融。
- 当前可引用的 StackCube learned-policy 数字必须同时带上 backend、数据来源、state-bank/seed
  split、horizon 和 success 口径；表格必须先按比较组分层，再看结果，不能把 CPU official、GPU
  native4d、GPU M1-W 7D、M2-W 和 LeRobot DP 合并成一张无条件排名表。

## 证据边界

- Guide 页面负责完整命令、协议、结果落点和复现条件；本 README 只保留“目的—结论”摘要。
- `Formal`、`Closed-loop`、`Offline`、`Smoke` 和 `Diagnostic` 不混写；没有正式测量的字段不补填数字。
- 单任务、单 seed、参与 checkpoint selection 的 validation 结果，只能支持对应实验范围内的判断，不能写成跨任务、跨 seed 或模型家族结论。
- ManiSkill 的 official 数据必须使用 CPU rollout，M1-W/native4d/M2-W/LeRobot DP 数据必须使用 GPU rollout；转换 backend 和正式 rollout backend 不混写。当前 4D/7D 对照受 split 与转换路径影响，只能作为诊断证据。
- 计划仍放在 [`plans/`](plans/README.md)，实际结果仍放在 [`reports/`](reports/README.md)；数据、checkpoint、视频、Rerun recording 和日志等生成物不提交到 Git。

项目代码和文档使用 [Apache License 2.0](LICENSE)；第三方软件和数据集说明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
