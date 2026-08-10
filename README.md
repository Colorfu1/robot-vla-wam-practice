# Robot VLA WAM Practice

这是一个以 LeRobot 为工作流层的机器人操作实验仓库。当前 README 只总结
`LIBERO`、`PushT`、`SmolVLA` 三条项目线中每个子任务的目的与结论。

完整实验命令、协议和产物位置见 [`reports/guides/`](reports/guides/README.md)，实际运行记录见
[`reports/experiment_log.md`](reports/experiment_log.md)。本页不把计划、Smoke 或诊断结果写成
Formal 结论。

> 说明：LIBERO 是 benchmark。下面的“LIBERO”章节指 ACT 在 LIBERO 上的实验；“SmolVLA”章节
> 是另一条在 LIBERO-Goal 上进行语言条件适配的 VLA 实验线，两者的结果不能互相替代。

## 三条线的当前结论

| 项目 | 目的 | 当前结论 |
| --- | --- | --- |
| LIBERO / ACT | 验证 ACT 的训练、动作 chunk、平滑控制和 LIBERO 闭环能力 | A1-VAE 达到 19/20（95%），A2 + Temporal Ensemble 达到 18/20（90%）；当前证据只覆盖单任务、单 seed |
| PushT | 研究 Diffusion 的随机闭环行为、采样效率和条件控制 | 完整 DDPM sampling 能产生多个成功闭环 mode；DDIM-5 是当前最佳速度配置，但固定 observation 多峰性、Prototype-ID 和 Route Field 语义控制均未建立 |
| SmolVLA | 验证 LIBERO 域适配、LoRA、paraphrase 和语言目标服从 | C3 selected `015000` 达到 canonical 88/90、held-out paraphrase 67/90、Prompt Swap 首次达成 30/30；background regression 为 0/210，而 B0 基线为 183/210，当前配置仍有严重 background forgetting |

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

## 证据边界

- Guide 页面负责完整命令、协议、结果落点和复现条件；本 README 只保留“目的—结论”摘要。
- `Formal`、`Closed-loop`、`Offline`、`Smoke` 和 `Diagnostic` 不混写；没有正式测量的字段不补填数字。
- 单任务、单 seed、参与 checkpoint selection 的 validation 结果，只能支持对应实验范围内的判断，不能写成跨任务、跨 seed 或模型家族结论。
- 计划仍放在 [`plans/`](plans/README.md)，实际结果仍放在 [`reports/`](reports/README.md)；数据、checkpoint、视频、Rerun recording 和日志等生成物不提交到 Git。

项目代码和文档使用 [Apache License 2.0](LICENSE)；第三方软件和数据集说明见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
