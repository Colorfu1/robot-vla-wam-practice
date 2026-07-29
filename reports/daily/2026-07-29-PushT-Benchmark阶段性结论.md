# 2026-07-29 PushT Benchmark 阶段性结论

## 重要进展

- 完成 BCChunk 与 Diffusion 各 200,000 updates 的 `Formal` 训练；两者使用相同
  165-episode 训练集和成对 step-zero 初始化，共享的 213 个 tensor 已验证一致。
- 在 20 个 development environment seed 上完成 checkpoint selection：BCChunk
  选择 150k，闭环成功率 55%；Diffusion 选择 175k，闭环成功率 80%。
- 选中模型的逐 seed `Closed-loop / Formal` 对比中，Diffusion 比 BCChunk 高
  25 个百分点，但配对检验 `p=0.125`，当前只能描述为方向性优势。
- 完成全量 E0 `Offline / Formal` 审计：206 条 demonstration、25,650 帧、
  33 个自动分类的全局策略组；这些全局分组不能证明严格条件多模态。
- 完成 E1 双协议 `Offline / Formal` 评测：只改变初始 noise 时差异几乎为零，
  改变完整 DDPM sampling seed 后出现可复现的 action-chunk 多样性。
- 正式 E2、E3 和最终报告尚未运行，因此还不能下独立泛化、成功多模态或
  success-latency Pareto 的最终结论。

## 实验协议

| 字段 | 固定设置 |
|---|---|
| Task suite | PushT |
| Dataset size | 206 episodes；165 train / 20 development / 21 held-out audit |
| Observation keys | 96×96 RGB `observation.image` + 2D `observation.state` |
| Observation history | `n_obs_steps=2` |
| Action space | 2D absolute target position |
| Action horizon | 内部 `horizon=16`；执行切片 `[1:9]`，共 8 步 |
| Control frequency | 10 Hz |
| Train seed | 1000 |
| Checkpoint interval | 25,000 updates，共比较 8 个 checkpoint |
| Diffusion selection setting | DDPM，100 inference steps，单次采样 |

BCChunk 和 Diffusion 使用同一数据、预处理、视觉编码器、temporal U-Net 容量、
optimizer 和 scheduler。成对初始化清单显示 213 个共享 tensor 一致，因此主要受控
差异是 clean-action regression 与 epsilon prediction/反向去噪。两种训练 loss
语义不同，不能比较绝对大小。

## E0：数据审计

模式：`Offline / Formal`。

| 指标 | 结果 |
|---|---:|
| Demonstrations | 206 |
| Recorded frames | 25,650 |
| 自动分类的全局策略组 | 33 |
| Block geometry coverage | 100% |
| 显式正 success label | 0 |
| Terminal-only episodes | 206 |
| Mean episode max reward | 0.8923 |

E0 证明数据中存在多种完整轨迹形态，并且移动块几何解析覆盖全部帧；但全局轨迹
分组不等于“相同 observation 对应多个有效 action mode”。同时，数据集没有显式
正 success signal，206 条 episode 都只能由 terminal 和 dense reward 辅助解释。
因此 E0 不单独支持“数据已经证明严格条件多模态”的说法。

结果：
[E0 metrics.md](../../outputs/pusht_diffusion/data_inspection_e0_full/metrics.md)。

## 正式训练

模式：`Formal`。

| 模型 | Updates | 最后记录的 train loss | Loss 含义 |
|---|---:|---:|---|
| BCChunk | 200,000 | 0.000 | clean-action MSE，日志只保留三位小数，不能解释为精确零 |
| Diffusion | 200,000 | 0.004 | epsilon-prediction MSE |

两个训练任务都正常结束并保存 25k 到 200k 的 checkpoint。这里的 loss 只证明训练
目标已收敛到较低值；它既不能跨模型直接比较，也不能代替闭环模型选择。

## Development checkpoint selection

模式：`Closed-loop / Formal`，用途是选模，不是独立 final 评测。

| 模型 | 选中 step | Success | Mean max reward | Offline denormalized action MSE |
|---|---:|---:|---:|---:|
| BCChunk | 150k | 11/20，55% | 0.8652 | 582.31 |
| Diffusion | 175k | 16/20，80% | 0.8790 | 1169.87 |

选模曲线明显非单调：

- BCChunk 的最低离线 MSE 出现在 100k（568.41），但成功率只有 35%；150k 的
  MSE 略高，却达到 55%；200k 又回落到 35%。
- Diffusion 的最低离线 MSE 出现在 100k（640.68），成功率为 60%；175k 的
  MSE 更高，却达到 80%；200k 为 65%。

这说明 train loss 和 teacher-forced/offline MSE 都不能可靠替代 closed-loop
checkpoint selection。选模应继续以 success 为第一排序项，而不是默认使用最后
checkpoint 或最低离线误差。

结果：
[checkpoint selection metrics.md](../../outputs/pusht_diffusion/pilot/checkpoint_selection/metrics.md)
和
[`selection.parquet`](../../outputs/pusht_diffusion/pilot/checkpoint_selection/selection.parquet)。

## 选中 checkpoint 的逐 seed 对比

模式：`Closed-loop / Formal`，仍使用选模时的 20 个 development seed。

| 结果 | 数量 |
|---|---:|
| BCChunk 成功 | 11/20，55%；Wilson 95% CI 34.2%–74.2% |
| Diffusion 成功 | 16/20，80%；Wilson 95% CI 58.4%–91.9% |
| 仅 BCChunk 成功 | 1 |
| 仅 Diffusion 成功 | 6 |
| 两者都成功 | 10 |
| 两者都失败 | 3 |

Diffusion 的 success 高 25 个百分点，并在 6 个 seed 上恢复了 BCChunk 的失败，
而只在 1 个 seed 上发生相反结果。对 7 个 discordant pair 做 exact two-sided
McNemar 检验得到 `p=0.125`：方向对 Diffusion 有利，但样本数不足以称为统计显著。

更重要的限制是，这 20 个 seed 本身参与了 checkpoint selection。这个结果用于确认
选模复跑一致性和定位差异案例，不能作为独立 held-out 泛化估计。

结果：
[development comparison metrics.md](../../outputs/pusht_diffusion/pilot/development_comparison/metrics.md)
和
[`rollouts.parquet`](../../outputs/pusht_diffusion/pilot/development_comparison/rollouts.parquet)。

## E1：固定 observation 的双协议采样

模式：`Offline / Formal`。E1 使用 10 个 held-out observation candidate，每个协议
生成 20 个 Diffusion action chunk，不执行环境动作。

| 指标 | E1a：只改变 initial noise | E1b：改变完整 sampling seed |
|---|---:|---:|
| Denormalized trajectory pairwise RMS | 0.00163 px | 1.66488 px |
| Denormalized endpoint pairwise distance | 0.00213 px | 2.41515 px |
| 相同条件 repeat max error | 0 | 0 |

BCChunk 三次调用的最大误差也是 0。E1b 相对 E1a 的 trajectory RMS 约为 1024 倍，
endpoint distance 约为 1134 倍。

这组结果把随机性的来源区分开了：

- 只改变初始高斯 noise、固定 DDPM 反向随机路径时，输出几乎不变；
- 同时改变初始 noise 和每一步反向采样随机流时，模型会生成可复现的不同
  action patterns；
- 因此不能把 Diffusion 的随机性简单归因于“初始 noise”，完整 reverse sampling
  stream 才是当前观察到的主要多样性来源。

E1 仍是 teacher-forced offline 结果。约 1.7–2.4 像素的 action 差异是否足以形成不同且
成功的闭环轨迹，必须由 E2 验证；这里不能直接声称存在多种成功行为。

结果：
[E1 metrics.md](../../outputs/pusht_diffusion/pilot/e1_fixed_observation/metrics.md)、
[`action_chunks.parquet`](../../outputs/pusht_diffusion/pilot/e1_fixed_observation/action_chunks.parquet)
和
[`diversity_metrics.parquet`](../../outputs/pusht_diffusion/pilot/e1_fixed_observation/diversity_metrics.parquet)。

## 必填指标状态

| 字段 | 当前状态 |
|---|---|
| Train loss | 已测量；BCChunk 末值日志显示 0.000（三位小数），Diffusion 0.004；语义不同 |
| Open-loop action error | 已测量；选中 BCChunk/DP 的 denormalized MSE 为 582.31/1169.87 |
| Closed-loop success rate | development 为 55%/80%；不是独立 final 结果 |
| Latency | 未测量；等待 E3 |
| Failure cases | development 中仅 BC 成功 1、仅 DP 成功 6、两者失败 3；失败机制尚未标注 |

外部预训练参考得到 7/10 成功、mean max reward 0.9926，但其数据划分和 checkpoint
来源不同，只能作为兼容性参考，不能与当前 55%/80% 做公平排名。

## 当前可以得到的结论

1. **闭环选模是必要的。** 两种模型的最低 offline MSE、最后 checkpoint 和最高
   development success 都不重合。
2. **Diffusion-175k 在 development 上有明确方向性优势。** 它达到 80%，高于
   BCChunk-150k 的 55%，但 `p=0.125` 且复用了 selection set，尚不是最终证据。
3. **完整 DDPM 随机流确实带来 action-chunk 多样性。** 仅改变 initial noise
   几乎无效，改变完整 sampling seed 后差异扩大约三个数量级。
4. **当前不能把 success 优势归因于“成功多模态”。** E0 只显示全局轨迹多样性，
   E1 只显示离线动作多样性；是否产生多种成功闭环轨迹仍未验证。
5. **当前不能比较部署效率。** DDPM-100 的延迟和减少采样步数后的成功率都未测量。

## 下一步

1. 从头运行正式 E2a（2 环境 × 5 policy seed）和主实验 E2b（20 环境 × 10 policy
   seed，另含 20 条 BCChunk baseline），分开统计 success variance、trajectory
   diversity、successful mode count 和 any@10 oracle coverage。
2. 运行正式 E3：比较 DDIM 5/10/20/50/100 与 DDPM-100，报告 p95 end-to-end
   action-chunk latency 和 success-latency Pareto。
3. E2/E3 完成后生成最终报告，再决定“Diffusion 是否优于 BCChunk”以及优势来自
   随机多样性、训练目标还是其他结构差异。

## 相关文档

- [2026-07-27 PushT 规划和训练](2026-07-27-PushT-BCChunk与Diffusion模型规划和训练.md)
- [PushT Diffusion Pilot 指南](../guides/pusht_diffusion/README.md)
- [E1 双协议指南](../guides/pusht_diffusion/e1_fixed_observation.md)
- [E2 闭环多 seed 指南](../guides/pusht_diffusion/e2_multi_seed_rollout.md)
- [E3 采样速度指南](../guides/pusht_diffusion/e3_sampling_speed.md)
