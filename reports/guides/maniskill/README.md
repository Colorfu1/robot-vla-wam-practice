# ManiSkill 实验指南

阶段零的 M0 已完成：它验证了 PickCube-v1 的官方演示、replay、LeRobotDataset、Diffusion 训练和原生
ManiSkill 闭环评测管路；它不是 Mini-WAM 或 Residual RL 实验。当前 M1-W 只保留 ManiSkill3
StackCube。CPU 和 GPU 都可独立运行，但不能跨平台混用状态/动作；M1-W 的 replay、数据生成和
rollout 统一走 GPU，因为 GPU 并行效率更高。PlugCharger 因 full replay survival 不足已排除。
M1-W 已关闭；Policy Decorator MS2/v2 M1-R 已完成 source/environment audit 和 50 条 smoke，但按用户要求
暂缓，待切换 `maniskill2-pidec` 环境后恢复。M1-R 是独立的 Residual 路线，不并入下面的 StackCube
模型对照表。

## 指南导航

1. [M0 PickCube 端到端探针](00_m0_pickcube.md)：已完成的 RL 原生 7D/GPU 诊断线、数据问题、replay/闭环可视化与三个 system 参照
2. [PickCube 4D 论文参照基线](01_paper_aligned_pickcube_4d.md)：已完成的 motion-planning、CPU、horizon 100 与官方 Table III 口径复现
3. [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md)：ManiSkill3 StackCube 的 GPU live RGB 数据链；baseline、failed-only EE repair、final replay、state banks 与 frozen protocol 已完成
4. [M1-R Policy Decorator MS2/v2 协议预检](03_m1_r_protocol_audit.md)：独立 MS2/v2 identity、environment 与 50 条 smoke 已完成；full replay/state-bank 暂缓，执行计划见 [M1-R 计划](../../../plans/2026-08-13-M1-R-PolicyDecorator-MS2v2协议预检.md)
5. [M2-W StackCube Action-only Flow](04_m2_w_action_only_flow.md)：M1-W frozen protocol 上的 Action-only Flow smoke、50k 正式训练、development 选模与 baseline freeze 执行指南
6. [StackCube 训练样本可视化](05_stackcube_visualization.md)：复用 Rerun 组件查看双相机、Panda skeleton、动作/状态曲线与诊断场景 overlay，并随机抽样 10 个 train case
7. [M2 StackCube Diffusion Policy 一键训练](06_m2_diffusion_policy.md)：生成独立 RGB LeRobot 数据并调用原生 Diffusion Policy pipeline，避免改动既有数据和训练管路
8. [Peg P1–P3 历史诊断记录](../../experiment_log.md)：conversion、MS2 exact replay、controller contract 与精密接触分叉证据；后续作为 M1-R 输入
9. [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md)：M1-W 协议之后的官方 990-demo RGB/4D Diffusion Policy 复现，含 50k/100k budget curve 与 `success_once` 验证
10. [官方 StackCube RGB Golden Anchor 执行命令](08_official_stackcube_rgb_anchor_commands.md)：源数据同步、分阶段 replay/训练/验证与日志观察命令
11. [M2 base Action-only Flow 4D/7D 适配](09_m2_base_action_flow.md)：官方 RGB H5 适配、4D/7D head、懒读 H5 与 4 卡训练命令
12. [官方 Diffusion Policy 接入 M1-W 7D 数据](10_official_dp_m1w_7d.md)：官方 DP、M1-W live RGB/18D state、懒读 H5 与 torchrun 多卡训练入口
13. [官方 Diffusion Policy 18D/4D 对照实验](11_official_dp_18d4_ablation.md)：官方数据 25D→18D 与 M1-W 7D→4D 的同协议四卡对照
14. [M1-W 原生 4D 数据重生成](12_m1_w_native_4d_data.md)：从原始 joint action 官方转换到 4D，再 GPU replay、修复并实时采集 RGB
15. [ManiSkill 复现实验与 M2-W 结果总览](13_reproduction_and_m2w_summary.md)：统一记录官方复现、M1-W 7D/原生 4D、LeRobot Diffusion Policy、M2-W 和当前 7D 诊断结论

## M1-R Policy Decorator MS2/v2 Residual 路线

M1-R 与 M1-W 是两条并行但不共享数据和环境的协议线：M1-W 固定 ManiSkill3 `StackCube-v1` 的
GPU live-RGB WAM 数据，M1-R 固定 Policy Decorator 原始 ManiSkill2/Peg contract，服务后续
M2-R、R0 和 R1。R2/R3 再迁移到 LIBERO；因此 M1-R 不属于 StackCube 模型横向比较，也不能用
M1-W 的 7D/4D 数据、controller、horizon、state bank 或 output root 替代。

### 重要进展

- M1-W 的 frozen protocol gate 已通过，满足 M1-R 的前置顺序要求。
- Policy Decorator 的 pinned checkout、H5/JSON identity、source schema 和 MS2 environment audit
  已通过；运行环境固定为 `maniskill2-pidec`，`mani-skill2==0.5.3`。
- 50 条 exact-replay smoke 已通过：native/strict 均 `50/50`，success-once 均 `50/50`，均在
  success 后 terminated、无 truncation，initial H5 state reconstruction 最大绝对误差为 `0`。
- 当前状态为 `blocked`（用户暂缓，不是已失败）：986 条 full replay、D_core、offline split、
  online-RL state banks 和 M1-R protocol freeze 尚未执行；因此尚无 M2-R、R0/R1 的 learned-policy
  success rate。

### 1. 固定协议

| 项目 | M1-R 冻结值 |
| --- | --- |
| task identity | dataset metadata 为 `PegInsertionSide-v0`；formal runtime 为 repo-local `PegInsertionSide-v2` |
| environment | 独立 `maniskill2-pidec`，`mani-skill2==0.5.3`；不与 ManiSkill3 同进程混用 |
| source/controller | Policy Decorator pinned checkout；native `pd_ee_delta_pose`，7D normalized action |
| observation | 50D privileged state：`qpos(9)+qvel(9)+base_pose(7)+tcp_pose(7)+peg_pose(7)+peg_half_size(3)+box_hole_pose(7)+radius(1)` |
| temporal contract | observation / prediction / action / execution = `2/16/4/4`；每次 decision 执行 4 个 low-level actions |
| episode contract | source/runtime TimeLimit 固定为 200；不能继承 MS3 Peg@300 |
| reward contract | reference metadata 未指定时记录 `normalized_dense`；formal replay/online residual 固定 `sparse` |
| formal data target | 986 episodes；D_core offline train/development/test 目标为 `786/100/100`，state banks 目标为 `2000/200/400`，均须通过 fingerprint leakage gate |

`PegInsertionSide-v0` 是 demonstrations 的原始 task ID，`v2` 只是 repo-local corrected runtime
environment，不能写成“v2 数据”。固定 checkout、H5/JSON SHA-256 和完整 audit contract 见
[M1-R 协议预检指南](03_m1_r_protocol_audit.md)。

### 2. R 路线阶段

```text
M1-R：MS2/v2 task-data-controller 协议冻结
  ↓
M2-R：state-conditioned Flow-Matching base + precision headroom
  ↓
R0：privileged-state chunk residual RL
  ↓
R1：同一 MS2/v2 contract 下的视觉 residual student
  ↓
R2 → R3：迁移到 LIBERO / frozen SmolVLA，并核对真实交互预算下的算法收益
```

| 阶段 | 作用 | 当前状态与结果口径 |
| --- | --- | --- |
| M1-R identity/source/environment audit | 固定 commit、H5/JSON、v0 metadata、v2 runtime、50D state、7D action、success/reward/termination 和 TimeLimit | 已完成；属于 `Offline` 协议审计 |
| M1-R 50 条 exact replay | 验证 native/strict reference reconstruction 和 success predicate | 已完成 `50/50`；属于 `Smoke`，不是 learned-policy benchmark |
| M1-R 全量 986 replay | 固定 expert-action replay、first-success 和 post-success tail | 暂停；通过后才能继续冻结 D_core |
| M1-R D_core / split / state banks | 生成 offline split 和 online residual 初始状态集合，并检查跨 split leakage | 暂停；尚未产生正式 frozen artifact |
| M1-R protocol freeze | 汇总所有 manifest，生成并验证 `frozen_protocol.json` | 未完成；不能进入 M2-R |
| M2-R | 训练 state-conditioned Flow base，先验证主体行为和 residual headroom | 未开始；不应写入训练成功率 |
| R0 → R1 | 先在 privileged state 上验证 chunk residual，再切视觉输入 | 未开始；base 必须冻结，residual 才能归因 |
| R2 → R3 | 在 LIBERO/frozen SmolVLA 上验证迁移与同交互预算对照 | 后续路线，未开始 |

### 3. 正式执行顺序

M1-R 的 runner 会在除 `validate-config` 外的阶段检查 M1-W frozen protocol；顺序 gate 未通过时
必须停止，不能创建空的 M1-R protocol 文件绕过依赖。用户解除暂停后按以下顺序恢复：

```bash
# 先做 identity/config 与环境审计
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=validate-config
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=source-audit
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=environment-audit

# 先保留 50 条 smoke，再扩展到全量 986 条
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-smoke --count=50
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-full --count=986

# full replay 通过后才冻结 D_core、split、state banks 和最终协议
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=trajectory-audit
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=freeze-data
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=state-banks
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=finalize
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=validate-frozen \
  --protocol=outputs/policy_decorator_ms2/m1_r_protocol_audit/frozen_protocol.json
```

正式产物统一落在 `outputs/policy_decorator_ms2/m1_r_protocol_audit/`，包括 source/environment
audit、`reference_replay_50/`、`reference_replay_full/`、trajectory audit、D_core、state banks、
split leakage report 和 `frozen_protocol.json`。其中 `reference_replay_50/` 的现有 smoke 证据
可以保留，但不应冒充 full replay 或 Residual RL 结果。

### 4. 当前结论与边界

- M1-R 当前已经证明的是“固定的 MS2/v2 reference contract 可进行 50 条 exact replay”，还没有
  证明 Flow base 或 Residual RL 的效果。
- R0 的第一版 base 应使用 low-dimensional state-conditioned Flow-Matching policy，而不是直接
  引入 RGB；这样才能把 residual-control pipeline 与 perception 问题分开。
- R0 进入前，base 必须明显高于 random、已经掌握 peg-to-hole 的主体行为，同时保留可由
  chunk-level residual 修正的 precision/alignment headroom；若 base 太弱，先修 M1-R/M2-R contract。
- 历史 Peg P1–P3、MS3 joint→EEF conversion 和 StackCube replay 只能作为诊断背景，不能替代
  M1-R 的 v0/v2 exact replay 或其 frozen artifact。

## StackCube 阶段结果

StackCube 的文档顺序固定为“数据协议 → 模型对照子表”；official 模型复现作为子表中的独立分组，
完整细节仍保留在 official guide 中。其中：

1. [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md) 已冻结 GPU live-RGB、7D action、18D proprioception、state bank 和 split；同一 source 的 `[0,1,2,6]` 投影 4D 全量 replay 为 `911/987` success-once、`844/987` success-at-end。
2. [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md) 已完成 990-demo 官方 4D/默认 57D state 的 50k 与 100k 单卡复现；主 verification 分别为 `0.340 / 0.330` 与 `0.520 / 0.470`。
3. 下面的 M2-W、M2 Base、LeRobot、official 和 native-4D 主线及 official 4-GPU 补充行构成 **StackCube 模型对照子表**；official 复现结果在表中单独分组，按训练 steps、数据来源和 rollout contract 记录，不把它误读成无条件总排行榜。

### StackCube 模型对照子表

下面汇总当前仓库中与 M2/官方 Diffusion Policy 直接相关的主实验线，并单列 official 4-GPU
工程变体。表中指标统一写成 `success_once / success_at_end`；第一列按协议/数据阶段合并，
第二列记录模型与数据组合，第三列记录训练 steps 和闭环结果。

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
<th rowspan="5">M1-W StackCube<br>数据协议</th>
<td><a href="04_m2_w_action_only_flow.md">M2-W Action-only Flow + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>43/200 / 43/200</code>（21.5% / 21.5%）。candidate 为 step 50k、<code>n_action_steps=8</code>；test 未访问。</td>
</tr>
<tr>
<td><a href="09_m2_base_action_flow.md">M2 Base Action-only Flow + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps / 4-GPU DDP</strong>；GPU development 100 states / horizon 200；selected step 30k、<code>n_action_steps=8</code>；<code>18/100 / 18/100</code>（18.0% / 18.0%）。<a href="../../../outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/development_gpu/selected_checkpoint.json">selected checkpoint</a></td>
</tr>
<tr>
<td><a href="09_m2_base_action_flow.md">M2 Base Action-only Flow + M1-W 7D 的 projected 4D success-at-end 数据</a></td>
<td><strong>50k steps / 4-GPU DDP</strong>；监督 action 只取 <code>[0,1,2,6]</code>；GPU development 100 states，selected step 50k、<code>n_action_steps=8</code> 为 <code>22/100 / 22/100</code>；held-out test 为 <code>21/100 / 21/100</code>。<a href="../../../outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/development_gpu/selected_checkpoint.json">selected checkpoint</a> · <a href="../../../outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/test_gpu/metrics.json">test metrics</a></td>
</tr>
<tr>
<td><a href="06_m2_diffusion_policy.md">LeRobot Diffusion Policy + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>19/200 / 19/200</code>（9.5% / 9.5%）。policy forward mean 约 719.3 ms。</td>
</tr>
<tr>
<td><a href="10_official_dp_m1w_7d.md">official Diffusion Policy + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>27/200 / 0/200</code>（13.5% / 0%）。train loss <code>0.0021</code>。</td>
</tr>
<tr>
<th rowspan="3">official 模型复现</th>
<td><a href="07_official_stackcube_rgb_anchor.md">official Diffusion Policy + 官方 motion-planning 4D replay（单卡主 anchor）</a></td>
<td><strong>50k / 100k steps</strong>；默认 <code>agent + extra</code>，实际 57D state；CPU verification 100 episodes / horizon 200：50k 为 <code>34/100 / 33/100</code>，100k 为 <code>52/100 / 47/100</code>。</td>
</tr>
<tr>
<td><a href="07_official_stackcube_rgb_anchor.md">official Diffusion Policy + 同一官方 motion-planning 4D replay（4-GPU DDP）</a></td>
<td><strong>同一 990-demo 数据、100k steps / GPU 4,5,6,7</strong>；同一默认 57D state、4D action 和 CPU verification contract；global batch <code>256</code>；<code>48/100 / 45/100</code>（48.0% / 45.0%）。仅训练并行方式不同，不替代单卡 100k anchor。</td>
</tr>
<tr>
<td><a href="11_official_dp_18d4_ablation.md">official Diffusion Policy + 同一官方 motion-planning 4D replay + 显式 18D state</a></td>
<td><strong>同一官方 motion-planning 4D replay 来源；state 改为显式 18D qpos+qvel；50k steps、787 demos</strong>；CPU development 200 states / horizon 200；<code>67/200 / 54/200</code>（33.5% / 27.0%）。train loss <code>0.00109</code>。数据来源相同，但 demos、state extractor 和 eval contract 不同，因此不是严格单变量 ablation。</td>
</tr>
<tr>
<th rowspan="2">M1-W 原生 4D<br>数据</th>
<td><a href="12_m1_w_native_4d_data.md">official Diffusion Policy + M1-W 原生 4D 数据</a></td>
<td><strong>50k / 100k steps</strong>；GPU development 200 states / horizon 500：50k 为 <code>100/200 / 0/200</code>，100k 为 <code>103/200 / 0/200</code>。不是官方 motion-planning 4D replay。</td>
</tr>
<tr>
<td><a href="09_m2_base_action_flow.md">M2 Base Action-only Flow + M1-W 原生 4D 数据</a></td>
<td><strong>100k steps / 4-GPU</strong>；global batch <code>256</code>；step 100k、<code>n_action_steps=4</code> 被选中；train mean tail loss <code>0.02532</code>。GPU development：<code>83/95 / 83/95</code>（87.4% / 87.4%）；post-selection test：<code>82/96 / 82/96</code>（85.4% / 85.4%）。<a href="../../../outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu/selected_checkpoint.json">selected checkpoint</a> · <a href="../../../outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/test_gpu/metrics.json">test metrics</a></td>
</tr>
</tbody>
</table>

### 从这些主线可以得到的结论

- 按训练步数绑定后的结果速览：M2-W（50k）+ GPU 7D 为 `0.215 / 0.215`；M2 Base GPU 7D
  per-channel route（50k / 4-GPU，selected step 30k、`n_action_steps=8`）为
  `0.180 / 0.180`；LeRobot Diffusion Policy
  （50k）+ GPU 7D 为 `0.095 / 0.095`；官方 Diffusion + 官方 4D（50k / 100k）分别为
  `0.340 / 0.330`、`0.520 / 0.470`；官方 Diffusion + 18D state + 官方 4D（50k）为
  `0.335 / 0.270`；官方 Diffusion + GPU 7D（50k）为 `0.135 / 0.000`。
- 在同一套 M1-W GPU 7D 数据和基本相同的 development state-bank 口径下，M2-W Action-only Flow
  的 `0.215 / 0.215` 明显高于 LeRobot Diffusion Policy 的 `0.095 / 0.095`；M2 Base GPU 7D
  per-channel route 为 `0.180 / 0.180`，没有超过 M2-W formal；由于 runner/config 也不同，当前不能把差异归因于 7D 各通道的尺度不一致。
  这条 M2 Base route 与 M2-W formal 是不同 runner/config，不能把两者写成同一个模型结果。
- M2 Base GPU 7D route 的 `ever_grasped=95%` 但 `ever_on_cube=28%`，说明策略大多能够接触并抓住目标，主要失败发生在
  搬运后的放置/堆叠阶段；这与 development success 仅 `18.0%` 的结果一致。当前更应优先检查 7D
  数据的动作条件、未来监督和 repair route 分布，而不是继续单纯延长训练步数。
- 官方 4D replay 的官方默认 state（57D）路线从 50k 的 `0.340 / 0.330` 提升到 100k 的
  `0.520 / 0.470`，说明这条官方 pipeline 继续训练确实有收益，不能用 50k 结果代表其上限。
- M2 Base 在 M1-W 原生 4D 数据上完成的 100k route 为 development `83/95`（87.4%），
  post-selection test `82/96`（85.4%）；这个结果明显高于 M2-W 7D 的 50k，但不能单独归因于
  action 维度，还要同时考虑训练预算、数据 provenance、control contract 和 rollout split。
- 57D→18D 的官方 state 对照目前不能下严格结论：两者同时改变了训练 demo 数（990→787）、
  eval episode 数（100→200）和选模流程。18D/4D 的 `0.335 / 0.270` 与 57D/4D 的 50k 结果接近，
  但不是已完成的单变量 ablation。
- 官方 DP 在 M1-W GPU 7D 上只有 `0.135 / 0.000`，而 M2-W 在同类数据上是 `0.215 / 0.215`；
  7D 数据路线确实存在泛化问题，但目前不能把差异归因给 action 维度本身，因为控制模式、模型和
  数据 pipeline 也一起变了。
- `[0,1,2,6]` 投影 4D 的全量 replay 为 `911/987` success-once、`844/987` success-at-end；
  因此当前不再把“投影 action 本身不可执行”作为主解释。M2 Base 投影 4D 训练的 development/test
  结果为 `22/100`、`21/100`，说明 replay 可执行不等于模型能从对应 RGB/state 学会同一动作。
- M2 Base GPU 7D 的 selected step 30k 为 `18/100`，投影 4D success-at-end route 的 selected
  step 50k 为 development `22/100`、test `21/100`；native 4D 100k 则为 test `82/96`。目前更稳妥
  的工作假设是：7D 与 native 4D 的 RGB/state 轨迹和时间对齐不一致，削弱了 learned policy 的跨路线复制。
- M2-W 的 train probe 是 `0.90 / 0.90`，development 只有 `0.215 / 0.215`，且 train/dev
  normalized MSE 为 `0.00778 / 0.10635`（约 13.7 倍差距）。因此当前最明确的瓶颈是泛化/数据覆盖，
  不能把训练集闭环成功率当作正式模型成绩。

可视化对照（同 episode / timestep 的 7D 与 native-4D RGB）见
[episode timeline](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_episode_timeline_ep0000_clean.png)、
[t=50](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_timestep_t050_clean.png) 和
[t=80](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_timestep_t080_clean.png)。这些图支持
“输入轨迹不同”的诊断方向，但不是新的模型 success 结果。

### 7D→4D 投影 replay 诊断

这不是 policy rollout，而是固定同一批 987 条 M1-W source episode，只取 action
`[0,1,2,6]`，切换到 `pd_ee_delta_pos` 后在 `physx_cuda` 中按时间步 replay。结果如下：

| 范围 | success-once | success-at-end |
| --- | ---: | ---: |
| 全量 987 | `911/987`（92.30%） | `844/987`（85.51%） |
| development 100 | `95/100` | `92/100` |
| test 100 | `90/100` | `84/100` |
| train 787 | `726/787` | `668/787` |

其中 `67` 条 episode 曾经 success 但在 episode 结束时未保持；baseline route 为
`736/805` once、`679/805` end，repair route 为 `175/182` once、`165/182` end。
这说明投影 action 的环境可执行性大体成立，但不能替代 learned policy 的 RGB/state→action
泛化评测。完整记录见
[action projection metrics](../../../outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/action_projection_0126_full/metrics.json)。

### 当前缺失或尚未闭环的结果

1. M2-W formal 的 development 结果 `43/200` 没有独立 held-out test；其 planned 7D per-channel
   formal rerun 没有形成 8-GPU training output，不能引用 `18/100`。当前 `18/100` 是 M2 Base GPU
   7D route 的结果；M1-W native-4D 的 M2 Base 100k route 已有 post-selection test `82/96`。
2. M2-W formal 的 `selected_checkpoint.json` 已记录 50k + `n_action_steps=8` 候选，但 selection gate 为
   false；没有 frozen base 或 test 结果，因此只能称为 development 最优候选，不能称为完整 frozen baseline。
3. M2 Base GPU 7D route 目前只完成 50k development 选模；projected-4D success-at-end route 已有
   50k development `22/100` 和 held-out test `21/100`，但仍没有与 native-4D route 完全相同的
   data provenance 和 input trajectory，因此不能作为动作维度的单变量结论。
4. 官方 18D/4D 对照目前只有 50k；还缺同一份数据、同一训练和评测协议下的 100k continuation。
5. 官方模型复现的 100k 已有 100-episode 官方 CPU verification；如果要和更大规模 sweep 对齐，还缺
   100k checkpoint 的 400-episode multigpu 汇总（现有 checkpoint sweep 到 90k）。
6. M2 Base 7D 与 projected-4D 的独立 test 已分别完成/部分完成（7D 仍只有 development，projected-4D test 为 `21/100`）；
   若要继续归因，还需要同一 reset、controller/backend、RGB、state 和 action source 的逐帧配对采集，
   而不是再把 replay success 当作 learned-policy 证据。

补充区分：`official_dp_stackcube_native_4d` 和其 `100k` 目录使用的是 M1-W 重生成的原生 4D 数据，
不是 official 模型复现组的“官方 4D replay”；它们已经在上面的 `M1-W 原生 4D 数据` 分组中单列，
development 结果另见
[50k native-4D metrics](../../../outputs/maniskill/official_dp_stackcube_native_4d/20_eval/development/metrics.json)
和 [100k selection](../../../outputs/maniskill/official_dp_stackcube_native_4d_100k/20_eval/development/selection.json)。

实际运行结果在远端产生后，统一登记到 [实验运行记录](../../experiment_log.md)。M1-R/RL 路线见
[Residual RL 母计划](../../../plans/2026-08-11-Mini-WAM与Residual-RL-完整实验总计划.md)；WAM 文献复现与
StackCube 迁移路线单独见 [文献模型复现实验指南](../literature/README.md)。
