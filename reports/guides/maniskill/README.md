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
9. [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md)：M1-W 协议之后的官方 990-demo RGB/4D Diffusion Policy 复现，含 50k/100k budget curve 与 `success_once` 验证
10. [官方 StackCube RGB Golden Anchor 执行命令](08_official_stackcube_rgb_anchor_commands.md)：源数据同步、分阶段 replay/训练/验证与日志观察命令
11. [M2 base Action-only Flow 4D/7D 适配](09_m2_base_action_flow.md)：官方 RGB H5 适配、4D/7D head、懒读 H5 与 4 卡训练命令
12. [官方 Diffusion Policy 接入 M1-W 7D 数据](10_official_dp_m1w_7d.md)：官方 DP、M1-W live RGB/18D state、懒读 H5 与 torchrun 多卡训练入口
13. [官方 Diffusion Policy 18D/4D 对照实验](11_official_dp_18d4_ablation.md)：官方数据 25D→18D 与 M1-W 7D→4D 的同协议四卡对照
14. [M1-W 原生 4D 数据重生成](12_m1_w_native_4d_data.md)：从原始 joint action 官方转换到 4D，再 GPU replay、修复并实时采集 RGB

## StackCube 阶段结果

StackCube 的文档顺序固定为“数据协议 → 模型对照子表”；official 模型复现作为子表中的独立分组，
完整细节仍保留在 official guide 中。其中：

1. [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md) 已冻结 GPU live-RGB、7D action、18D proprioception、state bank 和 split。
2. [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md) 已完成 990-demo 官方 4D/默认 57D state 的 50k 与 100k 单卡复现；主 verification 分别为 `0.340 / 0.330` 与 `0.520 / 0.470`。
3. 下面的五条主线及 official 4-GPU 补充行构成 **StackCube 模型对照子表**；official 复现结果在表中单独分组，按训练 steps、数据来源和 rollout contract 记录，不把它误读成无条件总排行榜。

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
<th rowspan="4">M1-W StackCube<br>数据协议</th>
<td><a href="04_m2_w_action_only_flow.md">M2-W Action-only Flow + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps</strong>；GPU development 200 states / horizon 500；<code>43/200 / 43/200</code>（21.5% / 21.5%）。candidate 为 step 50k、<code>n_action_steps=8</code>；test 未访问。</td>
</tr>
<tr>
<td><a href="09_m2_base_action_flow.md">M2-W Action-only Flow（7D per-channel normalization rerun） + M1-W GPU 7D 数据</a></td>
<td><strong>50k steps / 4-GPU DDP</strong>；GPU development 100 states / horizon 200；step 30k、<code>n_action_steps=8</code> 被选中；<code>18/100 / 18/100</code>（18.0% / 18.0%）。<code>ever_grasped=95%</code>、<code>ever_on_cube=28%</code>；test 未访问。<a href="../../../outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/development_gpu/selected_checkpoint.json">selected checkpoint</a></td>
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
<td><a href="09_m2_base_action_flow.md">M2-W Action-only Flow + M1-W 原生 4D 数据</a></td>
<td><strong>100k steps / 4-GPU</strong>；global batch <code>256</code>；step 100k、<code>n_action_steps=4</code> 被选中；train mean tail loss <code>0.02532</code>。GPU development：<code>83/95 / 83/95</code>（87.4% / 87.4%）；post-selection test：<code>82/96 / 82/96</code>（85.4% / 85.4%）。<a href="../../../outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu/selected_checkpoint.json">selected checkpoint</a> · <a href="../../../outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/test_gpu/metrics.json">test metrics</a></td>
</tr>
</tbody>
</table>

### 从这些主线可以得到的结论

- 按训练步数绑定后的结果速览：M2-W（50k）+ GPU 7D 为 `0.215 / 0.215`；本次 7D
  per-channel normalization rerun（50k / 4-GPU，selected step 30k、`n_action_steps=8`）为
  `0.180 / 0.180`；LeRobot Diffusion Policy
  （50k）+ GPU 7D 为 `0.095 / 0.095`；官方 Diffusion + 官方 4D（50k / 100k）分别为
  `0.340 / 0.330`、`0.520 / 0.470`；官方 Diffusion + 18D state + 官方 4D（50k）为
  `0.335 / 0.270`；官方 Diffusion + GPU 7D（50k）为 `0.135 / 0.000`。
- 在同一套 M1-W GPU 7D 数据和基本相同的 development state-bank 口径下，M2-W Action-only Flow
  的 `0.215 / 0.215` 明显高于 LeRobot Diffusion Policy 的 `0.095 / 0.095`；新的 per-channel
  normalization rerun 为 `0.180 / 0.180`，没有超过旧结果，说明当前问题不只是 7D 各通道的尺度不一致。
  在这条数据线上，M2-W 的模型/训练配置仍比 LeRobot Diffusion Policy 更有效。
- 新 rerun 的 `ever_grasped=95%` 但 `ever_on_cube=28%`，说明策略大多能够接触并抓住目标，主要失败发生在
  搬运后的放置/堆叠阶段；这与 development success 仅 `18.0%` 的结果一致。当前更应优先检查 7D
  数据的动作条件、未来监督和 repair route 分布，而不是继续单纯延长训练步数。
- 官方 4D replay 的官方默认 state（57D）路线从 50k 的 `0.340 / 0.330` 提升到 100k 的
  `0.520 / 0.470`，说明这条官方 pipeline 继续训练确实有收益，不能用 50k 结果代表其上限。
- M2-W 在 M1-W 原生 4D 数据上完成的 100k route 为 development `83/95`（87.4%），
  post-selection test `82/96`（85.4%）；这个结果明显高于 M2-W 7D 的 50k，但不能单独归因于
  action 维度，还要同时考虑训练预算、数据 provenance、control contract 和 rollout split。
- 57D→18D 的官方 state 对照目前不能下严格结论：两者同时改变了训练 demo 数（990→787）、
  eval episode 数（100→200）和选模流程。18D/4D 的 `0.335 / 0.270` 与 57D/4D 的 50k 结果接近，
  但不是已完成的单变量 ablation。
- 官方 DP 在 M1-W GPU 7D 上只有 `0.135 / 0.000`，而 M2-W 在同类数据上是 `0.215 / 0.215`；
  7D 数据路线确实存在泛化问题，但目前不能把差异归因给 action 维度本身，因为控制模式、模型和
  数据 pipeline 也一起变了。
- M2-W 的 train probe 是 `0.90 / 0.90`，development 只有 `0.215 / 0.215`，且 train/dev
  normalized MSE 为 `0.00778 / 0.10635`（约 13.7 倍差距）。因此当前最明确的瓶颈是泛化/数据覆盖，
  不能把训练集闭环成功率当作正式模型成绩。

### 当前缺失或尚未闭环的结果

1. 7D M2-W 的两个 development 结果（原 formal `43/200`、per-channel normalization rerun
   `18/100`）都没有独立 held-out test；rerun 还只访问了 development state-bank，不能作为最终测试成绩。
   M1-W native-4D 的 M2-W 100k route 是补充组，已有 post-selection test `82/96`。
2. 第 1 条的 `selected_checkpoint.json` 已记录 50k + `n_action_steps=8` 候选，但 selection gate 为
   false；没有 frozen base 或 test 结果，因此只能称为 development 最优候选，不能称为完整 frozen baseline。
3. 第 2 条只有 50k development 选模，没有独立 test，也没有当前 sidecar 可直接引用的 train loss。
4. 第 4 条只有 50k；还缺同一份 18D/4D 官方数据、同一训练和评测协议下的 100k continuation。
5. 第 3 条的 100k 已有 100-episode 官方 CPU verification；如果要和更大规模 sweep 对齐，还缺
   100k checkpoint 的 400-episode multigpu 汇总（现有 checkpoint sweep 到 90k）。
6. 7D 的 per-channel normalization 已完成一次 M2-W 对照，但同一 episode/split 下的 7D→4D 投影和
   baseline-only（排除 repair route）对照尚未完成，暂时不能把失败单独归因于 rotation channel 或
   repair route。

补充区分：`official_dp_stackcube_native_4d` 和其 `100k` 目录使用的是 M1-W 重生成的原生 4D 数据，
不是 official 模型复现组的“官方 4D replay”；它们已经在上面的 `M1-W 原生 4D 数据` 分组中单列，
development 结果另见
[50k native-4D metrics](../../../outputs/maniskill/official_dp_stackcube_native_4d/20_eval/development/metrics.json)
和 [100k selection](../../../outputs/maniskill/official_dp_stackcube_native_4d_100k/20_eval/development/selection.json)。

实际运行结果在远端产生后，统一登记到 [实验运行记录](../../experiment_log.md)。M0 的上游路线见
[Mini-WAM 与 Residual RL 完整实验总计划](../../../plans/2026-08-11-Mini-WAM与Residual-RL-完整实验总计划.md)。
