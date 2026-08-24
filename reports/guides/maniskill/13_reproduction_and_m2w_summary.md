# ManiSkill 复现实验与 M2-W / M2 Base 结果总览

## 重要进展

- 官方 Diffusion Policy + 官方 motion-planning 4D replay 已完成 50k、100k 单卡主 anchor，另有同数据的 100k、4-GPU DDP 工程变体。
- M1-W GPU 7D 数据上的 M2-W Action-only Flow 50k formal run 已完成；原计划的 M2-W 7D per-channel formal rerun 尚未形成 8-GPU training output。另有独立的 M2 Base GPU 7D per-channel route，结果已单列。
- M1-W 原生 4D 数据上的 Action-only Flow 100k 训练、development 选模和 held-out test 已完成，是目前最完整的 M2 Base 结果；同一 M2 Base trunk 的 GPU 7D 与 projected 4D 对照也已登记。
- M1-W 7D action 只取 `[0,1,2,6]` 的全量 replay 为 `911/987` success-once、`844/987` success-at-end，说明投影 action 大体可执行。当前阶段性归因收敛为：7D RGB/state 与 native 4D RGB/state 不是同一条可逐帧复现的输入轨迹，这会削弱 learned policy 的跨数据路线复制和泛化；不能表述为 7D action 维度本身无法训练。

## 1. 统一实验口径

这份文档是结果总览和复现入口；各模型的实现细节仍保留在对应专项 guide 中。

| 维度 | 当前统一记录方式 |
| --- | --- |
| task | ManiSkill3 `StackCube-v1` |
| 7D M1-W 数据 | GPU live-RGB，`pd_ee_delta_pose`，7D action，18D `qpos+qvel` |
| 原生 4D 数据 | 原始 joint demonstrations 转换为 `pd_ee_delta_pos`，再 GPU replay/repair/live-RGB capture |
| official 数据 | 官方 motion-planning replay，4D `pd_ee_delta_pos`；默认 state 实际为 57D `agent+extra`，另有显式 18D ablation |
| M2-W action-only | 最近 2 步双相机 RGB + 18D state，输出 16 步 action chunk；Flow Euler/10 steps |
| M2 Base 4D/7D | 同一 Action-only Flow trunk，按 config 切换 4D 或 7D action contract |
| rollout | M1-W 数据使用 GPU PhysX；official 数据使用 CPU PhysX；两者不能混用 |
| 指标 | 统一写为 `success_once / success_at_end`，并同时记录 steps、episode 数和 rollout backend |

不要把 official CPU rollout、M1-W GPU rollout、teacher-forced hybrid diagnostic 三种结果放在同一列直接排序。

## 2. 数据协议与前置产物

| 数据路线 | 数据规模与 split | 关键 observation/action contract | 前置结果 |
| --- | --- | --- | --- |
| M1-W GPU 7D | D_core 987 episodes；train/development/test = 787/100/100 | 两路 `128×128×3` RGB + 18D `qpos+qvel`；7D `pd_ee_delta_pose`；20 Hz；Flow `2/16/4` 或 `2/16/8` | GPU baseline → failed-only repair → fingerprint split/freeze；7D replay `987/987` success-at-end；projected `[0,1,2,6]` full replay `911/987` once、`844/987` end |
| M1-W projected 4D success-at-end | 原始 source 987；训练/开发/测试 index = 668/100/100 | 同一 7D RGB + 18D state；监督 action 只取 `[0,1,2,6]` 为 4D `pd_ee_delta_pos`；20 Hz | 4D projection replay is executable on most episodes；用于 M2 Base 4D contrast，不是 native-4D data |
| M1-W native 4D | final live-RGB 946 episodes；train/development/test = 748/95/96 | 两路 RGB + 18D `qpos+qvel`；4D `pd_ee_delta_pos`；20 Hz；GPU PhysX | CPU joint→4D conversion `1000/1000`；GPU baseline `760/1000`；repair `186/240`；accepted/final `946` |
| official motion-planning | 主 anchor 990 demos；18D/4D ablation 使用 787 demos | 两路 RGB + 官方默认 57D `agent+extra`，或显式 18D；4D `pd_ee_delta_pos`；20 Hz；CPU PhysX | 官方 replay/adapter gate 通过；official rollout 固定 CPU |

详细数据生成和协议审计见 [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md) 和 [M1-W 原生 4D 数据重生成](12_m1_w_native_4d_data.md)。

## 3. 当前实验结果矩阵

结果均为已经落盘的 measured result；`development` 结果不能替代 held-out test。

| 实验路线 | 训练 steps | 数据与模型 | rollout / split | 结果 `success_once / success_at_end` |
| --- | ---: | --- | --- | ---: |
| M2-W formal | 50k | M2-W Action-only Flow + M1-W GPU 7D | GPU development，200 states；step 50k，`n_action_steps=8` | `43/200 / 43/200`（21.5% / 21.5%） |
| LeRobot Diffusion Policy | 50k | LeRobot Diffusion Policy + M1-W GPU 7D | GPU development，200 states | `19/200 / 19/200`（9.5% / 9.5%） |
| official Diffusion Policy | 50k | 官方 motion-planning 4D replay + 默认 57D state | 官方 CPU verification，100 episodes | `34/100 / 33/100`（34.0% / 33.0%） |
| official Diffusion Policy | 100k | 同一官方 4D replay + 默认 57D state | 官方 CPU verification，100 episodes | `52/100 / 47/100`（52.0% / 47.0%） |
| official Diffusion Policy DDP | 100k | 同一官方 4D replay + 默认 57D state；仅改为 4-GPU DDP | 官方 CPU verification，100 episodes | `48/100 / 45/100`（48.0% / 45.0%） |
| official Diffusion Policy 18D ablation | 50k | 官方 4D replay + 显式 18D state | CPU development，200 episodes | `67/200 / 54/200`（33.5% / 27.0%） |
| official Diffusion Policy + M1-W 7D | 50k | M1-W GPU 7D，7D `pd_ee_delta_pose` | GPU development，200 states | `27/200 / 0/200`（13.5% / 0%） |
| official Diffusion Policy + M1-W native 4D | 50k | M1-W 原生 4D live-RGB | GPU development，200 states | `100/200 / 0/200` |
| official Diffusion Policy + M1-W native 4D | 100k | M1-W 原生 4D live-RGB | GPU development，200 states | `103/200 / 0/200` |
| M2 Base / Action-only Flow + M1-W GPU 7D | 50k | 7D `pd_ee_delta_pose`；987 episodes；同一 M2 Base 7D config | 4-GPU GPU development 100；selected step 30k，`n_action_steps=8` | `18/100 / 18/100` |
| M2 Base / Action-only Flow + M1-W projected 4D | 50k | 同一 7D source；监督 action 取 `[0,1,2,6]`；success-at-end index 668/100/100 | 4-GPU GPU development 100；selected step 50k，`n_action_steps=8`；held-out test 100 | dev `22/100 / 22/100`；test `21/100 / 21/100` |
| M2 Base / Action-only Flow + M1-W native 4D | 100k | 原生 4D `pd_ee_delta_pos`；946 条成功 live-RGB episode | GPU development 95；held-out test 96；selected step 100k，`n_action_steps=4` | dev `83/95 / 83/95`；test `82/96 / 82/96` |

说明：表中的“原生 4D”是重新转换、GPU replay/repair 并实时采集 RGB 的数据，不是把 7D action 简单切成 4D；“projected 4D”则明确只取 7D 的 `[0,1,2,6]` 监督，二者不能混称。M2 Base native-4D 100k、projected-4D 50k 和 M2-W 7D formal 使用同一类 Action-only Flow trunk，但数据、action contract 和训练预算不同，不能合并为一个单变量结论。

### 3.1 训练与失败字段补充

没有落盘的字段明确写成 `N/A`，不从终端速度或 smoke 结果反推正式指标。

| 路线 | train loss | open-loop action error | latency | 主要 failure cases |
| --- | ---: | ---: | ---: | --- |
| M2-W 7D formal | tail `0.0685` | `N/A` | `N/A` | 157/200 development 未成功；未完成独立 held-out test；planned 8-GPU per-channel rerun 尚无 formal output |
| LeRobot Diffusion Policy + 7D | `N/A` | `N/A` | policy forward mean 约 `719.3 ms` | 181/200 development 未成功 |
| official 4D/57D 50k / 100k | `0.00227` / `0.001` | `N/A` | `N/A` | success-once 失败分别为 66/100、48/100 |
| official 18D/4D 50k | `0.00109` | `N/A` | `N/A` | 133/200 development 未 success-once |
| official + M1-W 7D 50k | `0.0021` | `N/A` | `N/A` | success-at-end 为 0/200 |
| M2 Base + M1-W GPU 7D 50k | last `0.07552`；last 1k mean `0.07153` | `N/A` | `N/A` | development 82/100 未成功；当前只登记 development |
| M2 Base + projected 4D success-at-end 50k | last `0.04619`；last 1k mean `0.04240` | `N/A` | `N/A` | development 78/100、test 79/100 truncated；action clip fraction dev/test 约 `0.126` / `0.122` |
| M2 Base native 4D 100k | mean tail `0.02532`；last `0.01671` | `N/A` | `N/A` | development 12/95、test 14/96 truncated；action clip fraction 约 0.182/0.184 |

M1-W 7D 的冻结 batch probe 是目前唯一同时给出 input-side open-loop loss 对照的诊断：native pipeline `0.00245568`、完整 projected pipeline `0.01124220`、7D RGB + native 4D state + projected action `0.00557943`。这三个数是 one-step teacher-forced loss，不是闭环 success rate。

### 3.2 7D→4D 投影 replay 的协议诊断

固定 D_core 的 987 条 source episode，只取 action `[0,1,2,6]`，在 `physx_cuda` 中使用
`pd_ee_delta_pos` 做 action replay；没有模型推理，RGB/state 也不作为 policy input。结果为：

| 范围 | success-once | success-at-end |
| --- | ---: | ---: |
| 全量 987 | `911/987`（92.30%） | `844/987`（85.51%） |
| development 100 | `95/100` | `92/100` |
| test 100 | `90/100` | `84/100` |
| train 787 | `726/787` | `668/787` |

因此 replay 结果支持“投影 action 大体可执行”，不支持“模型只要输出 4D 就必然能复制该轨迹”。
完整记录：[action projection metrics](../../../outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/action_projection_0126_full/metrics.json)。

## 4. 复现入口

### 4.1 官方 Diffusion Policy anchor

官方主 anchor 是 990 demos、官方 4D replay、默认 57D state。完整 replay、训练、CPU verification 和 4-GPU DDP 变体见 [官方 StackCube RGB Golden Anchor](07_official_stackcube_rgb_anchor.md) 与 [官方执行命令](08_official_stackcube_rgb_anchor_commands.md)。

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=all \
  --output-root=outputs/maniskill/official_stackcube_rgb_990_100k \
  --num-demos=990 \
  --total-iters=100000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --exp-name=official_stackcube_rgb_990_100k \
  --gpu=0 \
  --model-device=cuda
```

### 4.2 官方 18D/4D 与 M1-W 7D 对照

这个入口保持官方 Diffusion Policy，只替换数据 adapter；official 数据的 rollout 必须是 CPU，M1-W 数据的 rollout 必须是 GPU。

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=official \
  --phase=all \
  --gpus=0,1,2,3 \
  --num-demos=787 \
  --total-iters=50000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_official
```

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=m1w \
  --phase=all \
  --gpus=0,1,2,3 \
  --num-demos=787 \
  --total-iters=50000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_m1w_projected
```

详见 [官方 Diffusion Policy 18D/4D 对照实验](11_official_dp_18d4_ablation.md) 和 [官方 Diffusion Policy 接入 M1-W 7D 数据](10_official_dp_m1w_7d.md)。

### 4.3 M2-W 7D formal

M2-W formal config 固定 8-GPU、global batch 256、50k optimizer steps；训练中 checkpoint eval 只读取 development，不访问 test。

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000
```

选模使用独立 development 输出：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --num-envs=64 \
  --train-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/development
```

per-channel normalization rerun 只能写入独立目录：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000_7d_per_channel
```

完整流程和 formal contract 见 [M2-W StackCube Action-only Flow](04_m2_w_action_only_flow.md)。

M2 Base 的 GPU 7D 与 projected 4D route 共用另一套 4-GPU runner；projected 4D 只读取
`[0,1,2,6]`，不能改写成 native 4D。完整训练、选模和 test 命令见
[M2 base Action-only Flow 4D/7D 适配](09_m2_base_action_flow.md)。

### 4.4 M2 Base native 4D 完整结果

这条路线是目前最完整的 Action-only Flow 训练结果：100k steps、4-GPU DDP、native 4D live-RGB 数据，选模和 held-out test 均已完成。

```bash
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json \
  --num-gpus=4 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k
```

选模与 test：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json \
  --train-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200 \
  --n-action-steps=4,8
```

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=evaluate-test \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json \
  --selection-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/test_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200
```

详见 [M2 base Action-only Flow 4D/7D 适配](09_m2_base_action_flow.md)。LeRobot Diffusion Policy 的独立入口见 [M2 StackCube Diffusion Policy](06_m2_diffusion_policy.md)。

## 5. 7D 数据的当前诊断结论

### 5.1 阶段性工作假设

当前暂时采用下面的解释来组织后续工作：

> M1-W 7D 数据与原生 4D 数据虽然来自相同 StackCube 任务和相近的 episode split，但 RGB、state、controller/backend 或时间轨迹并不是同一条可逐帧复制的轨迹。因此，用原生 4D checkpoint 直接读取 7D RGB、再替换成 native 4D state，不能被视为只改变 action 维度的干净 ablation。

这条假设现在只用于解释 learned-policy 的跨路线差异，不再把 action projection 本身当作主要失败源。完整 projected replay 在同一 source 上已经达到 `911/987` success-once、`844/987` success-at-end；这说明 `[0,1,2,6]` action 在环境中大体可执行。

它解释了为什么：

- 7D action 投影后的数值不等于 native 4D action，且两者对应的 RGB/state 不是同一条输入轨迹；
- 同一 episode/time window 的 RGB 和 state 仍可能有明显差异；
- native 4D checkpoint 的 true closed-loop test 可以达到 `86/100`，而固定 recorded 7D RGB + native 4D state 的 hybrid diagnostic 只有 `44/100`；这不是 projected replay 失败，而是 recorded input 与 live closed-loop input 的差异证据。

这条解释目前是工程上的阶段性归因，不是“7D 数据天然不能训练”的证明。真正的单变量验证仍需要同一 source episode、同一 reset state、同一 controller/backend 下重新采集两种 observation/action 表示。

### 5.2 已完成的冻结 batch 诊断

使用同一个 episode、同一个时间窗、同一个 native 4D 100k checkpoint，学习率设为 0，只计算一次 loss：

| 输入组合 | full loss |
| --- | ---: |
| native 4D RGB + native 4D state + native 4D action | `0.00245568` |
| 7D RGB + 7D state + 7D projected action，走 7D pipeline | `0.01124220` |
| 7D RGB + native 4D state + projected action | `0.00557943` |

这个诊断支持“RGB/state provenance 是重要变量”，但不能单独分离 RGB、state、normalization 和 action trajectory 的因果贡献。详细的 ablation 和 output 见 [M2 base guide](09_m2_base_action_flow.md) 以及 `outputs/maniskill/m2_base_action_flow/loss_probe_native4d_vs_projected4d/metrics.json`。

### 5.3 Hybrid test 与输入可视化结果边界

最近完成的 100 个 test hybrid diagnostic 配置为：7D recorded RGB + native 4D qpos/qvel + native 4D checkpoint 输出的 4D action；action 在 GPU PhysX 中执行，但 RGB/state 仍由记录数据 teacher-force 提供。

- 全量：`44/100 / 44/100`；
- 起始 state fingerprint 完全匹配的 96 条：`43/96 / 43/96`；
- 4 条 state 不匹配 episode：`9, 560, 582, 677`，已在 metrics 中单独标记；
- 这不是普通 closed-loop policy score，不能直接替代 native 4D checkpoint 的 `82/96` held-out test。

结果：[hybrid metrics](../../../outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d_hybrid_7d_rgb_native4dstate_test/metrics.json)。同 episode / timestep 的 RGB 对照见
[timeline](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_episode_timeline_ep0000_clean.png)、
[t=50](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_timestep_t050_clean.png) 和
[t=80](../../../outputs/maniskill/m1w_rgb_4d_vs_7d_visual_comparison/same_timestep_t080_clean.png)。这些是
offline diagnostic visualization，不是新的 rollout score。

## 6. 当前结论与下一步

1. **上游基线是可信的。** 官方 4D/57D state 的 50k→100k 从 `0.340/0.330` 提升到 `0.520/0.470`，说明官方 pipeline 和当前 CPU rollout contract 可以形成可复现参照。
2. **M2-W 7D formal 已完成，但泛化较弱。** 原始 7D formal development 为 `0.215/0.215`；planned M2-W 8-GPU per-channel rerun 尚未形成 formal output。另一个 M2 Base 7D per-channel route 的 development 为 `0.180/0.180`，不能与 M2-W formal 混称。
3. **原生 4D Action-only Flow 已有完整成功结果。** 100k native 4D route 的 held-out test 为 `82/96`；projected 4D 50k route 的 held-out test 为 `21/100`，GPU 7D 50k development 为 `18/100`。native-4D 的优势不能直接归因于 action 维度，因为 data provenance、controller、RGB/state trajectory 和训练预算都不同。
4. **投影 replay 与 learned-policy 泛化必须分开。** `[0,1,2,6]` 全量 replay 为 `911/987` once、`844/987` end，说明投影 action 大体可执行；当前工作归因是 7D 与 native 4D 的 RGB/state 轨迹和时间对齐不一致，削弱了 learned policy 的跨路线复制。后续若要做单变量验证，应保证 RGB、state、action、reset state 和 controller/backend 来自同一 source。
5. **结果登记规则。** 新实验必须同时写清楚 dataset provenance、训练 steps、split/episode 数、state/action contract、rollout backend、`success_once/success_at_end` 和 output root；development、teacher-forced、replay success 不得写成 held-out learned-policy result。

专项实现、可视化和命令继续分别维护在：

- [M1-W StackCube 数据与控制协议预检](02_m1_w_protocol_audit.md)
- [M1-W 原生 4D 数据重生成](12_m1_w_native_4d_data.md)
- [M2-W Action-only Flow](04_m2_w_action_only_flow.md)
- [M2 base Action-only Flow](09_m2_base_action_flow.md)
- [官方 Diffusion Policy anchor](07_official_stackcube_rgb_anchor.md)
- [LeRobot Diffusion Policy](06_m2_diffusion_policy.md)
