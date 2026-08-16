# M1 PegInsertionSide 数据 / Controller Contract 继续诊断计划

- 日期：2026-08-11
- 状态：`done`
- 范围：仅用于排查 `PegInsertionSide` 的 demonstration / controller conversion 问题。
- 目标：在继续修改 converter 之前，把问题严格拆成 **原数据可 replay 性 → controller action contract → 跨版本 trajectory divergence** 三层，确定真正的失败来源。
- 原则：**不放宽 M1 gate，不把当前 8/10 conversion 数据并入正式 `D_core`。**
- 2026-08-12 路线说明：本计划的诊断已完成；其中“生成 MS3 native EEF demos”是当时诊断树的
  候选后继，不再是当前待执行路线。现行母计划已冻结为 R0/R1 使用 Policy Decorator 的
  ManiSkill2 0.5.3 + repo-local `PegInsertionSide-v2`，MS3 Peg 不进入正式数据线。

---

## 0. 当前已知结论

### 0.1 ManiSkill3 joint → EEF conversion

原始 motion-planning 数据：

```text
8D pd_joint_pos
```

目标控制协议：

```text
7D pd_ee_delta_pose
```

当前有效 smoke 结果：

| Variant | 方法 | 成功率 | 失败 episode |
|---|---|---:|---|
| A | 原始 joint → `pd_ee_delta_pose` conversion | 8/10 | 1, 2 |
| B | rotation contract 修正 + EEF-error convergence inner-loop | 8/10 | 2, 9 |
| C | joint target → FK → absolute `pd_ee_pose` | 8/10 | 1, 2 |

说明：

- B/C 都没有通过 M1 的正式 gate；
- 因此当前 converted trajectories 不能作为正式 `D_core`；
- 此前出现的 0/10 是 probe reset / controller cache 问题，已经判定无效；
- 当前不能再把问题简单归因为单一 rotation sign、四元数乘法顺序或 delta accumulation。

### 0.2 Policy Decorator 数据

已知数据属性：

```text
ManiSkill2 PegInsertionSide
native 7D pd_ee_delta_pose
986 episodes
final success = 986/986
```

当前已做的 ManiSkill3 raw-action probe：

```text
Policy Decorator 7D action
        ↓
ManiSkill3 PegInsertionSide-v1
physx_cpu
horizon = 200
        ↓
native predicate: 0/10
strict predicate: 0/10
mid-trajectory success: 0/10
```

**该 0/10 暂时不能解释为“ManiSkill2 → ManiSkill3 physics/domain gap 很大”。**

它最多说明：

> Policy Decorator 数据中的 7D action 数值不能未经语义转换直接作为当前 ManiSkill3 的 `pd_ee_delta_pose` action 使用。

因为该实验可能同时混入：

- initial physical state 不匹配；
- action frame 不匹配；
- rotation parameterization / scaling contract 不匹配；
- IK / controller implementation 不匹配；
- contact dynamics / task implementation 差异。

因此必须继续拆解。

---

# 1. 总诊断路线

```text
P1  Policy Decorator / ManiSkill2 exact replay
 │
 │  先证明 reference dataset 在自己的原环境里可以复现
 ↓
P2  ManiSkill2 vs ManiSkill3 单步 controller-contract test
 │
 │  先把 frame / sign / scale / rotation representation 锁死
 ↓
P3  matched-state cross-version trajectory divergence
 │
 │  不先看最终 success，而是找“第一次从哪里开始分叉”
 ↓
Decision
 ├─ action semantics 问题 → 修 semantic converter
 ├─ IK/controller realization gap → 停止强转 joint dataset
 ├─ contact/precision gap → 停止强转 joint dataset
 └─ cross-version 可高度复现 → 再评估是否能复用 native EEF data
```

本轮最多做到 P3。

**禁止继续无边界新增 B2/B3/C2/C3 converter variant。**

---

# 2. P1 — Policy Decorator / ManiSkill2 exact replay

## 2.1 研究问题

先回答最基础的问题：

> Policy Decorator 的 native `pd_ee_delta_pose` trajectories，在它所属的 ManiSkill2 环境、controller 和原始 episode state 下，是否真的可以稳定 replay？

这是后续所有 MS2 → MS3 对比的 reference。

---

## 2.2 环境

优先建立与 Policy Decorator 数据一致的环境：

```text
mani-skill2 == 0.5.3
PegInsertionSide
Policy Decorator 对应 controller contract
pd_ee_delta_pose
```

需要复用 / 核对：

- Policy Decorator environment pin；
- dataset metadata；
- episode reset/state reconstruction；
- controller config；
- horizon；
- success predicate。

参考：

```text
https://github.com/tongzhoumu/policy_decorator
https://github.com/tongzhoumu/policy_decorator/blob/main/envs/maniskill_fixed.py
```

---

## 2.3 第一阶段 smoke

先固定抽取：

```text
20 episodes
```

要求：

1. 使用 dataset 中对应 episode 的原始 state / reset 信息；
2. 不用新的随机 reset 替代原始 episode state；
3. 原样执行 dataset action；
4. 不修改 action；
5. 不加 converter；
6. 同时记录 native success 与 dataset metadata 中的 success。

### 必须落盘

每条 episode：

```text
episode_id
trajectory_length
initial_robot_qpos
initial_robot_qvel
initial_peg_pose
initial_box/hole_pose
dataset_final_success
replay_final_success
first_success_step
final_eef_pose
final_peg_pose
```

如果 dataset 中还有 task randomization 参数，也一并保存。

---

## 2.4 Gate

### Strong pass

```text
20/20 replay success
```

则继续扩到：

```text
50 episodes
```

### Conditional

若出现极少失败，但失败可明确解释为：

```text
state reconstruction 缺字段
success predicate 差异
replay API 使用问题
```

先修 replay，再重复同一批 episode。

### Fail

如果 exact replay 仍出现稳定的大比例失败：

> 暂停所有 MS2 → MS3 推断。

此时优先说明我们还没有正确复原 Policy Decorator 的原始 replay contract。

---

# 3. P2 — ManiSkill2 vs ManiSkill3 单步 Controller Contract Test

## 3.1 研究问题

P2 不测试 Peg task success。

只回答：

> “在没有 contact 的情况下，给定同一个物理 EE 小位移 / 小旋转目标，MS2 与 MS3 controller 实际产生的 EE motion 是否一致？”

需要隔离：

```text
translation frame
rotation frame
rotation sign
rotation scale
rotation representation
normalization / clipping
```

---

## 3.2 测试状态

构造至少 3 个固定 robot configuration：

```text
S0: 接近 nominal reset
S1: EEF 有明显 yaw / pitch
S2: 接近 Peg pre-insertion 的典型姿态
```

要求：

- 无接触；
- peg / hole 不影响机器人；
- robot qpos 固定；
- MS2 / MS3 尽可能初始化到同一机械臂物理姿态。

---

## 3.3 6-axis elementary commands

分别测试：

```text
+x
-x
+y
-y
+z
-z

+rx
-rx
+ry
-ry
+rz
-rz
```

第一版物理目标：

```text
translation magnitude ≈ 0.01 m
rotation magnitude    ≈ 0.02 rad
```

这里的数值是诊断用小扰动，不是训练协议。

---

## 3.4 不比较 action number，比较 realized motion

每个 command 记录：

```text
input normalized action
decoded physical target
EEF pose before
EEF pose after
realized Δp
realized ΔR
target joint qpos
actual joint qpos
clipping flag
```

定义：

\[
e_p =
\left\|
\Delta p^{MS2}
-
\Delta p^{MS3}
\right\|_2
\]

旋转误差用统一 SO(3) geodesic distance：

\[
e_R =
\left\|
\log
\left(
(\Delta R^{MS2})^{-1}
\Delta R^{MS3}
\right)
\right\|_2
\]

---

## 3.5 显式 semantic converter

不要做：

```text
MS2 action number
→ 原样 feed MS3
```

而是建立：

```text
MS2 action
   ↓
decode 成 physical EE transform
   ↓
ΔT_physical
   ↓
encode 成 MS3 controller action
```

也就是：

\[
a_t^{MS2}
\rightarrow
\Delta T_t^{physical}
\rightarrow
a_t^{MS3}
\]

这个中间层必须作为独立函数实现，例如：

```python
decode_ms2_ee_action(...)
encode_ms3_ee_action(...)
convert_ms2_action_to_ms3(...)
```

禁止把 frame/sign 修补散落在 replay loop 中。

---

## 3.6 Gate

P2 的通过条件不是 success rate，而是：

1. 12 个基础 axis commands 方向全部正确；
2. translation sign / frame 无歧义；
3. rotation sign / frame 无歧义；
4. normalized action → physical motion scale 可解释；
5. 不依赖“两个错误互相抵消”才能得到正确方向；
6. 同一 physical command 下，MS2 / MS3 realized EE motion 达到稳定的小误差。

具体容差先根据无接触 repeat noise 统计后预注册，不在看到 Peg trajectory 结果后修改。

---

# 4. P3 — Matched-State Cross-Version Trajectory Divergence

## 4.1 研究问题

P1/P2 通过后，才回答：

> 在 initial state 和 action semantics 都尽可能对齐后，MS2 trajectory 与 MS3 trajectory 从什么阶段开始产生不可接受的分叉？

这一步的重点不是最终：

```text
success / failure
```

而是：

```text
first divergence point
```

---

## 4.2 Episode 选择

至少选择：

### Group A — Policy Decorator native trajectories

```text
5 条简单成功 trajectory
5 条较长 / 接近 precision boundary 的 trajectory
```

### Group B — 当前 joint conversion counterexample

单独保留：

```text
episode 2
```

因为它在：

```text
original
B
C
```

中都失败，是当前最稳定的 counterexample。

---

## 4.3 Initial state matching

对每条 Policy Decorator episode，尽可能恢复：

```text
robot qpos
robot qvel
peg pose
box pose
hole geometry / task randomization
其他可访问 episode state
```

然后分别建立：

```text
MS2 reference rollout
MS3 semantic-converted rollout
```

如果某些 MS2 state 字段无法一一映射到 MS3：

- 明确记录；
- 不允许默默用随机 reset；
- 将该 episode 标记为 `state_match_partial`。

---

## 4.4 每步记录

每个 timestep：

```text
t
source action
converted action

MS2 EEF pose
MS3 EEF pose

MS2 qpos
MS3 qpos

MS2 peg pose
MS3 peg pose

MS2 target EE pose
MS3 target EE pose

MS2 target qpos
MS3 target qpos

contact state
task stage
success predicate
```

如果能拿到 controller internal target，也一起记录。

---

## 4.5 核心误差

EEF position divergence：

\[
e_p(t)
=
\left\|
p_t^{MS3}
-
p_t^{MS2}
\right\|_2
\]

EEF orientation divergence：

\[
e_R(t)
=
d_{SO(3)}
\left(
R_t^{MS3},
R_t^{MS2}
\right)
\]

joint divergence：

\[
e_q(t)
=
\left\|
q_t^{MS3}
-
q_t^{MS2}
\right\|_2
\]

peg divergence：

\[
e_{peg}(t)
=
d
\left(
T_{peg,t}^{MS3},
T_{peg,t}^{MS2}
\right)
\]

---

## 4.6 必须标记关键事件

至少标记：

```text
approach
grasp/contact
lift
transport
pre-insertion
first hole/rim contact
insertion
success
```

输出一张 timeline：

```text
time ─────────────────────────────────────────────>

approach  grasp  transport  pre-insert  contact  insert
                 ↑
             first divergence
```

并画：

```text
EEF position error vs time
EEF rotation error vs time
joint error vs time
peg pose error vs time
```

---

# 5. Episode 2 专项诊断

## 5.1 为什么单独看 episode 2

目前：

```text
Original conversion: fail
B:                   fail
C:                   fail
```

因此 episode 2 是当前最有价值的 stable counterexample。

---

## 5.2 需要比较四条轨迹

```text
1. source pd_joint_pos actual trajectory
2. source pd_joint_pos controller target trajectory
3. B converted trajectory
4. C converted trajectory
```

从 source target qpos 计算：

\[
T^{src,target}_t
=
FK(q^{src,target}_t)
\]

定义 conversion realization error：

\[
e_{FK}(t)
=
d
\left(
T^{src,target}_t,
T^{converted,actual}_t
\right)
\]

同时比较 converted controller target qpos：

\[
e_{q,target}(t)
=
\left\|
q^{converted,target}_t
-
q^{src,target}_t
\right\|_2
\]

---

## 5.3 必须回答的四个问题

### Q1

第一次明显 EE divergence 出现在：

```text
contact 前
还是
contact 后
```

### Q2

在 divergence 发生时：

```text
EEF target 本身是否已经错误？
```

如果是：

> converter / action semantics 仍有问题。

### Q3

如果 EEF target 正确：

```text
新的 EEF controller 是否产生了明显不同的 target qpos？
```

如果是：

> IK / redundancy / controller realization 是主要候选。

### Q4

如果 contact 前一直能很好跟踪：

```text
是否在 peg-hole contact 后误差突然增长？
```

如果是：

> precision contact / controller compliance / dynamics gap 才是主要候选。

---

# 6. 决策树

```text
P1: MS2 exact replay
│
├─ FAIL
│   └─ 修 Policy Decorator state/replay reconstruction
│      不进入 MS3 cross-version 分析
│
└─ PASS
    ↓
P2: controller-contract test
│
├─ FAIL
│   └─ 修 action semantic converter
│      frame / sign / scale / representation 未锁死
│
└─ PASS
    ↓
P3: matched trajectory divergence
    │
    ├─ 很早 / contact 前明显分叉
    │   ├─ target transform 错
    │   │   └─ semantic converter 问题
    │   │
    │   └─ target transform 对，但 q / realized pose 分叉
    │       └─ IK / controller realization gap
    │
    └─ approach / grasp / transport 基本一致
        只在 insertion contact 后分叉
        └─ precision contact / dynamics / compliance gap
```

---

# 7. Stop Rule

本轮诊断完成 P1–P3 后，不再无限扩展 converter。

如果出现以下任一结果：

### Case A — IK / controller realization gap

```text
physical EE target 正确
但新的 EEF controller 在 contact 前就稳定产生不同 joint realization
```

则：

> 停止要求 `pd_joint_pos → EEF` conversion 达到 10/10。

转为：

```text
直接生成 ManiSkill3 native pd_ee_delta_pose demonstrations
```

### Case B — Contact / precision gap

```text
free-space trajectory 高度一致
但 insertion contact 后稳定分叉
```

则：

> 认定 converted joint expert 不适合作为正式 EEF `D_core`。

同样转为：

```text
直接生成 ManiSkill3 native pd_ee_delta_pose demonstrations
```

### Case C — Semantic conversion 能解决

如果：

```text
P2 semantic converter 通过
+
P3 trajectory divergence 大幅下降
+
正式 replay 达到预注册 M1 gate
```

才允许重新评估 converted dataset 是否可进入 `D_core`。

---

# 8. 本轮禁止事项

为了避免诊断发散，本轮不做：

- 不放宽当前 M1 10/10 smoke gate；
- 不把 8/10 数据用于正式 Flow-Matching base；
- 不继续盲调 quaternion multiplication order；
- 不继续只增加 retry 次数；
- 不做宽范围 controller gain sweep；
- 不同时修改 task geometry；
- 不因为 Policy Decorator raw-action 0/10 就直接归因于 physics gap；
- 不因为 `pd_ee_pose` 8/10 就直接否定 EEF controller；
- 不在 initial state 未对齐时用 cross-version final success 做强结论。

---

# 9. 最小输出文件

建议实现后至少落盘：

```text
reports/experiment_log.md

artifacts/peg_diag/p1_exact_replay.csv
artifacts/peg_diag/p2_controller_contract.csv
artifacts/peg_diag/p3_trajectory_metrics.csv

artifacts/peg_diag/plots/
    eef_position_error_*.png
    eef_rotation_error_*.png
    joint_error_*.png
    peg_pose_error_*.png
```

并在：

```text
reports/experiment_log.md
```

追加本轮结论。

---

# 10. 最终需要回答的结论模板

完成 P1–P3 后，只允许从下面几类结论中选择有证据支持的一类。

## Conclusion A — Replay reconstruction problem

> Policy Decorator 数据尚未在原始 ManiSkill2 contract 下被正确复现，因此当前不能进行 MS2 → MS3 controller/domain-gap 推断。

## Conclusion B — Action-contract problem

> Policy Decorator raw actions 的跨版本失败主要来自 action frame / rotation / scaling contract 不兼容；显式 physical-transform conversion 后 trajectory consistency 显著改善。

## Conclusion C — IK / controller realization problem

> 即使 initial state 与 physical EE target 已对齐，ManiSkill2 与 ManiSkill3 的 controller / IK realization 仍在接触前产生稳定 trajectory divergence，因此 native MS2 EEF trajectory 不能直接作为 MS3 EEF demonstration。

## Conclusion D — Precision-contact problem

> Free-space approach / grasp / transport 可以较好跨版本复现，但 trajectory 在 peg-hole contact / insertion 阶段系统性分叉，说明主要瓶颈位于精密 contact control / dynamics，而非简单 action-coordinate conversion。

对于 C / D：

> 不再继续强制转换 joint-space expert，改为生成 ManiSkill3 native `pd_ee_delta_pose` demonstrations。

---

# 11. 执行顺序

严格按以下顺序实现：

```text
[x] P1.1 建立 ManiSkill2 0.5.3 + Policy Decorator replay 环境
[x] P1.2 exact replay 固定 20 episodes
[x] P1.3 若 20/20，通过后扩 50 episodes
[x] P1.4 冻结 reference replay contract

[x] P2.1 建立 MS2/MS3 fixed-state no-contact probe
[x] P2.2 跑 12 个 ±xyz / ±rxyz elementary commands
[x] P2.3 实现 physical EE transform semantic bridge
[x] P2.4 锁死 frame / sign / scale / rotation contract
[x] P2.5 冻结 semantic converter

[x] P3.1 选择 Policy Decorator cross-version episodes
[x] P3.2 恢复 / 对齐 initial physical state
[x] P3.3 跑 MS2 reference trajectory
[x] P3.4 跑 MS3 semantic-converted trajectory
[x] P3.5 记录逐步 EEF / joint / peg divergence
[x] P3.6 标记 first divergence 与 task stage

[x] P3.7 对 episode 2 做 source/B/C 专项对比

[x] 根据 Decision Tree 写最终 M1 诊断结论
[x] 决定：继续 semantic conversion / 停止 conversion / 生成 MS3 native EEF demos
```

---

## 一句话目标

这轮不再问：

> “还能不能把 converter 从 8/10 调到 10/10？”

而是先回答：

> **“Policy Decorator reference 数据是否可在原环境 exact replay；MS2/MS3 的 EEF action contract 能否被显式对齐；在这些因素对齐后，trajectory 第一次真正不可复现的位置到底在哪里？”**

只有这个问题回答清楚，才决定是否值得继续 conversion。
