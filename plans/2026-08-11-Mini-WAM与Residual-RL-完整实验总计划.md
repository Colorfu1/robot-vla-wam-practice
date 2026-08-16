# 2026-08-11 Mini-WAM 与 Residual RL 完整实验总计划

- 日期：2026-08-11
- 状态：`in-progress`（M0 `done`；M1-W `done`；M1-R 用户暂缓；当前推进 M2-W，之后进入 W0/W1）
- 最后更新：2026-08-13；WAM 与 Residual 改为独立依赖链。M1-R 暂待切换 `maniskill2-pidec`
  环境后恢复。WAM 保留 ManiSkill3，R0/R1
  改在 Policy Decorator 的 ManiSkill2 0.5.3 + repo-local `PegInsertionSide-v2` 上执行，
  R2/R3 仍进入 LIBERO。
- 类型：综述性母计划；不直接作为实验执行单。
- 作用：把阶段零、Joint Mini-WAM、W4 Imagined Actor-Critic、Residual RL 和最终 LIBERO 迁移整理为一份唯一总路线。
- 执行原则：每个编号开跑前另建一份可直接执行的子计划；完整命令、机器配置和最终超参数写入子计划，不写死在总计划中。

## 0. 合并规则与版本优先级

本文件按以下优先级合并现有计划，后者覆盖前者的冲突内容：

1. 阶段零、M0/M1 和全局数据协议：以 `Mini-WAM与残差RL三阶段实验计划-M2修订版` 为底稿；
   2026-08-12 的 MS3-WAM / MS2-Residual 分线决议覆盖其中的三任务同栈假设。
2. Mini-WAM：以 `Mini-WAM Joint World-Action实验计划-W4-ActorCritic版` 为最终版本；它覆盖旧 W0–W3 reranking 路线，并新增 W4。
3. Residual RL：沿用 `Residual-RL-R0-R3-PolicyDecorator-FlowMatching修订版` 的算法结构，
   但 R0/R1 不再迁移到 ManiSkill3；二者固定在 Policy Decorator 的原始 MS2/v2
   environment/data/controller contract，R2/R3 才迁入 LIBERO。
4. `F1 → F2 → F3`：当前统一标记为 **TODO**，不在本轮展开。
5. `F0`：旧总计划中的 F0 仍建立在 candidate/reranking WAM 上，而新版 Joint Mini-WAM 文件没有提供新的 LIBERO F0；因此本文件对 F0 做了**衔接性重写**：保留“把 ManiSkill 上成立的 WAM 机制迁移到语言条件 VLA”这一目的，但改成 Action-only vs Joint world-action 的 direct-action 公平对照。该段是本次整合决策，开跑前仍需单独子计划冻结细节。

因此以下旧定义从主线中删除：

- Mini-WAM ≠ `Frozen policy → K proposals → world-model critic → argmax`；
- Oracle@K gap 不再是 W0–W3 的进入 gate；
- M3 success/failure rollout 不再是 W0 的前置条件；
- W0–W3 不使用 reward/value head，不把 test-time reranking 定义为 Mini-WAM；
- 旧 W3 的 ensemble / pessimism / data aggregation 不再属于 Mini-WAM 主实验。

---

## 1. 总研究目标

整个实验链回答三个彼此独立、最后再组合的问题：

1. **Mini-WAM / Joint World-Action Learning**：在相同 demonstrations、相同 action architecture 和相同 inference path 下，训练时加入 action-conditioned future/world supervision，能否直接学出更好的 Flow-Matching action policy？
2. **W4 / Model-Based RL**：如果 learned world representation 已经对 direct policy 有价值，能否进一步把 learned dynamics 当作 imagination environment，在不增加真实训练交互的前提下优化主 actor？
3. **Residual RL**：一个已经掌握主体行为的 frozen imitation/VLA policy，能否通过小型 residual 获得闭环提升；并且在相同真实交互预算下，包括 reward optimization、critic/bootstrapping 和 exploration/data collection 在内的完整 Residual RL 算法，是否比只用新数据继续模仿学习更有价值？

最终 LIBERO 阶段再研究迁移、组合、样本效率与泛化；其中 F1–F3 暂不展开。

---

## 2. 总路线

```text
阶段零：协议与基座
M0 ManiSkill3 端到端探针（done）
        ↓
M1-W：MS3 StackCube GPU 数据与控制协议冻结（done）
        ↓
M2-W：RGB Action-only Flow baseline + headroom
        ↓
       W0 → W1 → W2 → W3 ─────→ F0
                      │
                      ├→ W4（可选增强）
                      └→ W2-P（可选 reranking probe）

M3 / W-Data（可选数据扩展；不 gate W0）

M1-R：MS2 0.5.3 + Policy Decorator Peg v0-data/v2-env 协议冻结（独立 Residual 线）
        ↓
M2-R：MS2 state Flow baseline + precision headroom
        ↓
       R0 → R1 ───────────────→ R2 → R3
        MS2/v2                   LIBERO

F0 + R3 完成后：
F1 → F2 → F3    # TODO
```

### 2.1 当前任务执行顺序

当前只允许按以下顺序推进：

1. M1-W 已完成 StackCube state banks、finalize 和 frozen-protocol validation；M2-W 已解除该项
   协议阻塞，但仍按 M2-W 子计划执行。
2. M1-W 关闭后执行 M2-W：先冻结 Action-only Flow baseline 与 development headroom，再进入
   W0。W0 通过 action intervention gate 后才启动 W1；不能从 M2-W 跳过 W0 直接扫 horizon。
3. M1-R→M2-R 是独立 Residual 依赖链，不再阻塞 M2-W。当前优先完成 W 线；恢复 Residual 线时
   仍必须先完成 M1-R，且不得把 MS3/MS2 数据、controller、horizon、state bank 或指标并表。

M2-W 的可执行细节由
[2026-08-13 M2-W StackCube Action-only Flow 基线与 Headroom](2026-08-13-M2-W-StackCube-Action-only-Flow基线与Headroom.md)
冻结。

### 2.2 三条方法线的边界

```text
W0–W3  Mini-WAM:
同一 demonstrations
→ future/world auxiliary supervision
→ 更新 direct action policy
→ inference 只走 action path

W4  Imagined Actor-Critic:
learned world model
→ imagined transitions / return
→ 更新主 actor
→ 主版本额外真实训练交互 = 0

R0–R3  Residual RL:
real simulator feedback
→ SAC / residual correction
→ base 永久冻结
→ 只更新 residual policy / critic
```

W4 与 Residual RL 都带 RL 成分，但不能合并归因：前者依赖 learned dynamics 并更新主 actor，后者依赖真实环境并保持 base frozen。

---

## 3. 全局实验单元与任务选择

### 3.1 实验单元

后续不允许只写“StackCube 实验”或“PegInsertionSide 实验”。每个正式实验必须绑定：

```text
task
× demonstration source
× controller / action dimension
× backend
× horizon
× student observation
× action normalization / clipping contract
× success / termination / reward contract
× train/development/test split
× evaluation protocol
```

同一 task 在不同数据源、controller、horizon 和评测语义下可以表现完全不同，因此“任务名”不是实验条件的充分描述。

### 3.2 任务与数据先验

| 路线 / 阶段 | 任务 | 默认演示源 | 角色 |
| --- | --- | --- | --- |
| Mini-WAM | ManiSkill3 `StackCube-v1` | M1-W 冻结的 GPU-native 7D source / LeRobot 数据集 | 主候选 |
| M1-W 数据与协议 | ManiSkill3 `StackCube-v1` | GPU-native 7D source；官方 motion-planning 仅作 provenance | 主 WAM 候选；GPU route |
| M1-R 协议预检 | ManiSkill2 0.5.3 + Policy Decorator `PegInsertionSide-v2` | Policy Decorator 原生 7D EEF demonstrations（metadata 为 `PegInsertionSide-v0`） | M1-W 完成后执行的独立 Residual 协议 gate |
| Residual R0/R1 | ManiSkill2 0.5.3 + Policy Decorator `PegInsertionSide-v2` | Policy Decorator 原生 7D EEF demonstrations（metadata 为 `PegInsertionSide-v0`） | 保持数据、controller、环境与 residual recipe 同栈 |
| Residual R2/R3 | LIBERO + frozen SmolVLA | LIBERO 冻结 train demonstrations + 明确记账的在线交互 | VLA 迁移与算法归因 |

- `PickCube`：仅保留 M0、回归、harness smoke 和论文参照，不再承担方法结论。
- `PlugCharger-v1`：历史 full replay 和 actual-EEF retarget 均未通过，已从 M1-W 正式数据线排除；相关诊断只作为路线决策证据。
- `TurnFaucet`：ManiSkill 3.0.1 当前的官方可下载 demonstrations 和 motion-planning solver 都不提供对应主线，因此不再作默认 fallback。若未来使用，必须另建数据生成或 ManiSkill2 迁移子计划。
- 不做全任务横扫。M1-W 只冻结 `StackCube-v1`。MS3 Peg/Plug 的 joint→EEF
  conversion 与跨版本复用已从正式路线移除；既有诊断保留为分线依据，不再要求生成
  ManiSkill3 native EEF demonstrations 才能启动 R0/R1。

---

## 4. 数据与评测边界

### 4.1 ManiSkill3 WAM 路线

1. StackCube 使用固定 revision 的官方 motion-planning provenance 和 GPU-native 7D source；
   正式数据走 GPU baseline replay、失败集 EE repair、fingerprint merge 和 final live RGB replay，
   不切换到官方 RL demonstrations。这一规则只属于 ManiSkill3 WAM 路线，不外推到 MS2 Residual 路线。
2. 每个任务训练前必须审计并落盘：
   - replay survival；
   - first-success 分布；
   - post-success tail 比例；
   - action channel 范围、分位数、越界率；
   - controller / action dimension；
   - normalization / clipping 前后统计；
   - demonstration first-success p90 与 eval horizon 的兼容性；
   - student observation schema；
   - success predicate、termination 与 sparse/dense reward 的语义一致性；
   - train/development/test state 隔离报告。
3. PickCube 4D 的 `replay survival ≥95% / action 越界=0 / 成功前帧≥85% / first-success p90≤90` 只作为协议模板起点；StackCube 在 M1-W 子计划中已冻结自己的数据保留、horizon 和 fingerprint gate，不机械复用 PickCube 阈值。
4. W0–W3 的核心数据固定为：

```text
D_core = M1-W 清理并冻结的 task demonstrations
```

同一 trajectory 同时提供当前 observation、ground-truth action chunk 和 future target，因此 future supervision 不计作额外 interaction data。
5. W0–W3 的 Action-only / True-Future / Shuffled-Future 对照必须使用完全相同的 `D_core`、训练 budget 和 split。
6. 任何 frozen-policy success/failure/recovery rollout 都属于额外 data augmentation；必须单列为 `M3 / W-Data`，不能混进核心 Mini-WAM 因果实验。
7. development 用于选 checkpoint / horizon / architecture；正式 test 只在配置冻结后运行一次。

### 4.2 M1-R：ManiSkill2 Policy Decorator Residual 路线

- Python/environment 固定为 `mani-skill2==0.5.3` 与可复现的 Policy Decorator checkout；
  不在同一 Python 进程中混装 ManiSkill3。
- Policy Decorator checkout 固定 commit `92ba9ba442587ae5989c286355ace2200a0537fb`；native Peg
  H5 SHA-256 固定 `2483eb42a8995203b55e5c29795f5b3298c1724ab8fede3795de809636e7ef3b`，
  JSON SHA-256 固定 `2cc855c7fb6173fc59f1c0fdec72ecde3b900a151772ad4171e745b13a90245d`。
- runtime task 固定 Policy Decorator repo-local `PegInsertionSide-v2`；数据 metadata 中的
  `PegInsertionSide-v0` 是原始任务 ID，不得误写成“v2 数据”。
- demonstrations 固定 Policy Decorator 原生 `pd_ee_delta_pose` 7D 数据。不得再从 MS3
  motion-planning joint 数据转换，也不得把 MS2 action 跨版本喂给 MS3。
- R0/R1 前先冻结 checkout commit、H5/JSON hash、986 条 episode identity、50/50 exact
  replay reference、50D privileged state schema、action bounds、success/reward/termination、
  simulator backend 与 episode TimeLimit。MS3 formal horizon 300 不继承到这里；R0 子计划
  必须从 Policy Decorator source/runtime contract 解析并冻结实际 evaluation horizon。
- 时序接口保持 observation/prediction/action/execution `2/16/4/4`，每次 decision 实际执行
  4 个 low-level actions；若 source-level audit 发现不同，先停止并记录 deviation。
- MS2 的 dataset/split、checkpoint 和 outputs 与 MS3 WAM 物理隔离，跨线指标不得并表成同一
  benchmark。

### 4.3 LIBERO

- 训练只允许使用目标 train demonstrations、允许使用的背景 train demonstrations，以及专门新生成的 RL train init states / train rollouts。
- validation/test state、held-out paraphrase rollout 禁止进入训练。
- held-out paraphrase 保持固定 hard split，不根据 baseline 成绩重新造 OOD。
- 进入 LIBERO 方法实验前必须：
  - 修复 Prompt Swap 判定 bug；
  - 冻结明确的 C1/C2/SmolVLA base checkpoint；
  - 重测 canonical、held-out paraphrase 与必要的背景 baseline；
  - 为 RL 生成独立训练 init states，并证明其分布与官方状态协议相容。

### 4.4 真实交互统一记账

后续任何 sample-efficiency claim 都必须显式区分：

```text
real environment interactions
imagined model steps
offline demonstration transitions
```

imagined steps 不能和真实 steps 混称 sample efficiency；W4 主版本真实新增训练交互为 `0`，R0–R3 的真实 simulator interactions 必须逐项计账。对 chunk controller 同时报告 policy decisions/chunks 和实际 low-level simulator action steps，避免把“一次 4-step chunk”记成一次交互。

---

# 阶段零：ManiSkill 协议与基座冻结（M0–M3）

## M0 — 端到端探针 `done`

M0 不再追加实验，冻结结论：

- 正式结果与数值只保留在 `reports/` 的 M0 结论和 ManiSkill guide 中，本母计划不重复载入。
- 向后约束一：PickCube 仅保留为 regression/harness/paper-aligned 参照，不承担 WAM 或 Residual RL 方法结论。
- 向后约束二：已被诊断 sweep 反复查看的 state block 不再作正式 test，每个新任务重建独立 split。
- 向后约束三：每个新任务必须重新审计 post-success tail、action range/controller 和 horizon，不把“官方数据可 replay”当成协议已干净。
- 向后约束四：paper-aligned PickCube 的 motion-planning / 4D / CPU / horizon-100 contract 只作 regression 模板，不机械套用到 Stack/Peg/Plug。

**M0 → M1-W：M1-W 数据链、state banks 与 frozen protocol 已全部通过，解除 M2-W 的协议阻塞。**

---

## M1-W — ManiSkill3 WAM 数据与控制协议预检 `done`

### 要验证什么

不是“哪个任务最难”，而是 WAM 候选在当前 ManiSkill3 版本下能否形成干净、可解释、
可复现的数据与 action contract。

M1-W 只对 ManiSkill3 `StackCube-v1` 形成正式 WAM 输入协议。官方 motion-planning 数据只作
provenance；正式 source 是已有 GPU-native 7D `pd_ee_delta_pose` artifact，formal horizon 固定
500，native horizon 50 只作诊断。PlugCharger 因 full replay survival 和 actual-EEF retarget
均未通过而退出本路线，Peg 进入后续独立 M1-R。

### 必须落盘

1. replay survival；
2. first-success 分布与 post-success tail；
3. action channel 范围、越界率、rotation / gripper 活跃度；
4. controller / action dimension / normalization / clipping contract；
5. observation/prediction/action horizon、chunk slicing 与实际执行步数的兼容性；
6. student observation schema；
7. success predicate / termination / sparse-dense reward 语义；
8. train/development/test 状态隔离。

### Peg 的排除边界

Peg 的 P1 exact replay、P2 controller contract 与 P3 matched-state divergence 已作为诊断证据
完成。该证据只支持“MS2 原生 EEF contract 可用、跨到 MS3 的精密接触轨迹不可直接复用”，
不产生 ManiSkill3 Peg `D_core`。因此 M1-W 的 config、task loop、state bank、comparison 与
frozen protocol 中都不得出现 Peg；旧 predicate/conversion outputs 只保留，不作为新 gate。

### Gate

通过 M1-W 只表示：**StackCube 的 GPU 数据身份链和 WAM 输入协议干净到值得训练 base。**

如果不通过：先修 protocol；不能切 RL demonstrations，也不能把 Peg 拉回 MS3 作为自动替代。

可执行细节由
[2026-08-12 M1 WAM 数据与控制协议预检](2026-08-12-M1-WAM数据与控制协议预检.md)
冻结。旧三任务 M1 已标记 `dropped`。

---

## M1-R — Policy Decorator MS2/v2 Residual 协议预检 `blocked（用户暂缓）`

M1-R 是独立 Residual 里程碑，不复用 M1-W 的 config、data、controller、Python environment、
horizon 或 output root。它负责冻结本计划第 4.2 节定义的 Policy Decorator checkout、v0-data/
v2-env、原生 7D EEF action、50D state、episode TimeLimit、split 与 exact replay contract。

M1-R guide 已拆分为 `reports/guides/maniskill/03_m1_r_protocol_audit.md`；独立子计划、MS2/v2
formal audit runner 和可恢复的阶段产物已准备。source/environment audit 与 50 条 smoke 已通过，
但用户要求先暂停 full replay、D_core、state banks 与 protocol freeze，待切换到 `maniskill2-pidec`
环境后恢复。该暂停不阻塞 M2-W；恢复 Residual 线时仍按 M1-R→M2-R→R0 的顺序执行。

---

## M2 — 任务专属 baseline + headroom

M2 不再承担“Oracle gap 决定能否做 Mini-WAM”的职责。主线只做两件事：

1. 为 Mini-WAM 冻结正式 Action-only Flow baseline；
2. 为 Residual RL 冻结正式 state Flow base，并确认存在适合 residual correction 的 headroom。

### M2-W — Mini-WAM Action-only Flow baseline

默认先跑 ManiSkill3 `StackCube-v1` + M1-W 已冻结的 GPU-native 7D source / LeRobot 数据集；
官方 motion-planning 只作为 provenance。具体 archive、trajectory 数量与 hash 在 M1-W/M2-W
子计划中冻结。可执行合同见
[2026-08-13 M2-W StackCube Action-only Flow 基线与 Headroom](2026-08-13-M2-W-StackCube-Action-only-Flow基线与Headroom.md)。

#### Base

```text
RGB + robot state + task/goal
        ↓
visual/context encoder
        ↓
action expert
        ↓
Flow-Matching action chunk
```

训练目标：

$$
\mathcal L_{action}=\mathcal L_{fm}.
$$

M2-W 必须冻结：

- 数据源与 trim/weight 规则；
- observation schema；
- controller / action dimension；
- action normalization / clipping；
- chunk length / execution steps；
- Gaussian source-noise contract；
- Flow ODE solver 与 integration steps；
- train/dev/test split；
- checkpoint-selection protocol；
- 最终 baseline checkpoint。

#### M2-W Gate

进入 W0 只要求：

- Action-only Flow 明显高于 random，已经掌握任务主体；
- baseline 未接近 ceiling，存在稳定方法 headroom；
- 数据/controller/eval protocol 已冻结；
- checkpoint、训练预算和 split 可以作为后续严格公平对照。

**不再要求 Oracle@K gap。**

若 `StackCube-v1` base 已接近 ceiling 或明显不适合作为 headroom task，停止并另建候选任务
子计划；不自动切 Plug/Peg/TurnFaucet，不做全任务扫描。Plug 若要重开，必须先有新的独立数据与
horizon 协议计划，不能复用已退出 M1-W 的历史产物。

### M2-P — Proposal / Oracle / reranking mechanism probe `optional`

旧 M2-W 的 candidate/Oracle 审计降级到可选 probe，只服务后面的 W2-P，不 gate W0–W4。

若执行，保持以下协议：

```text
同一个 o_t
  ├─ ε1 → Flow sampler → A1
  ├─ ε2 → Flow sampler → A2
  └─ ...             → AK
```

- candidates 只改变 Gaussian source noise；
- Oracle continuation 使用 matched future-noise streams，避免把 candidate quality 与后续随机采样幸运程度混在一起；
- `Oracle@K / MixedOutcome@K / stage disagreement` 只作 privileged evaluation upper bound；
- Oracle labels 不进入 Mini-WAM / DirectScore 训练；
- 无论 Oracle gap 大小，都不改变 W0–W3 的主线定义。

### M2-R — Residual RL frozen base

固定独立的 Policy Decorator reference stack：

```text
mani-skill2 == 0.5.3
Policy Decorator pinned checkout
runtime env = repo-local PegInsertionSide-v2
dataset metadata env = PegInsertionSide-v0
native controller/data = pd_ee_delta_pose, 7D
```

checkout 与 H5/JSON hash 使用本计划第 4.2 节的固定值。子计划还必须冻结 986 条 episode identity、50D privileged state
schema、action bounds、success/reward/termination、backend、TimeLimit 和 split。不得用模糊的
“v2 数据”称呼：`v2` 是 repo-local corrected runtime env，demonstration metadata 仍为 `v0`。

M1-R 必须重验已有 exact replay contract，并明确 50/50 native/strict reference
仍成立；MS2 与 MS3 使用不同 conda environment 和 output root。MS3 的 motion-planning 数据、
Peg formal horizon 300、strict predicate 和 joint→EEF converter 都不进入此线。

R0 的第一版 base 使用 **low-dimensional state-conditioned Flow-Matching policy**，而不是 RGB；这是为了让 R0 只验证 residual-control pipeline，不同时承担 perception 问题。

进入 R0 前要求：

- base 明显高于 random；
- 已掌握“把 peg 移向 hole”等任务主体行为；
- 仍有明显 precision / alignment / insertion headroom；
- 失败主要是可被 chunk-level 闭环 residual 纠正的执行误差，而不是完全不知道任务怎么做。

若 base 太弱，先修 MS2 base/data/v2-predicate contract。若协议已确认干净但该 reference task
仍不适合 residual refinement，停止 R0/R1；不自动切到 MS3 Peg、Plug 或其他任务。R2/R3 的
LIBERO 任务另按其 frozen SmolVLA headroom 选择，不受 ManiSkill2 task identity 绑定。

---

## M3 / W-Data — Policy rollout 数据扩展 `optional / non-gating`

新版中，M3 **不再位于 M2 → W0 的主链路上**。

核心 W0–W3 与 W4 主版本首先只使用 `D_core`，保证 world supervision 的收益不和“多拿数据”混淆。

后续如果要研究 failure/off-policy data 是否增强 world supervision，可单独采集：

- frozen base success rollouts；
- frozen base failure rollouts；
- simulator 原生 stage labels；
- recovery / near-failure states；
- 唯一明确的采集 checkpoint。

任何使用 M3 数据的实验必须同时给 Action-only baseline 相同的数据预算，或明确标成 data-augmentation study，不能把结果归因给 Mini-WAM 本体。

---

# 阶段一：Joint Mini-WAM + W4

#### 1. 为什么重设计

##### 1.1 已存在的 mini-WAM 不是旧计划里的 reranker

`keivalya/mini-wam` 对 mini-WAM 的直接定义是：

```text
image_t + language + state_t
        ↓
action_t + state_{t+1} + image_{t+1}
```

它的核心思想是 **“act while predicting what its action will do to the world”**。当前仓库仍是 WIP，但其方向从一开始就是 action generation + future prediction，而不是：

```text
Frozen policy → K proposals → world model/value score → argmax
```

因此旧计划实际验证的是一个 **lightweight action-conditioned world model + test-time reranking**，不是原 mini-WAM 的主范式。

##### 1.2 当前更成熟的 WAM 工作也支持 joint world-action learning

- **Fast-WAM（arXiv:2603.16666）**：训练时保留 world/video co-training，action expert 与 world representation 联合学习；推理时可跳过显式 future generation，直接输出 action。
- **LiLa-WAM（arXiv:2608.03701）**：在一个轻量共享 stream 中联合 future-state prediction 与 Flow-Matching action generation；训练目标为

  $$
  \mathcal L = \mathcal L_{fm}+\lambda_{ff}\mathcal L_{ff},
  $$

  future decoder 只在训练时使用，部署时丢弃。
- **WLA（arXiv:2606.05979）**：world prediction 通过共享表示影响 Action Expert；world prediction 可在 inference 关闭，也可以作为额外 test-time scaling 打开。

因此本项目的 Mini-WAM 主问题改为：

> **future/world supervision 能否塑造一个更好的 direct action policy？**

而不是：

> 一个外接 world-model scorer 能否替 frozen policy 选 action？

---

#### 2. 新 Mini-WAM 的模型定义

##### 2.1 Action-only Flow baseline

先保留 M2 训练得到的任务专属 Flow-Matching policy 作为严格 baseline：

```text
RGB + robot state + task/goal
              ↓
       visual/context encoder
              ↓
         action expert
              ↓
     Flow-Matching action chunk
```

训练目标只有：

$$
\mathcal L_{action}=\mathcal L_{fm}.
$$

以 ground-truth action chunk $x_1$、Gaussian source noise $x_0$ 和 flow time $s$ 为例：

$$
x_s=(1-s)x_0+s x_1,
\qquad
v^*=x_1-x_0,
$$

$$
\mathcal L_{fm}=\mathbb E\|\hat v_s-v^*\|_2^2.
$$

##### 2.2 Joint Mini-WAM

第一版不生成 RGB，不加 reward/value head，不做 MPC。结构固定为：

```text
            RGB + robot state + task/goal
                         ↓
                 visual/context tokens
                         │
            noisy action tokens x_s
                         │
                         ▼
             shared action/world trunk
                 /                \
                /                  \
       Flow velocity head       future latent head
              ↓                       ↓
        action chunk A         z_hat_{t+Δ}
```

关键约束：

1. **future head 与 action head 必须共享真正参与 action generation 的 trunk**；不能只在视觉 encoder 外面挂一个与 action 无关的 auxiliary MLP。
2. future prediction 必须能够读取 action-token 信息，使其有机会表示

   $$
   (o_t, A_{t:t+H})\rightarrow o_{t+\Delta}
   $$

   的 consequence，而不是仅靠当前 observation 猜最可能的下一状态。
3. 第一版复用 Action-only baseline 的视觉 encoder / action expert 规模；**不同时更换成更大的 backbone**，避免把收益归因给模型容量变化。
4. future target 使用 latent feature，不预测像素。

##### 2.3 Future target

用 frozen / EMA target encoder 提取真实未来 observation：

$$
z^{target}_{t+\Delta}
=
\operatorname{sg}(E_{target}(o_{t+\Delta})).
$$

共享 trunk 产生 predicted future latent，再通过轻量 decoder/projector 对齐 target feature：

$$
\hat z_{t+\Delta}=F_{future}(h_t).
$$

第一版 future loss 使用 feature-space cosine loss：

$$
\mathcal L_{future}
=
1-\cos(\hat z_{t+\Delta},z^{target}_{t+\Delta}).
$$

总训练目标：

$$
\boxed{
\mathcal L
=
\mathcal L_{fm}
+
\lambda_{future}\mathcal L_{future}
}
$$

##### 2.4 Deployment contract

正式 Mini-WAM 部署时：

```text
observation
   ↓
Mini-WAM shared trunk + Flow action head
   ↓
action chunk
   ↓
execute
```

**future decoder 默认关闭/丢弃。**

主方法 inference 中不存在：

```text
K candidates
Oracle
DirectScore
reward head
value head
argmax reranking
ensemble pessimism
```

这些只能作为额外机制探针，不能再定义 Mini-WAM 本体。

---

#### 3. 对 M2 / M3 的结构性修改

##### 3.1 M2-W 不再要求 Oracle gap 才能进入 Mini-WAM

旧 M2-W 的 `Oracle@8 - First@1` 是 reranking 路线的前提，但 **joint Mini-WAM 直接输出 action，不依赖 candidate selection**。

因此主线 M2 只需确认：

- Flow-Matching Action-only baseline 已掌握任务主体；
- baseline 没有接近 ceiling；
- 数据/controller/eval protocol 已冻结；
- baseline checkpoint、训练预算和 split 可作为后续公平对照。

旧的 Flow candidate / matched continuation / Oracle@K 审计改名为：

> **M2-P：Proposal / reranking mechanism probe（可选）**

它只服务后面的可选 W2-P，不再是 W0–W3 Mini-WAM 主线或 W4 的进入 gate。

##### 3.2 M3 的 policy rollout 不再是 W0 前置条件

joint Mini-WAM 的主 claim 必须先在 **与 Action-only baseline 完全相同的数据**上成立。

因此核心 W0–W3 使用同一份数据；W4 的主版本也只从同一批数据构造 world/reward supervision：

```text
D_core = M1-W 清理并冻结的 task demonstrations
```

同一个 trajectory 同时天然提供：

- 当前 observation / robot state；
- ground-truth action chunk；
- 未来 observation / state target。

原 M3 的 success/failure frozen-policy rollout 移出主前置链路，后续如果要研究 off-policy / failure data 对 world supervision 的作用，再单独设 `W-Data` 数据扩展实验。

否则如果：

```text
Action-only 只看 demonstrations
Mini-WAM 额外看大量 policy rollout
```

即使 Mini-WAM 更好，也无法区分收益来自 `future supervision` 还是 `more data`。

---

### 4. 新 W0–W3 主实验 + W4 Model-Based RL 增强分支

#### W0 — Future branch 是否真的学到 action-conditioned future

##### 要证明什么

证明 future head 不是一个只靠当前 observation 做时间外推的普通 predictor，而是其预测**确实依赖 action tokens**。

核心命题：

$$
\hat z_{t+\Delta}
=
F(o_t,A_{t:t+H})
$$

中的 $A$ 不能被忽略。

##### 数据

只使用 `D_core` demonstrations；不使用 M2 Oracle terminal labels，不额外采 policy rollout。
第一版机制 gate 的 future target 固定为 M2-W 选中的 `Δ=n_action_steps`；W0 不在多个 horizon
之间选优。只有该锚点证明 action-conditioned 后，W1 才比较 `1/n_action_steps/H_chunk`。

##### 实验设置

在相同 observation 上做 action intervention：

1. **Original Action**：正确 action chunk；
2. **Action Replace**：替换成另一 trajectory / 不匹配状态的 action chunk；
3. **Action Noise**：对 action tokens 加扰动；
4. **Action→Future Mask**：阻断 future/readout 从 action tokens 获取信息；
5. **No-action / Current-latent baseline**：只根据当前 observation 预测未来。

这些 intervention 直接借鉴 LiLa-WAM 对 action-conditioned future tokens 的验证思路，但使用 M1-W/M2-W 最终冻结的 ManiSkill3 任务与数据协议。

##### 主要指标

- future latent cosine error / similarity；
- action replacement 前后 predicted future 的变化；
- 正确 action 与 mismatched action 对真实 future target 的预测差；
- 按 simulator stage 分层的 future error：grasp / transport / place / static 等；
- object pose / robot state probe 仅作解释性诊断，不作为唯一 gate。

##### 通过条件

必须同时看到：

- Original Action 的 future prediction 明显优于 No-action；
- Action Replace 后 future prediction 显著退化；
- 阻断 action→future 信息流后显著退化；
- 效应在多个 task stage 存在，而不是只由某个单一静态阶段贡献。

若不通过：

> 不能声称 Mini-WAM 学到了 action-conditioned world dynamics；先修共享 trunk / attention / future target，不进入 W1。

---

#### W1 — 什么 future horizon 才真正提供 foresight，而不是局部时间平滑

##### 要证明什么

W0 只证明 action 会影响 future prediction；W1 进一步问：

> **预测多远的 future 才会给 action learning 提供真正的 control-relevant foresight？**

如果只预测 $t+1$ 就和长 horizon 一样有效，那么我们的结果更接近“temporal auxiliary regularization”，不能强声称 multi-step foresight。

##### 设置

第一版只做小规模 development sweep，不做大范围扫参。

future target horizon：

```text
Δ = 1
Δ = n_action_steps        # 当前 chunk 实际执行到重新观测的位置
Δ = H_chunk               # action chunk 的末端
```

可选增加一个 multi-horizon head：

```text
Δ ∈ {1, n_action_steps, H_chunk}
```

共享同一 action-only architecture、训练数据、训练步数和 `λ_future` 初值。

##### 主要指标

- 各 horizon 的 future latent prediction error；
- action intervention sensitivity；
- development closed-loop success；
- 按阶段的 success / failure；
- action Flow loss 作为优化诊断，不作为最终 control 结论。

##### 选择规则

在 development 上预注册选择一个主 horizon，后续 W2/W3/W4 固定，不再用 test 调整。

优先选择：

> **能够改善 closed-loop action performance，同时仍具有稳定 action-conditioned future prediction 的最短 horizon。**

这样避免为了“更像 world model”而无意义地拉长预测距离。

---

#### W2 — Future supervision 是否真的改善 action learning

##### 要证明什么

这是第一个直接回答 Mini-WAM 核心因果问题的实验：

> **在完全相同的 demonstrations、action architecture 和 inference 路径下，正确 future supervision 是否让 action policy 变得更好？**

##### 必做对照

| Variant | Action loss | Future head | Future target | 作用 |
| --- | --- | --- | --- | --- |
| A0 Action-only Flow | ✓ | × | — | VLA / Flow baseline |
| A1 Joint Mini-WAM | ✓ | ✓ | 正确 `t+Δ` future | 主方法 |
| A2 Shuffled-Future | ✓ | ✓ | 其他 trajectory / mismatched future | 控制“多一个 head / 多一个辅助 loss” |
| A3 Current-State Aux（可选） | ✓ | ✓ | `z_t` | 控制“一般 feature reconstruction regularization” |

所有 variant 必须：

- 相同 demonstrations；
- 相同 train/dev split；
- 相同且预注册的单个 training seed；
- 相同 action expert 主体；
- 相同 optimizer / update steps / batch budget；
- 相同 Flow sampler / ODE steps；
- inference 都只走 action path。

##### 为什么 Shuffled-Future 是关键对照

如果：

```text
A0 Action-only      65%
A2 Shuffled-Future  66%
A1 True-Future      76%
```

才说明收益主要来自 **正确 temporal/world supervision**。

如果 A1 与 A2 相当，则只能说增加 auxiliary objective / regularization 有效，不能说 learned foresight 有效。

##### 主要指标

- development closed-loop success；
- stage-conditioned success；
- paired state-set bootstrap CI；
- action execution smoothness / invalid action rate 只作安全诊断；
- training memory / throughput；
- inference latency 必须与 A0 基本同级，因为 future decoder 不参与部署。

W2 不要求多个独立 training seeds。正式统计使用单个预注册 training seed、固定 paired initial-state bank 和 paired-state bootstrap；该 CI 只覆盖这个固定训练运行下的状态集差异，不声称覆盖训练随机性或跨 seed 稳健性。

##### Gate

进入正式 W3 至少要求：

- A1 > A0，且 paired bootstrap 95% CI 下界 > 0；
- A1 > A2，证明正确 future target 比 shuffled auxiliary supervision 更有价值；
- improvement 不能以明显更高 test-time latency 为代价。

建议将 `+5pp` 作为 practical strong-pass 参考，而不是领域通用阈值；统计显著性仍以预注册 paired CI 为主。

如果 A1 只提升 offline action loss、closed-loop 不涨，则记录：

> future supervision 改善了拟合，但没有转化成 control benefit。

停止把 Mini-WAM 推向后续组合实验。

---

#### W3 — Direct-action Mini-WAM 正式闭环验证

##### 要证明什么

这是新 Mini-WAM 主线的**最终方法实验**，取代旧 W3 的 ensemble/pessimistic-reranker 实验。

正式 claim：

> **一个训练时同时学习 future dynamics 的轻量 Flow action policy，在部署时不做 explicit future rollout、不做 K-candidate reranking，仍能比 action-only Flow policy 获得更好的闭环控制。**

##### 冻结配置

W3 开始前冻结：

- W1 选出的 `Δ`；
- `λ_future`；
- encoder / shared trunk / action expert 结构；
- Flow ODE solver 与 steps；
- checkpoint-selection protocol；
- development/test split；
- 数据规模和训练 step budget；
- 与 W2 一致的单个预注册 training seed 与 paired initial-state bank。

##### 正式比较

核心只保留：

```text
Action-only Flow
vs
Joint Mini-WAM
```

Shuffled-Future 保留为机制消融结果。W3 正式 test 不重复多个 training seeds，只对预注册的单个训练运行做一次 frozen paired-state 评测。

##### 正式评测

至少报告：

- overall closed-loop success；
- simulator 原生 stage success / failure；
- paired per-state outcome；
- development 与一次性 frozen test；
- inference latency；
- peak GPU memory；
- 参数量区分 train-time-only decoder 与 deployment-active parameters。

W3 的结论范围限定为“在预注册的单个 training seed 和固定 state bank 上”；paired bootstrap 不作为跨训练 seed 置信区间。

##### W3 成功时可以声称什么

若 Joint Mini-WAM 显著超过 Action-only，并且 W0/W2 的机制对照同时通过，可以声称：

> **action-conditioned future supervision 在共享 world-action representation 中提供了可用于控制的 foresight，并在不依赖 test-time world rollout / reranking 的情况下改善了直接 action policy。**

不能仅凭 W3 success 单独声称：

- latent 就是完整物理 simulator；
- 模型能进行长 horizon planning；
- future prediction 在 inference 是必要的。

这些需要额外实验。


#### W4 — Imagined Actor-Critic / Model-Based Policy Optimization（独立增强分支）

##### 定位

W4 **不是**把 critic 接到 Mini-WAM 后面，在 test time 从 `K` 个 proposal 中挑最高分动作。

它验证的是另一条经典 world-model control 路线：

> **既然 W0–W3 已经证明 Mini-WAM 学到的 future/world representation 对 action learning 有用，那么能否进一步把 learned world model 当作 imagination environment，在模型内部训练 actor + critic，再让优化后的 actor 直接输出动作？**

因此 W4 属于 **model-based RL / imagined policy optimization**，而不是 Mini-WAM 的基础定义。W0–W3 即使已经形成完整主结论，W4 仍可以独立失败。

##### 为什么不能直接把 W3 future head 当成 Dreamer world model

W3 的 future auxiliary objective 只要求：

$$
(o_t,A_{t:t+H})\rightarrow \hat z_{t+\Delta}.
$$

这足够验证 future supervision 是否改善 action policy，但还不足以支撑 actor-critic imagination。W4 至少需要额外建立：

1. **可递归 latent transition**

   $$
   \hat z_{t+1}=D_\psi(\hat z_t,a_t);
   $$

2. **reward / progress model**

   $$
   \hat r_t=R_\psi(\hat z_t,a_t);
   $$

3. **critic / value model**

   $$
   V_\phi(\hat z_t);
   $$

4. **可被 imagined return 优化的 actor**。

因此 W4 新增的是 **control model heads**，不是给 W3 加一个 scalar critic 就结束。

##### W4.0 — World-control model sanity check

先使用与 W0–W3 相同的 `D_core` / replay protocol 训练或适配：

```text
observation
   ↓
frozen / target encoder
   ↓
latent z_t
   │
   ├── action a_t ──→ latent transition Dψ ──→ ẑ_{t+1}
   │
   └── action a_t ──→ reward model Rψ ──────→ r̂_t
```

reward target 优先使用 ManiSkill simulator 原生 reward / task-progress signal，从**同一条 replayed trajectory**获得；第一版不额外通过在线交互扩大数据量。

进入 actor-critic 前必须确认：

- one-step latent transition 明显优于 copy/no-action；
- action shuffle 后 transition prediction 显著退化；
- reward / progress prediction 在 development 上具有稳定区分能力；
- recursive rollout 在预注册的短 imagination horizon 内没有立即失控。

这里不追求“完整 simulator”，只要求模型足够支持短 horizon policy optimization。

##### W4.1 — Actor 与 critic

主 actor **沿用 W3 的 Flow-Matching action policy 参数化**，而不是另起一个完全不同的控制器。

给定 latent state 和 Gaussian source noise：

$$
\epsilon_t\sim\mathcal N(0,I),
\qquad
A_t=F_\theta(z_t,\epsilon_t).
$$

Flow sampler 是可重参数化的；W4 第一版不依赖精确 `\log \pi(A|z)`，而是通过 differentiable Flow ODE + frozen world dynamics 对 imagined return 做 pathwise gradient。这样与后面的 PPO/SAC residual 线保持方法上的清晰区分。

critic 使用 imagined $\lambda$-return：

$$
G_t^\lambda
=
\hat r_t
+
\gamma\left[(1-\lambda)V_\phi(\hat z_{t+1})
+\lambda G_{t+1}^\lambda\right].
$$

critic loss：

$$
\mathcal L_V
=
\mathbb E\left[
\left(V_\phi(\hat z_t)-\operatorname{sg}(G_t^\lambda)\right)^2
\right].
$$

actor objective：

$$
J_{actor}(\theta)
=
\mathbb E\left[G_t^\lambda\right],
$$

梯度通过 sampled Flow action 和 frozen latent dynamics 回到 actor；**world model 在 actor update 时 stop-gradient / frozen**，防止 actor 与 model 同时漂移导致归因不清。

##### W4.2 — 第一版训练协议

W4 主版本固定为 **offline imagination first**：

```text
D_core demonstrations / replay
        ↓
train world transition + reward model
        ↓
freeze world model
        ↓
start from W3 Joint Mini-WAM actor
        ↓
imagined rollouts
        ↓
train critic + update actor
        ↓
real simulator only for development / final evaluation
```

第一版**不在训练过程中新增真实 environment interaction**。这样 W4 与 Residual RL 的差异最干净：

- W4：用 learned model 产生 imagined transitions；
- Residual RL：用真实 simulator interaction 学 residual。

若 offline-imagination W4 已经失败，不立刻增加 online model refresh；额外真实数据作为后续独立 `W4-Online` 扩展，不混入主结论。

##### W4.3 — 必做对照

| Variant | Future/world model | Critic | Actor optimization | 额外真实训练交互 | 用途 |
| --- | --- | --- | --- | ---: | --- |
| B0 Action-only Flow | × | × | BC / Flow Matching | 0 | 基线 |
| B1 Joint Mini-WAM (W3) | ✓ auxiliary | × | BC / Flow Matching | 0 | world supervision 主方法 |
| B2 Mini-WAM + Critic-only evaluation | ✓ | ✓ | **不更新 actor** | 0 | 控制“只是多了 value head” |
| B3 Mini-WAM + Imagined Actor-Critic | ✓ recursive | ✓ | **imagined return 更新 actor** | 0 | W4 主方法 |

`B2` 不做 K-candidate reranking；critic 只做诊断，真实部署动作仍由原 actor 直接输出。它用于确认 W4 的收益来自 **policy optimization**，而不是“训练了一个 value head”本身。

##### W4.4 — 正式评测

至少报告：

- closed-loop success；
- 相对 W3 Joint Mini-WAM 的绝对增益；
- stage-conditioned success / failure；
- imagined return 与真实 rollout return 的 calibration；
- actor update 前后 action distribution shift；
- world-model rollout error 随 imagination horizon 的增长；
- 训练所用**额外真实 environment interactions = 0**（主版本）；
- imagined model steps、训练 wall-clock 和显存开销单独记账。

W4 的核心比较是：

$$
\text{Mini-WAM + Imagined Actor-Critic}
\quad vs\quad
\text{Joint Mini-WAM direct action}.
$$

如果 W4 提升，才支持：

> **learned world model 不仅能作为 action-learning auxiliary supervision，还能作为 imagination environment 进一步优化 policy。**

如果 W4 不提升甚至下降，则记录：

> **在当前数据和模型精度下，world supervision 对 direct policy 有价值，但 model-based policy optimization 受到 model bias / rollout error 限制。**

这不会推翻 W0–W3 的 Mini-WAM 结论。

##### W4.5 — 与 Residual RL 的边界

W4 与后续 Residual RL 不合并，二者回答不同问题：

```text
Joint Mini-WAM
      │
      ├──────── W4: Model-Based RL
      │            learned world model
      │            → imagined rollout
      │            → critic
      │            → optimize main actor
      │
      └──────── Residual RL
                   real simulator interaction
                   → reward / critic
                   → optimize residual Δa
                   → base actor frozen
```

核心差异：

| 维度 | W4 Imagined Actor-Critic | Residual RL |
| --- | --- | --- |
| dynamics 来源 | learned world model | real simulator/environment |
| 主训练交互 | imagined | real |
| 优化对象 | 主 actor / action policy | residual policy `Δa` |
| base 是否冻结 | world model 冻结；actor 被更新 | base actor 冻结 |
| 主要风险 | model bias / compounding error | real interaction cost / exploration |
| 主要问题 | world model 能否减少真实控制学习成本 | real-feedback residual 能否纠正 frozen base |

后续若两条线都通过，可以再做跨线比较：在相同**真实 environment interaction budget**下比较 W4、Residual RL 和 Online BC；该比较属于最终组合/效率实验，不作为 W4 的进入 gate。

---

### 5. 可选机制实验：W2-P / Test-time imagination probe

旧计划的 Oracle / candidate reranking 不删除，但**降级为可选 probe**，不再属于 Mini-WAM 主架构。

只有在 W3 已经证明 Joint Mini-WAM direct action 有收益后，才值得问：

> 训练出的 future model 在 test time 进一步显式使用，是否还能提供额外收益？

可比较：

```text
Joint Mini-WAM direct action       # 主部署方式
Joint Mini-WAM + K proposal rerank # 可选 test-time scaling
Oracle@K                           # simulator upper bound，仅 reference
```

这个实验回答的是 Fast-WAM 式问题：

> **WAM 的收益主要来自 training-time world supervision，还是还需要 test-time future imagination？**

若 direct action 与 explicit rerank 相当：

> future/world modeling 的主要价值在训练 representation/action policy；不需要部署时 planning。

若 rerank 仍显著更好：

> 显式 future prediction 还有额外 test-time control value，可作为后续 scaling 分支。

无论结果如何，都不再把 reranking 定义成 Mini-WAM 本体。

---

### 8. 数据公平性原则

1. **W0–W4 的真实环境训练数据必须从同一套冻结 demonstrations / replay 协议出发；W0–W3 核心对照严格使用相同 demonstrations。**
2. future target 只是同一 trajectory 的未来帧/状态，不算额外 interaction data。
3. 任何新增 frozen-policy rollout / failure / recovery data，必须单列为后续 data augmentation 实验。
4. 如果 Mini-WAM 使用额外 rollout，而 baseline 没有使用，结果不能单独归因给 world supervision。
5. W4 主版本允许无限制的 **imagined model steps**，但额外真实 environment interaction 必须保持为 `0`；imagined steps 与真实 steps 分开记账，不能混称 sample efficiency。
6. W4 的 reward/progress labels 必须来自同一 `D_core` replay / simulator 原生信号；若新增真实 rollout，必须另列为 `W4-Online`。
7. test state 一次性冻结；W1/W2 的架构与 horizon 选择只用 development。

---

### 9. 第一版复杂度边界

为了保持 Mini，**W0–W3 核心主线**仍然：

- 不预测 RGB / video pixels；
- 不引入大规模 video DiT；
- 不加 reward/value head；
- 不把 Oracle terminal rollout 当训练 label；
- 不把 K-candidate reranking 当默认 inference；
- 不在第一版同时更换视觉 backbone、action expert 和数据源；
- 不做 5-model ensemble / pessimistic planner。

**reward/value/actor-critic 只允许出现在 W4 独立增强分支。** W4 第一版也不允许边训练边新增真实 rollout；先验证纯 offline-imagination 是否成立。

如果 Joint Mini-WAM 的 W0–W3 本身都不能稳定超过 Action-only Flow，则 W4 不启动。这样避免用更复杂的 RL 机制掩盖基础 world-action learning 本身不成立。

---

### 10. 需要落盘的最小结果矩阵

| Experiment | Action-only | True-Future | Shuffled-Future | 主要结论 |
| --- | ---: | ---: | ---: | --- |
| W0 future error | — / baseline | ✓ | ✓ | future 是否 action-conditioned |
| W1 `Δ=1` | ✓ | ✓ | — | 局部 future supervision |
| W1 `Δ=n_exec` | ✓ | ✓ | — | 与闭环 replan horizon 对齐 |
| W1 `Δ=H_chunk` | ✓ | ✓ | — | chunk 末端 foresight |
| W2 development success | ✓ | ✓ | ✓ | 正确 future supervision 是否有因果贡献 |
| W3 formal test success | ✓ | ✓ | 机制结果复用 | direct-action Mini-WAM 主结果 |
| W3 inference latency | ✓ | ✓ | — | future branch 是否真正零/低额外部署成本 |
| W4 world-control sanity | — | ✓ | — | recursive dynamics / reward model 是否足够支撑 imagination |
| W4 imagined actor-critic | baseline 复用 | W3 vs W4 | — | world model 是否能进一步优化 direct actor |
| W4 real interaction budget | 0 | 0（主版本） | — | 与后续 Residual RL 分离真实样本成本 |

---

---

# 阶段二：Residual RL（R0–R3）

#### 0. 研究问题

Residual RL 主线研究的不是“RL 能不能从零学会机器人操作”，而是：

> 一个已经掌握任务主体行为、但在精细执行上仍会失败的 imitation / VLA policy，能否在**不修改 base 参数**的前提下，通过一个小型、chunk-level 闭环 RL residual 获得可靠提升？

在 R0 中，统一动作单元是完整的执行 chunk：

$$
A_t^{final}=A_t^{base}+\Delta A_t^{res},
\qquad
A_t\in\mathbb R^{H_{exec}\times d_a}.
$$

其中：

- `base`：负责长程意图和主要动作结构；
- `residual`：在 chunk 决策时读取当前状态/表示，输出与 base execution slice 同形的有界 correction chunk，其中 `\Delta A_t^{res}=\alpha\tanh(u_t)` 已包含 residual scale；
- R0 冻结 `H_exec = 4`：组合后连续执行 4 个 low-level actions，然后重新观测并生成下一个 chunk；
- 第一原则：residual 是 refinement，不应逐渐接管整个控制问题。

最终四个问题依次是：

```text
R0  在 Policy Decorator 原始 MS2/v2 contract 内，recipe 换成 Flow-Matching base 后能否成立？
 ↓
R1  在同一 MS2/v2 contract 内，去掉 privileged state、换成视觉输入后还能成立吗？
 ↓
R2  接到真正 frozen SmolVLA / LIBERO 上还能成立吗？
 ↓
R3  同样真实交互预算下，Residual RL 是否真的优于“只是拿新数据继续学”？
```

---

### 1. R0 — Policy Decorator + Flow-Matching Base

#### 1.1 R0 的定位

`R0` 是 **same-environment generator/recipe validation experiment**，不是 MS2→MS3 环境迁移，
也不是 Policy Decorator 论文数值的严格复现。

我们不从 vanilla residual RL 开始，而直接保留 Policy Decorator 已公开验证的核心 recipe：

1. frozen offline imitation base；
2. 小型 Gaussian residual policy 输出完整 4-step residual chunk；
3. SAC；
4. bounded residual action；
5. progressive exploration schedule；
6. critic 对最终组合的完整 action chunk 建模；
7. observation horizon `2`、prediction horizon `16`、每次执行 `4` 个 actions 后重新观测。

R0 有意取消 simulator major-version、task geometry 与 controller migration：

```text
Policy Decorator reference implementation:
ManiSkill2 0.5.3 + repo-local PegInsertionSide-v2
+ Diffusion Policy base
+ original chunk residual recipe

我们的 R0:
同一 ManiSkill2 0.5.3 + repo-local PegInsertionSide-v2
+ 同一 Policy Decorator native EEF demonstrations / state schema
+ state Flow-Matching base
+ original chunk residual recipe
```

因此 R0 仍同时改变 generator 与其 integration，但不再混入跨版本 dynamics。准确 claim 是：

> **在 Policy Decorator 的原始 ManiSkill2/v2 task-data-controller contract 内，chunk-residual
> recipe 可以改善 frozen state Flow-Matching base。**

如果 R0 失败，先审计 exact reference、Flow-policy integration 与 chunk contract，再审计
residual RL 实现；不得切回 MS3 规避问题，也不把失败自动归因到单个 Flow 替换。

---

#### 1.2 任务与数据

##### 环境

固定：

```text
ManiSkill2 0.5.3
Policy Decorator repo-local PegInsertionSide-v2
```

原因：

- Policy Decorator 已直接在这一 runtime task/controller contract 上验证 residual online refinement；
- 任务需要高精度控制；
- base 可以掌握“把 peg 移向 hole”的主体意图，但仍保留 alignment / insertion headroom；
- 非常贴合“base 负责规划，residual 负责精度纠正”的假设。

环境的进入条件是 M1-R 已冻结 Policy Decorator checkout、repo-local v2 注册、
success/reward/termination、backend 与 runtime TimeLimit。R0 不允许在训练中途更换这些语义。

##### Demonstrations

使用 Policy Decorator 原生数据：

```text
metadata task: PegInsertionSide-v0
controller: pd_ee_delta_pose
action: native 7D
episodes: 986
```

具体 H5/JSON hash、episode identity、reset reconstruction 和 split 由 M2-R 子计划冻结。
repo-local `v2` 是 corrected runtime env，不是另一份 demonstrations；严禁加入 MS3
motion-planning 数据或转换产物。

---

#### 1.3 R0 Base：State-conditioned Flow-Matching Policy

##### 输入

第一版 base **不使用 RGB**，而使用 low-dimensional state observation。

这是有意设计：R0 只验证 residual-control pipeline，不同时承担 perception 问题。

```text
state observation
      ↓
Flow-Matching action policy
      ↓
action chunk
```

##### Action decoder

必须使用 Flow Matching，不使用 DDPM / Diffusion Policy decoder。

条件生成形式：

$$
A^{(0)}=\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

$$
\frac{dA^{(\tau)}}{d\tau}
=
v_\theta(A^{(\tau)},s_t,\tau)
$$

ODE integration 后得到 action chunk：

$$
A_{t:t+H}=F_\theta(s_t,\epsilon).
$$

##### 为了尽量贴近 Policy Decorator，第一版冻结这些接口

优先沿用其 Diffusion Policy base 的时序接口：

```text
observation horizon = 2
prediction horizon  = 16
action horizon      = 4
```

R0 子计划在训练前必须完成 source-level contract audit，不只对比配置字段名：

1. 核对 Policy Decorator reference implementation 的 observation/prediction/action horizon 与 sequence wrapper 执行逻辑；
2. 用 shape/assertion 证明 Flow base 生成 `16×7`，residual 生成 `4×7`，并且只与 base 的对应 4 步组合；
3. 用小规模 wrapper test 证明每个 policy decision 恰好触发 4 个 low-level `env.step`，第 4 步后才重新观测；
4. 分别记录 model prediction horizon、residual/action horizon 和 execution steps；不能只用一个含糊的 `chunk_size` 字段声称“已对齐官方配置”。

这里的目标不是声称这些 horizon 对 Flow Matching 最优，而是在只替换 generator 的前提下，
保留 Policy Decorator 的时序控制契约：

> **base 预测 16-step chunk，base 与 residual 都参与组合前 4 步，执行 4 步后再重新观测。**

若当前 MS2/Flow integration 必须使用不同 horizon，必须先停止 R0，在子计划中记录为显式
deviation 并重新定义复现边界；不能在代码中默默改成 per-step residual。episode evaluation
horizon 则单独从 pinned runtime env 解析冻结，不继承 MS3 的 300。

##### Base checkpoint gate

只在 development split 选 checkpoint，然后永久冻结。

进入 residual training 前要求：

1. 明显高于 random；
2. 已经掌握任务主体；
3. 仍有足够 headroom；
4. 失败以 alignment / insertion / execution error 为主，而不是完全不知道任务该怎么做。

如果 base 太弱，先修 Flow base；不能指望 residual 重新学整项任务。

---

#### 1.4 Policy Decorator Residual

##### Residual policy

使用小型 Gaussian policy：

$$
\operatorname{vec}(\Delta A_t^{res})\sim
\mathcal N(\mu_\phi(s_t),\operatorname{diag}(\sigma_\phi^2(s_t))),
\qquad
\Delta A_t^{res}\in\mathbb R^{4\times d_a}.
$$

经过 `tanh` 后再乘 residual bound：

$$
\Delta A_t^{res}
=
\alpha\tanh(u_t).
$$

最终动作：

$$
A_t^{final}
=
A_t^{base}+\Delta A_t^{res}.
$$

`tanh` 和 residual bound 逐元作用于完整 4-step chunk。组合后依次执行 4 个 actions，中间不召回 policy；第 4 步之后获取新状态并做下一次 chunk decision。

##### Residual observation

R0 主设置：

```text
residual input = current state observation
```

不默认加入 `base action chunk`。

理由：Policy Decorator 的消融发现 residual policy 只输入 observation 通常优于 `observation + base action`。

但记录一个很小的 implementation diagnostic：

```text
state only
vs
state + current base action chunk
```

这不是 R0 主 hypothesis，不做大规模超参扫描。

---

#### 1.5 Critic

使用 SAC critic：

$$
Q_\psi(s_t,A_t^{final}).
$$

Critic action 输入使用：

$$
A_t^{final}=A_t^{base}+\Delta A_t^{res},
$$

而不是只输入 residual chunk 或 chunk 的第一步。

这直接沿用 Policy Decorator 的设计结论：critic 看最终组合的完整 action sequence。

---

#### 1.6 Controlled Exploration

##### A. Bounded residual

R0 的 residual 不能自由覆盖整个 action space。

第一参考值直接采用 Policy Decorator 在：

```text
PegInsertionSide + Diffusion Policy + state
```

上的：

```text
alpha = 0.1
```

但这里必须标注：`0.1` 是 **Diffusion Policy base 上的 published prior**，不是 Flow-Matching base 的理论最优值。

正式做法：

1. 先统计 Flow base demonstration / rollout action scale；
2. 若尺度 contract 与原论文兼容，则以 `0.1` 为默认值；
3. 若尺度明显不同，只允许做一次预注册的小范围尺度校准，而不是宽搜索。

##### B. Progressive exploration

训练行为策略：

$$
\Pi_{behavior}(s)=
\begin{cases}
\Pi_{base}(s)+\Pi_{res}(s), & u<\epsilon\\
\Pi_{base}(s), & \text{otherwise}
\end{cases}
$$

其中：

$$
\epsilon:0\rightarrow1
$$

线性增长。

Policy Decorator 在：

```text
PegInsertionSide + Diffusion Policy + state
```

公开使用：

```text
H = 30K  # 原实现中 progressive-exploration schedule 参数
```

因此 R0 第一版以 `30K` 作为 progressive exploration prior。子计划必须审计 Policy Decorator 原代码中该 schedule 的计数分母，并同时写出对应的 chunk decisions 与 low-level simulator steps；不能把二者都简称为 `environment steps`。

若 Flow base 的初始成功率与原论文基座差异非常大，再根据“维持训练初期稳定成功信号”的原则调整，但必须先记录原因。

---

#### 1.7 RL 算法

R0 固定：

```text
SAC
```

不在 R0 同时比较 PPO / SAC。

原因：Policy Decorator 主实验就是 SAC，R0 目标是尽可能减少自由度。

第一版可参考其 ManiSkill 公共超参数：

```text
batch size              1024
learning rate           1e-4
policy update frequency 1
training frequency      64
update-to-data ratio    0.25
target update frequency 1
tau                     0.01
learning starts         8000
```

`gamma` 按 PegInsertionSide 原代码 / reward horizon contract 固定，不在看到结果后调。

---

#### 1.8 Gripper 处理

##### R0

为了尽量贴近 Policy Decorator 的 7D PegInsertionSide residual setting，R0 主实验的 residual 对每个 low-level action 都输出完整 7D correction，一次输出 4 步。

即：

$$
\Delta A^{res}\in\mathbb R^{4\times 7}.
$$

这是 R0 对原 chunk-residual action contract 的保留。

##### 后续 R2

迁移到 SmolVLA 时，第一版仍采用之前约定：

> **关闭 gripper residual 只表示 residual 分支不修改 gripper；base / SmolVLA 仍正常输出 gripper action。**

即：

$$
a^{final}
=
[
a^{base}_{EEF}+\alpha\Delta a^{res}_{EEF},
\; a^{base}_{gripper}
].
$$

---

#### 1.9 R0 必须比较什么

##### 核心组

1. **Frozen Flow Base**
2. **Flow Base + Policy Decorator Residual**

##### sanity controls

3. Flow Base + Zero Residual
4. Flow Base + Random Bounded Residual
5. RL from scratch（只作参考，不作为 R0 成败主基准）

##### 不在 R0 做

- RGB residual；
- SmolVLA；
- full base fine-tuning；
- PPO vs SAC 算法扫描；
- WAM / imagined RL；
- Online BC 数据归因。

这些都留到后续阶段。

---

#### 1.10 R0 要证明什么

R0 只允许得出一个核心结论：

> **Policy Decorator 的原版 chunk-residual recipe 在保持 ManiSkill2/v2 环境、原生 EEF 数据
> 和 controller 不变时，能改善 frozen state Flow-Matching base。**

该结论支持 Flow base 上的 recipe compatibility；没有 matched Diffusion control 时，仍不把它
写成 `Diffusion → Flow Matching` 的单变量因果结论。

也就是：

$$
SR(\text{Flow + Residual})
>
SR(\text{Flow Base}).
$$

建议 gate：

- development success 提升至少 `+10pp`；
- 多 seed 下方向一致；
- residual magnitude 未长期顶在 bound；
- jerk / action deviation 没有出现明显失控；
- improvement 主要出现在 Peg insertion / alignment 的 failure stage，而不是 residual 完全接管任务。

如果 R0 失败：

```text
第一优先：检查 pinned MS2/v2 exact replay、success/reward/termination 与 dataset identity
第二优先：检查 16-step prediction / 4-step residual / 4-step execution 的 chunk 契约
第三优先：检查 Flow base 质量、action scale / alpha / progressive exploration
第四优先：检查 SAC/replay/reward implementation
最后才考虑 residual hypothesis 本身不成立
```

---

### 2. R1 — Privileged State → Visual / Learned Representation

#### 2.1 研究问题

R0 只证明 privileged-state system 能工作。R1 只做一个系统级迁移：

```text
privileged state
        ↓
visual / learned representation
```

R1 继续使用与 R0 完全相同的 ManiSkill2 0.5.3、repo-local `PegInsertionSide-v2`、原生
EEF 数据、controller、TimeLimit、success/reward/termination 和 split；只迁移 observation/
representation contract。R1 不是 MS3 视觉迁移实验。

R1 不再拆成 R1-A/R1-B/R1-C，也不企图在这一节分别隔离“base perception”和“residual perception”的贡献。一次性将 base 与 residual 的 privileged-state 依赖替换为子计划冻结的视觉/学习表示 contract，然后判断完整系统是否保留 residual gain。

#### 2.2 冻结系统 contract

Policy Decorator 公共 artifact 的主 observation 是 50D privileged state，不能假设它已经包含
R1 所需 RGB。R1 子计划必须用相同 986 条 episode 的 reset/options、原生 action 与 v2 runtime
env 做 deterministic replay，按冻结 camera/schema 采集逐帧视觉 observation；不得新增、替换或
重新规划 expert actions。视觉数据必须满足 episode/action/frame 一一对齐、同一 train/dev/test
episode split、reference replay gate 不下降，才能进入训练。

R1 只允许选定一个主输入 contract，例如 `frozen visual encoder latent + robot state`，不在正式
实验中再做大规模 feature engineering。对应的视觉 Flow-Matching base 先单独达到 M2-R 类似的
主体行为/headroom gate，然后冻结；residual 只读取部署时可得的视觉/学习表示。若固定相机下
无法形成无泄漏、可 replay 的视觉数据，R1 标为 blocked，不切 MS3 补数据。

其他组件保持 R0 不变：

- SAC；
- 16-step prediction / 4-step residual / 4-step execution chunk contract；
- bounded residual；
- progressive exploration；
- frozen Flow base；
- 同一 success/reward/termination contract；
- 同一 task split。

#### 2.3 R1 要证明什么

定义：

$$
\Delta SR_{state}=SR_{R0}-SR_{base,state},
\qquad
\Delta SR_{visual}=SR_{R1}-SR_{base,visual}.
$$

目标：

$$
\frac{\Delta SR_{visual}}{\Delta SR_{state}}\ge 0.8.
$$

即：整个 visual/learned-representation residual system 保留 privileged-state residual 增益的约 80%。若明显低于该值，只能结论“从 privileged state 到 visual/learned representation 的系统迁移是瓶颈”；因为 R1 不做组件拆分，不能进一步声称瓶颈特定来自 base 或 residual，也不直接跳到 VLA。

---

### 3. R2 — Frozen SmolVLA + Residual RL

#### 3.1 研究问题

R2 才回答真正面向 VLA 的问题：

> **Policy Decorator-style residual refinement 能否作为外挂模块改善一个真正的 frozen VLA，而不修改 VLA 参数？**

---

#### 3.2 环境与 base

```text
LIBERO
+
frozen SmolVLA
```

选择一个经过预审、base 已会做但仍有稳定 headroom 的单任务作为 R2 主任务。

SmolVLA：

- vision/language backbone frozen；
- Flow-Matching action expert frozen；
- 不通过 residual loss 更新任何 SmolVLA 参数。

---

#### 3.3 Residual 输入

第一版：

```text
frozen visual-language embedding
+
robot state
+
必要时 current base action
```

base chunk 可记录用于诊断，但不默认作为 residual 必需输入。

---

#### 3.4 Action composition

第一版 residual 只修 6D EEF：

$$
\Delta a^{res}_{EEF}\in\mathbb R^6.
$$

最终动作：

$$
a^{final}
=
[
a^{base}_{EEF}+\alpha\Delta a^{res}_{EEF},
\; a^{base}_{gripper}
].
$$

再次强调：

> `gripper residual off` ≠ `gripper action off`。

SmolVLA 仍然正常产生 gripper command；只是 residual 没有权限修改该维度。

---

#### 3.5 R2 要证明什么

核心 claim：

> **Residual correction 不是 ManiSkill/state-policy 特例，而可以迁移到 frozen modern VLA。**

进入 R3 前要求：

- held-out evaluation 上相对 frozen SmolVLA 有稳定提升；
- canonical capability 不出现明显 catastrophic degradation；
- residual norm / jerk 保持可控；
- improvement 对应可解释的 execution / precision failure reduction。

---

### 4. R3 — Residual RL Algorithm Attribution

#### 4.1 为什么 R3 是必须的

如果：

```text
Frozen SmolVLA     35%
+ Residual RL      60%
```

仍不能直接说“Residual RL 算法比继续模仿学习更有用”。

因为 Residual RL 在训练中获得了额外真实环境 interactions。

必须回答：

> 在相同真实交互预算下，包括 reward optimization、critic/bootstrapping 和 exploration/data collection policy 在内的 Residual RL 完整算法，是否比只用新数据继续模仿学习更好？

R3 不把 exploration/collection policy 当成需要“排除”的干扰项；它本来就是 RL 算法优势的一部分。因此 R3 归因的是算法/系统整体，不是一个单独 loss 项。

---

#### 4.2 Fair-budget protocol

所有在线方法使用完全相同的：

```text
frozen base checkpoint / starting parameters
environment + success/reward/termination contract
real low-level environment interaction budget
train initial-state distribution
evaluation split
```

真实交互预算是 R3 的第一控制变量。每种方法允许使用自己自然的 collection/exploration policy 和 update rule；不强制不同方法看到同一个 replay buffer，否则会人为抹掉 RL 的在线探索优势。同时报告 policy decisions（若为 chunk controller 则报 chunk 数）、low-level simulator steps、wall-clock、参数更新量和训练 FLOPs，但后四者不替代真实交互预算控制。

---

#### 4.3 对照组

至少包括：

1. Frozen SmolVLA
2. **Residual RL / Policy Decorator**
3. Online BC
4. Weighted BC（例如成功/高回报样本加权）
5. Residual BC
6. 直接 fine-tune action expert（在同等数据预算下）

Online BC / Weighted BC / Residual BC 各自在相同交互预算内采集数据并更新，用来检验“只要有额外在线数据就能提升”这个解释。

可选增加“共享 Residual RL replay 数据的 BC”作为 diagnostic，辅助观察 update rule 的差异；但该组不是 R3 必做 gate，也不能取代各方法自然 collection/exploration 下的端到端对比。

---

#### 4.4 R3 要证明什么

真正允许的强结论只有：

$$
SR_{ResidualRL}
>
SR_{OnlineBC/WeightedBC/ResidualBC}
$$

在：

$$
N_{real\ interaction}
$$

完全相同的条件下成立。

如果 Residual RL 与 Online/Weighted/Residual BC 基本相当，则结论应写成：

> “online data collection 有价值，但没有证据证明 Residual RL 完整算法相比在线模仿学习提供额外收益。”

如果 Residual RL 明显更好，才支持：

> **在相同真实交互预算下，包括 reward-driven optimization、critic/bootstrapping 和 exploration/data collection 在内的 Residual RL 算法整体，比继续模仿学习具有额外控制价值。**

---

### 5. R0 的实现边界

R0 不做 simulator major-version transfer。Policy Decorator 的 ManiSkill2 0.5.3、repo-local
`PegInsertionSide-v2`、native EEF demonstrations/controller 与任务语义全部冻结；主要工程变化是
把 offline base 接口换成 state Flow-Matching，并把原 chunk-residual recipe 接到该 base。

R0 冻结不变的是 Policy Decorator 核心 recipe：

- frozen state-conditioned imitation base；
- Gaussian full-chunk residual；
- observation `2` / prediction `16` / residual+execution `4` 的时序契约；
- bounded residual + progressive exploration；
- SAC 与最终组合 chunk critic；
- sparse-reward online refinement。

显式改变的是：

- Diffusion Policy base → state Flow-Matching base。

不允许再额外改 simulator major version、task geometry、dataset、controller、residual algorithm、
exploration recipe、observation modality、action execution contract 或训练中的 reward/success
语义。R0 成功时支持“recipe 与 Flow base 在原始 MS2/v2 contract 中兼容”；若未加入 matched
Diffusion control，不支持“Flow Matching 是唯一原因”的单变量结论。

---

# 阶段三：LIBERO 迁移与最终组合

## F0 — Joint Mini-WAM 迁移到语言条件 VLA

### 研究问题

W3 在 ManiSkill 上回答的是“future/world supervision 能否改善 direct action policy”。F0 要验证这一机制能否迁移到真正的语言条件 VLA / LIBERO，而不是只在单任务 ManiSkill 成立。

### Base 与公平对照

固定一个经过预审、未饱和且具有稳定 headroom 的 frozen / trainable SmolVLA 配置；第一版仍保持：

```text
Action-only SmolVLA / Flow action expert
vs
Joint world-action SmolVLA
```

两者必须：

- 使用相同 LIBERO train demonstrations；
- 使用相同 language / vision / robot-state inputs；
- 使用相同 action expert 主体；
- 使用相同 optimizer、update steps、batch budget 和 Flow sampler；
- 仅 Joint 版本额外加入正确 future latent supervision；
- inference 默认关闭 future decoder，不做 K-candidate reranking。

### 必做机制对照

至少保留：

1. Action-only；
2. True-Future Joint；
3. Shuffled-Future；
4. 必要时 Current-State auxiliary diagnostic。

这样 F0 仍能回答“正确 world/future target 是否比一般 auxiliary regularization 更有价值”。

### 评测

至少报告：

- canonical closed-loop success；
- held-out paraphrase；
- paired per-state outcome；
- stage / failure-type breakdown（在已有 evaluator 能可靠提供的范围内）；
- inference latency；
- peak GPU memory；
- deployment-active parameter count；
- future decoder train-time-only 参数量。

### F0 Gate

进入后续组合阶段至少要求：

- Joint > Action-only；
- Joint > Shuffled-Future；
- improvement 不能依赖 test-time world rollout / reranking；
- canonical capability 不出现不可接受的退化；
- held-out paraphrase 方向与 canonical 至少不出现明显相反结论。

如果 F0 失败：记录“Mini-WAM direct-action 机制在当前 ManiSkill 设置成立，但尚未迁移到语言条件 VLA”；不通过增加 test-time planner 来掩盖迁移失败。

---

## F1 — WAM / W4 与 Residual RL 是否互补

**TODO**

待 F0 与 R3 的最终结果冻结后再设计，不在本轮预注册具体组合方式、任务数、seed 数或成功阈值。

---

## F2 — 是否减少真实环境交互

**TODO**

待 F1 确认存在互补性后再设计。届时必须统一记账：offline demonstrations、Mini-WAM/W4 训练数据、imagined model steps 与 Residual RL 真实 simulator interactions。

---

## F3 — 最终泛化与能力保留

**TODO**

待 F1/F2 通过后再设计。预期覆盖 canonical、held-out paraphrase、状态/视觉 OOD、恢复场景和背景能力保留，但本轮不冻结具体 protocol。

---

# 全局阶段闸门

| 闸门 | 通过条件 | 不通过时 |
| --- | --- | --- |
| M0 → M1-W | 已通过；基础 replay→dataset→policy→closed-loop 管路成立 | M0 关闭 |
| M1-W → M2-W | MS3 StackCube GPU 数据/协议、D_core、LeRobot dataset、state banks 与 frozen manifest 全部通过；M1-W 状态为 `done` | M2-W 保持 blocked；先收口 M1-W |
| M1-R → M2-R | MS2 0.5.3、pinned Policy Decorator checkout、v0-data/v2-env、原生 7D EEF、hash、TimeLimit 与 split 冻结；exact replay reference 通过 | 停止 Residual base；不借 MS3 数据填补 |
| M2-W → W0 | Action-only Flow 掌握任务主体、未 ceiling、baseline/config 已冻结 | 先修 base；如 StackCube 不合适，另立候选任务子计划，不自动切换 Plug |
| M2-R → R0 | MS2/v2 state Flow base 掌握主体且存在 precision/execution headroom | 先修 MS2 data/base/chunk contract，不让 residual 重学任务 |
| W0 → W1 | correct action 对 future prediction 有显著贡献；replace/mask 后退化 | 修共享 trunk / attention / future target |
| W1 → W2 | 至少一个 `Δ` 同时有稳定 future prediction 与较好 development control | 缩短 horizon 或停止 multi-step foresight claim |
| W2 → W3 | 单个预注册 training seed 下 True-Future > Action-only 且 > Shuffled-Future；paired-state bootstrap 95% CI 下界 > 0 | 不做正式 Mini-WAM claim；该 CI 不外推到跨 training seed 稳健性 |
| W3 → W4 | W3 frozen test 显著超过 Action-only | W4 不启动；W0–W3 结论独立保留 |
| W3 → F0 | direct-action Mini-WAM 主结论成立 | 不迁移到 LIBERO Joint WAM |
| R0 → R1 | MS2/v2 + Flow Base + 原版 chunk Residual recipe 相对 frozen base 提升约 ≥10pp，多 seed 方向一致且 residual 不失控 | 优先审计 exact replay/data/chunk contract，再查 Flow/scale/alpha/exploration/SAC |
| R1 → R2 | MS2/v2 visual/learned-representation residual system 保留 privileged-state residual 增益约 80% | 记录 representation-transfer bottleneck，不归因到单一组件，不迁移 LIBERO |
| R2 → R3 | frozen SmolVLA 被 residual 稳定改善，canonical degradation 可控 | 停止 Residual VLA 主线 |
| R3 → Final | 同真实 interaction budget 下，各方法使用自然 collection/exploration，Residual RL 显著超过 Online/Weighted/Residual BC | 只声称 online data 有用，不声称 Residual RL 完整算法有额外优势 |
| F0 → F1 | Joint Mini-WAM 在 LIBERO 仍有可归因的 direct-action 提升 | F1–F3 保持 TODO / 不启动 |
| F1 → F2 | TODO | TODO |
| F2 → F3 | TODO | TODO |

---

# 统一停止条件与风险控制

1. **Baseline floor / ceiling 都是停止条件。** 太弱会把方法实验变成重新学任务，太强则没有 headroom。
2. **禁止测试泄漏。** M0 已被 sweep 触碰的 test 永久降级；后续每个新任务建立新 split，正式 test 只在配置冻结后运行。
3. **post-success tail 必须显式处理。** 不能让成功后的 hold 动作未经审计地主导训练。
4. **动作契约必须逐 channel 验证。** 数值超 `[-1,1]` 不能自动判为 simulator bug，也不能未经解释就进入统一 normalization。
5. **MS2/v2 task identity 必须精确。** R0/R1 runtime 固定 Policy Decorator repo-local
   `PegInsertionSide-v2`，data metadata 固定 `PegInsertionSide-v0`；不得把二者合并称为“v2
   数据”，也不得在训练中途退回 native v0 predicate。
6. **Mini-WAM 核心因果实验禁止数据不公平。** Joint 不能比 Action-only 多看额外真实 rollout。
7. **防止 future branch 退化成 no-action predictor。** W0 必须包含 action replace / noise / mask / no-action controls。
8. **防止“更大模型”伪装成 world-supervision 收益。** W2/W3 不同时更换 backbone、action expert 和数据源。
9. **W2/W3 使用单个预注册 training seed。** 固定 paired state bank 和 paired bootstrap 用于衡量该训练运行下的状态差异，不对跨 training seed 稳健性作声称。
10. **防止 W4 model exploitation。** recursive rollout error 一旦在短 horizon 内失控，不扩大 imagination horizon；先修 world-control model。
11. **W4 imagined steps 与真实交互分开记账。** 主版本额外真实训练 interaction 必须为 `0`。
12. **Residual 必须保持 refinement。** residual magnitude 长期顶 bound、jerk/动作偏差失控时，视为失败，不允许 residual 接管整个任务。
13. **R3 不可跳过。** 没有同真实交互预算、但允许各方法自然 collection/exploration 的 BC-family 对照，就不能声称“Residual RL 完整算法有额外优势”。
14. **F1–F3 当前不提前承诺。** 先完成单线因果结论，再设计最终组合，避免用后期组合结果反向改早期 protocol。

---

# 产物路径与实验记录

建议沿用当前目录约定：

- ManiSkill 状态划分：`configs/experiments/maniskill/split.json`
- ManiSkill 演示与转换数据：`datasets/lerobot/maniskill/`
- M1-W 协议审计：`outputs/maniskill/m1_wam_protocol_audit/<任务>/`
- M1-R 协议审计：`outputs/policy_decorator_ms2/m1_r_protocol_audit/`
- Peg MS3 历史诊断：`outputs/maniskill/m1_protocol_audit/PegInsertionSide-v1/` 与
  `outputs/maniskill/m1_peg_continue_diagnosis/`（只读保留，不作为正式数据）
- M2-W frozen base：`outputs/maniskill/m2_task_base/<任务>/`
- M2-R frozen base：`outputs/policy_decorator_ms2/m2_r_base/`
- M2-P Oracle/proposal probe：`outputs/maniskill/m2_proposal_probe/<任务>/`
- M3 / W-Data：`datasets/generated/maniskill_policy_rollouts/<任务>/<checkpoint>/`
- W0–W4：`outputs/maniskill/<实验编号>/`
- Residual R0–R1：`outputs/policy_decorator_ms2/<实验编号>/`
- LIBERO R2–R3 / F0–F3：`outputs/smolvla_libero/{pilot,eval}/<实验编号>/`
- 每个实验的可执行子计划：`plans/YYYY-MM-DD-<编号与主题>.md`
- 实验结论：按项目报告约定写入对应 report，不把临时结果直接写回本总计划。

每个正式实验至少冻结并记录：

```text
git commit
config hash
dataset/split version
checkpoint id
seed / initial-state bank
training budget
real interaction count
imagined step count (if any)
evaluation protocol
raw metrics
bootstrap / statistical summary
```

---

# 最终一句话逻辑

```text
M0–M2 先把协议和公平 baseline 固定住；
W0–W3 证明 world supervision 本身能让 direct Flow policy 更好；
W4 再测试 learned world 能否在纯 imagination 中继续优化主 actor；
R0–R3 独立证明 real-feedback residual correction，并用 R3 检验同交互预算下完整 Residual RL 算法是否超过只用新数据继续模仿学习；
F0 把 Joint Mini-WAM 迁移到语言条件 VLA；
F1–F3 等单线结果冻结后再设计。  # TODO
```
