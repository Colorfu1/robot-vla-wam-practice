# 2026-08-06 Mini-WAM与残差RL三阶段实验计划

## 状态

- 状态：`todo`
- 范围：本文件是三阶段总路线（外加一个接通新环境的阶段零），只固定依赖、数据边界和阶段闸门。
  每个编号开跑前另建一份可直接执行的子计划，完整命令写在子计划中。
- 最后更新：2026-08-06

## 目标

在冻结的模仿策略上依次验证：Mini-WAM 能否学习可用于控制的动作动力学，Residual RL 能否在
相同真实交互预算下超过数据微调，以及两者组合后是否互补、是否减少真实交互。

## 已确定的路线

- **取消 Outcome Model。** 它需要为同一状态下的每个候选执行完整终局续跑，数据成本高；现有
  Diffusion 在固定 observation 下的候选多样性又有限，不再实现或评测这条路线。它要体现的内容
  由 M1 的候选空间审计和 W2 的 `H=0` 对照臂覆盖。
- **主环境从 PushT 换成 ManiSkill。** 换的不只是速度，更是任务结构，理由见下一节。PushT 退出
  新实验，只作为先验证据被引用，并保留为 M0 失败时的退路。
- **Mini-WAM 只预测隐空间状态、reward 和 value，不生成 RGB。** W0/W1 使用普通 transition 和
  最多 16 步的短分支训练，不采集候选到终局的 counterfactual GT。
- **WAM 与 Residual RL 可以并行。** 两条线分别通过后，才进入 LIBERO 和最终组合。

```text
阶段零:  M0 接通环境 → M1 候选空间审计 → M2 任务选型 → M3 策略自采数据
                                                          ↓
ManiSkill Mini-WAM:   W0 → W1 → W2 → W3 → F0 ─┐
                                                ├→ F1 → F2 → F3
ManiSkill/LIBERO RL:  R0 → R1 → R2 → R3 ───────┘
```

## 为什么换环境（已实测，不再讨论）

| 环境 | 吞吐 | 动作维度 | 结果结构 |
| --- | ---: | ---: | --- |
| LIBERO 单环境 256×256 | **77.5 步/秒** | 7 | 稀疏 0/1 + BDDL 谓词 |
| gym_pusht 单环境 | 2233 步/秒 | 2 | 只有一个标量 coverage |
| ManiSkill 256 并行 + RGB 128×128 | **5716 步/秒** | 8 | `info` 原生给阶段分解 |

LIBERO 那 77.5 是纯仿真步进、零动作、不含策略前向，真实差距比表里更大；ManiSkill 带渲染仍是
它的 74 倍。10⁶ 个环境步在 ManiSkill 上约三分钟纯仿真，在 LIBERO 上按天算。

但速度不是主要理由。**显式预测中间状态只在两种情况下划得来:一是奖励分段有闸门,二是存在不可逆
事件。** 有闸门时,中间状态是因果上必需的信息——`is_cubeA_grasped` 不成立,后面的
`is_cubeA_on_cubeB` 就无从谈起,一个把夹爪送到位却没抓住的动作块收益是零,直接打分器要从
`(o, A)` 硬拟合这种不连续的复合逻辑,而动力学模型可以分解表示。不可逆时,中间状态携带了终点预测
拿不到的信息——掉了或倒了就救不回来。

PushT 两条都不满足。它的 reward 是 `np.clip(coverage / 0.95, 0.0, 1.0)`,coverage 只是当前 T 块
位姿的函数,所以把 8 步的 `γʰ r̂` 加起来几乎由终点决定,**一个只预测终点 coverage 的直接打分器已经
拿到了几乎全部信息**;而它的状态是 `agent_pos (2,) + environment_state (16,)`,T 块准静态、无惯性、
无处可掉,推歪了推回来即可,动作到结果的映射是光滑的。所以在 PushT 上得出"多步动力学无控制价值",
主要是关于 PushT 的结论,不是关于世界模型的结论,W1/W2 的判据会被任务本身推向负结果。ManiSkill 有
抓取、有堆叠倒塌,`info` 里还原生给出阶段分解(`is_grasped` / `is_cubeA_on_cubeB` /
`is_cubeA_static` 等),正是 PushT 缺的两样。原生阶段信号顺带省掉自己写谓词的工作,而自己写判据
正是 RF7 和 Prompt Swap 两个 bug 的来源。

任务分工：**PickCube-v1** 用于 M0 验管路（最简单，实测 9345 步/秒）；**StackCube-v1** 作 WAM
主环境（8871 步/秒，多阶段且会倒塌）；**PegInsertionSide-v1**（5056 步/秒）留给需要精度的
残差 RL。

## 前置条件

1. **M0 通过**（见阶段零）。仓库现在对 ManiSkill 是零基础：`mani_skill 3.0.1` 装着，但代码、
   配置、产物里一处引用都没有，`envs/maniskill/` 是空占位；LeRobot 自己也没有 ManiSkill 适配
   （`lerobot/envs/` 只有 `libero.py` 和 `metaworld.py`）。所以在跑任何实验之前，先确认这条
   数据管路走得通。
2. 在 `configs/experiments/maniskill/split.json` 中新增并冻结可复现的状态划分：
   - `train_seeds`、`development_seeds`、`test_seeds` 三组两两无交集
   - test seeds 只在正式评测时跑一次
3. ManiSkill 候选生成器固定为 M0 训出的那个 diffusion checkpoint，并记录它在 development
   seeds 上的成功率作为后续一切比较的基线。
4. PushT 只作为先验证据引用，不进入训练或评测：E2b 的 `single_sample_success = 0.55` 对
   `success_any_at_10 = 0.90`（35 个百分点）说明这类任务存在可选择空间，但那批 rollout 用的是
   final seeds，且量的是"变整条 episode 的初始噪声"，与实际用法不同，必须在 ManiSkill 上重测。
5. 进入 LIBERO 前必须完成：
   - Prompt Swap 判定 bug 已修（`plans/2026-08-05-PromptSwap目标判定修复.md`）。未修则语言依从
     数字不可用——现在的 18/30 = 60% 是错的，正确读法接近"可判定目标上 18/20 = 90%"；
   - 冻结一个明确的 C1/C2 底座 checkpoint，并重新测量 canonical、held-out paraphrase 和背景基线；
   - 若最终要声称背景保留，底座必须使用背景回放训练结果；否则背景列只记录，不作成功判据。
     **不允许既用忘光的底座（C1 是 0/210、C2 是 8/210），又把不遗忘当成功标准**；
   - 为 RL 新生成训练 init states，并证明其分布与官方状态相容。官方 50 个 init state 已全部用于
     validation/test，禁止拿来训练。

## 数据边界

### ManiSkill

- **演示来源**：官方演示可从 HuggingFace 直接下载（已确认 StackCube / PickCube /
  PegInsertionSide / PlugCharger / PushT 等 16 个任务都有）。但它们录在某个控制模式下，需要用
  `replay_trajectory` 转成 RGB 观测 + 目标控制模式，**转换会有失败**，这就是工具自带
  `max_retry` 和 `allow_failure` 的原因。**存活率必须实测并记录，不许假设是 100%。**
- **转格式先用官方工具。** `mani_skill/trajectory/convert_to_lerobot.py` 输出 LeRobot v3.0，与本
  仓库的 `CODEBASE_VERSION = v3.0` 一致。已知不足是 `observation.state` 只有 `agent.qpos`（9 维）、
  视频用 OpenCV `mp4v`。先验它能否被 `lerobot-train` 直接吃下去，不够再补，别先自己写。
- **转 LeRobot 格式只为基座策略，不为 WAM。** 转过去是为了复用 LeRobot 的训练循环——动作块采样
  （`delta_timestamps`）、归一化统计量、以及和仓库现有入口一致的 checkpoint 格式；M0 第 5 步要训
  的就是一个 LeRobot diffusion policy。**WAM 的训练数据不需要转**，它用的 transition 和短分支是
  自己从 rollout 生成的，训练循环也自己写。理由只是框架适配，不涉及数据质量或正确性。
- WAM 基础数据：转换后演示的连续 transition。
- WAM 补充数据：train seeds 上冻结基线策略的成功/失败 rollout，以及必要时执行 1/4/8/16 步的
  短候选分支。分支依赖 `get_state_dict` / `set_state_dict`，已实测可精确回滚。
- WAM 选模：development seeds；test seeds 只运行一次正式闭环评测。
- 每条 WAM 样本至少包含 `o_t`、`a_t`、`o_{t+1}`、reward、机器人状态、物体位姿，以及 `info` 里的
  **原生阶段信号**。阶段信号不要自己写谓词，直接取仿真器的。
- 观测直接是 GPU 上的 torch 张量（`rgb: (N,128,128,3) uint8`），不过 CPU。

### PushT（仅退路）

M0 失败时的退路：回 PushT 只做 W0/W1 的机制核查，用现成的
`outputs/pusht_diffusion/pilot/diffusion/checkpoints/175000`，branch 种子取 `3000..3199` 与
`3200..3239`（与 development `1000..1019`、final `2000..2049` 两两无交集）。角度误差参考目标必须
是 `π/4`。**走这条退路时必须明确承认 W2 的结论在 PushT 上是弱证据**，不得在这种情况下硬下
"多步动力学没用"的判断。

### LIBERO

- WAM/RL 训练只能使用目标 train demonstration、允许使用的背景 train demonstration，以及新生成的
  RL 训练状态上的 policy rollout。
- validation/test state、目标离线 validation 和 held-out paraphrase rollout 禁止进入训练。
- held-out paraphrase 是固定 hard split（C1 在上面是 33.3%，正落在基线不饱和的区间），
  **不根据基线成绩重新造 OOD**——那等于拿基线成绩去调测试集。
- 背景任务的两个数字要分清：`background_split.json` 的 `num_tasks` 是 37（训练侧 1412 条演示），
  但**闭环背景评测只有 7 个任务**，210 条就是 7 × 30。"37 个任务的保留审计"需要新建不存在的
  评测入口，是新增成本。

## 执行步骤

### 阶段零：接通 ManiSkill（M0–M3）

第一件事不是做实验，是确认这条路走得通。这一步的价值是**早失败**。

#### M0 端到端探针

七件事，顺序不能换：

1. 审计已下载的演示（时间结构、成功后填充帧）
2. 固定状态划分并做初始状态泄漏审计
3. 用 `replay_trajectory` 生成 RGB 观测，**先跑 10 条试通**
4. 全量 replay 并**数一下活下来多少条**
5. 转成 LeRobotDataset（Full 与 Trim 两份）
6. 训一个 LeRobot diffusion policy
7. 测闭环成功率，同时跑随机基线与特权 PPO 参照

**主线演示用 `rl/pd_ee_delta_pose`（1013 条、原生 7 维），不用 motionplanning。** 后者首次成功
中位 73 步，而 PickCube 的 `max_episode_steps = 50`，只有 1% 落在 horizon 内，拿它训必然被截断。
详见 M0 计划第 1 步。因此**控制模式转换不在主线上**，第 3、4 步的风险点变成 GPU 并行 replay 的
状态恢复。选 PickCube 不是因为它有意思，是因为它最简单，验管路最快。

**建议不走 LeRobot 的 env 层。** ManiSkill 本身就是向量化的，而 LeRobot 那个 LIBERO wrapper 里
`if terminated: self.reset()` 的自动重置，正是 Prompt Swap 那个 bug 的成因——目标达成后环境先
重置了，判定才去看状态，于是整列判成失败。直接用 ManiSkill 原生 API 就不会继承这个坑。

**顺带的降风险手段**：ManiSkill 里有 `PushT-v1`，可以拿已经熟悉的任务验 harness。动力学与
gym_pusht 不同，数字不会对上——它验的是管路，不是数字。

#### M1 候选空间审计

**要验证什么**：从同一状态采 K 个候选，它们的结果差异大不大。

这一步以前挂在 Outcome Model 下面，但它跟用什么打分器无关：**WAM 规划同样是从候选里挑，没有可
选择空间就没得挑。** 所以它是阶段一的前置。

**具体做什么**：train seeds 上取若干分支点，每点采 K ∈ {1,4,8} 个 chunk，分别执行后各自交回
冻结基线跑到终止，比较 Oracle@K、first-sample、Random@K。

**这里要付一次贵的代价，而且只付这一次。** 取消 Outcome Model 的理由就是"为每个候选做终局续跑
太贵"，W2 之后也只真实执行被选中的那个候选。但**不做终局续跑就算不出 Oracle@K**，也就拿不到
天花板，后面无从知道自己捕获了可用空间的几成。所以 M1 在小规模上把 oracle 量一次作为固定参照，
W2 起不再重复。

**在 ManiSkill 上这笔代价其实不大**，按实测 5716 步/秒（256 并行 + RGB）算：

| 规模 | 分支条数 | 总环境步 | 纯仿真 |
| --- | ---: | ---: | ---: |
| 1000 状态 × K=8 × 50 步 | 8,000 | 400,000 | 1.2 分钟 |
| 2000 状态 × K=8 × 50 步 | 16,000 | 800,000 | 2.3 分钟 |
| 2000 状态 × K=8 × 100 步 | 16,000 | 1,600,000 | 4.7 分钟 |

各任务 horizon 实测：PickCube 与 StackCube 都是 **50** 步，PegInsertionSide **100** 步，
PlugCharger **200** 步（贵四倍）。策略前向才是真瓶颈（按 `n_action_steps=8` 约 10 万次），但量级
可控。**这是换 ManiSkill 的一项额外收益：在 PushT/LIBERO 上必须"先测单位耗时再定规模"的约束，
在这里基本解除了。**

#### M2 任务选型闸门

**不要凭感觉预先定主任务。** M0 只验通了 PickCube 一条管路，而 PickCube 的目标信号是
`goal_pos` (3,)——不含朝向，很可能太容易，也让 7 维动作的旋转通道形同虚设。所以进阶段一之前，
对四个候选做一次横向小规模 pilot：**PickCube、StackCube、PegInsertionSide、PlugCharger**。

每个任务测四项，都要落盘：

| 测什么 | 怎么读 |
| --- | --- |
| replay 存活率 | 越高越好，低于 50% 的任务出局 |
| Oracle@8 − first-sample | ≥ 10 个百分点才有可选择空间，这是阶段一能不能做的前提 |
| 结果方差 | 同状态下 K 个候选的结果分散度，要明显非零 |
| 有效动作维度 | 各动作通道的绝对值分位数；旋转三维是否活跃决定 7 维口径有没有意义 |

**base 成功率不自己训小模型去测。** 用一两百条 demo 训出来的弱基线，其成功率测的是"小预算下策略
有多强"，不是"任务有多难"，而 30%–85% 这个区间恰好是欠训策略在简单任务上会落进的区间——会把
"我训得不够"误判成"这个任务有好的 headroom"。**改为查 ManiSkill 官方基准的公开数字**作为 base
成功率的先验，pilot 只测上面那四项查不到的。

**预期的选型结果（是预期不是结论，照实记录）**：PickCube 大概只够当 M0 的管路验证；StackCube 最
可能作阶段一的 WAM 主环境（多阶段、会倒塌、50 步便宜）；PegInsertionSide 最可能作阶段二残差 RL
的精度主环境（`box_hole_pose` (7,) 含朝向，旋转通道真在工作）；PlugCharger 作最后的稀疏奖励压力
测试（200 步，最贵）。选中任务的演示量建议 500–1000 条（PickCube 官方量是 1000）。

#### M3 策略自采数据

**世界模型不能只吃成功演示。** 官方演示（无论运动规划还是 PPO rollout）彼此近乎同分布，而 WAM
必须见过策略自己产生的错误状态，否则一到规划器访问的新状态就失效。继续加几千条同分布的成功演示，价值低于采一批
策略自己的失败。

对 M2 选中的任务各采：

- 约 1000 条成功 rollout
- 约 1000 条失败 rollout（基线成功率 80% 的话要跑约 5000 条才凑够失败）
- 失败阶段标签，**直接取 `info` 里的原生阶段信号**（`is_grasped` / `is_obj_placed` /
  `is_cubeA_on_cubeB` / `is_robot_static` 等），不要自己写谓词

**必须记录采集用的是哪个 checkpoint。** 一个 80% 成功率的策略和一个欠训策略的失败模式是两回事，
混着采出来的失败分布没有意义，事后也无法归因。同一批数据里不允许混多个 checkpoint。

### 阶段一：ManiSkill Mini-WAM（W0–W3）

主环境 StackCube-v1。Mini-WAM 的计算路径固定为：
Mini-WAM 的计算路径固定为：

```text
z_t = E(o_t)
ẑ_{t+1} = D(z_t, a_t)
Ĝ(A) = Σ γ^h r̂_{t+h} + γ^H V̂(ẑ_{t+H})
```

| 编号 | 要验证什么 | 具体做什么 |
| --- | --- | --- |
| W0 | 模型是否真的学习 action-conditioned 单步动力学 | 先冻结或使用 target encoder，训练单步 latent、位姿和 reward head；对比 full、no-action、copy-latent、action-shuffle、state-only oracle |
| W1 | 递归预测多步后是否仍有控制价值 | 在不注入真实中间状态的 open-loop 模式下评测 horizon `1/4/8/16`；同时报告位姿误差、reward 误差和短候选排序准确率 |
| W2 | WAM 预测能否提高冻结基线的闭环表现 | 每个决策点生成 K 个 chunk，WAM 在 latent 中 rollout 后选择一个执行；对比 first sample、Random@K、**`H=0` 同模型**、WAM@K，记录 success、原生阶段信号、延迟和被选动作的真实短期预测误差 |
| W3 | 如何抑制模型漏洞和分布偏移 | 训练 5 模型 ensemble，对比均值打分、悲观打分 `E[Ĝ]-βσ` 和高不确定性时回退基座；在 train seeds 上做最多 2 轮失败数据聚合 |

**W2 里那个 `H=0` 对照臂**是同一个 WAM、reward 头直接吃 `(z, A)` 不做 rollout。它替代了被取消的
Outcome Model：没有它，W2 只能跟 first sample 和 Random@K 比，那你只能说"学过的打分器有用"，
说不清"显式建模中间状态"贡献了什么。一条消融臂就够，不需要单独一个阶段。

W2 只真实执行被选中的候选，不为其余候选生成终局 GT，避免重新引入 K 条完整续跑；天花板用 M1 量
好的那个。若拿测试专用的分布外场景做 W3 训练，必须另建互不重叠的正式测试 split。

### 阶段二：Residual RL（R0–R3）

统一形式：`a_t = a_base + α·Δa_RL`。基座策略冻结，第一版不修改 Diffusion/SmolVLA action expert。

**残差策略是什么、为什么它能训而基座不方便训。** 残差不是另一个 flow matching 模型，而是一个
带显式密度的单步随机策略（一般就是小 MLP 输出对角高斯的均值和 log-std），`log π(Δa|s)` 是闭式，
PPO 的似然比和 SAC 的熵项都现成。基座那边采样容易但**精确 `log π(A|o)` 不容易**——SmolVLA 是沿
ODE 积分 10 步生成整块动作（`num_steps=10`，`chunk_size=50`，50 个动作步是一起更新的，不是自
回归），要精确密度得追路径上的 Jacobian log-det，又贵又只能近似。**梯度也完全不穿过基座**：从残差
看，`a_base` 就是随观测一起进来的一个数，不需要对那 10 步积分反传。注意"残差"说的是动作参数化，
用 PPO 还是 SAC 是另一个选择，前者是似然比策略梯度，后者是穿过 critic 的重参数化梯度。

**残差补的是什么能力。** 基座一次出 50 个动作、执行 `n_action_steps=10` 步才重新看观测（PushT
diffusion 是 8 步），这段时间它开环、看不见新观测；残差每个环境步都算一个，是闭环的。所以第 h 步
的动作是"基座 chunk 的第 h 个动作 + 当下算出的一个残差"，不是 chunk 加 chunk。分工是基座给长程
意图，残差给逐步纠正。

**方法依据（已核对摘要，未读实验章节）。** 结构出自 Residual Reinforcement Learning for Robot
Control（arXiv 1812.03201），最终策略是传统控制器与 RL 残差的叠加。最贴近本阶段的是 ResiP
（arXiv 2407.16677，MIT），它给**冻结的分块 BC 模型**外挂**全闭环残差**，论证与上一段一致——
"分块预测让 BC 策略更像轨迹规划器而不是闭环控制器"，并且明确指出 BC 在需要精度的任务上
**随数据增加会饱和**，这正是 R3 那个控制实验的前提。冻结 VLA 上做残差的近期工作有 PLD
（arXiv 2511.00091）、Object-Centric Residual RL（arXiv 2606.18953）、ReCoVLA
（arXiv 2606.09630）。

换 ManiSkill 在这一阶段收益最直接：10⁶ 个环境步约三分钟纯仿真，在 LIBERO 上按天算，每步还要一次
VLA 前向。**在准静态二维推动上验出来的残差，推不到七自由度带抓取的操作**，这也是 R0/R1 不该留在
PushT 的原因。

| 编号 | 要验证什么 | 具体做什么 |
| --- | --- | --- |
| R0 | RL 管路在最理想状态输入下是否成立 | ManiSkill 使用低维状态、base action 和 base chunk 训练 residual；只固定一种算法，对比冻结基座、RL from scratch、零 residual 和随机 residual |
| R1 | 视觉 residual 能否接近状态 residual | 对比 RGB+机器人状态、冻结 encoder latent、latent+base chunk；只消融更新频率和 residual 幅度，持续监控 norm、jerk 和动作夹角 |
| R2 | SmolVLA 单任务能否被 residual 修正 | 选择一个 LIBERO 目标任务，输入冻结视觉/language embedding、8D state 和 base chunk；只修 6D EEF，第一版关闭 gripper residual |
| R3 | 收益是否真的来自 RL | 三个目标任务使用完全相同的真实交互预算，对比 Residual RL、Online BC、Weighted BC、Residual BC 和直接微调 action expert |

**R0 的 reward 不能只给稀疏成功。** Foresight Residual RL（arXiv 2607.16506）报了一个正是这个设定
下的失效：在冻结 VLA 底座上做残差 RL，恒定的稀疏成功奖励能各自改善每个子任务，但**技能串起来后
收益很小甚至没有，因为终态质量没被控制**。StackCube 是多阶段的（抓住 → 放上 → 稳住），可能撞上；
LIBERO 的目标任务 1/4/8 是单阶段，R2 大概不会。所以 R0 的 reward 必须包含终态质量项（例如放置后
物体是否稳住），而不是只有成功位。这条也是 F1 失败阶段分层要盯的地方。

R0/R1 与 W0/W1 并行。**R3 是整个阶段最重要的控制实验，跳过它就没法声称"RL 有用"。**
需要精度的话，R1 之后可以在 PegInsertionSide-v1 上再跑一遍——那是残差 RL 最对口的设定。
R2 只能在 R1 通过且 LIBERO 新训练状态完成分布检查后开始。

### 阶段三：LIBERO 组合验证（F0–F3）

| 编号 | 要验证什么 | 具体做什么 |
| --- | --- | --- |
| F0 | ManiSkill 上成立的 WAM 能否迁移到语言条件任务 | 用 LIBERO train demonstration 和新训练状态上的成功/失败 policy trajectory 训练语言条件 WAM；候选来自不同 flow noise，先报告候选多样性再谈排序收益 |
| F1 | WAM 与 residual 是否互补 | 固定四组：VLA、VLA+WAM、VLA+Residual、VLA+WAM+Residual；除总成功率外，按 approach、contact/grasp、transport、release 分层失败原因 |
| F2 | WAM 是否减少 residual RL 的真实交互 | 对比纯真实训练、WAM latent 表征、WAM value 初始化和短 imagined rollout；所有方法真实交互预算相同，并把训练 WAM 的真实数据计入总预算 |
| F3 | 最终泛化和能力保留 | 只对通过 F1/F2 的方法评测 canonical、held-out paraphrase、状态/视觉 OOD、恢复场景和背景保留；先用单条耗时计算总预算，再冻结方法数和 seed 数 |

## 判断标准

| 闸门 | 通过条件 | 不通过时 |
| --- | --- | --- |
| M0 → M1 | 演示 replay 存活率已实测并记录；转出的 LeRobot 数据能训出闭环成功率明显高于随机的策略 | 走退路：回 PushT 只做 W0/W1 机制核查，并明确承认 W2 结论是弱证据 |
| M1 → M2 | chunk 粒度上 Oracle@8 比 first-sample 高至少 10 个百分点 | 停止。问题在候选生成器，不在打分器，另开 action proposal 计划 |
| M2 → M3 | 四个候选任务的存活率、Oracle@8 差、结果方差、动作通道活跃度已全部实测落盘，主任务已按预注册口径选定 | 无任务满足 Oracle@8 差 ≥ 10 个百分点则停止阶段一 |
| M3 → W0 | 成功与失败 rollout 各约 1000 条，失败阶段标签取自 `info` 原生信号，采集 checkpoint 已记录且未混用 | 补采。混用 checkpoint 的批次作废重采 |
| W0 → W1 | Full 的单步误差显著低于 no-action/copy-latent，且 action-shuffle 显著退化 | 停止 WAM，先修表示或数据，不做多步规划 |
| W1 → W2 | 至少一个 `H>1` 的短候选排序准确率显著高于 `H=1`，且 open-loop 误差未随 horizon 失控 | 记录多步动力学无控制价值，停止 W2/W3 |
| W2 → W3 | WAM@K 在 development 上的 success 配对差值，对 first sample、Random@K 和 `H=0` 三者的 bootstrap 95% CI 下界均大于 0 | 不做数据聚合；若只是打不过 `H=0`，如实记录"显式建模中间状态无额外价值" |
| W3 → F0 | 悲观/聚合后的 WAM 提高 success，或在 success 不降时显著降低 false-confidence rate | 不把 WAM 迁移到 LIBERO |
| R0 → R1 | residual 在 development 上超过冻结基座至少 10 个百分点，且 residual norm 未越过预注册上限 | 停止，先修 RL 管路，不要换到视觉输入 |
| R1 → R2 | 视觉 residual 达到 state residual 增益的至少 80% | 记录视觉表示瓶颈，不进入 LIBERO |
| R2 → R3 | held-out paraphrase 相对冻结 C1 的 33.3% 提升至少 10 个百分点，canonical 退化不超过 5 个百分点 | 停止 Residual RL 线 |
| R3 → F1 | Residual RL 在相同真实交互预算下显著超过 Online BC | 如实记录收益来自额外数据，不声称 RL 有优势 |
| F1 → F2 | WAM+Residual 同时超过两个单组件系统，且失败阶段分析显示改善来源不同 | 记录二者不互补，跳过 F2 |
| F2 → F3 | 达到同一成功率所需的总真实交互量下降；总量包含 WAM 数据采集 | 不声称模型数据提高样本效率 |

## 产物路径

- ManiSkill 状态划分：`configs/experiments/maniskill/split.json`
- ManiSkill 演示与转换数据：`datasets/lerobot/maniskill/`
- M2 选型结果：`outputs/maniskill/m2_task_selection/{metrics.json,run_manifest.json}`
- M3 策略自采数据：`datasets/generated/maniskill_policy_rollouts/<任务>/<checkpoint>/`
- WAM 数据：`datasets/generated/maniskill_wam/`、`datasets/generated/libero_wam/`
- ManiSkill 结果：`outputs/maniskill/<实验编号>/`
- LIBERO 结果：`outputs/smolvla_libero/{pilot,eval}/<实验编号>/`
- 退路用的 PushT 划分与结果：`configs/experiments/pusht_diffusion/split.json`、
  `outputs/pusht_diffusion/pilot/<实验编号>/`
- 每个实验的可执行子计划：`plans/YYYY-MM-DD-<编号与主题>.md`
- 实验结论：按 `reports/AGENTS.md` 写入对应报告，不写回本计划

## 风险与停止条件

- **M0 最可能卡在 replay 上，两个独立成因。** 一是状态恢复：rl 演示的 `episode_seed` 复现不出录制
  状态（四种试法全 0 命中，机制未定位，见 M0 计划 3.1 第三条），而并行 replay 路径正是用它做 reset，
  只能靠 `--use-first-env-state` 兜。二是跨版本动力学：三份演示生成于三个非 release commit，
  本机是 3.0.1，初始状态分布已验同分布，但同一串动作会不会走到同一个地方未知。
  存活率过低时的退路已写在数据边界里：回 PushT
  只做 W0/W1 机制核查，并**明确承认 W2 的结论在 PushT 上是弱证据**，不得在这种情况下硬下"多步
  动力学没用"的判断。
- **候选多样性仍是 W2/F0 的硬限制。** WAM 能评价动作，不能凭空创造不同动作；若 K 个 chunk
  近乎重复，停止 reranking，另开 action proposal 计划。M1 就是提前发现这件事的地方。
- **防止 latent collapse。** W0 必须使用冻结/target encoder 或可解释辅助 head，并保留
  copy-latent、no-action 和 action-shuffle 对照。
- **防止 model exploitation。** WAM 预测高但实际 selected-action reward 低时，不得继续扩大
  rollout horizon；先进入 W3 的悲观打分和数据聚合。
- **禁止测试泄漏。** ManiSkill test seeds、PushT 的 final/E2b/E4、LIBERO validation/test 只用于
  各自声明的评测；若参与训练，必须另建互不重叠的正式测试集。
- **Residual 不得覆盖基座。** translation、rotation、gripper 分开限幅，监控 residual norm、jerk
  和与 base action 的夹角。残差把基座整个盖掉时闭环成绩可能反而变好，但那已经不是"残差修正"，
  结论会被错误归因。
- **稀疏成功奖励在多阶段任务上会不串。** Foresight Residual RL（arXiv 2607.16506）报的失效正是
  这个设定：冻结 VLA 底座上的残差 RL，稀疏成功奖励能分别改善各子任务，串起来却不涨，因为终态质量
  没被控制。R0 的 reward 必须含终态质量项；若观察到"各阶段都改善、总成功率不动"，先查这条，不要
  当成残差无效。
- **每阶段先 smoke。** 报告数据生成、训练和单 episode 的实测耗时，再冻结全量规模；不得看到结果后
  修改 split、阈值或测试难度。F3 是最贵的单项，按原设计约 11,340 条 LIBERO episode，必须先用实测
  单条耗时反推总时长再裁剪方法数或 seed 数。
- **指标与 simulator 原生信号交叉检查。** ManiSkill 的阶段判定直接用 `info` 里的原生信号，不要自己
  写谓词；PushT 角度目标使用 `π/4`；LIBERO 语言依从必须使用 Prompt Swap 修复后的判定入口。这条是
  两次教训换来的——RF7 的旋转参考角用了 0 弧度而非 `π/4`，唯一成功那条 episode 报 45.83°、改对后
  是 0.83°；Prompt Swap 在环境重置**之后**才评估，把一整列静默清零成 0/10。
- **所有样本效率声明都计算总真实交互。** 包括 WAM 训练轨迹、RL rollout 和数据聚合新增轨迹。
- **`--overwrite` 会 `shutil.rmtree` 整个 run 目录**（`train_b0.py:74` 的
  `_clear_run_artifacts`），对已完成的正式 run 没有保护。B0 的背景 test 和选模结果已经因此丢过
  一次，183/210 现在只存在于文档里。多 seed 会把这个风险乘以 seed 数，F3 开跑前先处理这个入口。

## 收尾

- [ ] 所有阶段完成或明确 `dropped` 后，将状态改为 `done`
- [ ] 更新 `plans/README.md`
- [ ] 结果只写入 `reports/`，本文件不补实验结果
