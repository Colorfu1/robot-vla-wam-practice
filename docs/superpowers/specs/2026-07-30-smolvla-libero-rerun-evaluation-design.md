# SmolVLA LIBERO Rerun 闭环评测设计

## 目标

为 SmolVLA 增加本地 LIBERO 闭环评测入口，并复用现有 ACT 评测已经实现的
Rerun 图像、关节、三维机械臂、末端轨迹、动作曲线、奖励、成功状态和延迟可视化。

评测必须支持：

- Hugging Face policy repo ID，例如 `HuggingFaceVLA/smolvla_libero`；
- 本地完整 `pretrained_model`；
- LeRobot 原生保存的本地 PEFT/LoRA checkpoint；
- 成功时保存 reset 到成功终止帧的完整 rollout；
- 失败时运行到 LIBERO suite 步数上限并保存完整 rollout。

旧 Smoke 只有 MP4 和汇总 JSON，不能据此伪造逐步状态或关节数据。验收时重新运行
公共 SmolVLA-LIBERO checkpoint。

## 入口与兼容性

保留现有 ACT 入口及其参数语义：

```text
scripts/experiments/act/eval_libero.sh
```

新增 SmolVLA 入口：

```text
scripts/experiments/smolvla_libero/eval_libero.sh
python -m robot_practice.evaluation.libero_smolvla
```

两个入口共享 LIBERO 环境、逐步数据采集、输出生命周期和 Rerun 记录基础设施，但使用
独立的策略执行适配器。ACT 的 `ActionPlanTracker`、Temporal Ensemble 和 checkpoint
限制继续只属于 ACT。

Python 入口公开 `evaluate_smolvla_libero()`。`--policy-path` 接受字符串形式的 Hub repo
ID 或本地目录；解析时先检查它是否是现存本地目录，否则按 Hub ID 处理。显式
`--output-dir` 优先。未提供时：

- 标准本地 run checkpoint 继续写入该 run 的 `eval/libero/.../<timestamp>`；
- Hub ID 或其他本地目录写入
  `outputs/smolvla_libero/eval/<safe-policy-name>/<suite>/task_<id>/<timestamp>`。

`safe-policy-name` 只由 policy reference 的末段经过字母、数字、点、下划线和连字符
白名单处理得到，不能形成任意路径。

## 架构

```text
LIBERO closed-loop engine
├── task、seed、init state 与环境生命周期
├── observation、图像与实测关节采集
├── success/environment-done/time-limit 终止
├── Parquet、MP4、JSON、Markdown 输出
└── policy-neutral Rerun recording
      ↑
policy execution adapter
├── ACT adapter
│   └── ActionPlanTracker 与 Temporal Ensemble
└── SmolVLA adapter
    ├── Hub、本地普通或本地 PEFT 加载
    ├── 单次 predict_action_chunk
    └── 同一 chunk 同时驱动执行队列与可视化
```

优先抽取现有实现中的模型无关部分，不复制完整 ACT evaluator。改造只覆盖本目标需要的
边界，不扩展到其他 policy family。

## SmolVLA 加载

使用 LeRobot 原生接口：

- `PreTrainedConfig.from_pretrained()` 解析 policy 配置；
- 随后显式设置 `config.pretrained_path = Path(policy_reference)`；仅解析 config 不会让
  `make_policy()` 自动加载权重；
- `make_policy()` 加载普通或 PEFT policy；
- `make_pre_post_processors()` 加载 checkpoint normalization 与语言/图像处理；
- `make_env_pre_post_processors()` 处理 LIBERO observation 和 controller action。

本地 PEFT checkpoint 由配置中的 `use_peft` 与 `adapter_config.json` 驱动 LeRobot 原生
`PeftModel` 加载，不手工合并 adapter 权重。

加载后要求：

- config 类型为 `SmolVLAConfig`；
- 普通 policy 为 `SmolVLAPolicy`；PEFT policy 可以是 `PeftModel`，但其 base model
  必须是 `SmolVLAPolicy`；
- 两路 256×256 RGB policy 图像可用；
- runtime observation state 为有限 8D；
- controller action 为有限 7D relative end-effector action；
- processor 文件完整可加载。

## 单次查询与动作执行

SmolVLA 的 flow-matching sampling 具有随机性。不得分别调用一次模型用于执行、再调用
一次用于可视化。

每次外部执行队列为空时：

1. 调用一次公开的 `predict_action_chunk()`；
2. 对完整 chunk 执行 checkpoint 与 LIBERO action postprocessing；
3. 保存该 controller-space chunk；
4. 将前 `n_action_steps` 放入外部执行队列；
5. 当前步弹出并执行队首 action；
6. 将仍可执行的 queue 和不会执行的 chunk tail 记录到 Rerun。

队列非空时不调用模型。`model_queried`、`plan_step_index` 与 query latency 必须反映真实
行为。默认采用 checkpoint 的 `n_action_steps`，允许评测入口显式覆盖，但必须满足
`1 <= n_action_steps <= chunk_size`。

## 逐步数据

所有 episode 的每个控制 transition 都在 Parquet 中保存以下数值数据：

- 当前及下一步 8D observation state；
- 当前及下一步 7D simulator-measured arm joints；
- 本次查询产生的完整 action chunk（若有）；
- 实际执行的 controller-space 7D action；
- reward、terminated、truncated、done 和 success；
- 明确的 `success`、`environment_done` 或 `time_limit` 终止原因；
- policy-step latency 与真实 model-query latency；
- seed、init state、episode 和 step identity。

前 `num_visualized_episodes` 个 episode 额外写入 Rerun 和 MP4。Rerun 按时间记录两路
policy 输入图像和 post-action agentview；MP4 只保存 agentview。非可视化 episode
不保存图像，避免声称 Parquet 持久化了它并不包含的像素。

Rerun 使用实测关节做 Panda forward kinematics，显示：

- 两路输入图像与 post-action agentview；
- Panda 三维骨架和关节运动；
- 实测 EEF 历史轨迹；
- 当前 controller command 与实测下一步 response；
- SmolVLA executable queue 与 discarded tail；
- 7D action、8D state、reward、success、done 和 latency 曲线。

通用面板和说明不得写死 “ACT”。ACT 专属 Temporal Ensemble 内容仍由 ACT adapter
按原语义提供。

每个可视化 episode 在 terminal transition 后额外写入一个只表示终止 observation 的
Rerun 时间点 `step=num_steps`。这个 endpoint 必须包含下一步 8D state、7D measured
joints、Panda pose、两路下一步图像以及最终 done/success；不得伪造 action、reward 或
model latency。这样成功帧或失败终止帧不是只有一张 post-action agentview。

## 输出与生命周期

```text
<output-dir>/
├── rollout.rrd
├── rollouts.parquet
├── videos/episode_000.mp4
├── metrics.json
├── metrics.md
├── eval_info.json
└── run_config.json
```

每个可视化 episode 是独立 Rerun Recording，可在 Viewer 左侧切换。输出路径必须在
Hub/config/weights/processor 加载前解析、加锁并写入 `.incomplete`。只有逐步记录、指标、
视频、Rerun 合并及验证全部成功后才能标记完成。加载或执行异常保留诊断状态，不发布
完成指标。

终止状态只读取 vector environment 的权威 `final_info.is_success`：

- 成功 terminal：保存到成功帧；
- 未成功且环境没有提前 terminal：执行到 suite 的完整步数上限；
- 环境权威地提前 terminal、但 `is_success=false`：记录为 `environment_done` 失败并停止，
  因为继续 step 会在 autoreset 后进入另一个 episode；
- 短 `--max-steps`：记录为 `time_limit` Smoke，并在 metrics 中明确标记非正式结果。

## 错误处理

以下情况必须在发送无效环境 action 前失败：

- checkpoint 不是 SmolVLA；
- processor 或 PEFT adapter 文件不完整；
- observation、joint、chunk 或 action shape 不符合契约；
- observation、chunk 或 action 含 NaN/Inf；
- action postprocessing 改变了预期维度；
- 显示的首个执行 action 与实际发送 action 不一致。

Hub 下载或加载失败、视频/Rerun 写入失败以及 archive 验证失败都保留 `.incomplete`。
为保证这一点，输出目录创建和锁定必须早于 policy config、权重和 processor 加载。不得把
部分产物描述为完成的闭环评测。

## 测试

### 单元测试

- 一个 SmolVLA queue 周期只调用一次 `predict_action_chunk()`；
- 执行 action 与 Rerun chunk 来自同一个模型返回值；
- `n_action_steps=1` 时只执行第一步，其余为 discarded；
- queue hit 不产生 model-query latency；
- Hub ID、本地普通 checkpoint、本地 PEFT checkpoint 使用正确加载路径；
- 非 SmolVLA、错误 shape 和非有限值被拒绝。

### 假环境集成测试

- 两步成功 episode 生成 Parquet、RRD、MP4 和完整指标；
- 失败 episode 运行到 time limit 并完整保存；
- RRD archive 含相机、Panda、EEF、action/state、reward/success 和 latency 实体；
- terminal endpoint 含最终两路图像、8D state、7D joints 和 Panda pose；
- early unsuccessful environment terminal 使用 `environment_done`，且不在 autoreset 后继续；
- 每个 episode 有唯一 Recording ID、名称和独立时间线；
- 现有 ACT evaluator 和 Rerun 测试保持通过。

### 真实 Smoke

使用：

```text
policy=HuggingFaceVLA/smolvla_libero
suite=libero_goal
task_id=1
init_state_id=0
n_episodes=1
num_visualized_episodes=1
n_action_steps=1
```

验收条件：

- rollout 闭环运行到成功 terminal 或 300 步 time limit；
- Parquet transition 数与实际控制步数一致；
- MP4 覆盖 reset 后首帧到 terminal observation；
- `rollout.rrd` 可加载且包含一个独立 Recording；
- 两路图像、实测关节、EEF 轨迹、action/state 与终止状态时间同步，且 terminal
  endpoint 完整；
- 完成后自动打开新 `rollout.rrd`。
