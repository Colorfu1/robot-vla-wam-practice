# PushT E5 Oracle-Mode Checkpoint Selection 与 Audit 闭环评测设计

## 状态

`Approved design / Not implemented / Not executed`

本文冻结 E5 Mode-BCChunk 与 Mode-Diffusion 的 checkpoint selection、开环诊断和
dataset-derived 闭环评测协议。它只描述计划；不得把设计中的预期产物或
preflight 结果写成正式实验结论。

## 目标

1. 复用现有 PushT 双模型 checkpoint-selection 基础设施，为 Mode-BCChunk 与
   Mode-Diffusion 分别选择一个最佳 checkpoint。
2. 使用 development demonstrations 的完整 GT 轨迹得到 oracle Prototype ID，并从每条
   demonstration 的首帧重建 PushT 初始状态。
3. 选模只比较 oracle-mode conditioned closed-loop task success；GT action 不进入闭环。
4. 冻结 checkpoint 后，在 audit demonstrations 上执行一次正式开环与闭环评测。
5. checkpoint selection 支持显式多 GPU；audit evaluation 本阶段使用单 GPU
   顺序执行。

## 非目标

- 不要求模型根据 observation 自行预测 Prototype ID。
- 不对每条 episode 枚举 P0/P1/P2/NULL；每条 episode 只输入其 GT demonstration
  对应的 oracle Prototype ID。
- 不把 GT action、medoid sketch 或未来 observation 输入闭环 policy。
- 不用 audit split 选择 checkpoint、阈值或实现策略。
- 不恢复逐 bit 的原始 Pymunk snapshot；dataset 没有保存该 snapshot。
- 不在本协议中评价 NULL 分支的多 seed mode coverage；该问题仍属于原 E5
  完整网格。
- 不修改安装的 LeRobot 或 `gym-pusht` 源码。

## 数据与 Oracle Mode

冻结的 episode split 来自：

```text
configs/experiments/pusht_diffusion/split.json
```

| Split | Episodes | 用途 |
|---|---:|---|
| train | 165 | prototype fit 与模型训练 |
| development | 20 | checkpoint selection |
| audit | 21 | checkpoint 冻结后的 final evaluation |

Prototype 只在 train split 上拟合。训练 sidecar 位于：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/
├── prototype_manifest.json
├── episode_assignments.parquet
└── prototype_sketches.json
```

`episode_assignments.parquet` 中：

- `prototype_id` 是完整 demonstration 到最近 train medoid 的 ID；
- `training_condition_id` 只对 train episode 非空；
- development/audit 的 `training_condition_id=None`，确保它们不能进入训练 wrapper；
- development/audit 的 `prototype_id` 可在本协议中作为 oracle task condition。

当前 K=3。development 的 P0/P1/P2 数量为 6/11/3，audit 为 3/14/4。全部
development/audit episode 都能构造完整 trajectory sketch。development 中四条轨迹
位于最近 prototype 的 train q95 rejection radius 外；它们仍保留最近
`prototype_id` 并参与 success-rate 统计，同时用 `gt_mode_classifiable=false`
单独标注。不得为了提高结果而删除这些 episode。

Oracle Prototype ID 表示调用方已经告诉 policy 希望采用哪一种全局行为模式。
它不是模型从当前 observation 推断出来的预测，也不是把 GT trajectory 直接输入
policy。

## Dataset-Derived 初始状态重建

LeRobot PushT dataset 每帧包含 RGB、二维 agent position、二维绝对 action、reward 和
success，但不包含完整 simulator snapshot。每条 development/audit episode 使用首帧
构造五维初始状态：

```text
[agent_x, agent_y, block_body_x, block_body_y, block_angle]
```

重建步骤固定为：

1. 从首帧 `observation.state` 读取 agent workspace position。
2. 使用 E0 codec-tolerant 图像几何从首帧恢复 T 的几何中心和主方向。
3. 将图像几何方向转换为 PushT block body angle；方向差使用 wrapped angle。
4. 使用 T 的固定局部几何中心 `[0, 285/7]`，从世界几何中心反算 block body
   origin。
5. reset 环境后按 angle → position 的顺序设置 block，避免 legacy position-then-angle
   导致 body origin 漂移。
6. 将 agent/block linear velocity 和 block angular velocity 设为零，reindex shapes，
   再执行一个环境 `dt` 使状态生效。

该初始化称为 `dataset_first_frame_reconstruction`，不能写成 exact simulator replay。
episode ID 用作结果主键和 Diffusion RNG namespace，不与既有 environment seed 列表做
虚构映射。

### 重建 Preflight

任何 checkpoint rollout 前，先对完整 development 与 audit split 做一次无模型
preflight。重建后的 96×96 observation 与 dataset 首帧比较：

- block pose 必须可解析；
- block-mask IoU 必须至少 0.75；
- 重建 block 几何中心误差必须不超过 2 workspace pixels；
- wrapped block angle 误差必须不超过 5 degrees；
- agent position 误差必须不超过 `1e-6`。

任一 episode 失败时停止整个评测，不静默丢弃、不自动放宽阈值。preflight
结果写入 Parquet 和摘要；阈值只能通过新的显式设计变更修改，不能查看 audit
success 后调整。

## 开环评测

开环评测使用 dataset observation、oracle `prototype_id` 和逐帧 GT action。它不推进
环境，也不用于主要 checkpoint 排序。

checkpoint selection 阶段沿用现有 lightweight development open-loop diagnostic，
但为每个 sample 注入 sidecar `prototype_id`。完整 audit open-loop 只对两个冻结后的
selected checkpoints 运行，并报告：

- normalized action MSE；
- denormalized action MSE；
- 每个 Prototype ID 的 MSE；
- action padding 数量和有效 action 数量。

Mode-Diffusion 开环使用固定 `policy_seed=0`、DDPM-100。开环结果只说明 teacher-forced
action error，不能代替闭环 success。

## Development Checkpoint Selection

### 输入

一个入口同时接收：

```text
--bc-run=MODE_BC_RUN_ROOT
--diffusion-run=MODE_DIFFUSION_RUN_ROOT
--prototype-manifest=PROTOTYPE_MANIFEST
--dataset-root=LEROBOT_DATASET_ROOT
--num-gpus=N
--output-dir=OUTPUT_DIR
```

两个 run root 都是必填项。入口使用现有 `discover_checkpoints()` 规则发现所有
numeric LeRobot checkpoints，并在启动 worker 前验证每个 checkpoint 的权重、
config、processor 和 policy type。缺少任一模型或 checkpoint 时整体停止；不得只
选择其中一个模型。

### 闭环协议

对每个模型、checkpoint 和 development episode：

1. 读取该 episode 的 oracle `prototype_id`；
2. 从首帧重建初始环境；
3. reset policy，并在完整 episode 内锁定该 ID；
4. 执行最多 300 environment steps；
5. 不再使用该 episode 的后续 GT observation/action；
6. 保存 success、reward、实际 action targets、agent/block path、latency 和终止原因。

Mode-BCChunk 每条 episode 是一次确定性 rollout。Mode-Diffusion 使用固定
`policy_seed=0`、DDPM-100；选模阶段不使用多 seed oracle coverage。

若两个模型各有八个 checkpoints，预期 development 闭环数为：

```text
Mode-BCChunk:   8 × 20 = 160
Mode-Diffusion: 8 × 20 = 160
total:                   320
```

### 排序规则

每个模型独立按以下 lexicographic 顺序选一个 checkpoint：

1. closed-loop success rate 降序；
2. mean max reward 降序；
3. checkpoint step 升序。

Prototype adherence、open-loop MSE 和 classifiable 状态只作为诊断，不参与排序。
这样可直接回答“给定 GT mode 指令后是否完成任务”，不把聚类拒识规则混入 task
success。

## 多 GPU 执行

复用现有 checkpoint selector 的 spawn process pool、checkpoint discovery、progress
events、atomic output 和共享任务队列：

- 一个 worker 绑定一张 `cuda:<index>`；
- 一个 task 是一个模型 checkpoint；
- worker 加载 checkpoint 后顺序完成全部 20 个 development episodes；
- Diffusion tasks 优先入队以改善负载均衡；
- `worker_count=min(num_gpus, checkpoint_tasks)`；
- parent 进程负责聚合，worker 不竞争写汇总文件；
- 任一 worker 失败时终止其他 worker，atomic output 不发布半成品。

单 GPU 与多 GPU 必须产生相同的 episode/checkpoint 行集合和相同选择结果。

Shell 入口新增：

```text
scripts/experiments/pusht/select_e5_checkpoints.sh
```

Python 实现优先给现有 checkpoint-selection engine 增加 mode-aware evaluator
strategy，保持原 BCChunk/Diffusion 默认行为与现有 CLI 向后兼容。禁止复制一份
独立且逐渐漂移的多 GPU 调度器。

## Audit Final Evaluation

只有 development selection 成功写出并验证 `selected_checkpoints.json` 后才能
开始 audit。final evaluator 从 selection 文件解析两个冻结 checkpoint，不接受手工
替换其中一个路径而不更新 manifest。

Audit 本阶段固定为单 GPU 顺序执行两个 selected checkpoints。它只有 42 条 primary
rollouts，不新增第二套多 GPU 分片或调度协议；`--num-gpus` 只属于 development
checkpoint-selection 入口。未来若 audit 运行时间成为实际瓶颈，再通过单独设计扩展，
不能在实现阶段临时改变 episode/RNG 顺序。

对 21 个 audit episodes：

- 使用各自 GT `prototype_id`；
- 使用相同的首帧重建算法与冻结 preflight 阈值；
- Mode-BCChunk 每条 episode 一次；
- Mode-Diffusion primary result 使用固定 `policy_seed=0`、DDPM-100，每条 episode一次；
- 总计 42 条 primary closed-loop rollouts；
- 同时对两个 selected checkpoints 运行完整 audit open-loop。

正式 primary 指标是两个模型各自的 21-episode success rate。报告 Wilson 95% CI、
mean max reward、每个 prototype 的 success、classifiable/unclassifiable 分层
success、open-loop action MSE、latency 和失败 episode。样本较小时不把百分点差异
写成统计显著。

rollout 完成后可用实际轨迹计算 prototype adherence，帮助解释模型是否遵循指令；
它不改变 task success，也不追溯修改 checkpoint 选择。

## 产物

Development selection：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/checkpoint_selection_oracle_mode/
├── selection.parquet
├── selected_checkpoints.json
├── rollouts.parquet
├── reconstruction_metrics.parquet
├── metrics.json
├── run_manifest.json
└── recording.rrd
```

Audit evaluation：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/audit_oracle_mode/
├── open_loop.parquet
├── rollouts.parquet
├── condition_metrics.parquet
├── reconstruction_metrics.parquet
├── metrics.json
├── manifest.json
└── recordings/
```

manifest 必须记录 split hash、prototype sidecar hash、checkpoint hashes、policy
seed、scheduler、inference steps、重建算法版本、阈值、GPU 数和代码 provenance。
文档和摘要只使用仓库相对路径，不复制 raw action、observation、media 或 model
weights。

## 错误处理

- sidecar hash、split hash 或 dataset revision 不一致：启动前失败。
- development/audit episode 缺少 oracle ID：启动前失败。
- initial-state reconstruction 未通过任一阈值：所有 checkpoint rollout 前失败。
- checkpoint type/K/config 与 sidecar 不一致：对应 worker 失败并取消整体输出。
- 产生重复或缺失的 `(model, checkpoint, episode)` 行：聚合失败。
- audit selection 文件不是 complete development 产物：拒绝执行。
- 目标输出已存在且未显式 `--overwrite`：拒绝覆盖。

## 测试与验证

实现至少覆盖：

1. 首帧几何中心到 block body origin 的坐标变换；
2. angle wrap 与 T 主方向的 90-degree convention；
3. development/audit oracle ID 注入且 GT action 不进入闭环 batch；
4. 重建 preflight 阈值及整批 fail-fast；
5. 两模型 checkpoint discovery 与缺失模型拒绝；
6. success → reward → earliest-step 排序；
7. 单 GPU/多 GPU 任务集合一致；
8. worker error 取消 atomic output；
9. audit evaluator 只能读取冻结 selection；
10. 原有非 E5 checkpoint-selection tests 保持通过。

验证顺序为 focused unit tests、CLI `--print-command`、一个 development episode 的
双模型 Smoke、完整 reconstruction preflight、完整 development selection，最后才是
一次 audit evaluation。

## 结论边界

该协议若成功，只能支持：

> 给定由完整 GT demonstration 得到的 oracle Prototype ID，并从其首帧重建初始
> 状态时，selected policy 的闭环任务成功率为何。

它不能证明 policy 能从 observation 自行选择 mode，也不能证明重建 rollout 会严格
复现 GT trajectory。实际 rollout 偏离 commanded prototype 但完成任务时，task
success 仍为 true，adherence 单独记录。
