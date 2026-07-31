# SmolVLA LIBERO B0 训练入口设计

## 目标与范围

本工作为 SmolVLA LIBERO 实验实现可复现的 B0 domain adaptation 训练入口。B0 从
`lerobot/smolvla_base` 开始，只使用 `background_split.json` 中 37 个非目标任务的
Train episodes，并产出供 Z0、C1、C2、C3 共用的 checkpoint 候选。

本工作只覆盖 B0 训练、训练记录和显式 checkpoint resume，不实现 B0 checkpoint
selection、Z0 闭环评测或 C1/C2/C3 训练。

## 冻结训练契约

正式 B0 固定为：

| 项目 | 值 |
| --- | --- |
| Dataset | `lerobot/libero` |
| Split | `configs/experiments/smolvla_libero/background_split.json` |
| Tasks | 37 个非目标任务 |
| Train episodes | 1,412 |
| Train frames | 236,470 |
| 排除 task indices | `10,17,18` |
| 初始化 | `lerobot/smolvla_base` |
| PEFT | 关闭 |
| `train_expert_only` | `true` |
| `freeze_vision_encoder` | `true` |
| `n_obs_steps` | 1 |
| `chunk_size` | 50 |
| `n_action_steps` | 10 |
| 每卡 batch | 8 |
| GPU/world size | 8 |
| Global batch | 64 |
| Mixed precision | BF16 |
| Optimizer | AdamW |
| Optimizer LR | `1e-4` |
| AdamW betas / eps | `(0.9,0.95)` / `1e-8` |
| Weight decay | `1e-10` |
| Gradient clipping | global norm `10` |
| Scheduler | cosine decay with warmup |
| Warmup / decay steps | 1,000 / 30,000 |
| Final LR | `2.5e-6` |
| Optimizer updates | 30,000 |
| Checkpoint cadence | 每 5,000 steps |
| Train seed | 1000 |

训练保持 LeRobot 0.4.4 的原生全局按帧 shuffle，不增加 task-balanced、episode-balanced
或 replacement sampler。`DatasetConfig.episodes` 只限定 B0 Train episodes。

Normalization 保持 LeRobot 默认行为：图像使用 ImageNet stats，state/action 使用完整
`lerobot/libero` metadata 中的默认 stats。本工作不重新计算 B0 Train-only stats。
`run_config.json` 必须明确记录该 scope，避免以后将其误解为 train-only normalization。

Optimizer、scheduler 和 gradient clipping 使用并显式复核 SmolVLA 0.4.4 的上述完整
training preset，`use_policy_training_preset=true`。不能只依赖将来可能变化的默认值；
resolved config 必须把所有数值持久化，fresh 和 resume 均按这些值 fail closed。

## 方案选择

采用本地 Python launcher 构造官方 `TrainPipelineConfig`，随后调用安装版
`lerobot.scripts.lerobot_train.train()`。本地代码只负责实验约束、输入验证、运行记录和
resume 适配，不复制或修改 LeRobot 的 forward、backward、optimizer、scheduler、DDP
和 checkpoint 保存逻辑。

不采用以下方案：

- 纯 Shell 拼接 1,412 个 episode IDs：实验契约难以独立验证和测试；
- 复制 LeRobot 训练循环：会形成不必要的上游代码分叉。

## 组件

### Shell 入口

新增：

```text
scripts/experiments/smolvla_libero/train_b0.sh
```

Shell 入口负责：

- 解析容器内可直接执行的参数；
- 从仓库根目录解析相对输入和输出路径；
- fresh 和 resume 都调用 `accelerate launch`；
- fresh 将显式的 `--num-processes` 与 BF16 传给 Accelerate；
- resume 从原 run 的 `run_config.json` 读取并重建相同的 process count、mixed precision
  和 rendezvous 参数，不接受这些字段的 CLI override；
- `--print-command` 时打印完整、可复制的最终命令；
- 保持退出码和信号语义，不在后台吞掉训练错误。

### Python 入口

新增：

```text
src/robot_practice/experiments/smolvla_libero/train_b0.py
```

Python 入口负责：

- 加载并验证 background split；
- 验证本地 dataset metadata、features 和 episode/frame 计数；
- 加载 SmolVLA base config 并覆盖冻结的 B0 参数；
- 构造官方 `DatasetConfig`、`TrainPipelineConfig` 和 Accelerate runtime；
- 验证正式运行的 world size、global batch 和 BF16；
- 调用 LeRobot 原生训练函数；
- 写入主进程训练日志、逐 step 指标和各 rank runtime 指标。

可复用的 split/config/runtime 辅助代码留在
`src/robot_practice/experiments/smolvla_libero/` 内，不与 PushT 实验模块耦合。

### 指标截获边界

LeRobot 0.4.4 没有训练 callback。为了在不复制训练循环的前提下生成逐 step JSONL，
本地 runtime adapter 在调用训练函数前临时包装模块级
`lerobot_train.update_policy`：

1. wrapper 使用完全相同的参数调用并返回原始 `update_policy`；
2. 原调用成功返回后，只由 rank 0 从返回的 `MetricsTracker` 读取本 step 的 loss、
   grad norm、LR、update time 和调用前已设置的 dataloading time；
3. wrapper 不修改 batch、loss、gradient、optimizer、scheduler、tracker 或返回值；
4. step 从 fresh 的 0 或 resume training state 的 saved step 初始化；
5. 训练退出后恢复原始函数对象。

该 adapter 是进程内观测包装，不修改 site-packages，也不解析不稳定的控制台文本。
LeRobot 自身 `log_freq` 固定为 100；`train_metrics.jsonl` 仍由 adapter 每个成功
optimizer update 写一条。测试必须证明包装前后原函数调用次数、参数、返回值和更新结果
一致。

## Fresh 训练流程

正式入口为：

```bash
bash scripts/experiments/smolvla_libero/train_b0.sh \
  --mode=formal \
  --base-policy=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/background_split.json \
  --num-processes=8 \
  --batch-size=8 \
  --steps=30000 \
  --save-freq=5000 \
  --learning-rate=0.0001 \
  --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation
```

流程如下：

1. 输出目录存在时立即拒绝，避免覆盖已有 run；
2. 校验 split schema、聚合计数、任务排除和 episode 唯一性；
3. 用 dataset metadata 复核每个 selected episode 的 task 与帧范围；
4. 将 Hub base policy 解析到不可变 revision，保存 repo ID、resolved revision 和关键
   config/weight 文件 SHA-256；本地 base policy 则保存内容 SHA-256；
5. 加载 base config，并确认最终 policy 是 Non-PEFT、默认冻结边界；
6. 构造只含 B0 Train episodes 的 `DatasetConfig`；
7. 显式复核完整 optimizer、scheduler 和 gradient clipping preset；
8. Accelerate 初始化 8 个进程，并确认 mixed precision 为 BF16；
9. 安装只读指标 adapter 并调用 LeRobot 原生训练循环；
10. 每 5,000 steps 保存完整 policy、processor 和 training state；
11. 正常完成后发布运行记录，失败时保留诊断日志。

相机键继续使用已验证的映射：

```json
{
  "observation.images.image": "observation.images.camera1",
  "observation.images.image2": "observation.images.camera2"
}
```

正式模式对冻结字段 fail closed。`--mode=smoke` 允许覆盖 GPU 数、batch、steps 和
save frequency，用于单卡两步验证，但产物和文档必须标记为 `Smoke`，不能作为正式
checkpoint。

## Resume 语义

入口支持显式：

```bash
bash scripts/experiments/smolvla_libero/train_b0.sh \
  --resume-from=outputs/smolvla_libero/pilot/b0_domain_adaptation/checkpoints/005000
```

Shell 仍以 8 进程、BF16 方式启动同一个 Python 入口；这些启动参数从原
`run_config.json` 重建，而不是依赖当前 Accelerate 默认配置。

不自动猜测“最新 checkpoint”。`--resume-from` 必须精确指向 step checkpoint 根目录，
并映射为：

```text
config_path = <resume-from>/pretrained_model/train_config.json
checkpoint_path = <resume-from>
training_state = <resume-from>/training_state
```

Resume 前必须验证：

- checkpoint 同时具有完整的 `pretrained_model/` 和 `training_state/`；
- saved train config 指向原 B0 output directory；
- 它是该 run 当前 step 最大的完整 checkpoint；显式选择更旧 checkpoint 会被拒绝；
- split SHA-256、episode 列表、resolved base revision/内容 hash、world size、
  per-GPU/global batch、完整 optimizer/scheduler/gradient clipping、seed 和 mixed
  precision 与原 run contract 一致；
- 当前 split 文件及 dataset metadata 仍与 saved run config 一致；
- resume step 是合法的已保存 step，且小于 30,000。

通过后使用 LeRobot 原生 resume 路径恢复 model、optimizer、scheduler、RNG 和 training
step。Resume 不接受 output directory、batch、LR、seed、steps 或 scheduler override。

每次 fresh/resume 都分配单调递增的 `attempt_id`。训练日志和 JSONL 指标以 append
方式继续，并先写入包含 `attempt_id`、`resume_from_step` 和原 metrics tail step 的
resume event。任务可能在 checkpoint 之后又完成若干未保存 updates，因此 resume 后允许
不同 attempt 出现相同 training step；记录的唯一键是 `(attempt_id, step)`，历史记录不
截断。后续汇总必须按 attempt 明确处理这些真实重跑 step，不能把它们伪装成一次连续
运行。

## 输出与记录

Fresh run 的目标结构为：

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation/
├── run_config.json
├── train_metrics.jsonl
├── train.log
├── runtime_attempt_000_rank_0.json
├── ...
├── runtime_attempt_000_rank_7.json
└── checkpoints/
    ├── 005000/
    ├── 010000/
    ├── 015000/
    ├── 020000/
    ├── 025000/
    └── 030000/
```

`run_config.json` 至少记录：

- split 路径和 SHA-256；
- selected task/episode/frame 数；
- base policy reference、resolved revision 和内容 SHA-256；
- resolved SmolVLA 冻结边界；
- sampler 和 normalization scope；
- per-GPU/global batch、world size、BF16、完整 optimizer/scheduler/gradient clipping、
  steps、seed；
- 完整 resolved `TrainPipelineConfig`；
- LeRobot、Torch、CUDA 和 Accelerate 版本。

`train_metrics.jsonl` 使用一个对象一行，至少记录 step、loss、grad norm、LR、
dataloading time、update time、wall-clock time 和 `attempt_id`。Fresh/resume event
使用独立 record type。

每个 `runtime_attempt_<id>_rank_<rank>.json` 至少记录 attempt、rank、device、peak
allocated/reserved VRAM、开始/结束时间、完成状态和总耗时。聚合吞吐只能基于同一
attempt 的所有 rank 均成功完成的数据计算，resume 不覆盖早期 attempt。

运行记录先写到输出目录旁的临时文件。LeRobot 创建正式输出目录后再移动进去。失败时
保留临时日志并明确标记失败，不发布“完成”的 runtime summary。

## 错误处理

以下情况必须在第一个 optimizer update 前失败：

- split schema、hash、计数或目标任务排除不符合契约；
- selected episode 不存在、重复、task 不匹配或帧数不一致；
- dataset feature/action contract 不是两路图像、8D state、7D action；
- base policy 不是 SmolVLA；
- PEFT 被启用，或冻结边界、chunk/action horizon 漂移；
- optimizer、scheduler 或 gradient clipping 与冻结 preset 不一致；
- formal world size 不是 8、per-GPU batch 不是 8、global batch 不是 64；
- formal mixed precision 不是 BF16；
- fresh output 已存在；
- resume checkpoint 不完整或与原 contract 不一致。

各 distributed rank 的异常必须传播为非零退出码。任一 rank 失败时不得把整个 run 标记为
完成。

## 测试与验收

新增：

```text
tests/test_smolvla_libero_b0_training.py
reports/guides/smolvla_libero/02_b0_domain_adaptation.md
```

单元测试覆盖：

- 正确 split 得到 37 tasks、1,412 episodes、236,470 frames；
- 目标 task、重复 episode、错误帧数和 schema 漂移被拒绝；
- resolved config 保持原生 frame shuffle 和默认 normalization；
- Non-PEFT、冻结边界、camera rename、batch、完整 optimizer/scheduler/gradient
  clipping 和 checkpoint cadence 正确；
- 指标 wrapper 不改变原 `update_policy` 的参数、调用次数、返回值或更新结果；
- formal 参数漂移被拒绝；
- fresh 和 resume 的完整命令生成；
- resume 重建相同的 8 进程 BF16 Accelerate 命令，并把 checkpoint 映射到准确的
  config/training-state 路径；
- 已有输出、缺失 training state、非最新完整 checkpoint、错误 resume step 和 contract
  不一致被拒绝；
- JSONL/runtime 按 attempt 追加，允许可解释的重跑 step 且不会截断历史。

Fresh Smoke 使用单卡 BF16、batch 1、2 steps；另用 total steps 3、save frequency 2
的受限运行在 step 2 完整发布后终止并续训到 step 3。要求：

- loss、grad norm 和 LR 均有限；
- step 2 checkpoint 的 policy、processor 和 training state 完整；
- `run_config.json`、逐 step `train_metrics.jsonl`、`train.log` 和 attempt/rank runtime
  文件存在；
- 中断续训 Smoke 能从 step 2 恢复 step、optimizer、scheduler 和 RNG，并完成 step 3；
- Smoke 与 Formal 标签明确区分。

最终代码还必须通过相关 pytest、Ruff、Shell 语法检查和 `git diff --check`。8 卡正式运行
之前，在远程容器内执行 GPU/BF16/NCCL 环境验证。本设计验收不把两步 Smoke 描述为
B0 正式训练结果。
