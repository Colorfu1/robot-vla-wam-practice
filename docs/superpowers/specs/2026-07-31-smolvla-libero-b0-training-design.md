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
| Optimizer LR | `1e-4` |
| Optimizer updates | 30,000 |
| Checkpoint cadence | 每 5,000 steps |
| Train seed | 1000 |

训练保持 LeRobot 0.4.4 的原生全局按帧 shuffle，不增加 task-balanced、episode-balanced
或 replacement sampler。`DatasetConfig.episodes` 只限定 B0 Train episodes。

Normalization 保持 LeRobot 默认行为：图像使用 ImageNet stats，state/action 使用完整
`lerobot/libero` metadata 中的默认 stats。本工作不重新计算 B0 Train-only stats。
`run_config.json` 必须明确记录该 scope，避免以后将其误解为 train-only normalization。

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
- fresh run 时调用 `accelerate launch --mixed_precision=bf16`；
- 将 `--num-processes` 传给 Accelerate；
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
4. 加载 base config，并确认最终 policy 是 Non-PEFT、默认冻结边界；
5. 构造只含 B0 Train episodes 的 `DatasetConfig`；
6. Accelerate 初始化 8 个进程，并确认 mixed precision 为 BF16；
7. 调用 LeRobot 原生训练循环；
8. 每 5,000 steps 保存完整 policy、processor 和 training state；
9. 正常完成后发布运行记录，失败时保留诊断日志。

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

不自动猜测“最新 checkpoint”。Resume 前必须验证：

- checkpoint 同时具有完整的 `pretrained_model/` 和 `training_state/`；
- saved train config 指向原 B0 output directory；
- split SHA-256、episode 列表、base policy identity、world size、per-GPU/global batch、
  LR、scheduler、seed 和 mixed precision 与原 run contract 一致；
- 当前 split 文件及 dataset metadata 仍与 saved run config 一致；
- resume step 是合法的已保存 step，且小于 30,000。

通过后使用 LeRobot 原生 resume 路径恢复 model、optimizer、scheduler、RNG 和 training
step。Resume 不接受 output directory、batch、LR、seed、steps 或 scheduler override。
训练日志和 JSONL 指标以 append 方式继续，写入明确的 resume event；已有历史记录不得
截断或重复伪造。

## 输出与记录

Fresh run 的目标结构为：

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation/
├── run_config.json
├── train_metrics.jsonl
├── train.log
├── runtime_rank_0.json
├── ...
├── runtime_rank_7.json
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
- base policy reference；
- resolved SmolVLA 冻结边界；
- sampler 和 normalization scope；
- per-GPU/global batch、world size、BF16、LR、scheduler、steps、seed；
- LeRobot、Torch、CUDA 和 Accelerate 版本。

`train_metrics.jsonl` 使用一个对象一行，至少记录 step、loss、grad norm、LR、
dataloading time、update time 和 wall-clock time。Resume event 使用独立 record type。

每个 `runtime_rank_*.json` 至少记录 rank、device、peak allocated/reserved VRAM、
开始/结束时间、完成状态和总耗时。聚合吞吐只能基于所有 rank 均成功完成的数据计算。

运行记录先写到输出目录旁的临时文件。LeRobot 创建正式输出目录后再移动进去。失败时
保留临时日志并明确标记失败，不发布“完成”的 runtime summary。

## 错误处理

以下情况必须在第一个 optimizer update 前失败：

- split schema、hash、计数或目标任务排除不符合契约；
- selected episode 不存在、重复、task 不匹配或帧数不一致；
- dataset feature/action contract 不是两路图像、8D state、7D action；
- base policy 不是 SmolVLA；
- PEFT 被启用，或冻结边界、chunk/action horizon 漂移；
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
- Non-PEFT、冻结边界、camera rename、batch、scheduler 和 checkpoint cadence 正确；
- formal 参数漂移被拒绝；
- fresh 和 resume 的完整命令生成；
- 已有输出、缺失 training state、错误 resume step 和 contract 不一致被拒绝；
- JSONL/runtime 记录可恢复追加且不会截断历史。

真实 Smoke 使用 Docker 内单卡 BF16、batch 1、2 steps，要求：

- loss、grad norm 和 LR 均有限；
- step 2 checkpoint 的 policy、processor 和 training state 完整；
- `run_config.json`、`train_metrics.jsonl`、`train.log` 和 rank runtime 文件存在；
- 从 step 2 resume 的受限 Smoke 能恢复 step、optimizer、scheduler 和 RNG；
- Smoke 与 Formal 标签明确区分。

最终代码还必须通过相关 pytest、Ruff、Shell 语法检查和 `git diff --check`。8 卡正式运行
之前，在远程容器内执行 GPU/BF16/NCCL 环境验证。本设计验收不把两步 Smoke 描述为
B0 正式训练结果。
