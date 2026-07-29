# SmolVLA LIBERO：准备工作与 Smoke

## 实验目标

在制作数据 split 和启动正式训练前，确认 SmolVLA base、PEFT、两路相机、8 维 state、
7 维 action、BF16 和 LIBERO 闭环接口能够连通。

结果类别：`Smoke`。

## 实验思路

先补齐 SmolVLA/PEFT 依赖，再用三个目标任务各一条 demonstration 执行两步 LoRA
训练。原生命令只验证 loss 和 checkpoint；随后由 instrumented wrapper 记录 trainable
parameters、LoRA target 和 peak VRAM。最后使用 LIBERO dataset stats 运行一条 300-step
C0 闭环 rollout。

## 环境审计

审计日期：2026-07-30。

| 项目 | 当前状态 | 实测结果或影响 |
| --- | --- | --- |
| Conda | 已就绪 | `robot-practice`，Python 3.10.20 |
| LeRobot | 已就绪 | 0.4.4，包含 SmolVLA、LIBERO env 和通用 PEFT 包装 |
| PyTorch/CUDA | 已就绪 | PyTorch 2.10.0+cu128，RTX 3090 24 GB |
| BF16 | 已就绪 | GPU 支持 BF16；本实验固定使用 Accelerate BF16，但它不是框架强制要求 |
| LIBERO | 已就绪 | `libero_goal` 可加载，每个任务有 50 个官方初始状态 |
| SmolVLA extra | 已就绪 | `num2words==0.5.14`，已通过 `lerobot[smolvla]` 持久化 |
| PEFT/LoRA | 已就绪 | `peft==0.20.0`，SmolVLAPolicy 与 LoraConfig 导入通过 |
| SmolVLA base | 已缓存 | 两步 LoRA Smoke 已完成首次下载 |
| 本地闭环 evaluator | 未实现 | 当前 evaluator 只支持 ACT checkpoint |

依赖安装后，`doctor.sh`、`quickstart_info.sh`、SmolVLA/PEFT import、CUDA 和 BF16
检查均已通过。当前仍不能直接执行完整实验；阻塞项是首次下载 SmolVLA base，以及
SmolVLA 专用训练/评测胶水。

## 目标数据快照

三个目标任务都来自本地 combined LeRobot 数据集
`datasets/lerobot/lerobot/libero`：

| LIBERO-Goal task ID | Canonical instruction | Combined task index | Episodes | Frames | 时长（10 Hz） |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `put the bowl on the stove` | 17 | 48 | 4,854 | 485.4 秒 |
| 4 | `put the bowl on top of the cabinet` | 18 | 46 | 4,574 | 457.4 秒 |
| 8 | `put the bowl on the plate` | 10 | 49 | 4,563 | 456.3 秒 |
| 合计 | — | — | 143 | 13,991 | 1,399.1 秒 |

这里统计的是 demonstrations，不代表已经证明存在同样数量的 unique init states。
正式 split 仍需按照[数据准备](01_data_preparation.md)完成 provenance 映射。

## Feature 与 Processor Contract

本地 LIBERO 数据：

- `observation.images.image`：256×256 RGB
- `observation.images.image2`：256×256 RGB
- `observation.state`：8 维
- `action`：7 维
- 频率：10 Hz

SmolVLA base 使用三路 camera 名称和 6 维 state/action。本实验统一 rename：

```json
{
  "observation.images.image": "observation.images.camera1",
  "observation.images.image2": "observation.images.camera2"
}
```

第三路 camera 不提供。模型可以使用实际存在的两路图像。

训练入口必须使用 LIBERO dataset features/stats 覆盖 action feature 和 normalization
stats。C0 不能直接使用 base checkpoint 自带的 6 维 normalization stats，否则动作
反归一化不是有效的 LIBERO action contract。

## 准备命令

先把 `requirements.txt` 中的 LeRobot 依赖持久化为：

```text
lerobot[libero,pusht,smolvla,peft]==0.4.4
```

更新依赖后执行：

```bash
conda run --no-capture-output -n robot-practice \
  python -m pip install -r requirements.txt

conda run --no-capture-output -n robot-practice ./scripts/doctor.sh

conda run --no-capture-output -n robot-practice python - <<'PY'
import importlib
import torch

for name in ("lerobot", "libero", "num2words", "peft"):
    importlib.import_module(name)

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
print(torch.cuda.get_device_name(0))
print("SmolVLA/PEFT environment: ok")
PY
```

## 两步 LoRA Smoke

依赖就绪后，用每个任务各一条 demonstration 执行。Episodes `379/392/398` 分别对应
plate、stove、cabinet：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"

conda run --no-capture-output -n robot-practice \
  accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    -m lerobot.scripts.lerobot_train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --policy.n_obs_steps=1 \
    --policy.n_action_steps=10 \
    --policy.optimizer_lr=0.0001 \
    --dataset.repo_id=lerobot/libero \
    --dataset.root=datasets/lerobot/lerobot/libero \
    --dataset.episodes='[379,392,398]' \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --peft.method_type=LORA \
    --peft.r=32 \
    --batch_size=1 \
    --num_workers=0 \
    --steps=2 \
    --log_freq=1 \
    --eval_freq=0 \
    --save_checkpoint=true \
    --save_freq=2 \
    --wandb.enable=false \
    --output_dir="outputs/smolvla_libero/smoke/lora_${RUN_ID}"
```

Episodes `379/392/398` 属于 development-used 数据。后续 provenance 映射必须找到它们
对应的初始状态，并将这些状态排除在正式 test split 之外。

这里显式使用 `--mixed_precision=bf16` 是本实验为了降低显存、提高吞吐并保持 C1/C2/C3
精度配置一致而作出的固定选择，不是 SmolVLA 或 LeRobot 的功能性硬要求。正式训练前
必须由 Smoke 验证该配置下 loss 有限且没有 dtype 错误。

该命令应记录逐 step loss，并在 step 2 保存 train config 和 adapter checkpoint。它不能
记录 PyTorch peak VRAM，也不能完整审计实际 LoRA target。

## Instrumented LoRA Smoke

以下入口尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"

bash scripts/experiments/smolvla_libero/smoke_train.sh \
  --base-policy=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --episodes=379,392,398 \
  --steps=2 \
  --batch-size=1 \
  --learning-rate=0.0001 \
  --lora-rank=32 \
  --mixed-precision=bf16 \
  --output-dir="outputs/smolvla_libero/smoke/instrumented_lora_${RUN_ID}"
```

它必须额外记录：

- trainable/total parameter 数量和比例
- 实际 LoRA target modules 与 adapter config
- step 0/1 loss
- peak allocated/reserved VRAM
- 单 step 时间
- 冻结 VLM/vision 参数是否意外获得梯度

LeRobot 0.4.4 当前训练入口没有可配置的 gradient accumulation。不得把 batch 1 描述成
effective batch 32；如需 accumulation，必须先实现并测试。

## C0 Base 闭环 Smoke

以下入口同样属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"

bash scripts/experiments/smolvla_libero/smoke_eval.sh \
  --policy-path=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-id=1 \
  --init-state-id=0 \
  --policy-seed=2000 \
    --episode-length=300 \
    --output-dir="outputs/smolvla_libero/smoke/base_closed_loop_${RUN_ID}"
```

`task_id=1, init_state_id=0` 同样属于 development-used 状态，不得进入正式 test split。
以后新增 Smoke episode 或 init state 时，必须同步加入数据准备阶段的排除列表。

该 evaluator 必须显式加载 LIBERO stats，不能继续使用 base 的 6 维 processor stats。

## 结果落点

```text
outputs/smolvla_libero/smoke/
├── lora_<timestamp>/
├── instrumented_lora_<timestamp>/
└── base_closed_loop_<timestamp>/
```

原生 `lora_<timestamp>/` 只要求保存 LeRobot 原生 train config、step-2 adapter
checkpoint 和逐 step 控制台日志。它是 bootstrap 检查，不承担完整 Phase 0 验收。

`instrumented_lora_<timestamp>/` 必须保存：

- `run_config.json`
- `metrics.json`
- `trainable_parameters.json`
- `peft_targets.json`
- 逐 step loss
- peak allocated/reserved VRAM
- 错误日志

`base_closed_loop_<timestamp>/` 必须保存 `run_config.json`、`metrics.json`、视频和失败原因。

## 判断标准

通过条件：

- 两步 LoRA loss 有限，trainable parameters 非零
- 实际 target 与预期 action expert/projections 相符
- 未意外训练冻结的 VLM/vision 参数
- 没有 OOM、feature mismatch 或 state/action normalization 错误
- C0 能完成一条 300-step 闭环 rollout，不要求成功

只有 instrumented LoRA Smoke 和 C0 都通过，Phase 0 才算完成。原生两步命令单独成功
不能替代完整验收。任一条件失败时，停止后续数据准备和 30,000-step 训练。

下一步：[数据、Split 与 Paraphrase 准备](01_data_preparation.md)。
