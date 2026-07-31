# SmolVLA LIBERO：准备工作与 Smoke

## 实验目标

在制作数据 split 和启动正式训练前，确认 SmolVLA base、PEFT、两路相机、8 维 state、
7 维 action、BF16 和 LIBERO 闭环接口能够连通。

结果类别：`Smoke`。

## 实验思路

先补齐 SmolVLA/PEFT 依赖，再用三个目标任务各一条 demonstration 执行两步 LoRA
训练。原生命令只验证 loss 和 checkpoint；随后由 instrumented wrapper 记录 trainable
parameters、LoRA target 和 peak VRAM。闭环接口先用公共 SmolVLA-LIBERO checkpoint
跑完整 episode，再用本地两步 LoRA adapter 跑一条受限步数兼容性 Smoke。

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
| 单任务闭环 evaluator | 已就绪 | 支持 Hub、本地普通和本地 PEFT checkpoint，并输出 Rerun |

依赖安装后，`doctor.sh`、`quickstart_info.sh`、SmolVLA/PEFT import、CUDA 和 BF16
检查均已通过。SmolVLA 单任务闭环训练/评测链路已连通；当前 Phase 0 的剩余阻塞项是
instrumented LoRA Smoke，而不是模型下载或闭环 evaluator。

## 目标数据快照

三个目标任务都来自本地 combined LeRobot 数据集
`datasets/lerobot/lerobot/libero`：

| LIBERO-Goal task ID | Canonical instruction | Combined task index | Episodes | Frames | 时长（10 Hz） |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `put the bowl on the stove` | 17 | 48 | 4,854 | 485.4 秒 |
| 4 | `put the bowl on top of the cabinet` | 18 | 46 | 4,574 | 457.4 秒 |
| 8 | `put the bowl on the plate` | 10 | 49 | 4,563 | 456.3 秒 |
| 合计 | — | — | 143 | 13,991 | 1,399.1 秒 |

这里统计的是 demonstrations。它们通过 action provenance 划分 Train/Offline
Validation；官方 benchmark init states 使用另一套独立 closed-loop split，详见
[数据准备](01_data_preparation.md)。

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
stats。C0 不直接使用 base checkpoint 自带的 6 维 normalization stats，否则动作
反归一化不是有效的 LIBERO action contract。公共 `HuggingFaceVLA/smolvla_libero`
checkpoint 用于闭环 contract Smoke；本地 LoRA checkpoint 则恢复训练时保存的
rename map 和 processor state。

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

Episodes `379/392/398` 只用于 Smoke，不能作为正式训练或实验结果。训练 demonstrations
与官方 benchmark init states 是独立集合，因此不通过 episode 反推或修改 closed-loop
Test states。

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

## C0 公共 checkpoint 完整闭环 Smoke

以下命令执行一个无人工截断的 LIBERO-Goal episode，并保存数值、视频和 Rerun：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"

bash scripts/experiments/smolvla_libero/eval_libero.sh \
  --policy-path=HuggingFaceVLA/smolvla_libero \
  --suite=libero_goal \
  --task-id=1 \
  --n-episodes=1 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --seed=1000 \
  --output-dir="outputs/smolvla_libero/smoke/public_closed_loop_${RUN_ID}" \
  --no-open-viewer
```

这条命令没有 `--max-steps`，成功时保存到权威成功终止帧，失败时执行到 suite 上限。
虽然覆盖完整 episode，但只有一个开发状态，因此仍标为 `Smoke / Closed-loop`，不能用来
报告模型总体成功率。该运行对应 `task_id=1, init_state_id=0, seed=1000`；
`init_state_id=0` 必须排除在正式 test split 之外。

已验证运行 `outputs/smolvla_libero/smoke/hf_libero_goal_task1_rerun_20260730`：

- 第 94 个 transition 成功终止，共保存 94 条 Parquet transition；
- Rerun 两路相机、实测关节、8 维 state 和 done/success 时间线均为 0–94；
- MP4 为 95 帧，包含 reset observation 和成功终止 observation；
- 每次实际发送的 action 与同次模型查询 chunk 的第 0 步完全一致。

## 本地 PEFT 加载 Smoke

两步 LoRA 训练完成后，可自动选择最近的 step-2 adapter 运行 1 步兼容性检查：

```bash
PEFT_POLICY="$(
  find outputs/smolvla_libero/smoke \
    -path '*/checkpoints/000002/pretrained_model' \
    -type d \
    -print |
    sort |
    tail -n 1
)"
test -n "${PEFT_POLICY}"

bash scripts/experiments/smolvla_libero/eval_libero.sh \
  --policy-path="${PEFT_POLICY}" \
  --suite=libero_goal \
  --task-id=1 \
  --n-episodes=1 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --seed=1000 \
  --max-steps=1 \
  --output-dir=outputs/smolvla_libero/smoke/local_peft_rerun_1step \
  --overwrite \
  --no-open-viewer
```

该命令结果类别为 `Smoke`。它只验证本地 adapter、保存的 camera rename map、
processor、7D action 和 Rerun 链路，不提供任务成功率。

## 结果落点

```text
outputs/smolvla_libero/smoke/
├── lora_<timestamp>/
├── instrumented_lora_<timestamp>/
├── public_closed_loop_<timestamp>/
└── local_peft_rerun_1step/
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

两个闭环目录保存：

- `run_config.json`
- `metrics.json` 与 `metrics.md`
- `eval_info.json`
- `rollouts.parquet`
- `rollout.rrd`
- `videos/episode_000.mp4`

## 判断标准

通过条件：

- 两步 LoRA loss 有限，trainable parameters 非零
- 实际 target 与预期 action expert/projections 相符
- 未意外训练冻结的 VLM/vision 参数
- 没有 OOM、feature mismatch 或 state/action normalization 错误
- C0 能完成一条未人工截断的闭环 rollout，不要求成功
- 本地 PEFT 能恢复 adapter、rename map 和 processor，并在发出首个 action 前通过 shape/
  finite 检查

C0 和本地 PEFT 加载已通过；instrumented LoRA Smoke 尚未完成，所以 Phase 0 仍为
进行中。原生两步命令单独成功不能替代完整验收。任一条件失败时，停止后续数据准备和
30,000-step 训练。

下一步：[数据、Split 与 Paraphrase 准备](01_data_preparation.md)。
