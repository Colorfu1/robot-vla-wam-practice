# M2-W StackCube Action-only Flow 基线执行指南

## 实验目标

在 M1-W 已冻结的 ManiSkill3 `StackCube-v1` GPU/7D/RGB 协议上，训练唯一的
Action-only Flow baseline，并只用 development split 选择 checkpoint 与 `n_action_steps`，
最后生成可复核的 frozen baseline。这个流程不实现 future/world head，不读取 offline test
episode，也不读取 state-bank test。

本指南对应计划：
[`plans/2026-08-13-M2-W-StackCube-Action-only-Flow基线与Headroom.md`](../../../plans/2026-08-13-M2-W-StackCube-Action-only-Flow基线与Headroom.md)。

## 实验思路

- 输入是最近 2 步的 `base_camera`、`hand_camera` RGB 和 18D `qpos+qvel`。
- 输出是长度 16 的原生 7D `pd_ee_delta_pose` action chunk。
- 视觉编码器为无预训练 ResNet-18，BatchNorm 替换为 GroupNorm，32-keypoint SpatialSoftmax。
- token trunk 固定为 6 层、8 heads、`d_model=256`、MLP ratio 4、dropout 0.1。
- 训练目标是 `x0~N(0,I)` 到 clean action 的线性 Flow Matching velocity MSE。
- 推理固定 Euler/10 steps；`n_action_steps` 只比较 4 和 8。
- development 先评估 zero/random reference，再评估 5 个 checkpoint × 2 个执行步数并机械选模。

### 数据读取与启动缓存

训练的权威输入是冻结的 D_core `trajectory.h5`，不是把约 10 GiB H5 全部载入内存：
`StackCubeWindowDataset` 按需读取每个样本的 2 帧观测、18D state 和最多 16 步 action。
正式训练每个 rank 使用 4 个 DataLoader worker；每个 worker 在进程内复用一个只读 H5 句柄，
并启用 persistent workers/prefetch，避免每个 sample 重开文件。

LeRobot 数据集仍会被前置 hash/feature gate 校验，并用于兼容性、转换产物和审计；M2-W 的
训练读取暂不直接走 `LeRobotDataset`。原因是当前 LeRobot 产物的 RGB 是 AV1 视频，随机抽取
2 帧窗口需要视频解码，而 D_core H5 保存的是与 action/state 对齐的原始 RGB，按窗口切片更
适合这条 action-window 训练路径，也不会改变冻结数据语义。

train-only normalization stats 会缓存到 D_core 目录下的 `.m2_w_normalization_cache/`，缓存
绑定 D_core 文件大小/修改时间、D_core manifest、offline split 和 train episode 数；命中时
启动阶段不再扫描 787 条 train episodes。每次 run 仍会把实际使用的 stats 复制到自己的输出目录。

## 1. 前置检查

在仓库根目录执行。环境必须是 Python 3.10 的 `robot-practice`；首次安装按
根目录 [`AGENTS.md`](../../../AGENTS.md) 的命令完成。

```bash
cd "$(git rev-parse --show-toplevel)"
export PYTHONNOUSERSITE=1

conda run --no-capture-output -n robot-practice bash scripts/doctor.sh

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=validate-frozen \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --protocol=outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json

bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json
```

最后一条会检查 M1-W frozen hash、D_core/LeRobot manifest hash、offline split、state-bank
manifest、观察/动作 shape、Flow solver/steps、global batch 和 privileged key。失败时不要
启动 smoke 或正式训练。

## 2. 单卡 smoke

smoke 只验证实现，不是训练结果。它固定执行 20 个 optimizer steps、保存并加载一次
checkpoint、检查 `[B,16,7]` Flow 输出与 fingerprint-derived noise determinism，然后在
development state bank 前 10 条上用 `n_action_steps=4/8` 各跑最多 10 步。

使用一个新的空输出目录；已有非空目录会被 runner 拒绝。本仓库已保留一次可复核的
`outputs/.../smoke` smoke 结果，若要重新执行，请把最后一个参数改成新的目录名，例如
`outputs/maniskill/m2_task_base/StackCube-v1/smoke_rerun`：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=smoke \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=1 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/smoke
```

通过条件是命令返回 0，且 `run_manifest.json` 中同时出现：

- `status: "complete"`
- `completed_steps: 20`
- `smoke_checks.checkpoint_reload: true`
- `smoke_checks.flow_output_shape: [2, 16, 7]`
- `smoke_checks.noise_deterministic: true`
- `smoke_checks.test_access_count: 0`

smoke 的 loss、成功率和 latency 不得当作 M2-W 正式指标。

## 3. 唯一正式训练

正式合同是 seed 1000、50,000 optimizer steps、global batch 256、AdamW `1e-4`、linear
warmup 1000 后 cosine、bf16、每 10,000 steps 保存 checkpoint；候选固定为
10k/20k/30k/40k/50k。

正式 config 固定 `world_size=8`、`per_process_batch_size=32`，因此正式训练必须使用 8 张可见
GPU。先做 8 卡预检：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8
```

预检通过后启动唯一正式 run：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000
```

正式 run 的 `run_manifest.json` 应明确记录 `global_batch_size=256`、`world_size=8`、
`per_process_batch_size=32`、`completed_steps=50000` 和 10k–50k 候选 checkpoint。当前
runner 不会把单卡/四卡静默当成这份正式合同；若目标机器不是 8 卡，应另建 config 和子计划，
保持 global batch、optimizer steps、模型和 sampler 合同不变。

训练过程会同时持久化到训练输出目录：`training.log` 保存启动、每 100 steps 的进度、
checkpoint、完成/失败信息；`loss_history.jsonl` 每个 optimizer step 保存一条结构化记录，
包括 `loss`、`lr`、`grad_norm`、step/累计耗时、ETA 和吞吐。`run_manifest.json` 的 `logs`
与 `loss_history` 字段会指向这些文件，因此不需要依赖终端滚动输出；训练中可直接执行：

```bash
tail -f outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000/training.log
```

旧 run 没有逐 step 的历史，新增日志无法从已结束的终端输出补回；需要重新训练才会生成
完整的 `loss_history.jsonl`。

如果需要用同一个输出目录重跑，显式增加 `--overwrite`；它只允许清理 `outputs/` 下指定的
目录，默认不覆盖已有或未完成 run：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --overwrite
```

## 4. development reference

只使用 200 条 development exact states；zero/random 使用同一批 state 顺序和相同 500 步
horizon。输出目录必须是新的空目录：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=evaluate-references \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --num-envs=64 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/development
```

结果落点为：

```text
outputs/maniskill/m2_task_base/StackCube-v1/development/reference_metrics.json
outputs/maniskill/m2_task_base/StackCube-v1/development/references/zero/metrics.json
outputs/maniskill/m2_task_base/StackCube-v1/development/references/iid_gaussian_random/metrics.json
```

`--num-gpus=8` 会启动 8 个 evaluator worker，每个子进程只暴露一张物理 GPU
（通过 `CUDA_VISIBLE_DEVICES` 隔离），进程内统一使用逻辑设备 `cuda:0`，避免多个进程的
`torch.random.fork_rng` 同时初始化全部 GPU。合并后的 `metrics.json` 会记录逻辑
`worker_devices`/`environment_devices`、物理卡号 `physical_devices`，以及视觉模型的
`policy_devices`；reference 没有 policy，所以 `policy_devices` 为空。
环境创建使用 ManiSkill 的标准 `sim_backend="gpu"` 入口，不再在 worker 内拼接
`physx_cuda:N`/`sapien_cuda:N`；物理卡选择完全由每个子进程的 `CUDA_VISIBLE_DEVICES` 完成。
评测 worker 数是 rollout 并行度，不是训练 world size，因此可以因故障 GPU 被排除而使用
7 个 worker；global batch size 的整除约束只适用于正式训练。

评测期间父进程会显示单行总进度，按“已完成 episode + 当前 batch 的仿真 step 比例”估算，
因此即使 episode 尚未到达 500 步，也会持续显示进度和 ETA。每个 worker 的详细 stdout/stderr
保存在对应 `references/*/worker_*.log` 下，过程状态保存在 `progress_worker_*.json`；进度行中的
`workers=a/b active=...` 会区分 `running`、`finalizing`、`writing_metrics` 等收尾状态。
worker 超过 300 秒没有更新心跳时，父进程会停止本次评测并打印对应 worker log，避免无限等待。
worker log 还会记录 CUDA 可见性、环境创建、reset、`env.close()` 和 metrics 写入的阶段；
远端无法直接观察设备时，以这些日志和 worker 退出码为准。

## 5. development 联合选模

`select-development` 会先在 offline development episode 上计算固定 Flow noise 的
normalized action MSE，再对 5 个 checkpoint × `n_action_steps={4,8}` 做 200-state
闭环评测。它不读取 offline test，也不读取 state-bank test。

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --num-envs=64 \
  --train-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/development
```

机械排序固定为：

```text
success_at_end ↓
success_once ↓
ever-on-cube ↓
normalized action MSE（小优先）
amortized policy latency p95（小优先）
checkpoint step（早优先）
n_action_steps（小优先）
```

结果落点为：

```text
outputs/maniskill/m2_task_base/StackCube-v1/development/candidate_metrics.json
outputs/maniskill/m2_task_base/StackCube-v1/development/selected_checkpoint.json
outputs/maniskill/m2_task_base/StackCube-v1/development/candidates/step_*/metrics.json
```

## 6. 冻结 baseline

只有 `selected_checkpoint.json` 的全部 development gates 通过，才可以执行冻结：

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=freeze \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --train-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --selection=outputs/maniskill/m2_task_base/StackCube-v1/development/selected_checkpoint.json \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/frozen_base

bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=validate-frozen \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --protocol=outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/frozen_base.json
```

冻结目录必须包含：

```text
outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/frozen_base.json
outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/checkpoint/model.pt
outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/checkpoint/normalization_stats.json
```

`frozen_base.json` 记录 checkpoint/config/M1-W protocol/dataset/split/state-bank/normalization
hash、training seed、Flow solver/steps、选中 `n_action_steps` 和完整 development ranking。
checkpoint 是实体复制文件，不是 `latest` symlink。

## 判断标准与停止条件

下表是 development 数字出现前就已经写入 config/runner 的固定 gate：

| Gate | 通过条件 |
| --- | --- |
| 数据身份 | M1-W、dataset、split、state-bank hash 一致；`test_access_count=0` |
| Flow 学习 | selected normalized action MSE 比 train-action-mean predictor 低至少 10% |
| 任务主体 | `success_at_end≥0.30`、`success_once≥0.40`、ever-grasped `≥0.70`、ever-on-cube `≥0.40` |
| 高于参考 | 相对 zero/random 中较强者至少 `+0.20`，paired bootstrap 95% CI 下界 `>0` |
| Headroom | `success_at_end≤0.90` |
| 动作合法 | finite rate 100%，送入环境前裁剪元素比例 `≤1%` |
| 实时性 | policy chunk-forward p95 / selected `n_action_steps≤50 ms` |
| 冻结完整性 | `validate-frozen` 返回 `gate_pass: true` |

任一 gate 失败时停止，不看 test、不修改阈值、不混入 RL/失败数据、不直接进入 W0。若要改
模型架构、数据 schema、训练 budget、Flow sampler 或 ODE steps，必须新建 config id 并从训练
重新开始。

M2-W 通过只允许进入 W0；它不证明 future/world representation 有效，也不允许跳过 W0 直接
做 W1 的 horizon sweep。

## 当前验证边界

正式 50k 训练和 200-state development candidate sweep 已完成，产物位于
`outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000/` 与
`outputs/maniskill/m2_task_base/StackCube-v1/development/`。训练共 50,000 steps，global batch
为 256，tail loss 为 0.0685；最佳 candidate 是 step 50k、`n_action_steps=8`，development
`success_once=43/200`、`success_at_end=43/200`、on-cube 为 33.0%。同一 checkpoint 在训练集
抽样 probe 为 90/100，但 train/dev normalized action MSE 为 0.00778/0.10635，存在 13.7×
泛化 gap。

当前仍未完成两件事：`selected_checkpoint.json` 没有写入最终 selected candidate，且没有执行
state-bank test 或生成 `frozen_base`。因此 M2-W 可以报告为“正式训练 + development candidate
结果”，不能报告为已冻结 baseline 或已通过 W0。smoke 输出仍属于 `Smoke`，不能替代上述正式
结果。
