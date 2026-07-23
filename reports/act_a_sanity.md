# ACT A-Sanity 实验指南

这个实验用五条 LIBERO demonstration 检查 ACT 是否能完成最基本的小数据过拟合。

## 1. 实验描述

实验问题：

> 从同一个零更新 ACT checkpoint 出发，训练 20,000 updates 后，模型能否更准确地预测
> 五条训练轨迹的动作，并在对应 LIBERO 任务中获得更高的闭环成功率？

固定配置如下：

| 项目 | 配置 |
|---|---|
| LIBERO task | `libero_object` task 8 |
| Language | `pick up the chocolate pudding and place it in the basket` |
| LeRobot dataset | `lerobot/libero` |
| Dataset root | `datasets/lerobot/lerobot/libero` |
| Dataset task index | 29 |
| Episodes | 823, 826, 834, 840, 841 |
| 数据量 | 5 episodes，859 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action | 7D relative EEF action |
| Control frequency | 10 Hz |
| ACT | `chunk_size=20`，`n_action_steps=5`，`use_vae=false` |
| Training | batch size 8，seed 1000，20,000 updates |

LeRobot 负责数据格式、ACT 训练和评测工具；LIBERO 提供机器人任务、仿真环境和任务成功判定。

## 2. 实验思路

这是一个配对的训练前后对比：

1. 创建零次 optimizer update 的 B0 checkpoint。
2. 评测 B0。
3. 从这个确切的 B0 checkpoint 训练 20,000 updates。
4. 使用相同参数评测 trained checkpoint。
5. 比较 B0 和 trained 的结果。

B0 不是“全随机模型”：视觉骨干使用 ImageNet 初始化，但尚未在这五条轨迹上进行 optimizer
update。这样可以把训练前后的差异归因于本次 20,000-step 训练。

两种评测回答不同问题：

| 评测 | 做什么 | 看什么 |
|---|---|---|
| Offline | 在五条 demonstration 的全部 859 帧上预测动作，不执行动作 | `first_action_mae`、`delta_action_mae` |
| Closed-loop | 在 LIBERO 中真正执行策略，20 episodes，seed 1000 | `success_rate` |

期望结果：

- `loss@20000 <= 0.2 × loss@100`；
- trained `first_action_mae` 至少比 B0 降低 20%；
- trained `delta_action_mae` 不高于 B0；
- trained closed-loop `success_rate` 高于 B0。

Offline action error 下降不代表任务一定成功；任务成功率必须通过 LIBERO closed-loop rollout
判断。

## 3. 具体操作步骤

以下命令从 repo 根目录执行。

### 3.1 准备环境和输出目录

先激活环境并进入 Bash：

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate robot-practice
export PYTHONNOUSERSITE=1
bash
```

在新的 Bash prompt 中执行：

```bash
set -euo pipefail

experiment_id="$(date +%Y%m%d_%H%M%S)"
dataset_root="$PWD/datasets/lerobot/lerobot/libero"
log_dir="$PWD/outputs/act/a_sanity/logs/${experiment_id}"
eval_root="$PWD/outputs/act/a_sanity/eval/${experiment_id}"

mkdir -p -- "${log_dir}"
```

四个评测目录由 evaluator 创建，因此执行前不要手动创建
`${eval_root}/b0_offline`、`b0_closed`、`trained_offline` 或 `trained_closed`。
若某次评测中断后需要用同一个目录重跑，在原命令末尾添加 `--overwrite`。

### 3.2 创建 B0 checkpoint

```bash
b0_log="${log_dir}/b0_create.log"
scripts/experiments/act/a_sanity.sh --steps=0 2>&1 | tee "${b0_log}"
b0_run="$(sed -n 's/^Output directory: //p' "${b0_log}" | tail -n 1)"
b0_policy="${b0_run}/checkpoints/000000/pretrained_model"
test -d "${b0_policy}"
echo "B0 policy: ${b0_policy}"
```

### 3.3 评测 B0：offline

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${b0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root="${dataset_root}" \
  --agentview-image-convention=lerobot_180 \
  --episodes 823 826 834 840 841 \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir="${eval_root}/b0_offline"
```

### 3.4 评测 B0：LIBERO closed-loop

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${b0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir="${eval_root}/b0_closed"
```

不要添加 `--max-steps`，否则只会得到 smoke evaluation，不能计算正式成功率。

Closed-loop 的 `rollout.rrd` 不加载 demonstration，也不伪造逐帧 GT。每个
`global_step` 同时展示：

- 模型在 `t` 时刻接收的 agentview 与 eye-in-hand；
- 基于模拟器真实 7D arm joints 绘制的 Panda 和真实 EEF 历史；
- 实际发送的 action、模拟器产生的 `EEF[t] -> EEF[t+1]` 响应；
- ACT 之后会执行的 queue，以及模型预测但会丢弃的 chunk 尾部；
- `env.step(action[t])` 返回的 post-action agentview；
- 可点击选择的 7D action、8D simulator state、reward/success、latency 和 metrics。

这里的未来 action 线是控制命令预览，不是模拟出的未来机器人状态。Temporal Ensemble
模式显示的是当前新预测 chunk，不把它标成执行 queue。

在 `Free 3D` 标签中，左侧默认展开的 Blueprint Panel 提供眼睛开关，可分别隐藏或显示
坐标系、相机标定、机器人、实测 EEF 历史、当前命令、真实响应和各类 plan。细橙线是发送给
控制器的当前命令；较粗青色箭头及其青色终点是模拟器实测的
`EEF[t] -> EEF[t+1]` 响应。

### 3.5 从 B0 训练 20,000 updates

```bash
train_log="${log_dir}/train_20000.log"
scripts/experiments/act/a_sanity.sh \
  --steps=20000 \
  --policy-path="${b0_policy}" \
  2>&1 | tee "${train_log}"
trained_run="$(sed -n 's/^Output directory: //p' "${train_log}" | tail -n 1)"
trained_policy="${trained_run}/checkpoints/020000/pretrained_model"
test -d "${trained_policy}"
echo "Trained policy: ${trained_policy}"
```

### 3.6 评测 trained checkpoint：offline

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${trained_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root="${dataset_root}" \
  --agentview-image-convention=lerobot_180 \
  --episodes 823 826 834 840 841 \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir="${eval_root}/trained_offline"
```

### 3.7 评测 trained checkpoint：LIBERO closed-loop

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${trained_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir="${eval_root}/trained_closed"
```

### 3.8 查看结果

训练 loss：

```bash
rg 'step:(100|20K).*loss:' "${train_log}"
```

四份评测指标：

```bash
for result in b0_offline b0_closed trained_offline trained_closed; do
  echo "=== ${result} ==="
  python -c \
    'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["aggregate"], indent=2))' \
    "${eval_root}/${result}/metrics.json"
done
```

重点比较：

- `b0_offline` 与 `trained_offline` 的 `first_action_mae`、`delta_action_mae`；
- `b0_closed` 与 `trained_closed` 的 `success_rate`；
- `train_20000.log` 中 step 100 和 step 20K 的 loss。

按需打开可视化：

```bash
rerun "${eval_root}/b0_offline/prediction.rrd"
rerun "${eval_root}/trained_offline/prediction.rrd"
rerun "${eval_root}/b0_closed/rollout.rrd"
rerun "${eval_root}/trained_closed/rollout.rrd"
```

Offline Rerun 的 3D 坐标以机器人 base 为原点，标出 `X`、`Y Forward`、`Z Up`。
LIBERO 的 LeRobot/OpenVLA 数据约定会把渲染图像旋转 180 度，但 state 和 action 仍保持
机器人 base 坐标系，不做相同翻转。训练和闭环 processor 使用同一图像约定，因此已有 ACT
checkpoint 仍然有效，不能只把训练输入改回物理相机方向；若要更换约定，必须同步重建数据、
rollout 预处理、normalization statistics 并重新训练。

左侧两路 recorded images 保留模型所见内容和方向；显示时只是从 CHW/float 转成 HWC/uint8，
不会改 policy tensor。右上角默认打开 `Physically calibrated agentview`：它从当前 task 的
MuJoCo 模型读取固定 `agentview` 外参和 FOV，并为 `lerobot_180` 仅在可视化副本上做一次水平
反射，使 3D 物理投影与图像一致。该修正不会进入 ACT inference、指标或保存的数据。第二个
标签 `Free 3D` 是可自由旋转的独立视角，青色视锥表示真实 agentview。由于数据没有完整关节
状态，eye-in-hand 相机不做不可靠的动态标定叠加。

深蓝连线是 recorded GT EEF 历史。每个 recorded EEF 上保留一对互不相连的短线：浅蓝是
GT 首动作平移指令，深橙是 teacher-forced、未执行的 Pred 首动作平移指令；不同帧之间不会
连成一条伪 Pred 轨迹。半透明橙线只表示当前帧的完整 Pred future chunk 指令预览。短线端点
都是 OSC controller position goal，不是真实下一帧位置，也不是仿真执行结果。蓝色 Panda
骨架由 recorded 8D EEF state 连续 IK 重建，不是数据集中实测的关节位置。下方 action chart
有 14 条可独立点选的曲线，每个维度的 GT 和 Pred 相邻显示且颜色不同。
为避免遮挡，3D 画面只常驻三个坐标轴标签。点击关节点、EEF 点或轨迹后，在 Rerun
的 Selection 面板中查看对应标签。

闭环视频位于：

```text
<eval_root>/b0_closed/videos/episode_000.mp4
<eval_root>/trained_closed/videos/episode_000.mp4
```

实验完成后，将 task、数据量、observation/action、horizon、loss、offline error、
closed-loop success rate、latency 和 failure cases 写入 `reports/experiment_log.md`。

## 4. 实际结果（2026-07-23）

20,000-step checkpoint 在五条训练 demonstration 的全部 859 帧上完成了 offline
evaluation，并在相同 LIBERO task 上完成了 20 次正式 closed-loop rollout。

| 指标 | 100-step 参考 | 20,000-step |
|---|---:|---:|
| First-action MAE | 0.16989 | 0.02076 |
| Delta-action MAE | 0.02365 | 0.00972 |
| Teacher-forced Action TV | 0.02039 | 0.02011 |
| Teacher-forced Action Jerk | 0.02991 | 0.01146 |
| Closed-loop success rate | 未评测 | 18/20（90%） |

相对 100-step checkpoint，20,000-step checkpoint 的 first-action MAE 降低约
87.8%，delta-action MAE 降低约 58.9%。正式 closed-loop evaluation 成功
18 次、失败 2 次。执行动作的 Action TV 为 0.02195，Action Jerk 为 0.02210；
model-query latency p50/p95 分别为 18.28/24.47 ms。

本次训练关闭了 W&B，且训练终端日志没有保留，因此无法补录完整 loss 曲线，也不能验证
`loss@20000 <= 0.2 × loss@100`。零更新 B0 checkpoint 没有保留下来，表中的
100-step 结果只能作为近似参考，不能写成严格的 B0 对照。两次 closed-loop 失败尚未做
逐帧失败类型标注。

在这些证据限制下，A-Sanity 可以判定为完成：20,000-step 模型在五条训练轨迹上的
open-loop error 明显下降，并在对应任务取得 90% closed-loop success rate。

结果文件：

```text
outputs/act/a_sanity/20260721_171753/offline_5_recordings/metrics.json
outputs/act/a_sanity/20260721_171753/offline_5_recordings/prediction.rrd
outputs/act/a_sanity/20260721_171753/eval/libero/libero_object/task_008/20260723_152040/metrics.json
```
