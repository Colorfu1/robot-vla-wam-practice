# ManiSkill M0 PickCube 端到端探针指南

## 重要进展

- M0 已完成：RL 原生 7D/GPU 管路和 motionplanning 4D/CPU 论文参照管路均从演示走到了正式闭环结果。
- RL 7D 演示可完整 replay，评测状态与训练状态在 `1e-4` 容差下无 exact duplicate；其问题是数据结构与动作契约，不是 replay 管路损坏。
- 原始 RL 数据 73.0% 帧处于首次成功之后，且 gripper 与 `translation_z` 存在超出 `[-1, 1]` 的动作值；它不再作为后续 ManiSkill 项目的唯一主线。
- 4D motionplanning 参照在 CPU、horizon 100、1000 demonstrations 下达到固定 holdout `397/400`（99.25%），说明 LeRobot diffusion 管路已达到论文量级。
- PickCube 因 4D 基座已超过 95% 停止阈值，只保留作回归与论文参照；后续 WAM/RL 不应直接在它上面立项。

## 当前状态

- 状态：`done`。M0 的目标“确认 ManiSkill 官方演示 → RGB replay → LeRobot Dataset → diffusion → 原生闭环评测
  能稳定走通”已完成。
- 原始 7D RL/GPU 线的正式冻结结果是 20k checkpoint `020000`：development `120/200`（60.0%），
  test `234/400`（58.5%）。50k 的十个 checkpoint 随后都在同一 test block 上做了诊断 sweep，最高为
  `045000` 的 `340/400`（85.0%）；因此这块 test 已不再是未触碰的正式 holdout，不能用来再选 winner。
- 后续的独立论文参照线已完成：motionplanning → 4D `pd_ee_delta_pos` → CPU → horizon 100。其固定 CPU
  holdout 为 `397/400`（99.25%），与论文 Table III 的 1000-demo DP RGB PickCube 数量级一致。
- 所属路线：阶段零第一项 M0，来自[Mini-WAM 与残差 RL 三阶段实验计划](../../../plans/2026-08-06-Mini-WAM与残差RL三阶段实验计划.md)。
- 对应子计划：[2026-08-06 M0 接通 ManiSkill 端到端探针](../../../plans/2026-08-06-M0接通ManiSkill端到端探针.md)。
- 收尾结论见[2026-08-10 ManiSkill M0 收尾与 PickCube 论文参照](../../daily/2026-08-10-ManiSkill-M0收尾与PickCube论文参照.md)。
- 所有 replay、reset-only 指纹采集、训练和闭环命令都在远端执行；本地只做静态测试和离线 HDF5 审计。

## 已完成过程、结果与结论

### RL 原生 7D 线（Offline / Smoke / Formal）

这条线的目的已经完成：验证原生 7D `pd_ee_delta_pose`、GPU replay、21D student state、LeRobot
diffusion 和自写 ManiSkill 闭环 evaluator 的兼容性。它是有效的诊断基线，但不再被解释为后续 WAM/RL
唯一应继承的数据方案。

| 阶段 | 已验证结果 | 结论 |
| --- | --- | --- |
| Offline 原始审计 | 1013 条、50,360 帧；首次成功 median 14 steps；成功后尾帧 36,772 条，占 73.0% | 大部分训练帧是完成任务后的保持动作，不是到达目标的控制段。 |
| Reset-only 隔离审计 | 600 个 development/test 状态与 1013 个训练首帧在 `1e-4` 下 0 条 exact duplicate | RL 的 `episode_seed` 不能单独复原状态；以实际首帧指纹做隔离是正确口径。 |
| Replay | 1013/1013 写入 replay，1008/1008 原生成功 episode 写入训练集 | GPU 原生 replay 管路通过。 |
| Smoke | Full 为 0/50，Trim 为 1/50；均为 2k-step 欠训 smoke | 两者只差 2 个百分点，按预注册规则正式数据采用 Full；此 smoke 不作性能结论。 |
| Formal 20k | development 选中 `020000` 为 120/200（60.0%）；独立 test 为 visual 234/400（58.5%）、random 0/400、privileged PPO 396/400（99.0%） | 视觉学生显著高于随机，M0 的最小闭环基线门槛通过。 |

50k 训练的 test sweep 是额外诊断，不是一次新的正式选模。十个 checkpoint 在同一 400 条状态上的
`success_once` 为 `0.5%、2.5%、14.0%、43.0%、35.8%、62.5%、71.0%、75.0%、85.0%、84.8%`
（step `5k` 至 `50k`），峰值为 `045000` 的 85.0%。它说明 7D 线并非学不会，但因为 test 被反复查看，
该 block 从此只能叫“诊断曲线”；任何需要报告的 7D 最终数字必须使用新的、未接触状态块。

#### RL 数据的限制，而非“数据无效”

| 现象 | 已测证据 | 后续含义 |
| --- | --- | --- |
| 成功后尾帧占比高 | 73.0%（36,772/50,360）；真正首次成功前只有 13,588 帧 | Full 数据将大量权重给“已经完成后静止/保持”的动作；之后若复用此类数据，必须重新审计 tail 处理。 |
| 动作契约不干净 | 相对目标控制的 `[-1,1]` 范围，gripper 有 98.8%、`translation_z` 有 11.07% 的动作超界 | 不把它当作 simulator bug；但任何后续 7D 基座都必须明确动作单位、归一化和限幅，不能默认与 4D 数据可比。 |
| 教师与学生信息不同 | PPO actor 使用 42D state；学生仅用 RGB + 21D `qpos+qvel+goal_pos`，不含物体位姿/相对向量 | PPO 仅是特权参照，不能把与视觉策略的差值直接叫作“纯感知代价”。 |
| seed 不是 episode 状态 ID | PPO 采样中存在部分重置；RL `episode_seed` 对批内非首环境未必参与过实际 reset | 训练/评测隔离继续使用状态指纹或明确的 CPU 单环境 seed 协议，不能只检查 seed 整数是否不重叠。 |
| 旋转并未失活 | replay action 的旋转/平移绝对值中位数比为 7.73，超过 10% 门槛 | 7D 本身仍可用于需要朝向的任务；PickCube 不是证明 7D 无意义的证据。 |

### motionplanning 4D 论文参照（Formal）

这条是 M0 完成后专门做的归因复现，不与上面的 GPU/7D/50-step 数字直接相减。固定口径是
motionplanning 演示、4D `pd_ee_delta_pos`、`physx_cpu`、horizon 100、1000 demonstrations 和
官方 `success_once` 语义；完整命令与协议见[4D 论文参照指南](01_paper_aligned_pickcube_4d.md)。

| 项目 | 实测结果 |
| --- | --- |
| Replay 与数据闸门 | 1000/1000 replay 成功；4D 动作越界 0%；成功前帧占比 93.14%；首次成功 p90 为 85 steps；全部闸门通过。 |
| 1000-demo 官方协议 | 选中 `022500`，`official_best=100/100`；五次重复均值 99.8%，范围 99.0%–100%。 |
| 1000-demo 固定 CPU holdout | `success_once=397/400`（99.25%），`success_at_end=313/400`（78.25%）。 |
| 100-demo 官方协议 / holdout | `official_best=73/100`；五次重复均值 57.8%（48%–67%）；固定 holdout `206/400`（51.5%）。 |
| 同一固定 holdout 参照 | random `0/400`；privileged PPO `300/400`（75.0%）。 |

`official_best` 与论文都是“无固定 reset seed、多个 checkpoint 取最高”的比较口径；固定 CPU holdout 才是
仓库内部的隔离评测。该协议成功后继续跑满 100 steps，所以 `success_at_end` 较低，不能与 7D 线的成功即停
指标直接比较。

### M0 最终结论与后续边界

1. **M0 完成且通过。** 两条独立数据/后端协议都验证了本仓库的 ManiSkill 数据、训练、选模、评测与可视化
   管路；4D 复现还排除了“LeRobot diffusion 实现根本错误”的解释。
2. **不能把 99.25% 对 85.0% 解读为 4D 胜过 7D。** 同时改变了专家来源、有效帧比例、控制模式、horizon、
   后端和评测终止语义；该对比只能说明数据与协议会主导 PickCube 结果。
3. **PickCube 不进入 WAM 的 M1 主实验。** 4D 基座已超过原计划的 95% 停止阈值，Oracle reranking 的理论
   上限太小。7D 85% 线可以作为 M1 harness smoke，但不能作为任务选型结论。
4. **后续 ManiSkill 工作应为每个候选任务建立独立协议。** 至少分开记录演示来源、action、backend、horizon、
   student observation、动作契约和 development/test 状态生成方式；StackCube 与 PegInsertionSide 的基座不能
   直接借用 PickCube checkpoint。

M0 到此不再追加 PickCube 训练或在既有 test 上调参。下一步是重定标后的任务预检、任务专属基座和候选空间
审计，而不是直接开始 WAM 或 Residual RL。

## 实验目标与固定口径

```text
RL 特权 PPO 演示 → ManiSkill replay 生成 RGB → Full/Trim LeRobotDataset
→ LeRobot Diffusion policy → ManiSkill 原生向量化闭环
```

M0 只验证 `PickCube-v1`，不做 WAM、Residual RL、多任务选型或 M1 候选审计。学生输入是 RGB 加本体感受
和目标坐标；PPO 演示者使用 42 维特权 state，两者不能作纯感知因果比较。

| 项目 | 固定口径 |
| --- | --- |
| 主数据 | `datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5` |
| replay | `-c pd_ee_delta_pose -o rgb -b physx_cuda`，不做控制模式转换 |
| action | 7D：平移 3 + 旋转 3 + 夹爪 1 |
| student state | `agent.qpos[0:9] + agent.qvel[0:9] + extra.goal_pos[0:3]`，21D |
| 排除字段 | `is_grasped`、`obj_pose`、`tcp_to_obj_pos`、`obj_to_goal_pos`、`tcp_pose` |
| 图像/频率 | `uint8` 128×128×3，20 Hz |
| policy horizon | `n_obs_steps=2`、`horizon=16`、`n_action_steps=8` |
| 主评测 | `physx_cuda`、50 steps |
| Trim | 首次 native success 后再保留 5 帧 |

## 固定状态划分

配置：[configs/experiments/maniskill/split.json](../../../configs/experiments/maniskill/split.json)。

| 集合 | 批首 seed | 一次 reset 条数 | 用途 |
| --- | ---: | ---: | --- |
| `train` | 0 | 2000 | M1 分支采样和在线 RL |
| `development` | 2000 | 200 | 选模、Full/Trim 对照、Smoke |
| `test` | 3000 | 400 | 7D 20k 的正式评测；50k checkpoint sweep 后降级为诊断 block |

一次 reset 传入一个 seed 列表；首元素是批首 seed，后面的数只是行标签。状态由批首 seed 和行号决定，
所以不能把 200/400 拆成更小 batch；显存不够时新增批首 seed 并记录成新的 block。状态重复判断容差固定为 `1e-4`。
其中 7D 的 `test` block 已在 50k checkpoint sweep 中被反复查看，后续若再做 7D 正式结论必须新建 block，
不能复用这一组状态。

## 远端前置检查

```bash
set -euo pipefail
repo_root="$(git rev-parse --show-toplevel)"
cd -- "${repo_root}"
conda run --no-capture-output -n robot-practice python -c "import mani_skill; print(mani_skill.__version__)"
conda run --no-capture-output -n robot-practice python -c "from robot_practice.experiments.maniskill.split import load_seed_sets; print(load_seed_sets('configs/experiments/maniskill/split.json'))"
```

版本必须是 `mani_skill 3.0.1`。

## 执行步骤

### 1. 检查 RL 演示元数据（`Offline`）

```bash
repo_root="$(git rev-parse --show-toplevel)"
cd -- "${repo_root}"
conda run --no-capture-output -n robot-practice python - <<'PY'
import json
from pathlib import Path
meta = Path("datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.json")
data = json.loads(meta.read_text(encoding="utf-8"))
env = data["env_info"]
kwargs = env.get("env_kwargs", {})
episodes = data["episodes"]
print("env_id:", env.get("env_id"))
print("control_mode:", kwargs.get("control_mode"))
print("obs_mode:", kwargs.get("obs_mode"))
print("sim_backend:", kwargs.get("sim_backend"))
print("source_type:", data.get("source_type"))
print("max_episode_steps:", env.get("max_episode_steps"))
steps = [item["elapsed_steps"] for item in episodes]
print("episodes:", len(episodes), "frames:", sum(steps), "mean:", sum(steps) / len(steps))
print("frame range:", min(steps), "..", max(steps))
print("episode_seed:", sum(item.get("episode_seed") is not None for item in episodes), "/", len(episodes))
print("episode_id range:", min(item["episode_id"] for item in episodes), "..", max(item["episode_id"] for item in episodes))
PY
```

预期是 1013 条、50,360 帧、原生 7D、最长 50 步、`physx_cuda`。RL 文件没有 RGB；`obs_mode=state` 是 PPO
教师录制配置。不要用 `episode_seed` 整数证明状态隔离，后面必须比实际指纹。

### 2. 原始成功尾帧审计（`Offline`）

只读 HDF5，不启动环境：

```bash
repo_root="$(git rev-parse --show-toplevel)"
cd -- "${repo_root}"
bash scripts/experiments/maniskill/audit_demos.sh \
  --source-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --output-dir=outputs/maniskill/m0_probe
```

产出 `raw_data_audit.json` 和 `success_tail_stats.json`，包括每条 episode 的首次/末次 success、成功持续性、
每个 action 维度的分位数、成功前后动作范数和按 `τ=t-t_first_success` 对齐的曲线。已知待复核量是成功后填充
帧约占 73%。源文件未记录 TCP pose 时，TCP 位移写 null，不能用 qpos 冒充。

### 3. 初始状态指纹审计（reset-only，远端 GPU）

这一步只 reset、不执行 action，并且和第 2 步分成独立进程：

```bash
repo_root="$(git rev-parse --show-toplevel)"
cd -- "${repo_root}"
bash scripts/experiments/maniskill/audit_demos.sh \
  --source-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --output-dir=outputs/maniskill/m0_probe \
  --split-file=configs/experiments/maniskill/split.json \
  --state-set=development --state-set=test --skip-raw
```

训练侧和评测侧都提取 `[Panda qpos(9), cube xyz(3), cube quaternion(4), goal xyz(3)]`。报告记录最近训练状态
的整体和分量距离，并在 `1e-4` 下报告 exact duplicate。产物是：

```text
outputs/maniskill/m0_probe/initial_state_fingerprints.parquet
outputs/maniskill/m0_probe/split_leakage_report.json
```

有重复时换新的批首 seed，整块重取，不要挑单条删除。

### 4. Replay 10 条试通（`Replay smoke`）

必须使用 `--use-first-env-state`，不能使用 `--use-env-states`，也不做控制模式转换：

```bash
repo_root="$(git rev-parse --show-toplevel)"
cd -- "${repo_root}"
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.replay_trajectory \
  --traj-path datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgb -b physx_cuda \
  --save-traj --allow-failure --max-retry 3 --num-envs 10 --count 10
```

10/10 必须通过，并确认 RGB 为 `(128,128,3)`；否则停止，不跑全量。

### 5. Replay 全量

试通通过后去掉 `--count 10`，`--num-envs` 从 10 逐步上调到显存可承受值：

```bash
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.replay_trajectory \
  --traj-path datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5 \
  --use-first-env-state -c pd_ee_delta_pose -o rgb -b physx_cuda \
  --save-traj --allow-failure --max-retry 3 --num-envs 256
```

确认实际输出文件名后设置：

```bash
replay=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5
source=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5
```

第 7 步会把 `replay_ok`（文件存在率）和 `replay_success`（native success 率）分别写入
`outputs/maniskill/m0_probe/replay_survival.json`，并把源/replay 两边 action 分位数写入
`action_channel_stats.json`。旋转三维绝对值中位数至少达到平移三维中位数的 10% 是固定通道门槛；不满足
不判 M0 失败，但在 W/R 前换需要朝向的任务。

### 5.1 Replay 可视化（本地 `Offline` app）

Replay 产物已经包含 RGB、21D policy state、action 和 native success，不需要再次启动 ManiSkill 或重放动作。
可用专用入口生成每条 episode 一个 `.rrd`，并可选生成原始相机 MP4：

```bash
bash scripts/experiments/maniskill/visualize_replay.sh \
  --replay-h5=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
  --output-dir=outputs/maniskill/m0_probe/visualization/replay_0_1 \
  --episode=0 --episode=1 \
  --include-privileged-overlay --write-video
```

默认只取第一个 episode；`--episode` 可重复传入。`--include-privileged-overlay` 会在 Rerun 的 3D 诊断面板
显示 replay 中的 cube、goal 和 TCP 轨迹，但这些 overlay 不属于相机画面，也不能当作策略输入。相机面板仍然
只显示真实 RGB，因此相机里看不到的目标不会被伪装成可见。产物为：

```text
outputs/maniskill/m0_probe/visualization/replay_0_1/episode_0000_<success|failure>.rrd
outputs/maniskill/m0_probe/visualization/replay_0_1/episode_0000_<success|failure>.mp4
outputs/maniskill/m0_probe/visualization/replay_0_1/visualization_manifest.json
```

文件名中的 `success` / `failure` 使用 `success_at_end` 判定：最后一个动作之后仍为成功才标记为
`success`，中途曾成功但最后没有保持成功仍标记为 `failure`。`.rrd` 内部的 recording name 和
manifest 的 `result` 字段也使用同一口径。3D 诊断面板还会显示蓝色 Panda 骨架；它根据记录的
`extra.tcp_pose` 做数值 IK，并以记录的 7 个 arm qpos 作为初值，只用于可视化，不是策略输入。

ManiSkill 的 `.rrd` 使用 `episode/...` 路径，而本机两个 app 可能已经保存了
LIBERO/ACT 的持久化布局。若直接复用那个布局，Rerun 会打开窗口但所有面板为空。
首次查看 ManiSkill recording 时使用一次性干净状态，让 Rerun 采用 recording 内嵌的
Blueprint：

```bash
rerun_state="$(mktemp -d)"
mkdir -p "$rerun_state/data" "$rerun_state/config"
XDG_DATA_HOME="$rerun_state/data" \
  XDG_CONFIG_HOME="$rerun_state/config" \
  conda run --no-capture-output -n robot-practice \
  rerun --port=9881 \
  outputs/maniskill/m0_probe/visualization/replay_0_1/episode_0000_success.rrd
```

闭环 recording 也使用同样的方法；把上面的路径替换为
`outputs/maniskill/m0_probe/smoke/eval_trim_visual/visualizations/episode_0021_success.rrd`。
如果该条闭环没有在最后一步成功，则对应文件名会是 `episode_0021_failure.rrd`。
`--port=9881` 必须是空闲端口。`mp4` 只是原始相机 RGB，不包含 3D overlay。

### 6. 官方转换兼容性 smoke

官方工具只给 9D qpos state 和 `mp4v`，单独存放，只验 `lerobot-train` 能否读取：

```bash
replay=datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.convert_to_lerobot \
  --traj-path "${replay}" \
  --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_official_smoke \
  --fps=20 --image-size=128x128 --task-name=PickCube-v1 --robot-type=panda

conda run --no-capture-output -n robot-practice lerobot-train \
  --dataset.repo_id=local/maniskill_pickcube_rl_7d_official \
  --dataset.root=datasets/lerobot/maniskill/pickcube_rl_7d_official_smoke \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.n_obs_steps=2 \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --output_dir=outputs/maniskill/m0_probe/official_smoke_train \
  --job_name=m0_pickcube_official_smoke --seed=1000 --batch_size=64 --num_workers=4 \
  --steps=2000 --save_freq=1000 --log_freq=200 --eval_freq=0 --wandb.enable=false
```

### 7. 自定义转换：Full 与 Trim

先用 20 条 smoke，确认后去掉 `--max-episodes=20` 做正式转换。目录已存在时入口拒绝覆盖：

```bash
set -euo pipefail
base=datasets/raw/maniskill_demos/PickCube-v1/rl
replay="${base}/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
source="${base}/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"

bash scripts/experiments/maniskill/convert_demos.sh \
  --replay-h5="${replay}" --source-h5="${source}" --task=PickCube-v1 --fps=20 \
  --max-episodes=20 --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_full_smoke \
  --repo-id=local/maniskill_pickcube_rl_7d_full_smoke \
  --survival-report=outputs/maniskill/m0_probe/replay_survival.json \
  --action-stats-output=outputs/maniskill/m0_probe/action_channel_stats.json

bash scripts/experiments/maniskill/convert_demos.sh \
  --replay-h5="${replay}" --source-h5="${source}" --task=PickCube-v1 --fps=20 \
  --max-episodes=20 --success-tail-frames=5 \
  --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_trim_smoke \
  --repo-id=local/maniskill_pickcube_rl_7d_trim_smoke \
  --survival-report=outputs/maniskill/m0_probe/replay_survival.json
```

smoke 通过后，用下面两条命令生成正式 Full/Trim 数据（正式目录不能已存在）：

```bash
base=datasets/raw/maniskill_demos/PickCube-v1/rl
replay="${base}/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5"
source="${base}/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"

bash scripts/experiments/maniskill/convert_demos.sh \
  --replay-h5="${replay}" --source-h5="${source}" --task=PickCube-v1 --fps=20 \
  --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --repo-id=local/maniskill_pickcube_rl_7d_full \
  --survival-report=outputs/maniskill/m0_probe/replay_survival.json \
  --action-stats-output=outputs/maniskill/m0_probe/action_channel_stats.json

bash scripts/experiments/maniskill/convert_demos.sh \
  --replay-h5="${replay}" --source-h5="${source}" --task=PickCube-v1 --fps=20 \
  --success-tail-frames=5 \
  --output-dir=datasets/lerobot/maniskill/pickcube_rl_7d_trim \
  --repo-id=local/maniskill_pickcube_rl_7d_trim \
  --survival-report=outputs/maniskill/m0_probe/replay_survival.json
```

正式目录固定为 `datasets/lerobot/maniskill/pickcube_rl_7d_full/` 和
`datasets/lerobot/maniskill/pickcube_rl_7d_trim/`。Full 保留全部 replay 帧；Trim 使用
`min(T_i, first_success_step_i + 5)`。两份只写入 replay 原生 success episode。

### 8. Full/Trim smoke 训练与前 50 条闭环

两份训练除数据集外全部相同，训练由用户在远端执行：

```bash
conda run --no-capture-output -n robot-practice lerobot-train \
  --dataset.repo_id=local/maniskill_pickcube_rl_7d_full \
  --dataset.root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.n_obs_steps=2 \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --output_dir=outputs/maniskill/m0_probe/smoke/train_full_retry \
  --job_name=m0_pickcube_full_smoke --seed=1000 --batch_size=64 --num_workers=4 \
  --steps=2000 --save_freq=1000 --log_freq=200 --eval_freq=0 --wandb.enable=false

conda run --no-capture-output -n robot-practice lerobot-train \
  --dataset.repo_id=local/maniskill_pickcube_rl_7d_trim \
  --dataset.root=datasets/lerobot/maniskill/pickcube_rl_7d_trim \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false --policy.n_obs_steps=2 \
  --policy.horizon=16 --policy.n_action_steps=8 \
  --output_dir=outputs/maniskill/m0_probe/smoke/train_trim_retry \
  --job_name=m0_pickcube_trim_smoke --seed=1000 --batch_size=64 --num_workers=4 \
  --steps=2000 --save_freq=1000 --log_freq=200 --eval_freq=0 --wandb.enable=false
```

development 前 50 条也必须一次 reset 50 个环境：

```bash
bash scripts/experiments/maniskill/eval_closed_loop.sh \
  --system=visual --policy-path=outputs/maniskill/m0_probe/smoke/train_full_retry/checkpoints/last/pretrained_model \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --max-episodes=50 --num-envs=50 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_full \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --output-dir=outputs/maniskill/m0_probe/smoke/eval_full

bash scripts/experiments/maniskill/eval_closed_loop.sh \
  --system=visual --policy-path=outputs/maniskill/m0_probe/smoke/train_trim_retry/checkpoints/last/pretrained_model \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --max-episodes=50 --num-envs=50 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_trim \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_trim \
  --output-dir=outputs/maniskill/m0_probe/smoke/eval_trim
```

差异 `<5pp` 用 Full；Trim 明显更好用 Trim；只有差异
超过 20pp 才考虑 Balanced，M0 不扫描更多 k。

### 8.1 Smoke 闭环可视化（需要重新 rollout）

第 8 步不带额外参数时只保存指标。要看训练策略实际执行的 RGB 画面，必须在远端 rollout 时加上
`--visualize-episodes=50`；它会保留这 50 条 episode 的 action、21D state、native signals 和 RGB，
并在 rollout 结束后写出 MP4 与 `.rrd`：

```bash
bash scripts/experiments/maniskill/eval_closed_loop.sh \
  --system=visual --policy-path=outputs/maniskill/m0_probe/smoke/train_full_retry/checkpoints/last/pretrained_model \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --max-episodes=50 --num-envs=50 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_full \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --visualize-episodes=50 \
  --output-dir=outputs/maniskill/m0_probe/smoke/eval_full_visual

bash scripts/experiments/maniskill/eval_closed_loop.sh \
  --system=visual --policy-path=outputs/maniskill/m0_probe/smoke/train_trim_retry/checkpoints/last/pretrained_model \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --max-episodes=50 --num-envs=50 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_trim \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_trim \
  --visualize-episodes=50 \
  --output-dir=outputs/maniskill/m0_probe/smoke/eval_trim_visual
```

远端产物位于各自的 `visualizations/` 子目录：

```text
eval_trim_visual/visualizations/episode_0021_success.mp4
eval_trim_visual/visualizations/episode_0021_success.rrd
eval_trim_visual/visualizations/visualization_manifest.json
```

这里的 `success` / `failure` 是 `success_at_end` 的结果标签，不是 `success_once`：中途抓取或放置
成功、但在 horizon 结束时没有保持任务成功的 episode，仍然会写成 `_failure`。每条 RRD 的 3D
面板包含由 `tcp_pose` 数值 IK 重建的 Panda 节点、连杆线和夹爪线；它不会改变 rollout 指标。

把 `visualizations/` 下载到本地后，用一次性干净状态打开 `.rrd`：

```bash
rerun_state="$(mktemp -d)"
mkdir -p "$rerun_state/data" "$rerun_state/config"
XDG_DATA_HOME="$rerun_state/data" \
XDG_CONFIG_HOME="$rerun_state/config" \
conda run --no-capture-output -n robot-practice \
  rerun --port=9881 \
  outputs/maniskill/m0_probe/smoke/eval_trim_visual/visualizations/episode_0021_success.rrd
```

闭环 rollout 的 `.rrd` 仍带有 `robot_practice_act_libero` 身份，但 ManiSkill 的
`episode/...` schema 不应复用已经保存的 LIBERO 布局。`visualization_manifest.json`
中的 `result`、`success_once` 和 `success_at_end` 可用于挑选要看的成功或失败 episode。已有的 `eval_full` /
`eval_trim` 没有帧数据，不能事后补出视频，必须使用新的输出目录重跑。

### 9. 8 卡 50k 正式训练

> **归档说明。** 本节至第 11 节保留的是当时为 7D 线准备的可复现命令，不是 M0 的下一步命令。
> 实际运行中，50k 的十个 checkpoint 都在同一 `test` block 上做了诊断 sweep；因此不得再按这些命令
> 在该 block 上选模、重跑或宣称独立泛化。M0 已完成，后续任务需要新建独立状态块与任务协议。

Full/Trim smoke 的 development 闭环差异为 2 个百分点（小于预注册的 5pp），所以正式数据固定为
Full。20k 版本已经完成并得到 test=58.5%；为了判断是否只是训练不足，重新启动一个独立的 50k run。
使用仓库 launcher；它把 `--num-gpus=8` 转成 Accelerate 的 8 进程启动参数，不会把未知的
`--num-gpus` 传给 `lerobot-train`。只训练一个种子，选模只看 development：

```bash
bash scripts/experiments/maniskill/train_formal.sh \
  --num-gpus=8 \
  --dataset-variant=full \
  --output-dir=outputs/maniskill/m0_probe/train_50k \
  --job-name=m0_pickcube_50k \
  --batch-size=8 \
  --steps=50000 \
  --save-freq=5000 \
  --log-freq=200 \
  --seed=1000 \
  --num-workers=4
```

`--batch-size=8` 是每张卡的 batch，因此 8 卡全局 batch 为 64，与 smoke 保持一致；`--steps=50000`
是新的 optimizer step 预算，`--save-freq=5000` 会保存 `005000, 010000, ..., 050000` 十个候选。
只检查命令时运行：

```bash
bash scripts/experiments/maniskill/train_formal.sh --num-gpus=8 --print-command
```

### 10. 原定的 development 选模流程（归档）

这一节有两个目的，不能混成一个“看 loss 选模型”的步骤：

1. 在固定的 development 200 条状态上比较正式训练保存的候选 checkpoint，冻结一个视觉策略供 test 使用。
2. 在同一批状态上测 random 和 privileged PPO，给视觉策略的成功率提供下界和特权教师参照。

#### 10.1 候选 checkpoint rollout（远端）

50k 正式训练会保存 `005000,010000,015000,020000,025000,030000,035000,040000,045000,050000`。
先只运行这十个候选 checkpoint 的闭环 rollout 和 worker 合并：

```bash
bash scripts/experiments/maniskill/rollout_checkpoint_selection.sh \
  --run-dir=outputs/maniskill/m0_probe/train_50k \
  --checkpoint-steps=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000 \
  --num-gpus=8 \
  --selection-dir=outputs/maniskill/m0_probe/checkpoint_selection_50k \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_full \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --split-file=configs/experiments/maniskill/split.json \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --num-envs=200 --max-episodes=200 --max-steps=50
```

这个脚本只负责 rollout 和合并，不做 checkpoint 比较，也不发布 winner。它为每个候选 checkpoint
调用真正执行环境交互的 `scripts/experiments/maniskill/eval_closed_loop.sh`，再由
`src/robot_practice/experiments/maniskill/merge_closed_loop_workers.py` 生成每个候选的
`metrics.json`。如果某个候选的 aggregate `metrics.json` 已经存在，可加
`--reuse-completed` 跳过它；这个选项仍不会做排名。

`--num-gpus=8` 会为每张卡启动一个远端 worker。每个 worker 都保留完整的 200 条
reset batch，只对自己负责的 episode 做 policy forward，避免把 ManiSkill 的 batch seed
语义错误地切成独立标量 seed。worker 的详细 stdout/stderr 写入
`outputs/maniskill/m0_probe/checkpoint_selection_50k/logs/`，主终端只显示一条总进度条；汇总后的
`metrics.json` 才用于 checkpoint 排名。

#### 10.2 候选 checkpoint 比较、排名与发布（不 rollout）

rollout 全部完成后，再执行单独的 comparison script：

```bash
bash scripts/experiments/maniskill/select_checkpoint.sh \
  --run-dir=outputs/maniskill/m0_probe/train_50k \
  --checkpoint-steps=005000,010000,015000,020000,025000,030000,035000,040000,045000,050000 \
  --selection-dir=outputs/maniskill/m0_probe/checkpoint_selection_50k \
  --expected-episodes=200
```

这个脚本只读取上一步生成的 `metrics.json`，按 `success_at_end_rate`、
`success_once_rate`、较早 checkpoint 的规则比较，写出排名并把 winner 复制到
`checkpoint_selection_50k/selected/`。它不创建 ManiSkill 环境、不调用 rollout，也不需要 GPU。

固定排名规则是：先按 `success_at_end_rate` 降序；相同再按 `success_once_rate` 降序；仍相同选更早的
checkpoint。selector 会校验每份结果确实是 `visual / PickCube-v1 / pd_ee_delta_pose / development / 200`
且 `run_manifest.json` 指向对应 checkpoint。产物是：

```text
outputs/maniskill/m0_probe/checkpoint_selection_50k/evaluations/checkpoint_<step>/development/metrics.json
outputs/maniskill/m0_probe/checkpoint_selection_50k/checkpoint_results.json
outputs/maniskill/m0_probe/checkpoint_selection_50k/selected_checkpoint.json
outputs/maniskill/m0_probe/checkpoint_selection_50k/selected/pretrained_model
```

此前 20k 基线的 development 选模结果（固定 200 条状态，主判据为终态成功率）如下；这张表只作历史记录，
不替代 50k run 的重新选模：

| checkpoint | success_once | success_at_end | 首次成功数 | 终止情况 |
| ---: | ---: | ---: | ---: | --- |
| `010000` | 6/200（3.0%） | 6/200（3.0%） | 6 | success 6，truncated 194 |
| `020000` | 120/200（60.0%） | 120/200（60.0%） | 120 | success 120，truncated 80 |

因此此前 20k 基线冻结的是 `020000`；50k run 必须在十个 checkpoint 上重新排名并重新发布 winner。
最终泛化结论必须等 50k 冻结 checkpoint 后的 test 400 条评测。

如果候选 rollout 已经手动完成，只想生成排名，直接执行上面的 comparison command；它不会启动环境。
selector 会把 winner 复制成 `checkpoint_selection_50k/selected/` 下的自包含 checkpoint；后续评测
直接把这个副本传给 `--policy-path`，不再指向训练目录，也不再通过 shell 变量读取
`selected_checkpoint.json`：

```text
outputs/maniskill/m0_probe/checkpoint_selection_50k/selected/pretrained_model
```

#### 10.2.1 用已发布的 selected checkpoint 启动 visual rollout

`select_checkpoint.sh` 只比较并发布 checkpoint，不会启动选中策略的 rollout。要实际启动
选中策略，和 PPO 一样直接调用 `eval_closed_loop.sh`，把上面的实体副本传给
`--policy-path`：

```bash
bash scripts/experiments/maniskill/eval_closed_loop.sh --system=visual \
  --policy-path=outputs/maniskill/m0_probe/checkpoint_selection_50k/selected/pretrained_model \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_full \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --num-envs=200 --max-episodes=200 --max-steps=50 --num-gpus=8 \
  --policy-seed=2000 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval_50k/selected_visual/development_selected
```

这条命令是一个完整的 development 200 条评测；`--num-gpus=8` 由
`eval_closed_loop.py` 内部管理 worker、日志和结果合并，guide 不需要展开并发 shell 循环。
如只做 smoke，可将条数改为 50 并使用新的输出目录。

#### 10.3 同一 development 状态上的 random/PPO 参照

random 和 PPO 不参与视觉 checkpoint 排名，只用于解释结果；三者必须使用同一个 development 状态批：
本节的命令只生成 `eval/*/development/metrics.json`，不会生成 `eval/*/test/metrics.json`。
如果要做最终 test 400 条评测，跳到第 11 节，不要只把本节输出目录改名。

PPO 的 42D 输入必须严格复现 ManiSkill `obs_mode="state"` 的展平顺序：`qpos`、`qvel`、
`is_grasped`、`tcp_pose`、`goal_pos`、`obj_pose`、`tcp_to_obj_pos`、`obj_to_goal_pos`。
字段顺序修复前生成的 PPO metrics 无效；同步修复后只重跑 PPO 即可，random 和 selected visual
不受影响。

```bash
bash scripts/experiments/maniskill/eval_closed_loop.sh --system=random \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --num-envs=200 --max-episodes=200 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval/random/development

bash scripts/experiments/maniskill/eval_closed_loop.sh --system=ppo_privileged \
  --ppo-checkpoint=datasets/raw/maniskill_demos/PickCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=development \
  --num-envs=200 --max-episodes=200 --max-steps=50 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval/ppo_privileged/development
```

将 selector 选中的 visual metrics 与两个参照做离线比较：

```bash
conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.eval_closed_loop \
  --compare-metrics \
  outputs/maniskill/m0_probe/eval/random/development/metrics.json \
  outputs/maniskill/m0_probe/eval/ppo_privileged/development/metrics.json \
  outputs/maniskill/m0_probe/eval_50k/selected_visual/development_selected/metrics.json \
  --comparison-output=outputs/maniskill/m0_probe/eval_50k/development_selected_comparison.json
```

这里的 `--compare-metrics` 接受已经生成的 `metrics.json`；这里使用的是 50k run 复制后的
`selected/pretrained_model` 实际 rollout 结果，而不是历史 20k checkpoint 的旧结果。

PPO 是 privileged teacher reference，不是上限；差值只作描述，不做纯感知因果归因。

### 11. 原定的 50k selected checkpoint test 400 条评测（归档，不再执行）

原设计要求冻结 Full/Trim 和 checkpoint 后只执行一次下面三条命令：使用 `--seed-set=test`、
`--num-envs=400`、`--max-episodes=400`，并把输出目录写到各自的 `test`；development
结果不能改名充当 test 结果。实际 50k 工作改为在这组状态上查看全部 checkpoint，故这套命令现在仅用于
记录原定协议，**不能在既有 `split.json` 的 `test` block 上继续执行**：

```bash
bash scripts/experiments/maniskill/eval_closed_loop.sh --system=random \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=test \
  --num-envs=400 --max-episodes=400 --max-steps=50 --num-gpus=8 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval_50k/random/test

bash scripts/experiments/maniskill/eval_closed_loop.sh --system=ppo_privileged \
  --ppo-checkpoint=datasets/raw/maniskill_demos/PickCube-v1/rl/ppo_pd_ee_delta_pose_ckpt.pt \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=test \
  --num-envs=400 --max-episodes=400 --max-steps=50 --num-gpus=8 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval_50k/ppo_privileged/test

bash scripts/experiments/maniskill/eval_closed_loop.sh --system=visual \
  --policy-path=outputs/maniskill/m0_probe/checkpoint_selection_50k/selected/pretrained_model \
  --dataset-repo-id=local/maniskill_pickcube_rl_7d_full \
  --dataset-root=datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --task=PickCube-v1 --control-mode=pd_ee_delta_pose \
  --split-file=configs/experiments/maniskill/split.json --seed-set=test \
  --num-envs=400 --max-episodes=400 --max-steps=50 --num-gpus=8 \
  --policy-seed=2000 \
  --leakage-report=outputs/maniskill/m0_probe/split_leakage_report.json \
  --output-dir=outputs/maniskill/m0_probe/eval_50k/diffusion_visual/test

conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.eval_closed_loop \
  --compare-metrics \
  outputs/maniskill/m0_probe/eval_50k/random/test/metrics.json \
  outputs/maniskill/m0_probe/eval_50k/ppo_privileged/test/metrics.json \
  outputs/maniskill/m0_probe/eval_50k/diffusion_visual/test/metrics.json \
  --comparison-output=outputs/maniskill/m0_probe/eval_50k/comparison.json
```

## 判断标准

| 项目 | 固定标准 |
| --- | --- |
| replay smoke | 10/10 通过，否则停止 |
| 全量 `replay_ok` / `replay_success` | 各自 ≥95%，否则先查状态恢复和跨版本动力学 |
| 初始状态重复 | `1e-4` 下 0 条 exact duplicate，否则换整块首 seed |
| 旋转通道 | 旋转/平移绝对值中位数 ≥10%；不满足不判 M0 失败，但换任务 |
| 官方转换 | `lerobot-train` 能读取并完成 2000 steps |
| Full/Trim | 差异 <5pp 用 Full；Trim 明显更好用 Trim |
| visual development | 预注册成功口径达到 ≥50%；同时报告 `success_once` 和 `success_at_end` |
| visual ≥95% | 管路通但 PickCube 可能太简单，M1 前改 StackCube |
| test | 20k checkpoint 的一次 400 条结果有效；50k 诊断 sweep 后，同一 block 不再是未触碰 test |

管路失败与学不会分开：replay 不存活是管路问题；replay 正常但 visual 低于 50% 且 PPO 明显更高，是视觉
BC 基线问题，应检查 Full/Trim、归一化、图像预处理、时序、chunk 和 checkpoint。成功只认原生
`info["success"]`，不写自定义谓词，也不使用 LeRobot 自动 reset wrapper。

## motionplanning 4D 参照的定位

motionplanning 不再只是“RL replay 坏掉时的退路”：其独立的 4D/CPU/horizon-100 论文参照已完整执行，
并给出了 1000-demo 固定 holdout `397/400`（99.25%）。这证明 motionplanning 是可用的数据来源，也证明
旧 7D 分数不能被归因为整个实现错误。

但它仍不是与 7D/GPU/50-step 可直接相减的替代实验：控制模式、专家、后端、horizon、成功终止语义和数据
时间结构都不同。需要复现实验或查看完整协议时，使用独立的
[PickCube 4D 论文参照基线指南](01_paper_aligned_pickcube_4d.md)，不要把其结果混进 7D test 表。

## 产物路径与本地回归

主要产物：`outputs/maniskill/m0_probe/{raw_data_audit.json,success_tail_stats.json,initial_state_fingerprints.parquet,split_leakage_report.json,replay_survival.json,action_channel_stats.json}`；
数据集是 `datasets/lerobot/maniskill/pickcube_rl_7d_{full,trim}/`，评测在
`outputs/maniskill/m0_probe/eval/{random,ppo_privileged,diffusion_visual}/`。

以下只做静态/离线检查，不启动 ManiSkill，不 replay，不训练：

```bash
conda run --no-capture-output -n robot-practice python -m pytest \
  tests/test_maniskill_convert_demos.py tests/test_maniskill_eval_closed_loop.py \
  tests/test_maniskill_training_launcher.py -q
conda run --no-capture-output -n robot-practice ruff check \
  src/robot_practice/experiments/maniskill tests/test_maniskill_*.py
conda run --no-capture-output -n robot-practice ruff format --check \
  src/robot_practice/experiments/maniskill tests/test_maniskill_*.py
```
