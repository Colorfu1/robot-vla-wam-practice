# Robot VLA WAM Practice

## 当前工作总览

> 更新时间：2026-08-10。当前只保留三条主线；每条线单独说明实验编号、目的、状态和下一步。

### 1. LIBERO：主实验线已完成

这里的目标是验证 SmolVLA 在 LIBERO 目标任务上的适配能力、LoRA 效果和语言改写泛化。

| 编号 | 它是什么 | 为什么做 | 状态 / 结论 |
| --- | --- | --- | --- |
| B0 | 先用背景任务做 domain adaptation，再直接测试目标任务的基座模型 | 提供统一的 zero-shot 基线，作为后续 C1/C2/C3 的起点 | 已完成；已发布 selected B0 |
| C1 | 用目标任务的 canonical instruction 做全参数微调 | 测量最直接的目标任务适配效果 | 已完成训练、选模和正式评测 |
| C2 | 用同一批 canonical 数据做 LoRA 微调，冻结大部分基座参数 | 判断低秩适配能否达到 C1 的效果 | 已完成训练、选模和正式评测 |
| C3 | 在 LoRA 训练中加入目标任务的 paraphrase instruction | 测试对未见过的语言改写是否更鲁棒 | 已完成；修正选模后 winner 为 `015000`，held-out paraphrase 明显改善 |
| Prompt Swap | 评测工具，不是一个训练阶段；只替换指令，检查模型是否完成被要求的目标 | 检查语言条件是否真的被遵循 | 判定 bug 已修复；旧 stove `0/10` 已废弃 |

学习率缩放、背景回放和 background retention 是补充诊断，不改变 LIBERO 主实验已完成的结论。

入口：[LIBERO 评测指南](reports/guides/smolvla_libero/06_evaluation.md) · [C1](reports/guides/smolvla_libero/03_c1_non_peft.md) · [C2](reports/guides/smolvla_libero/04_c2_canonical_lora.md) · [C3](reports/guides/smolvla_libero/05_c3_paraphrase_lora.md) · [C3 结论](reports/daily/2026-08-07-C3改写增强LoRA正式评测结论.md)。

### 2. PushT：基本完成，降为非重点

| 模块 | 它是什么 | 状态 |
| --- | --- | --- |
| Diffusion baseline | PushT 上的基础 action-chunk policy 与评测管路 | 核心实验已完成 |
| RouteField / RF7 | 测试模型是否读取路线的顺序和方向条件 | 核心诊断已完成并形成结论 |
| 后续 RF7 收尾 | 可能还有正式闭环或下一版设计 TODO | 可选，不再占用当前主线资源 |

入口：[PushT 计划目录](plans/README.md) · [RF7 计划](plans/2026-08-03-RF7正式训练与条件遵循评测.md)。

### 3. ManiSkill：当前主线

#### 阶段零：M0–M3

| 节点 | 它是什么 | 当前状态 | 下一步 / 闸门 |
| --- | --- | --- | --- |
| M0 | 先把官方演示 replay、RGB 数据、LeRobot 训练和 ManiSkill 闭环全部接通 | 进行中；7D 20k 基线已完成选模和 test | 完成 50k 训练 → development 选模 → random / PPO / visual test |
| 4D 论文参照基线 | 用 motion-planning、4D `pd_ee_delta_pos`、CPU backend 和论文评测语义跑一条可比基线 | 并行进行中 | 8 卡训练 → `official_best` 选模 → `holdout_once`；不替换 M0 7D 主线 |
| M1 | 审计同一状态下 action chunk 候选是否足够不同 | 待开始 | 测候选多样性、`first-sample`、`Random@K`、一次性 `Oracle@K` |
| M2 | 从候选空间和任务稳定性中选出 Mini-WAM 主任务 | 待开始 | `Oracle@K` 增益达到预注册闸门后冻结任务 |
| M3 | 用冻结策略采集成功/失败轨迹和 simulator 原生阶段信号 | 待开始 | 生成 WAM / Residual RL 的训练数据 |

入口：[M0 计划](plans/2026-08-06-M0接通ManiSkill端到端探针.md) · [M0 指南](reports/guides/maniskill/00_m0_pickcube.md) · [4D 基线指南](reports/guides/maniskill/01_paper_aligned_pickcube_4d.md)。

#### M3 之后：Mini-WAM、Residual RL 和组合验证

| 路线 | 每个节点做什么 | 状态 |
| --- | --- | --- |
| Mini-WAM | **W0** 学单步 action-conditioned 动力学；**W1** 验证多步 open-loop 预测与候选排序；**W2** 比较 WAM@K、first sample、Random@K、`H=0`；**W3** 加入 ensemble、不确定性回退和失败数据聚合 | 待开始 |
| Residual RL | **R0** 低维 residual；**R1** 视觉 residual；**R2** 单个 LIBERO 任务修正冻结策略；**R3** 与 Online BC 等在相同真实交互预算下比较 | 待开始 |
| 组合验证 | **F0** WAM 迁移；**F1** WAM 与 residual 的互补性；**F2** 真实交互节省；**F3** 泛化、OOD、恢复和能力保留 | 待开始 |

Outcome Model 已取消：它需要为每个候选执行完整终局续跑，成本高且不适合训练期使用；候选空间由 M1 审计，`H=0` 由 W2 覆盖。

详情：[Mini-WAM/Residual RL 总计划](plans/2026-08-06-Mini-WAM与残差RL三阶段实验计划.md)。

LeRobot-first workspace for learning robot manipulation, VLA policy workflows, and
simulation benchmarks without modifying upstream LeRobot.

The main workflow uses LeRobot as the unifying layer for datasets, policies,
training, evaluation, hardware interfaces, VLA policies, and simulation benchmark
experiments. ManiSkill is the first simulation benchmark target. LIBERO is reserved
for language-conditioned manipulation and VLA benchmark work.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate robot-practice
```

Equivalent manual setup:

```bash
conda create -n robot-practice python=3.10 -y
conda activate robot-practice
pip install -r requirements.txt
pip install -e .
```

The environment sets `PYTHONNOUSERSITE=1` so packages from `~/.local` do not leak
into this project.

## Verify

Run the local doctor script:

```bash
./scripts/doctor.sh
```

Run LeRobot discovery commands:

```bash
lerobot-info
./scripts/quickstart_info.sh
```

## Visualize an ALOHA Episode

Open the synchronized Rerun viewer for the cached ALOHA cabinet dataset:

```bash
conda activate robot-practice
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0
```

The viewer keeps all camera frames, named joint position/action/effort/velocity/error
curves, a nominal bimanual 3D skeleton, and both end-effector paths on one frame timeline.
Use Space to play or pause, Left/Right to move by one frame, or drag the timeline.

For a quick signals-and-3D check without decoding video:

```bash
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0 \
  --no-images \
  --max-frames 100
```

Add `--save outputs/aloha_episode_0.rrd` to write a recording without opening a viewer.
The 3D skeleton uses nominal ViperX 300S geometry; it is not calibrated or registered to
the RGB cameras.

## Run the ACT A-Sanity Experiment

Create the training-before baseline without performing an optimizer update:

```bash
scripts/experiments/act/a_sanity.sh --steps=0
```

The resulting `checkpoints/000000/pretrained_model` uses the ImageNet-pretrained
ResNet18 backbone and the normal seeded random initialization for the remaining ACT
layers. To make a strict before/after comparison, evaluate that checkpoint first and
then explicitly use the same checkpoint to initialize training:

```bash
scripts/experiments/act/a_sanity.sh \
  --steps=20000 \
  --policy-path=outputs/act/a_sanity/<baseline_timestamp>/checkpoints/000000/pretrained_model
```

Each invocation is independent and writes a new timestamped directory. Omitting
`--policy-path` starts a separate fresh model, while omitting `--steps` preserves the
20,000-step default. Use `--print-command` to inspect a resolved command without
creating output.

## Evaluate an ACT Checkpoint on One LIBERO Task

Both evaluators require the complete LeRobot `pretrained_model` directory, not a standalone
`.pth` or `model.safetensors` file. For example:

```bash
policy_path="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
```

Run teacher-forced prediction on every available demonstration for one task:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${policy_path}" \
  --suite=libero_object \
  --task-id=8 \
  --agentview-image-convention=lerobot_180
```

This saves machine-readable `metrics.json`, a compact `metrics.md` summary,
`predictions.parquet`, and `prediction.rrd`, then opens Rerun.
The original camera panes remain unchanged in the model-facing LeRobot/OpenVLA orientation.
The upper-right pane defaults to a physically calibrated agentview overlay built from the
selected task's fixed MuJoCo camera pose and intrinsics. For the `lerobot_180` convention,
that visualization-only image is horizontally reflected so physical projection agrees;
its second tab is a free 3D orbit view whose cyan frustum marks that real camera. The
eye-in-hand image remains image-only because exact moving-camera reconstruction would need
joint states absent from this dataset. The 3D frame puts the robot base at the origin,
labels `X`, `Y Forward`, and `Z Up`, and
shows the recorded EEF history in deep blue. At every recorded EEF, a light-blue segment
shows the GT first-action translation command and a deep-orange segment shows the
teacher-forced Pred first-action command; pairs from different frames are retained but never
connected. A translucent orange line previews only the current Pred future chunk. These are
OSC command goals, not measured next poses or executed trajectories. The blue Panda skeleton
is reconstructed from the recorded 8D end-effector state with continuous IK; it is not
recorded joint-state ground truth. The action chart contains 14 individually toggleable
series, with each dimension's GT and Pred curves adjacent in the legend and assigned distinct
colors.
Only the three coordinate-axis labels remain permanently visible in 3D. Select a joint,
end-effector point, or trajectory to inspect its retained label in Rerun's Selection panel.
The evaluator first reuses a matching local dataset. If the task is missing, it downloads
only that task's upstream HDF5, replays it into a 256x256 LeRobot dataset, and retains both
caches. Use `--episodes 823 826` to select dataset-native episode IDs, `--batch-size N` to
change model-call batching, `--num-visualized-episodes N` to control Rerun volume, or
`--no-open-viewer` for a headless run. `--max-frames N` marks the output as smoke-only.
When intentionally reusing an explicit result path, pass `--overwrite`; without it an
existing directory is still protected. Active runs and unrelated directories are refused.

Run simulator-grounded evaluation for the same task:

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${policy_path}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none
```

This additionally writes simulator success/reward data, executed-action TV and jerk,
policy/model-query latency, `rollouts.parquet`, `eval_info.json`, Rerun, and MP4. The
closed-loop Rerun aligns the policy inputs at `t` with the executed command and post-action
agentview at `t+1`. It shows the Panda from measured simulator joints, measured EEF history
and response, the remaining executable ACT queue, the discarded chunk tail (or Temporal
Ensemble's new prediction chunk), selectable 7D action/8D state curves, and metrics. It does
not present a demonstration as frame-aligned closed-loop GT. Only the simulator run reports
task success; offline prediction cannot establish success because it never executes an
action. In the Free 3D tab, use the eye icons in the expanded Blueprint Panel to show or
hide the coordinate frame, camera calibration, robot, measured trajectory, current command,
actual response, executable queue, discarded tail, and Temporal Ensemble chunk
independently. The thin orange segment is the controller command; the thicker cyan arrow
and cyan endpoint are the measured EEF response at `t+1`.

Both commands require `--suite` and `--task-id`. Supported suites are `libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`, and `libero_90`; one invocation evaluates one
zero-based task ID. A checkpoint/task mismatch is warned because ACT has no language input,
while incompatible 8D-state/7D-action/two-camera checkpoint structure fails immediately.

To reopen a saved recording later:

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun \
  outputs/act/a_sanity/<train_timestamp>/eval/offline/libero_object/task_008/<eval_timestamp>/prediction.rrd
conda run --no-capture-output -n robot-practice \
  robot-rerun \
  outputs/act/a_sanity/<train_timestamp>/eval/libero/libero_object/task_008/<eval_timestamp>/rollout.rrd
```

If someone else checks out this repo, a one-shot visualizer setup is:

```bash
cd /path/to/robot-practice
./scripts/install_rerun_desktop_package.sh 0.1.1
```

That installs:
- `rr_of` (Offline icon, O)
- `rr_li` (LIBERO icon, L)
- `rerun` (fallback icon, R)

`prediction.rrd` will prefer `rr_of`, `rollout.rrd` will prefer `rr_li`.

## Workspace Shape

```text
envs/
  libero/
  maniskill/
datasets/
  raw/
  lerobot/
policies/
  bc/
  act/
  diffusion_policy/
  smolvla_or_pi0/
  wam_tiny/
eval/
  rollout.py
  metrics.py
  perturbations.py
plans/
  README.md
  TEMPLATE.md
reports/
  README.md
  experiment_log.md
  metrics.md
  daily/
  guides/
    act/
    pusht_diffusion/
```

Keep reusable local Python helpers under `src/robot_practice/`.
Work that still needs to be executed goes in `plans/`; see
[`plans/README.md`](plans/README.md).

## Benchmark Roles

- ManiSkill: simulation training and fast closed-loop evaluation.
- LIBERO: language-conditioned manipulation and VLA benchmark suites.
- LeRobot: dataset format, policy/training/evaluation workflow, and future hardware
  integration path.

## Experiment Log

Record every experiment in `reports/experiment_log.md` from the beginning. Each
entry should include:

- task suite
- dataset size
- observation keys
- action space
- action horizon
- control frequency
- train loss
- open-loop action error
- closed-loop success rate
- latency
- failure cases

## License and Data Provenance

Project-owned code and documentation are licensed under the
[Apache License 2.0](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for upstream software, ALOHA geometry/calibration provenance, and dataset notices.

The datasets referenced by examples, including
[`lerobot/aloha_mobile_cabinet`](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)
and [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero), are distributed
separately and are not included in this repository. Users are responsible for following each
dataset card and license. Generated checkpoints, videos, Parquet files, Rerun recordings, and
evaluation outputs are also excluded from version control.

## Artifact Policy

Generated datasets belong under `datasets/raw/` or `datasets/lerobot/`.
Training/evaluation outputs belong under `outputs/`. These paths are ignored by
git except for `.gitkeep` placeholders.
