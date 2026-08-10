# Robot VLA WAM Practice

## 当前任务与路线图

> 更新时间：2026-08-10。`进行中` 表示正在跑或正在补齐产物，`已完成` 表示已有可引用结果，
> `待开始` 表示计划已冻结但尚未执行。详细命令、前置条件和判断标准以各计划/指南为准。

### 当前优先任务

| 任务 | 状态 | 当前小节点 |
| --- | --- | --- |
| ManiSkill M0：PickCube 7D 主线 | 进行中 | 20k 基线已完成选模和 test；继续完成独立 50k 训练 → development 选模 → random / PPO / visual 三系统 test |
| ManiSkill PickCube 4D 论文参照基线 | 进行中 | motion-planning replay → 4D `pd_ee_delta_pos` 数据转换 → 8 卡 Diffusion 训练 → 官方协议 `official_best` 选模 → 固定 `holdout_once` |
| LIBERO C1 背景保留与回放诊断 | 进行中 | C1 背景回退 test；补齐 replay `k=2/8`、`k=6/8`；重跑修正选模后的 C3 winner background test |
| PushT RF7 条件遵循评测 | 进行中 | development checkpoint selection → 离线 condition gate → held-out smoke → 正式闭环评测 |

对应入口：[M0 指南](reports/guides/maniskill/00_m0_pickcube.md)、[4D 基线指南](reports/guides/maniskill/01_paper_aligned_pickcube_4d.md)、[LIBERO 评测指南](reports/guides/smolvla_libero/06_evaluation.md)、[RF7 计划](plans/2026-08-03-RF7正式训练与条件遵循评测.md)。

### ManiSkill 阶段零：M0 → M1 → M2 → M3

这是 Mini-WAM 和 Residual RL 之前的环境与数据闸门：

- **M0（进行中）**：验证官方演示 replay、RGB LeRobotDataset、Diffusion 训练和 ManiSkill 闭环评测整条管路；冻结基线和 train / development / test 划分。
- **M1（待开始）**：审计不同任务的 action-chunk 候选空间，测候选多样性、`first-sample`、`Random@K` 和一次性 `Oracle@K` 上限。
- **M2（待开始）**：根据候选空间是否有足够差异、`Oracle@K` 增益和任务稳定性，选定 Mini-WAM 的主任务。
- **M3（待开始）**：用冻结 checkpoint 采集成功/失败 rollout，记录 simulator 原生阶段信号，形成 Mini-WAM 与 Residual RL 的训练数据。

详情：[Mini-WAM/Residual RL 总计划](plans/2026-08-06-Mini-WAM与残差RL三阶段实验计划.md) · [M0 计划](plans/2026-08-06-M0接通ManiSkill端到端探针.md)。

### 后续实验：Mini-WAM、Residual RL 与 LIBERO 组合

M0–M3 完成后，按以下闸门推进；每一阶段先做小规模 smoke，再决定是否进入下一阶段。

| 阶段 | 节点与要回答的问题 |
| --- | --- |
| Mini-WAM | **W0**：验证 action-conditioned 单步 latent / pose / reward 动力学真的学到动作影响；**W1**：验证 open-loop `H=1/4/8/16` 的多步预测和短候选排序；**W2**：对 K 个 chunk 做 WAM 选择，比较 first sample、Random@K、同模型 `H=0` 与 WAM@K；**W3**：用 ensemble、不确定性回退和最多两轮失败数据聚合抑制模型漏洞。 |
| Residual RL | **R0**：低维状态上的 residual RL 管路；**R1**：RGB / latent residual 与 state residual 对比；**R2**：在一个 LIBERO 任务上修正冻结 SmolVLA；**R3**：在相同真实交互预算下与 Online BC、Weighted BC 等比较，确认收益是否来自 RL。 |
| LIBERO 组合 | **F0**：将 WAM 迁移到语言条件任务；**F1**：比较 VLA、VLA+WAM、VLA+Residual、组合系统；**F2**：测 WAM 是否减少真实交互；**F3**：评估 canonical、held-out paraphrase、OOD、恢复场景和 background retention。 |

Outcome Model 已从路线中取消：它需要为每个候选执行完整终局续跑，成本高且不适合训练期使用；候选空间问题由 M1 的一次性上限审计覆盖，`H=0` 作为 W2 的消融对照。

### LIBERO C 系列当前状态

- **C1 Default Non-PEFT：已完成正式目标评测**；当前只继续做背景回退和背景数据回放诊断，不把 diagnostic checkpoint 当作正式 C1。
- **C2 Canonical LoRA：已完成**；canonical 目标能力与 C1 接近，但 held-out paraphrase 较弱。
- **C3 LoRA + Paraphrase：已完成正式目标评测**；修正选模后 winner 为 `015000`，held-out paraphrase 明显改善；winner 的 background retention 仍需有效重跑。
- **Prompt Swap：已修复并合并结果**；stove 的旧 `0/10` 不再引用，当前按首次达成目标统计。

详情：[C1 指南](reports/guides/smolvla_libero/03_c1_non_peft.md) · [C2 指南](reports/guides/smolvla_libero/04_c2_canonical_lora.md) · [C3 指南](reports/guides/smolvla_libero/05_c3_paraphrase_lora.md) · [C3 结论](reports/daily/2026-08-07-C3改写增强LoRA正式评测结论.md) · [遗忘诊断结论](reports/2026-08-06-C1遗忘诊断与背景数据回放结论.md)。

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
