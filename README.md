# Robot VLA WAM Practice

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

## Evaluate an ACT A-Sanity Checkpoint

Both evaluators require the complete LeRobot `pretrained_model` directory, not a standalone
`.pth` or `model.safetensors` file. For example:

```bash
policy_path="outputs/act/a_sanity/<train_timestamp>/checkpoints/020000/pretrained_model"
```

Run teacher-forced prediction on the five A-Sanity demonstration episodes:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${policy_path}"
```

This saves `metrics.json`, `predictions.parquet`, and `prediction.rrd`, then opens Rerun.
Use `--episodes 823 826` to select a same-task subset, `--batch-size N` to change model-call
batching, or `--no-open-viewer` for a headless run. `--max-frames N` is only for smoke tests
and marks the output as non-formal.

Run simulator-grounded evaluation on `libero_object` task 8:

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${policy_path}" \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none
```

This additionally writes simulator success/reward data, executed-action TV and jerk,
policy/model-query latency, `rollouts.parquet`, `eval_info.json`, Rerun, and MP4. Only the
simulator run reports task success. Offline prediction cannot establish success because it
never executes an action.

To reopen a saved recording later:

```bash
rerun outputs/act/a_sanity/<train_timestamp>/eval/offline/<eval_timestamp>/prediction.rrd
rerun outputs/act/a_sanity/<train_timestamp>/eval/libero/<eval_timestamp>/rollout.rrd
```

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
reports/
  experiment_log.md
```

Keep reusable local Python helpers under `src/robot_practice/`.

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
