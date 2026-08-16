# M2 training-data rollout probe

This standalone probe samples episodes from the frozen M2 offline training split,
resets ManiSkill from each episode's exact `env_states[0]`, and runs the selected
M2 checkpoint closed loop for at most 500 steps. It is a debugging/visualization
probe, not a held-out evaluation: the sampled states and action trajectories come
from the training data.

The implementation lives in this directory so it does not touch the parallel M2
LeRobot/Diffusion-Policy integration work. It reads the existing M2 Flow checkpoint
loader and StackCube observation adapter, but does not modify those files.

The rollout now derives the Flow source-noise seed from the canonical M1
`state_fingerprint(env_states[0])`, rather than from the episode ID. This makes
the train probe use the same identity convention as the development state bank.

Outputs include:

- `rollout_summary.json`: sampled episode IDs and aggregate success metrics;
- `episodes.jsonl`: one result per sampled training episode;
- one `episode_*.json` file per sampled training episode.

Use a new output directory for every run.

To test the overfitting hypothesis without a closed-loop rollout, compare the
same checkpoint's normalized action reconstruction on the complete train and
development offline splits:

```bash
conda run --no-capture-output -n robot-practice \
  python tools/m2_train_data_rollout/compare_offline_mse.py \
  --checkpoint=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000/checkpoint_step_00050000.pt \
  --output=outputs/maniskill/m2_task_base/StackCube-v1/diagnostics/train_vs_development_mse.json \
  --device=cuda --batch-size=64 --num-workers=4
```

The result reports train/dev MSE, zero-action baselines, relative gains, and
the development-to-train MSE ratio. A large ratio is evidence for a train/dev
fit gap, not by itself proof of closed-loop overfitting.

For a larger closed-loop train-side sample using the exact existing M2
development evaluator (same reset, observation, action queue, termination and
metric code), run:

```bash
conda run --no-capture-output -n robot-practice \
  python -m tools.m2_train_data_rollout.closed_loop_same_evaluator \
  --checkpoint=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000/checkpoint_step_00050000.pt \
  --output-dir=outputs/maniskill/m2_task_base/StackCube-v1/diagnostics/train_closed_loop_100 \
  --sample-count=100 --seed=2026 --num-envs=64 --n-action-steps=8 --device=cuda
```

Its `same_evaluator/metrics.json` is directly comparable to the regular M2
development metrics; the outer `rollout_summary.json` adds the original train
episode IDs.
