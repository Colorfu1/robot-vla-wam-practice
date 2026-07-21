# ACT Evaluation Launcher Headers Design

## Goal

Make both ACT evaluation Bash entry points self-documenting when opened in an editor:

- `scripts/experiments/act/eval_offline.sh`;
- `scripts/experiments/act/eval_libero.sh`.

Each file header will explain the script's role, show copyable launch commands, enumerate
every accepted CLI argument with defaults and constraints, and summarize its outputs. The
change is comments only: the Bash wrappers, Python parsers, evaluation behavior, and output
formats remain unchanged.

## Common Header Rules

The shebang stays on line 1. The documentation comment block follows it and precedes
`set -euo pipefail`. Both launchers state that commands are run from the repository root
and that `--policy-path` must point to a complete LeRobot `pretrained_model` directory,
not an individual `.pth` or `.safetensors` file. The directory must contain a compatible
7D ACT config, model weights, preprocessor and postprocessor configs, and every processor
state file referenced by those configs.

Examples use one reusable variable:

```bash
# policy="outputs/act/a_sanity/<run_timestamp>/checkpoints/<step>/pretrained_model"
```

Both headers document the automatically provided `-h` / `--help` options. All other
arguments are forwarded unchanged by the Bash wrapper to its Python module.

## Offline Evaluation Header

### Purpose and launch examples

The header explains that offline evaluation performs teacher-forced prediction on local
`lerobot/libero` demonstration frames. It compares ACT action chunks with recorded target
actions and reports open-loop error, temporal smoothness, and model-query latency. It does
not execute actions in a simulator and therefore cannot report task success.

It shows a normal five-episode launch and a one-frame, headless smoke launch:

```bash
# scripts/experiments/act/eval_offline.sh --policy-path="${policy}"
# scripts/experiments/act/eval_offline.sh \
#   --policy-path="${policy}" --max-frames=1 --no-open-viewer
```

### Accepted arguments

The header lists the complete `offline_act.py` parser surface:

- `--policy-path PATH` is required and must meet the common complete 7D ACT checkpoint
  requirements;
- `--episodes INDEX [INDEX ...]` defaults to `823 826 834 840 841`; values must be
  non-negative, unique, present in the dataset, and all belong to dataset task index 29;
- `--dataset-root PATH` defaults to `datasets/lerobot/lerobot/libero` under the repository
  and must contain `meta/info.json`;
- `--batch-size N` defaults to 1 and must be positive;
- `--output-dir PATH` is optional. An explicit path is the result directory itself and must
  not already exist. When omitted, the policy path must follow the standard
  `<run>/checkpoints/<step>/pretrained_model` layout and output is created under
  `<run>/eval/offline/<YYYYMMDD_HHMMSS>`;
- `--max-frames N` is an optional positive global frame bound for smoke tests and marks the
  result as non-formal;
- `--no-open-viewer` saves the Rerun recording without opening the viewer;
- `-h` / `--help` prints parser help and exits.

### Outputs

The header lists `metrics.json`, `predictions.parquet`, `prediction.rrd`, and
`run_config.json`. The result begins with an `.incomplete` marker, removes it only after
atomic completion, and retains it when evaluation fails. Unless `--no-open-viewer` is
used, the evaluator attempts to open Rerun; a missing executable or launch failure emits a
warning but does not fail an otherwise completed evaluation.

## LIBERO Closed-Loop Header

### Purpose and launch examples

The header explains that this launcher executes ACT actions in the fixed LIBERO Object
task 8 simulator at 10 Hz, with an episode limit of 280 steps. It reports environment
success/reward, executed-action smoothness, gripper toggles, and latency. This is the
evaluation that determines whether the policy actually completes the task.

It shows a normal 20-episode launch and a one-episode, one-step, headless smoke launch:

```bash
# scripts/experiments/act/eval_libero.sh \
#   --policy-path="${policy}" --n-episodes=20 --num-visualized-episodes=1
# scripts/experiments/act/eval_libero.sh \
#   --policy-path="${policy}" --n-episodes=1 --num-visualized-episodes=1 \
#   --max-steps=1 --no-open-viewer
```

### Accepted arguments

The header lists the complete `libero_act.py` parser surface:

- `--policy-path PATH` is required and must meet the common complete 7D ACT checkpoint
  requirements;
- `--n-episodes N` defaults to 20 and must be positive;
- `--num-visualized-episodes N` defaults to 1 and must be between 0 and `--n-episodes`;
- `--perturbation NAME` defaults to `none`; `none` is currently the only implemented value;
- `--seed N` defaults to 1000 and must be non-negative. Episode `i` uses seed `N + i`;
- `--output-dir PATH` is optional. An explicit path is the result directory itself and must
  not already exist. When omitted, the policy path must follow the standard
  `<run>/checkpoints/<step>/pretrained_model` layout and output is created under
  `<run>/eval/libero/<YYYYMMDD_HHMMSS>`;
- `--max-steps N` is an optional per-episode bound from 1 through 280. Values below 280
  produce a smoke result rather than a formal success-rate evaluation;
- `--no-open-viewer` saves the Rerun recording without opening the viewer;
- `-h` / `--help` prints parser help and exits.

The header separately identifies the fixed settings as non-CLI arguments:
`suite=libero_object`, `task_id=8`, control frequency 10 Hz, episode length 280, relative
control, and two 256 x 256 RGB cameras. Episode `i` uses `init_state_id=i` in addition to
seed `--seed + i`.

### Outputs

The header lists `metrics.json`, `eval_info.json`, `rollouts.parquet`, `rollout.rrd`,
`run_config.json`, and zero-padded files such as `videos/episode_000.mp4` for the first
`--num-visualized-episodes` episodes. The result uses the same `.incomplete` lifecycle as
offline evaluation. Unless `--no-open-viewer` is used, the evaluator attempts to open
Rerun; a missing executable or launch failure emits a warning but does not fail an
otherwise completed evaluation.

## Validation

- compare each documented argument, default, and constraint with its Python parser and
  `validate_config` implementation;
- run `bash -n` for both Bash launchers and verify both remain executable;
- run each launcher with `--help` in the `robot-practice` environment;
- run `pytest -q tests/test_evaluation_interfaces.py` and the normal test suite;
- run `git diff --check` and confirm the diff changes comments only, with no generated
  evaluation artifacts.
