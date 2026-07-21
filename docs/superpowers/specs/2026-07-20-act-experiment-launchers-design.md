# ACT Experiment Launcher Design

## Goal

Create a scalable home for reproducible experiment launch scripts and add the first
launcher for the ACT A-Sanity overfitting experiment. The launcher must work from any
current directory, use the project Conda environment automatically, preserve every run,
and avoid external logging or Hub uploads by default.

## Chosen Layout

Use a policy-oriented hierarchy:

```text
scripts/
└── experiments/
    └── act/
        └── a_sanity.sh
```

This keeps the existing `scripts/` convention and leaves an obvious location for the
future A0, A1, A2, and A3 launchers without introducing a generic parameter framework
before it is needed.

## Launcher Behavior

The launcher is a Bash script with `set -euo pipefail`. It resolves the practice root
relative to its own location, so it behaves the same whether invoked from the repository
root or another directory.

It checks that:

- `conda` is available;
- `flock` is available;
- the local LIBERO dataset contains `meta/info.json`;
- the output base can be created and is writable;
- the generated output directory does not already exist.

The script does not activate or mutate the caller's shell. It runs LeRobot through:

```text
conda run --no-capture-output -n robot-practice lerobot-train
```

`--no-capture-output` preserves live training output in the terminal.

All LeRobot arguments are stored in a quoted Bash array and passed as
`lerobot-train "${train_args[@]}"`. In particular, this prevents the bracketed episode
list from undergoing pathname expansion when the launcher is called from an arbitrary
directory.

The only optional launcher argument is `--print-command`. It prints the fully resolved,
shell-escaped command and exits before creating the output base, taking a lock, or
starting training. Unknown launcher arguments are rejected. This provides a no-run seam
for configuration verification without maintaining a second copy of the training
arguments.

## Experiment Configuration

The launcher selects five complete demonstrations from one task: LIBERO Object task 8,
“pick up the chocolate pudding and place it in the basket.” The global dataset episode
indices are:

```text
[823, 826, 834, 840, 841]
```

The training configuration is:

```text
dataset.repo_id=lerobot/libero
dataset.root=<practice>/datasets/lerobot/lerobot/libero
dataset.episodes=[823,826,834,840,841]
policy.type=act
policy.chunk_size=20
policy.n_action_steps=5
policy.use_vae=false
policy.push_to_hub=false
batch_size=8
steps=20000
seed=1000
eval_freq=0
log_freq=100
save_checkpoint=true
save_freq=20000
wandb.enable=false
```

The deterministic seed and explicit logging/checkpoint settings make the sanity result
easier to reproduce and inspect. Evaluation is disabled during training because no
simulation environment is part of this launcher. The final checkpoint is retained for a
separate rollout against the training initial states.

## Output Isolation

Each invocation computes a local timestamp in `YYYYMMDD_HHMMSS` form and passes this
output directory to LeRobot:

```text
<practice>/outputs/act/a_sanity/<YYYYMMDD_HHMMSS>
```

The launcher must not create the timestamped leaf directory before invoking LeRobot,
because LeRobot rejects an existing output directory for a fresh run. It creates only
the parent `<practice>/outputs/act/a_sanity`, verifies that the parent is writable, and
uses a nonblocking `flock` on a lock file in that parent. `flock` runs with `--verbose`
and a dedicated conflict exit code so lock contention produces a visible diagnostic
instead of a silent status-1 exit. The lock is held for the whole
`conda run` process and deliberately disallows concurrent A-Sanity launches. This closes
the check-then-create race without reserving the timestamped leaf itself. If another
A-Sanity run holds the lock, or if a same-second output path already exists, the launcher
exits clearly rather than overwriting or resuming it.

## Failure Handling

Preflight failures produce a short message on stderr and a nonzero exit status. Once
training starts,
`exec flock --nonblock --no-fork --verbose --conflict-exit-code=75 ...` replaces the
launcher and then executes `conda run`, so the lock lifetime, signals, and final exit
status follow the training command. The launcher never deletes output, automatically
resumes training, logs into W&B, or pushes artifacts to the Hub.

## Verification

Verification must not start the 20,000-step experiment. It consists of:

1. `bash -n scripts/experiments/act/a_sanity.sh`;
2. checking that the file is executable;
3. invoking the launcher from outside the practice root with `--print-command`, parsing
   its shell-escaped output with `shlex.split`, and asserting the expected `conda run`
   prefix;
4. importing `lerobot.scripts.lerobot_train` to register ACT, then parsing the captured
   LeRobot argument suffix into `TrainPipelineConfig` with `draccus.parse` and calling
   `cfg.validate()` without constructing the dataset or entering the training loop;
5. resolving each selected episode through `meta/episodes/chunk-*/file-*.parquet` to its
   data Parquet file, reading only `episode_index` and `task_index` from that file, and
   confirming all five episodes have task index 29; then checking task index 29 in
   `meta/tasks.parquet` against the expected chocolate-pudding instruction;
6. confirming that the verification leaves the workspace without a `.git` directory.

## Version-Control Constraint

At design time, the workspace was not a Git working tree. The design and launcher could be
written and verified locally, but could not be committed unless version control was initialized
later. This work did not initialize Git as a side effect.

## Acceptance Criteria

- The launcher exists at `scripts/experiments/act/a_sanity.sh` and is executable.
- It can be invoked from outside the practice root.
- It automatically uses the `robot-practice` Conda environment.
- It selects exactly the five approved single-task episodes.
- It passes exactly `chunk_size=20`, `n_action_steps=5`, `use_vae=false`, batch size 8,
  20,000 steps, and seed 1000.
- Every fresh run receives an output path matching
  `outputs/act/a_sanity/[0-9]{8}_[0-9]{6}`.
- W&B and Hub uploads are disabled.
- Static shell validation and LeRobot configuration validation pass without starting
  training.
- A successfully completed training run retains its final step-20,000 checkpoint.
- Verification and launcher creation do not initialize Git.
