# ACT A-Sanity Step-Control and Initialization Benchmark Design

## Goal

Make the existing A-Sanity launcher accept a caller-controlled number of training
updates while keeping every invocation independent:

- `--steps=0` creates and saves an ACT initialization checkpoint without an optimizer
  update;
- `--steps=N`, for any positive integer `N`, runs exactly `N` optimizer updates and
  saves the final checkpoint;
- evaluation remains a separate, manually triggered operation through the existing
  offline and LIBERO evaluation launchers.

There is no command that automatically chains step 0, evaluation, training, and final
evaluation. The user chooses when to run each operation and which checkpoint to evaluate.

## Experiment Definition

The experiment continues to use the fixed A-Sanity data and policy definition:

| Field | Value |
|---|---|
| Task suite | LIBERO Object |
| Simulator task ID | 8 |
| Dataset | `lerobot/libero` |
| Dataset task index | 29 |
| Episodes | `823, 826, 834, 840, 841` |
| Dataset frames | 859 |
| Observation state | 8D: end-effector position 3D, axis-angle 3D, gripper qpos 2D |
| Cameras | external agent view and wrist view, both 256 x 256 RGB |
| Action | 7D: relative translation 3D, relative rotation 3D, gripper command 1D |
| Control frequency | 10 Hz |
| ACT chunk size | 20 |
| Executed action steps | 5 |
| VAE | disabled |
| Batch size | 8 |
| Seed | 1000 |

The step-0 checkpoint is an **initialization checkpoint**, not a fully random model. Its
ResNet18 backbone starts from `ResNet18_Weights.IMAGENET1K_V1`; the ACT transformer,
state projections, decoder queries, and action head use their normal random
initialization.

## Command Interface

The existing launcher becomes:

```text
Usage: a_sanity.sh [--steps=N] [--policy-path=PATH] [--print-command]
```

Arguments are order-independent. `--steps` defaults to `20000` to preserve the current
no-argument behavior. An explicit value must use the canonical decimal form
`^(0|[1-9][0-9]*)$` and be no greater than 2,147,483,647. Leading-zero forms such as
`00`, `08`, and `020000` are rejected instead of being interpreted through Bash octal
arithmetic. The launcher validates and normalizes the value without evaluating unchecked
text as a shell arithmetic expression.

Create an initialization checkpoint:

```bash
scripts/experiments/act/a_sanity.sh --steps=0
```

Run an independent fresh 20,000-update experiment:

```bash
scripts/experiments/act/a_sanity.sh --steps=20000
```

Run 20,000 updates from a previously saved A-Sanity initialization checkpoint:

```bash
scripts/experiments/act/a_sanity.sh \
  --steps=20000 \
  --policy-path=outputs/act/a_sanity/<baseline_timestamp>/checkpoints/000000/pretrained_model
```

`--policy-path` is optional for positive-step runs. Without it, LeRobot creates a fresh
policy from `--policy.type=act` using seed 1000. With it, the launcher loads exactly the
saved policy weights and starts a new optimizer; this is weight initialization, not
training-state resume. For a strict before/after comparison, the evaluated step-0
checkpoint must be passed explicitly.

A relative `--policy-path` is interpreted relative to the caller's current working
directory, not relative to the repository or script directory. The launcher resolves it
to an existing canonical absolute directory before entering Conda and passes it as one
shell-array element, preserving spaces and other non-control path characters safely.

`--policy-path` is rejected for `--steps=0`, because step 0 is specifically the operation
that creates a new initialization. The accepted path for a positive-step A-Sanity run
must be a complete, compatible ACT `pretrained_model` directory. It need not contain
optimizer state.

`--print-command` performs all non-mutating preflight validation and prints the command
that would run. It does not create an output directory, checkpoint, or lock file.

## Independent Execution Semantics

Every invocation creates a new timestamped run root:

```text
outputs/act/a_sanity/<YYYYMMDD_HHMMSS>/
```

The launcher never searches for or automatically selects the newest checkpoint. It also
never starts evaluation or another training invocation after completing its requested
operation.

For `--steps=0`, the result is:

```text
outputs/act/a_sanity/<timestamp>/
├── initialization_manifest.json
└── checkpoints/
    └── 000000/
        └── pretrained_model/
            ├── config.json
            ├── model.safetensors
            ├── policy_preprocessor.json
            ├── policy_postprocessor.json
            └── <processor state files>
```

The checkpoint is inference- and fine-tuning-ready but is not a resumable LeRobot
training checkpoint: it intentionally has no optimizer, scheduler, or RNG training state.

For a positive value `N`, the existing LeRobot trainer owns the output layout and saves
the final checkpoint under its normal zero-padded step directory, for example:

```text
outputs/act/a_sanity/<timestamp>/checkpoints/020000/pretrained_model
```

`--steps=N` means `N` optimizer updates in the new invocation. When the source is the
step-0 checkpoint, that is also the total number of learned updates in the policy
lineage. The launcher is not a resume interface.

## Components

### Shell launcher

`scripts/experiments/act/a_sanity.sh` remains the public entry point. It owns:

- argument parsing and usage text;
- non-negative step validation;
- timestamped output selection;
- dataset, Conda, `flock`, and output preflight checks;
- dispatch to the step-0 Python module or to `lerobot-train`;
- the existing non-blocking process lock so two A-Sanity operations do not compete for
  the same experiment resources.

For a fresh positive-step run, the LeRobot command keeps the current ACT arguments and
replaces the fixed `--steps=20000` and `--save_freq=20000` with the requested positive
value. For a checkpoint-initialized run, it uses `--policy.path=<PATH>` instead of
`--policy.type=act`; LeRobot applies the compatible saved ACT configuration and loads the
weights. Passing both policy selectors is forbidden by LeRobot and by the launcher.
Both branches explicitly pass `--policy.push_to_hub=false`. The step-0 config also
persists `push_to_hub=false`, so loading that config can never activate Hub upload or
require a policy `repo_id`.

The final checkpoint is saved even for a value that is not a multiple of the normal log
frequency. The log frequency remains at most 100 steps and is reduced to `N` for runs
shorter than 100 steps so a short experiment emits a terminal training log.

### Step-0 checkpoint module

A small reusable module under `src/robot_practice/` owns initialization checkpoint
creation. It constructs the same `TrainPipelineConfig` used by the positive-step command
and follows the same LeRobot construction path used by training:

1. construct the ACT config with chunk size 20, five executed action steps, VAE disabled,
   `push_to_hub=false`, and the ImageNet-pretrained ResNet18 backbone, then place it in
   the same fixed `TrainPipelineConfig` used by a positive-step run;
2. validate that training config before any RNG-dependent model construction;
3. set Python, NumPy, and Torch RNGs to seed 1000, matching the beginning of LeRobot's
   training function;
4. call LeRobot's `make_dataset(cfg)` with the fixed local dataset, episodes, and default
   `dataset.use_imagenet_stats=true`, then validate the selected episodes. The policy
   config must already exist because `make_dataset(...)` resolves action delta timestamps
   from `cfg.policy.action_delta_indices`;
5. infer input and output features from the resulting dataset metadata through LeRobot's
   policy factory, producing two image inputs, an 8D state input, and a 7D action output;
6. create the normal ACT preprocessor and postprocessor from the resulting
   `dataset.meta.stats`. This is significant because `make_dataset(...)` replaces each
   camera's raw dataset mean and standard deviation with LeRobot's ImageNet statistics
   when `dataset.use_imagenet_stats=true`;
7. save policy weights, policy config, processor configs, and processor state files;
8. verify that every saved ACT backbone tensor equals the corresponding tensor produced
   by torchvision's requested `ResNet18_Weights.IMAGENET1K_V1` initialization before any
   ACT training update;
9. validate the complete checkpoint with the existing evaluation checkpoint validator;
10. load the saved safetensors back, compare every policy tensor with the in-memory model,
   and verify the model fingerprint before publishing the output atomically.

The module does not implement a training loop and does not copy or modify LeRobot source.

### Initialization manifest

`initialization_manifest.json` records enough provenance to distinguish the baseline
from a fully random or learned policy:

- schema version and creation timestamp;
- `initialization_step: 0` and `optimizer_updates: 0`;
- dataset repo, relative dataset root, task index, selected episodes, and frame count;
- observation keys and shapes, action shape, FPS, chunk size, and action horizon;
- seed and backbone weight identifier;
- `dataset.use_imagenet_stats: true`, the two effective camera mean/std values, and a
  successful pretrained-backbone verification marker;
- LeRobot, Torch, and Python versions;
- SHA-256 of `model.safetensors`;
- the checkpoint path relative to the run root.

No absolute personal path is written to this manifest.

## Manual Evaluation

The existing evaluators remain unchanged and are invoked explicitly after either run:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path=<pretrained_model>
```

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path=<pretrained_model> \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none
```

Because both step-0 and positive-step checkpoints use the standard
`<run>/checkpoints/<step>/pretrained_model` layout, the evaluators can derive their normal
timestamped output directories without an explicit `--output-dir`.

A strict before/after comparison uses the same five offline episodes, closed-loop seeds,
episode count, perturbation setting, and inference settings for both checkpoints. The
relevant existing metrics are:

- offline: first-action MAE, chunk MAE by horizon, delta-action MAE, teacher-forced TV,
  teacher-forced jerk, and model-query latency;
- closed loop: success rate, reward, executed-action TV and jerk, gripper toggles,
  policy-step latency, and model-query latency.

The actual results are added to `reports/experiment_log.md` only after the user runs the
experiments. This design does not fabricate placeholder metrics.

## Validation and Failure Handling

The launcher or step-0 module fails before publishing a result when:

- `--steps` is negative, empty, repeated, not canonical decimal, or exceeds
  2,147,483,647;
- an unknown or repeated CLI option is supplied;
- `--policy-path` is used with step 0;
- the local LIBERO metadata is missing;
- any selected episode is absent or belongs to a task other than dataset task index 29;
- inferred inputs are not exactly the expected two 256 x 256 RGB cameras plus 8D state;
- the inferred action is not 7D;
- a supplied policy path is missing required policy or processor files;
- a supplied policy is not ACT or does not match chunk size 20, action steps 5, VAE off,
  the observation features, and the 7D action feature;
- a supplied relative policy path cannot be resolved from the caller's working directory;
- the timestamped output directory already exists or is not writable;
- the effective camera processor statistics differ from the normal training setup;
- the backbone differs from the requested torchvision pretrained weights;
- checkpoint round-trip tensor comparison or the model fingerprint check fails.

Step-0 creation writes into a temporary sibling directory with an incomplete marker. It
renames the completed run into its final timestamped path only after every required file
and fingerprint check succeeds. A failed attempt leaves no path that looks like a complete
run; diagnostic temporary output may remain clearly marked incomplete for inspection.

Generated checkpoints, manifests, evaluation artifacts, videos, and Rerun recordings stay
under the already ignored `outputs/` tree and are never committed.

## Testing

Unit tests cover:

- CLI parsing for omitted/default steps, zero, representative positive values, invalid
  values, option ordering, and `--print-command`, including explicit cases for `00`,
  `08`, and an overflow-sized decimal string;
- rejection of a policy path for step 0;
- fresh positive commands using `--policy.type=act` and checkpoint-initialized commands
  using `--policy.path` without combining the selectors;
- the fixed episode/task/frame contract and inferred 8D state, 7D action, and two-camera
  feature contract;
- deterministic initialization under seed 1000 with a compact test configuration;
- equality of the effective step-0 processor configuration and statistics with a normal
  fresh-training setup created through `make_dataset(...)`;
- an executable helper-path test that constructs ACT and training configs before calling
  `make_dataset(cfg)`, then builds the policy and processors in the specified order;
- pretrained-backbone application by comparing backbone tensors with the requested
  torchvision weights; ordinary offline unit tests mock the weight provider, while one
  explicitly gated integration check may use the cached/downloaded official weights;
- checkpoint save/load tensor equality, required processor state files, manifest schema,
  relative paths, and SHA-256 correctness;
- caller-relative policy-path canonicalization and shell-safe handling of a path that
  contains spaces;
- exclusive output creation and incomplete-output behavior;
- preservation of the existing A-Sanity defaults for a no-argument positive run.

The existing gated integration tests continue to provide a one-frame real offline
prediction and a one-step real LIBERO rollout from a generated ACT checkpoint. The normal
test suite does not run a 20,000-step training job or download generated experiment data.

## Non-Goals

This change does not:

- automatically chain initialization, evaluation, training, or final evaluation;
- automatically discover the latest checkpoint;
- resume optimizer or scheduler state;
- change ACT loss functions, temporal ensembling, perturbations, or evaluation metrics;
- add arbitrary datasets, tasks, policies, action dimensions, or camera layouts;
- modify installed LeRobot files;
- commit generated models or experiment artifacts.
