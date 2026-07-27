# PushT Diffusion Multimodality and Sampling-Speed Pilot Design

## Status

Approved design. This document defines the Pilot only. It does not authorize
implementation of the optional fixed-middle-state experiment or a full Pymunk
snapshot serializer.

## Goal

Build a reproducible PushT Pilot that answers three primary questions:

1. Given exactly the same two-observation history, does Diffusion Policy produce
   different 8-step action chunks when only the initial diffusion noise changes?
2. Given exactly the same PushT episode initial state, can different policy sampling
   seeds produce multiple distinct successful closed-loop rollouts?
3. How do DDIM denoising steps trade closed-loop success against action-chunk
   inference latency?

A deterministic `BCChunkPolicy` is the strict main baseline. The public
`lerobot/diffusion_pusht` checkpoint is an external environment and evaluator
reference only. It is not a split-comparable baseline because it was trained on the
complete public dataset.

The Pilot follows a strict progression:

```text
interface and reproducibility checks
    -> small-data smoke training
    -> full-split training
    -> fixed-observation sampling
    -> repeated full rollouts
    -> sampling-step/latency benchmark
    -> optional E3 decision
```

Small-data runs exist only to verify code, data flow, saved outputs, and
visualization. No configured experiment matrix is run on the small subset, no formal
claim uses a smoke checkpoint, and the full-data runs start again from random
initialization.

## Current Project and Dependency Baseline

The approved design targets the locally installed stack inspected on 2026-07-27:

| Component | Current value |
|---|---|
| LeRobot | 0.4.4, installed dependency |
| Diffusers | 0.35.2 |
| PyTorch | 2.10.0 with CUDA 12.8 |
| GPU | NVIDIA RTX 3090, 24 GiB |
| PushT dataset | `lerobot/pusht` |
| Dataset revision | `7628202a2180972f291ba1bc6723834921e72c19` |
| Dataset episodes | 206 |
| Dataset frames | 25,650 |
| Dataset frequency | 10 Hz |
| Dataset image | one 96 x 96 RGB stream |
| State | 2D agent position |
| Action | 2D absolute target position |

`gym-pusht` is not installed in the current environment and must be added through
the LeRobot PushT extra. Implementation must not modify the installed LeRobot source.

Project dependencies should expose the equivalent of:

```text
lerobot[libero,pusht] == 0.4.4
gym-pusht >= 0.1.5, < 0.2
diffusers == 0.35.2
scikit-learn < 2
```

The exact resolved environment remains recorded per experiment with `pip freeze`.
If the repository chooses not to globally pin a transitive dependency, the run
manifest must still record its exact installed version.

## Required Terminology

The following three counts are independent and must never share a field:

| Field | Meaning |
|---|---|
| `num_inference_steps` | Denoising iterations used to generate one action chunk |
| `num_action_samples` | Independent initial-noise tensors sampled for one fixed observation |
| `num_rollouts` | Independent environment episodes actually executed |

For example, `DDIM 20 steps` means 20 internal denoising iterations that produce one
action chunk. Sampling 20 times means 20 independent initial-noise tensors, each
followed by a complete denoising process, producing 20 candidate chunks.

Fixed-observation candidates are compared and not executed in the same environment.
Repeated rollout candidates each run in a separately reset environment. The
implementation must never execute 20 candidate chunks sequentially in one
environment and call that 20 rollouts.

## Scope

### Required

| Experiment | Purpose |
|---|---|
| Phase 0 | Verify interfaces, explicit-noise behavior, environment replay, and provenance |
| Smoke | Verify BCChunk and Diffusion training/evaluation paths on a small train-only subset |
| E0-lite | Qualitatively inspect demonstrations and select held-out observation histories |
| E1 | Generate multiple action chunks for frozen model input tensors |
| E2 | Run multiple complete rollouts from the same episode initial states |
| E4 | Compare DDIM/DDPM steps, success, and latency using one Diffusion checkpoint |

### Optional

E3 is a fixed-middle-state local branching experiment. It is designed only after E1
or E2 reveals a concrete case worth investigating. Prefix replay is the first
restoration mechanism.

### Explicit Non-goals

The Pilot does not prioritize or require:

- a complete Pymunk snapshot serializer;
- exact restoration of a dataset middle frame into the simulator;
- a paper-grade conditional-multimodality KNN benchmark;
- a full visual block-pose reconstruction benchmark;
- a best-of-M online selector or critic;
- executing an oracle-selected action sample;
- three training seeds;
- 200-500 final episodes per condition;
- ACT as a required model baseline;
- changes to installed LeRobot source.

## Repository Architecture

PushT is implemented as a separate vertical experiment slice:

```text
configs/experiments/pusht_diffusion/
├── split.json
├── bc_chunk.json
├── diffusion.json
├── e0_lite.json
├── e1_sampling.json
├── e2_rollout.json
└── e4_sampling_steps.json

scripts/experiments/pusht/
├── check_environment.sh
├── prepare_dataset.sh
├── benchmark_pretrained.sh
├── train_bc_chunk.sh
├── train_diffusion.sh
├── smoke_eval.sh
├── inspect_demonstrations.sh
├── select_checkpoints.sh
├── eval_fixed_observations.sh
├── eval_multi_seed_rollouts.sh
├── benchmark_sampling_steps.sh
└── make_report.sh

src/robot_practice/
├── policies/bc_chunk/
│   ├── __init__.py
│   ├── configuration_bc_chunk.py
│   └── modeling_bc_chunk.py
└── experiments/pusht_diffusion/
    ├── __init__.py
    ├── config.py
    ├── provenance.py
    ├── dataset_split.py
    ├── data_inspection.py
    ├── policy_loading.py
    ├── noise.py
    ├── rollout.py
    ├── fixed_observation_eval.py
    ├── diversity_metrics.py
    ├── latency.py
    ├── statistics.py
    ├── rerun_visualization.py
    └── report.py
```

Do not generalize the existing ACT/LIBERO evaluator as part of this Pilot. That code
contains valid task-specific assumptions about seven-dimensional actions, two camera
streams, LIBERO calibration, and Panda visualization. PushT may reuse only genuinely
generic behavior such as atomic output creation, incomplete-run markers, Markdown
metric style, and the Rerun launcher.

Generated content remains under the ignored output root:

```text
outputs/pusht_diffusion/
├── metadata/
├── benchmark_pretrained/
├── smoke/
├── data_inspection/
└── pilot/
    ├── bc_chunk/
    ├── diffusion/
    ├── checkpoint_selection/
    ├── e1_fixed_observation/
    ├── e2_rollouts/
    ├── e4_sampling_steps/
    └── report/
```

Datasets, weights, noise arrays, Parquet outputs, Rerun recordings, videos, raw
logs, and machine manifests must not be committed.

## Dataset Split

The public dataset contains one upstream `train` split. This project creates one
immutable episode-level split over episode IDs `0..205`:

```text
split_seed = 20260727
train = 165 episodes
development = 20 episodes
audit = 21 episodes
```

The implementation must:

1. sort all episode IDs;
2. apply a documented deterministic permutation using the split seed;
3. assign the first 165 to train, the next 20 to development, and the last 21 to
   audit;
4. save the exact lists, algorithm version, dataset revision, metadata hash, and
   split seed;
5. commit the resulting public `split.json`;
6. fail if the local dataset identity no longer matches it.

Frames are never randomly split. Training consumes only train episodes. Development
episodes support offline checks and checkpoint selection. E1 fixed inputs are chosen
from audit episodes.

Environment seeds are a separate concept:

```text
development environment seeds: 1000..1019
final Pilot environment seeds: 2000..2049
```

The lists are inclusive and disjoint. Model conditions reuse the same ordered seed
lists. E2 uses the first ten final seeds, `2000..2009`; E4 uses all 50 final seeds.

## Phase 0: Interface and Reproducibility Checks

### Runtime Contract

The expected Diffusion configuration is:

```text
n_obs_steps = 2
horizon = 16
n_action_steps = 8
action_dim = 2
noise shape = (batch, 16, 2)
returned chunk shape = (batch, 8, 2)
```

LeRobot 0.4.4 currently defines:

```text
observation indices = [-1, 0]
full action indices = [-1, 0, 1, ..., 14]
returned action slice = full_actions[1:9]
```

Relative to current observation time `t`, the policy conditions on
`O[t-1], O[t]`, trains against `A[t-1]..A[t+14]`, and exposes
`A[t]..A[t+7]` for execution. Unit and runtime tensor tests must prove this behavior;
the implementation must not rely only on this document.

### Reference Scheduler and Controlled Sampling

Training uses the configured 100-step DDPM forward process. Development checkpoint
selection, E1, and E2 use `DDPM-100` as the reference inference scheduler.

LeRobot 0.4.4 exposes an initial `noise` argument on
`predict_action_chunk()`, but that public method does not expose the
`torch.Generator` already accepted by the inner `conditional_sample()` method.
This distinction matters: DDPM draws additional variance noise during reverse
steps. Fixing only the initial noise tensor is therefore not a complete
reproducibility contract.

Implement a local controlled-sampling adapter without editing installed LeRobot.
The adapter must:

1. accept a frozen policy input, explicit initial noise tensor, and explicit reverse
   RNG seed;
2. create a fresh `torch.Generator` on the policy device for every model query;
3. seed it with the reverse RNG seed;
4. pass it through every scheduler `step()` by using the existing inner
   `conditional_sample(..., generator=..., noise=...)` path;
5. apply the same `[1:9]` public action slice;
6. record the initial noise tensor, its SHA256, the reverse RNG seed, the SHA256 of
   the generator's initial state, device, dtype, PyTorch version, CUDA version, and
   Diffusers version.

For DDIM, use `eta=0`. It is deterministic once the initial noise is fixed, so the
reverse generator is accepted and recorded by the common adapter but does not add
reverse-step noise.

### Explicit Noise

For an identical frozen input and noise:

```python
action_a = controlled_sample(observation, noise=noise_0, reverse_seed=reverse_seed_0)
action_b = controlled_sample(observation, noise=noise_0, reverse_seed=reverse_seed_0)
```

the maximum normalized-action error must be at most `1e-6`. Recreate the reverse
generator from its initial seed for the second call; do not reuse an advanced
generator object. A distinct `noise_1`, with `reverse_seed_0` held fixed, must
produce a maximum normalized-action difference greater than `1e-6`.

The test must freeze the complete model input and observation queue. Repeated E1
sampling must not mutate an action queue.

### Environment Reset Replay

For one environment seed:

1. reset the environment;
2. save the raw observation;
3. execute a fixed action sequence;
4. reset with the same seed;
5. execute the same sequence again.

Initial observations, agent paths, block paths, rewards, success, done state, and
episode length must reproduce within explicit tolerances. Failure blocks paired E2
and E4 evaluation.

Require exact equality for rendered integer images, booleans, and episode length.
Require `rtol=0, atol=1e-8` for floating observations, paths, and rewards. These
tolerances are verification gates, not values to relax after observing failures.

### Provenance

Create:

```text
outputs/pusht_diffusion/metadata/
├── git_commit.txt
├── environment.txt
├── dataset_info.json
├── diffusion_config.json
├── bc_config.json
└── interface_checks.json
```

Record the repository commit, dirty status, LeRobot distribution version and source
path, gym-pusht version, PyTorch, Diffusers, CUDA, GPU, dataset revision, policy
checkpoint identity, and all resolved shapes. If a pip distribution has no source
commit metadata, record `null` rather than inventing one.

## External Pretrained Reference

Load `lerobot/diffusion_pusht` and run a small fixed-seed benchmark before local
training. Its purposes are:

- test package and checkpoint compatibility;
- test PushT environment construction;
- test processors and public policy output;
- verify reward and success recording;
- provide a recognizable external performance reference.

It must be labeled:

```text
external pretrained reference — not split-comparable
```

Its checkpoint is approximately one gigabyte, so ordinary unit tests must not depend
on downloading it.

## Small-data Smoke

Select eight fixed episodes from the train split. Train both models for 1,000 updates
with batch size eight:

```text
BCChunk: random initialization
Diffusion: random initialization
```

Smoke checks:

- input and target shapes;
- full horizon and execution slice;
- checkpoint save/reload;
- BC repeat determinism;
- Diffusion same-noise reproducibility and different-noise sensitivity;
- two complete environment rollouts;
- a reduced E1 path with three Diffusion samples;
- CSV/Parquet, JSON, Markdown, Rerun, and static-plot output.

Smoke checkpoints are not resumed into Pilot training. Smoke outputs carry
`status=smoke`; formal reports must reject them as measured Pilot evidence.

## E0-lite: Demonstration Inspection

E0-lite is qualitative and never blocks training or E4.

Inspect at least 50 demonstrations and record:

- rendered frames;
- agent path;
- absolute action-target path;
- block path when it can be obtained through an existing reliable source;
- episode max reward;
- demonstration outcome.

The current Hub viewer indicates that `next.success` may not contain useful positive
labels. Implementation must inspect the actual data. If it is unusable, derive
demonstration outcome from `next.done` and maximum reward and record that rule.

Look for examples of left/right approaches, direct pushes, rotation-first behavior,
different contact points, failures, and recovery. Choose 5-10 two-frame observation
histories from audit episodes. Each saved candidate contains:

```text
candidate_id
episode_index
frame_index
previous raw observation
current raw observation
current agent state
future 8-step actions
future rendered frames
demonstration outcome
```

If visually/geometrically similar histories have different future directions, group
them for display. The report must distinguish:

1. the task theoretically admits multiple solutions;
2. the dataset shows different trajectories;
3. strict conditional multimodality has not been statistically proven.

No color-segmentation benchmark, formal KNN threshold search, or exact block-pose
reconstruction is required.

## Full Pilot Training

Both models restart from random initialization and use the 165 train episodes:

```text
batch size = 64
maximum updates = 200,000
checkpoint interval = 25,000
train seed = 1000
```

Development checkpoint selection:

1. calculate development offline metrics for every 25k checkpoint;
2. run the ordered development environment seeds `1000..1019`;
3. use `DDPM-100`, `policy_seed=0`, and the controlled initial/reverse RNG protocol;
4. derive noise streams only from the environment seed, policy seed, replan index,
   stream type, and a fixed protocol namespace—never from checkpoint identity;
5. reuse the resulting per-replan streams across every checkpoint;
6. rank by development success;
7. break success ties with mean maximum reward;
8. break remaining ties by selecting the earlier checkpoint;
9. write the full selection table, RNG manifest, and selected checkpoint SHA256.

Audit episodes and final environment seeds may not participate in checkpoint
selection.

## Strict BCChunk Baseline

### Shared Architecture

BCChunk should use the same observation encoder and conditional temporal U-Net class
as Diffusion:

```text
two images -> shared DiffusionRgbEncoder ----\
two states ----------------------------------+-> global condition
fixed zero trajectory query -----------------+
fixed timestep embedding, t = 0 -------------/
                         |
             DiffusionConditionalUnet1d
                         |
                16 x 2 clean actions
```

It is a deterministic direct-regression control:

| BCChunk | Diffusion |
|---|---|
| fixed zero query | Gaussian initial noise |
| one forward pass | iterative denoising |
| predicts clean action | predicts epsilon |
| MSE against clean action | MSE against sampled epsilon |
| deterministic | stochastic conditional generator |

Using a tiny MLP against a large Diffusion U-Net is outside the approved design,
because model capacity would become a major confound.

The fixed BC timestep is integer zero for every item:

```python
timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
```

It does not depend on the scheduler or training step.

### Shared Data and Optimization

Both models fix:

```text
n_obs_steps = 2
horizon = 16
n_action_steps = 8
crop_shape = (84, 84)
random crop during training
center crop during evaluation
VISUAL normalization = MEAN_STD
STATE normalization = MIN_MAX
ACTION normalization = MIN_MAX
pretrained_backbone_weights = None
GroupNorm enabled
SpatialSoftmax keypoints = 32
Adam lr = 1e-4
betas = (0.95, 0.999)
weight_decay = 1e-6
cosine schedule
warmup = 500
```

Compute statistics from train episodes only and store one shared stats file. Both
processors must load exactly that file.

Create one common initial encoder/U-Net state per train seed and load it into both
models. Hash and compare every common tensor before training. The trained weights
remain independent.

### Episode Tail

The Pilot preserves the official semantics in both policies:

```text
drop_n_last_frames = 7
do_mask_loss_for_padding = false
```

The execution window `[1:9]` stays outside padding, while non-executed positions in
the 16-step training horizon can contain copy-padding. Record:

```text
samples_with_padding_ratio
overall_padding_position_ratio
padding_count_p50
padding_count_p95
maximum_padding_count
```

A masked/unmasked comparison is a later ablation, not a Pilot variable.

### Fairness Reporting

Record parameter count, initial shared-weight hashes, optimizer settings, samples
seen, wall-clock training time, and peak GPU memory. Do not compare BC clean-action
MSE numerically with Diffusion epsilon-prediction MSE as if they were the same
objective.

## E1: Fixed-observation Multi-noise Sampling

### Input Freeze

For every audit candidate, store:

- previous and current raw image;
- previous and current raw state;
- the fully processed image and state tensors;
- processor/normalization metadata;
- frozen observation queue contents;
- SHA256 of every fixed input tensor.

All samples for one candidate must use identical tensors and queue state.

### Sampling

BCChunk predicts one chunk and repeats three calls to verify determinism.

Diffusion:

```text
num_action_samples = 20
inference scheduler = DDPM
num_inference_steps = 100
noise shape = (1, 16, 2)
returned action shape = (1, 8, 2)
num_rollouts = 0
```

Generate and save the 20 explicit initial-noise tensors before inference. For one
candidate, all 20 samples use the same recorded reverse RNG seed, recreated from its
initial state before every sample. Thus the measured variation isolates the initial
noise tensor rather than additional DDPM reverse-step randomness.

### Metrics

Report metrics in normalized and denormalized action space:

- trajectory pairwise distance;
- eighth-target endpoint pairwise distance;
- first-five-step relative-target direction angle;
- target path length;
- target total variation;
- in-bounds action-target ratio;
- BC repeated-call maximum error;
- Diffusion same-noise repeated-call maximum error.

PushT actions are absolute targets. Use:

```python
relative_target[k] = action[k] - agent_position_t
target_delta[k] = action[k] - action[k - 1]
```

Do not integrate action targets as velocities. Hierarchical clustering may support
visualization but cluster count cannot be the sole conclusion.

E1 can establish that different noise produces different structured action patterns.
It cannot establish that samples are valid or successful.

## E2: Multiple Full Rollouts from Identical Initial States

Pilot scale:

```text
10 environment seeds: 2000..2009
x 5 Diffusion policy sampling seeds
= 50 Diffusion rollouts
```

BCChunk runs one deterministic rollout for each of the ten environment seeds.

Every rollout:

```text
env.reset(seed=environment_seed)
policy.reset()
set rollout policy sampling seed
execute until success or time limit
```

Only an actual replan consumes a new noise tensor. Executing the remaining actions
from an existing eight-action queue does not advance `replan_index`.

Derive noise seeds with a stable algorithm such as:

```python
PROTOCOL_NAMESPACE = 20260727

initial_noise_seed = int(
    numpy.random.SeedSequence(
        [PROTOCOL_NAMESPACE, environment_seed, policy_seed, replan_index, 0]
    ).generate_state(1, dtype=numpy.uint32)[0]
)

reverse_noise_seed = int(
    numpy.random.SeedSequence(
        [PROTOCOL_NAMESPACE, environment_seed, replan_index, 1]
    ).generate_state(1, dtype=numpy.uint32)[0]
)
```

Stream type zero is the explicit initial noise. Stream type one is the DDPM reverse
generator and omits `policy_seed`, so reverse noise remains fixed across policy seeds
for one environment and replan. This makes the initial noise the only intentional
sampling difference between policy seeds. E1 substitutes the committed candidate
ordinal for `environment_seed` and the sample ordinal for `policy_seed`. Do not
include `condition_id`, checkpoint identity, scheduler identity, or a
language-runtime hash.

Per rollout record:

```text
model and checkpoint
train seed
environment seed
policy seed
replan index and derived noise seed
success
maximum and final reward
episode length
agent path
block position/angle path
absolute action targets
rendered video
chunk latency
```

Separate all-rollout, successful-rollout, and failed-rollout diversity. Resample
variable-length trajectories to 100 normalized-progress points before pairwise path
comparison.

Primary E2 summaries:

```text
single_sample_success
success_any@5
mean_successes_per_environment_seed
fraction_with_2plus_successful_rollouts
fraction_with_2plus_distinct_success_modes
successful_rollout_diversity
failed_rollout_diversity
```

`single_sample_success` uses fixed `policy_seed=0`. `success_any@5` is an oracle
coverage diagnostic and not deployment performance. Multiple successful behavior
requires at least two successful rollouts from one environment seed and a
pre-registered minimum trajectory distance.

For the Pilot, define the trajectory-mode distance on the joint agent and block
path. Stop each path at success or episode termination, resample it to 100 points by
normalized elapsed-step progress, linearly interpolate positions, and unwrap block
angle before interpolation. At each resampled point compare:

```text
agent_x / 512
agent_y / 512
block_x / 512
block_y / 512
wrapped_block_angle_difference / pi
```

The pairwise distance is the root mean square over all 100 points and all five
components. The pre-registered primary threshold is `0.05`. Within one environment
seed, apply complete-linkage agglomerative clustering to successful rollouts with
distance threshold `0.05`; two or more resulting clusters count as two or more
operational success modes. A sensitivity table at `0.025` and `0.075` may be shown,
but it cannot replace the primary threshold or control the E3 decision.

First-contact side is not required for the Pilot's mode classifier. It may be shown
as qualitative metadata only if a later implementation defines and tests a
geometric rule.

Environment seed is the primary statistical unit. Policy seeds are repeated samples
within one initial condition. Use Wilson intervals for binary success and
environment-level cluster bootstrap for repeated-sampling summaries.

## Optional E3: Prefix-replay Local Branching

Do not implement E3 automatically. The final Pilot report first identifies whether a
specific E1/E2 case warrants it.

Preferred restoration:

```text
same environment seed
+ same fixed action prefix
= same intermediate state
```

Before branching, repeated prefix replay must reproduce observation state, rendered
frame, agent pose, block pose, reward, elapsed steps, and the future trajectory under
a fixed test action sequence.

Only consider a complete Pymunk snapshot implementation if:

1. a clear and important local case exists;
2. prefix replay cannot reconstruct it;
3. the stronger local claim materially affects the conclusion;
4. the additional engineering cost is approved separately.

## E4: DDIM/DDPM Steps and Latency

Use the selected local Diffusion checkpoint for:

```text
DDIM 5
DDIM 10
DDIM 20
DDIM 50
DDIM 100
DDPM 100
```

If the resolved training timesteps are not 100, the complete reference must follow
the actual config and the manifest must explain the adjusted matrix.

Reconstruct each scheduler condition from the checkpoint config, load the same state
dict, require no missing/unexpected keys, and prove all model parameter hashes are
identical. Changing a config field on an already constructed scheduler is not an
accepted scheduler switch.

Every condition shares:

- 50 distinct final environment seeds;
- fixed `policy_seed=0`;
- the same per-replan explicit initial noise;
- the same recorded reverse RNG seed; DDPM consumes it and deterministic DDIM ignores
  it;
- `n_action_steps=8`;
- checkpoint weights;
- processor, device, dtype, and compile setting.

This yields 50 single-sample rollouts per condition. It is intentionally separate
from E2's `10 environment x 5 policy` repeated-sampling structure.

Neither initial nor reverse seed derivation includes scheduler type, inference-step
count, or `condition_id`. `DDPM-100` is the pre-registered high-quality reference
for the success-drop and speedup criterion because it matches the reference
inference process used for development selection, E1, and E2. `DDIM-100` remains a
reported comparison rather than silently replacing that reference.

Latency protocol:

```text
batch size = 1
policy.eval()
torch.no_grad()
rendering disabled
50 warm-up iterations
1,000 measured iterations
CUDA synchronization before and after timing
```

Report:

- model chunk latency after preprocessing, including visual encoding and denoising;
- end-to-end chunk latency from raw observation through denormalized eight-action
  chunk;
- amortized end-to-end latency per executed action;
- mean, p50, p90, p95, p99, standard deviation, and peak GPU memory.

The primary plot is p95 end-to-end chunk latency versus success. The target
criterion—an evaluation rule, not a promised outcome—is at least 2x lower p95 with no
more than a five-percentage-point success reduction from the high-quality reference.

## Result Schemas

Every experiment output contains:

```text
run_manifest.json
metrics.json
metrics.md
recording.rrd
```

Additional outputs:

```text
E0-lite:
  trajectories.parquet
  candidates.json

E1:
  observations/
  noises/
  action_chunks.parquet
  diversity_metrics.parquet

E2/E4:
  rollouts.parquet
  replans.parquet
  videos/

Latency:
  latency.parquet
```

All applicable records expose:

```text
experiment_id
condition_id
model
checkpoint_id
train_seed
environment_seed
policy_seed
replan_index
initial_noise_seed
initial_noise_sha256
reverse_noise_seed
reverse_generator_state_sha256
num_inference_steps
num_action_samples
num_rollouts
dataset_revision
code_commit
```

Use null for a field that does not apply; do not overload it with another meaning.

## Visualization

PushT is a two-dimensional task. Do not add a three-dimensional robot skeleton.

E0-lite Rerun:

- demonstration selector;
- frame-by-frame rendered image;
- agent, action-target, and optional reliable block paths;
- reward and outcome metadata.

E1 Rerun:

- candidate selector;
- previous/current image pair;
- 20 toggleable Diffusion action chunks on a 2D PushT plane;
- BCChunk trajectory;
- endpoint scatter;
- sample/cluster table and diversity metrics.

E2/E4 Rerun:

- rollout selector labeled by environment seed, policy seed, scheduler, and steps;
- simulator playback;
- agent, block, and action-target paths;
- reward/coverage;
- replan/noise timeline;
- latency and success/failure metadata.

Static reports generate:

```text
fixed_observation_samples.png
successful_rollout_modes.png
success_conditioned_diversity.png
sampling_steps_vs_success.png
latency_distribution.png
success_latency_pareto.png
```

E2 stores videos for all 50 Diffusion and ten BC rollouts. E4 stores all numeric
trajectories and defaults to five representative videos per condition; a CLI option
may increase that number.

## Public Launchers

Each launcher contains its own actual command construction. It must not delegate the
real operation through a chain of opaque wrappers.

Every script header documents:

- purpose;
- accepted parameters;
- defaults;
- the Python module or `lerobot-train` entrypoint invoked;
- smoke and Pilot examples;
- output locations;
- overwrite behavior.

Training launchers accept:

```text
--mode=smoke|pilot
--steps=N
--train-seed=N
--split-file=PATH
--output-dir=PATH
--print-command
```

`--print-command` validates and prints the complete shell-readable command without
creating an output directory, downloading data, or starting training.

Canonical workflow:

```bash
bash scripts/experiments/pusht/check_environment.sh
bash scripts/experiments/pusht/prepare_dataset.sh
bash scripts/experiments/pusht/benchmark_pretrained.sh \
  --n-episodes=10 \
  --output-dir=outputs/pusht_diffusion/benchmark_pretrained
```

```bash
bash scripts/experiments/pusht/train_bc_chunk.sh \
  --mode=smoke \
  --steps=1000 \
  --output-dir=outputs/pusht_diffusion/smoke/bc_chunk

bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=smoke \
  --steps=1000 \
  --output-dir=outputs/pusht_diffusion/smoke/diffusion
```

```bash
bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=50 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection
```

```bash
bash scripts/experiments/pusht/train_bc_chunk.sh \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/pilot/bc_chunk

bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/pilot/diffusion
```

```bash
bash scripts/experiments/pusht/select_checkpoints.sh \
  --bc-run=outputs/pusht_diffusion/pilot/bc_chunk \
  --diffusion-run=outputs/pusht_diffusion/pilot/diffusion \
  --output-dir=outputs/pusht_diffusion/pilot/checkpoint_selection
```

```bash
bash scripts/experiments/pusht/eval_fixed_observations.sh \
  --bc-policy=<selected-bc-checkpoint> \
  --diffusion-policy=<selected-diffusion-checkpoint> \
  --num-action-samples=20 \
  --output-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation
```

```bash
bash scripts/experiments/pusht/eval_multi_seed_rollouts.sh \
  --bc-policy=<selected-bc-checkpoint> \
  --diffusion-policy=<selected-diffusion-checkpoint> \
  --num-environment-seeds=10 \
  --num-policy-seeds=5 \
  --output-dir=outputs/pusht_diffusion/pilot/e2_rollouts
```

```bash
bash scripts/experiments/pusht/benchmark_sampling_steps.sh \
  --diffusion-policy=<selected-diffusion-checkpoint> \
  --schedulers=ddim,ddpm \
  --ddim-steps=5,10,20,50,100 \
  --ddpm-steps=100 \
  --n-episodes=50 \
  --latency-iterations=1000 \
  --output-dir=outputs/pusht_diffusion/pilot/e4_sampling_steps
```

```bash
bash scripts/experiments/pusht/make_report.sh \
  --e0=outputs/pusht_diffusion/data_inspection \
  --e1=outputs/pusht_diffusion/pilot/e1_fixed_observation \
  --e2=outputs/pusht_diffusion/pilot/e2_rollouts \
  --e4=outputs/pusht_diffusion/pilot/e4_sampling_steps \
  --output-dir=outputs/pusht_diffusion/pilot/report
```

## Failure Handling

Fail before expensive work when:

- the dataset revision does not match the split manifest;
- output shapes are not `(B, 8, 2)`;
- noise shape is not `(B, 16, 2)`;
- same-noise inference does not reproduce;
- DDPM is invoked through a path that cannot accept the recorded reverse generator;
- fixed-seed environment replay does not reproduce;
- train-only stats differ between models;
- a scheduler evaluator changes parameter hashes;
- E4 pairing metadata is incomplete;
- a requested output exists without explicit overwrite permission.

An E0-lite failure to find visually similar, different-action histories does not
block E1, E2, or E4. E1 may use generic held-out histories, while the report states
that the dataset inspection did not establish conditional multimodality.

Use atomic output creation and an incomplete marker. A failed run must not look like
a complete result.

## Test Strategy

### Unit

Test:

1. split determinism and disjoint episode sets;
2. train-only statistics excluding development/audit;
3. BC/Diffusion delta-index equality;
4. 16-step horizon and `[1:9]` execution slice;
5. padding statistics;
6. BC clean-action MSE;
7. shared initial parameter hashes;
8. stable noise-seed derivation;
9. condition-independent E4 initial/reverse seed derivation;
10. DDPM same-initial-noise and same-reverse-seed reproducibility;
11. DDPM different-initial-noise sensitivity with fixed reverse seed;
12. fixed BC timestep equal to integer zero;
13. frozen E1 queues and input tensor hashes;
14. replan-index advancement only when the action queue is empty;
15. environment reset replay;
16. scheduler replacement without weight changes;
17. trajectory-mode distance, wrapped angle, and complete-linkage threshold fixtures;
18. diversity metrics and trajectory resampling;
19. Wilson interval and environment-cluster bootstrap fixtures;
20. Parquet schemas and non-overloaded count fields;
21. minimal readable Rerun recordings.

### CLI

Test:

- complete `--help`;
- side-effect-free `--print-command`;
- invalid, duplicate, and unknown arguments;
- default refusal to overwrite;
- explicit overwrite behavior;
- prevention of smoke/Pilot output mixing.

### Integration

Test:

- BC and Diffusion forward passes on synthetic tensors;
- one real local PushT dataset batch;
- one fixed-seed environment replay;
- a two-step rollout;
- one DDIM/DDPM scheduler reconstruction.

GPU and one-gigabyte official-checkpoint tests are opt-in and not required for the
ordinary unit suite.

Implementation verification runs focused Pytest, Ruff on changed Python files,
shell syntax checks, and launcher `--print-command` checks. It must not start a
200,000-step job.

## Pilot Reporting

Write the shareable guide/result document to:

```text
reports/pusht_diffusion_pilot.md
```

Append each actual run to `reports/experiment_log.md`. Every completed entry must
include:

- task suite;
- dataset size and exact split;
- observation keys;
- action space;
- training and execution horizons;
- control frequency;
- train loss;
- open-loop action error/diversity;
- closed-loop success;
- latency;
- failure cases.

Reports distinguish plans, smoke checks, measured Pilot results, limitations, and
remaining work. They contain no generated observations, weights, raw logs, machine
paths, or private identifiers.

## Pilot Completion Criteria

The Pilot is complete when:

1. identical fixed observation and noise reproduce the same action chunk;
2. different noise on one fixed observation produces measurable action differences
   or a defensible measured null result;
3. repeated policy seeds run independently from identical episode initial states;
4. successful behavior diversity is separated from failure randomness;
5. BCChunk and Diffusion share the approved dataset, temporal alignment, processor,
   model capacity class, initialization, and episode-tail semantics;
6. DDIM/DDPM conditions produce a paired success-latency curve;
7. all results reproduce from recorded split, seeds, noise, checkpoint hashes, and
   configs;
8. the report answers whether E3 is justified.

The report must separately answer:

1. Does Diffusion produce multiple action patterns for one observation history?
2. Are those differences structured or merely invalid random noise?
3. Does one initial environment state yield multiple successful rollouts?
4. Does BCChunk show qualitative mode-averaging evidence in any held-out case?
5. Does Diffusion diversity improve ordinary single-sample success?
6. Which DDIM step setting offers the best observed success-latency trade-off?
7. Is there a specific case worth a separately approved E3 prefix-replay study?

No answer is required to favor Diffusion. A measured absence of useful multimodality
or a failure of low-step DDIM is a valid Pilot result.
