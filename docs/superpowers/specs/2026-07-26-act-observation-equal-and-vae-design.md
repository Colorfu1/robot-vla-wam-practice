# ACT Observation-Equal Loss and VAE Experiment Design

## Goal

Add two isolated ACT experiments that both compare directly with the existing A1
action-chunking baseline:

1. **A1-ObsEqual** tests whether giving every sampled observation equal total L1
   weight improves terminal release behavior.
2. **A1-VAE** tests whether enabling ACT's standard VAE improves closed-loop behavior
   under otherwise unchanged A1 settings.

The implementation must remain local to this repository and must not modify the
installed LeRobot package.

## Fixed Baseline

All three runs use the same LIBERO Object task and the same 20 demonstration
episodes:

```text
823, 826, 834, 840, 841, 845, 847, 869, 875, 878,
910, 918, 920, 929, 944, 948, 949, 953, 966, 972
```

The exact dataset identity is:

```text
dataset.repo_id=lerobot/libero
dataset.root=datasets/lerobot/lerobot/libero
```

The canonical control launcher is:

```text
scripts/experiments/act/a1_chunking.sh
```

The shared training contract is:

| Setting | Value |
|---|---:|
| policy family | ACT |
| `chunk_size` | 20 |
| checkpoint inference default `n_action_steps` | 5 |
| image/state/action feature inference | LeRobot dataset metadata |
| ImageNet dataset statistics | enabled |
| pretrained visual backbone | ACT default ImageNet ResNet-18 |
| batch size | 8 |
| optimizer updates | 20,000 by default |
| seed | 1000 |
| evaluation during training | disabled |
| log frequency | `min(steps, 100)` |
| checkpoint saving | enabled |
| checkpoint save step | final step |
| policy Hub push | disabled |
| W&B | disabled |

The complete shared command settings, before applying a variant's policy-specific
arguments, are:

```text
--dataset.repo_id=lerobot/libero
--dataset.root=datasets/lerobot/lerobot/libero
--dataset.episodes=[823,826,834,840,841,845,847,869,875,878,910,918,920,929,944,948,949,953,966,972]
--policy.chunk_size=20
--policy.n_action_steps=5
--policy.push_to_hub=false
--batch_size=8
--steps=<positive integer>
--seed=1000
--eval_freq=0
--log_freq=min(steps,100)
--save_checkpoint=true
--save_freq=steps
--wandb.enable=false
```

The existing A1 run and launcher remain unchanged and are the control. Any ACT
configuration not listed above or in a variant section must retain the value emitted
by the canonical A1 launcher and the installed LeRobot ACT defaults.

## Experiment 1: A1-ObsEqual

### Hypothesis

The upstream ACT masked L1 loss averages over the fixed `[batch, chunk, action_dim]`
tensor after padded positions are zeroed. A sample close to the episode boundary
therefore contributes less total loss because it contains fewer valid target actions.
Giving every sampled observation equal total weight may improve learning of release
and post-release behavior near the episode boundary.

### Exact loss

For a batch with absolute elementwise action error `E` of shape
`[batch, chunk, action_dim]`, convert LeRobot's padding field exactly as follows:

```python
V = ~batch["action_is_pad"].bool()
```

`V` has shape `[batch, chunk]`; `True` means that the target action is real and must
contribute to training. `False` means padding and must not contribute.

The loss is:

```python
per_action_l1 = E.mean(dim=-1)
valid_count = V.sum(dim=1).clamp_min(1)
per_observation_l1 = (per_action_l1 * V).sum(dim=1) / valid_count
l1_loss = per_observation_l1.mean()
```

LeRobotDataset always provides at least the anchor action, so every sample is
expected to contain at least one valid action. The policy must raise `ValueError` if
any batch row has zero valid actions instead of silently assigning that observation a
zero loss. The `clamp_min(1)` may remain only after this validation as a numerical
guard.

Consequences are intentional:

- Every sampled observation contributes one equally weighted scalar loss.
- Within a sample, its valid target actions are averaged.
- A sample with one valid action gives that action greater relative weight than each
  individual action in a sample with 20 valid actions.
- Padded target values never contribute.

The experiment uses `use_vae=false` and does not add delta-action loss. This isolates
the observation-level normalization change from A2.

Its policy-specific training arguments are:

```text
--policy.type=act_obs_equal
--policy.use_vae=false
--policy.discover_packages_path=robot_practice.policies.act_obs_equal
```

### Local policy

Create an independently registered policy type:

```text
type=act_obs_equal
discover package=robot_practice.policies.act_obs_equal
```

It subclasses the installed `ACTPolicy`, reuses the ACT model and inference behavior,
and overrides only the training `forward()` loss calculation. Its config subclasses
`ACTConfig` without adding tunable behavior. Its config validation must reject
`use_vae=true`; combining the observation-equal loss with VAE is outside these two
controlled experiments. Checkpoints must remain loadable by the existing offline and
LIBERO evaluators through the discover-package mechanism.

The forward result must retain LeRobot's expected loss dictionary:

- `l1_loss`
- `loss`

It must not emit `kld_loss`, because this policy only permits `use_vae=false`.

## Experiment 2: A1-VAE

### Hypothesis

The standard ACT VAE may model ambiguity in valid future action chunks better than
the deterministic A1 policy and may improve closed-loop release behavior.

### Controlled change

Use the installed standard ACT policy and original ACT loss. Relative to A1, change
only the VAE-related configuration:

| Setting | A1 | A1-VAE |
|---|---:|---:|
| `use_vae` | false | true |
| `latent_dim` | inactive | 32 |
| `kl_weight` | inactive | 10.0 |

`latent_dim=32` and `kl_weight=10.0` must be explicit in the launch command so the
experiment does not silently depend on package defaults.

Because VAE changes the model architecture, A1-VAE starts from its own seed-1000
random initialization rather than loading the deterministic A1 checkpoint.

Its policy-specific training arguments are:

```text
--policy.type=act
--policy.use_vae=true
--policy.latent_dim=32
--policy.kl_weight=10.0
```

## Launchers

Add two readable public scripts:

```text
scripts/experiments/act/a1_observation_equal.sh
scripts/experiments/act/a1_vae.sh
```

Each script must contain its actual `lerobot-train` command construction instead of
delegating training to another opaque launcher. Both accept:

```text
--steps=N
--output-dir=DIR
--print-command
-h / --help
```

Requirements:

- Default to 20,000 steps and a timestamped directory when `--output-dir` is absent.
- Require `--steps` to be a canonical positive integer; step-zero checkpoint creation
  is outside this experiment.
- Reject an existing output directory.
- Validate the local dataset before launch.
- Use a non-blocking per-output launch lock.
- Print the dataset episodes, policy type, loss behavior, VAE settings, training
  horizon, inference default and resolved output directory.
- `--print-command` performs validation and prints a shell-readable complete command
  without creating the output directory or starting training.
- Set `log_freq=min(steps, 100)` and `save_freq=steps`.

Default output roots:

```text
outputs/act/a1_observation_equal/
outputs/act/a1_vae/
```

## Evaluation Contract

Use the existing offline and LIBERO closed-loop evaluators without adding evaluator
behavior.

The first formal comparison fixes:

```text
suite=libero_object
task_id=8
n_episodes=20
n_action_steps=5
temporal ensemble disabled
perturbation=none
seed=1000
same ordered initialization set
```

Temporal Ensemble is excluded from this first comparison because it would add a
second inference-time variable.

Required reported fields:

- task suite
- dataset size
- observation keys
- action space
- action horizon
- control frequency
- train loss
- open-loop action error, including gripper error
- closed-loop success rate
- latency
- failure cases
- mean gripper toggles
- action jerk
- VAE KLD for A1-VAE

Primary interpretation:

- Compare A1-ObsEqual with A1 to test the terminal-loss hypothesis.
- Compare A1-VAE with A1 to test the VAE hypothesis.
- Do not attribute improvement to either change unless the corresponding controlled
  comparison improves closed-loop release behavior under the fixed evaluation
  contract.

## Experiment Guide

Add:

```text
reports/2026-07-26-ACT末尾样本等权与VAE实验指南.md
```

It must follow `reports/AGENTS.md` and contain only:

- experiment goals and hypotheses;
- the three-way control matrix;
- complete copy-and-run training commands;
- complete copy-and-run Offline and Closed-loop evaluation commands;
- result locations;
- acceptance criteria and a blank results table.

Plans and expected outcomes must be labeled as unverified until runs complete.

## Tests

### Policy tests

Use deterministic tensors to prove:

1. A full-horizon sample and a one-valid-action sample receive equal
   observation-level weight.
2. Changing padded target or predicted values does not change the loss.
3. The loss dictionary contains the expected fields.
4. The registered config and policy names are `act_obs_equal`.
5. `use_vae=true` is rejected.
6. A batch row with zero valid actions is rejected.

The numeric loss oracle must use unequal action-dimension errors and unequal valid
horizons so it distinguishes the specified sequence of reductions—mean over action
dimensions, valid-step mean per observation, then batch mean—from global
valid-element normalization.

### Launcher tests

For both scripts, verify:

1. The printed command contains the fixed dataset, episodes and shared A1 settings.
2. A1-ObsEqual discovers `robot_practice.policies.act_obs_equal`, uses
   `policy.type=act_obs_equal`, and fixes `use_vae=false`.
3. A1-VAE uses `policy.type=act`, `use_vae=true`, `latent_dim=32`, and
   `kl_weight=10.0`.
4. Invalid, duplicate and unknown arguments fail clearly.
5. Explicit output directories and short training runs resolve the intended log and
   save frequencies.

### Verification

Run the focused policy and launcher tests, Ruff on changed Python files, shell syntax
checks, and `--print-command` smoke checks for both launchers. Do not start a
20,000-step training run as part of implementation verification.
