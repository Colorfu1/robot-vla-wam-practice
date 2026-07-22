# ACT A-Sanity Experiment Guide Design

## Status

Approved in conversation on 2026-07-22. This document defines the structure and
verification contract for a living experiment guide at `reports/act_a_sanity.md`.

## Goal

Create one authoritative, executable guide for the ACT A-Sanity experiment. It must
consolidate the experiment purpose, fixed task and data contract, exact before/after
workflow, pass/fail criteria, artifact locations, troubleshooting rules, and a reusable
result record. A reader should be able to run the experiment without reconstructing the
procedure from old progress reports or design documents.

The guide is both a runbook and a long-lived detailed result record. Its procedure remains
stable near the top, while dated results are appended at the end. Every formal, failed, or
invalid experiment must also receive the required summary entry in
`reports/experiment_log.md`; the two records link to each other rather than competing for
ownership.

## Audience and Scope

The primary reader is a repository user running the experiment locally in the
`robot-practice` Conda environment. The guide assumes basic shell familiarity but does
not assume knowledge of the repository's implementation history.

The guide covers one strict paired experiment:

1. create a zero-update B0 checkpoint;
2. evaluate B0 offline and closed loop;
3. initialize a 20,000-update run from that exact B0 checkpoint;
4. evaluate the trained checkpoint with identical inputs;
5. compare the results against explicit gates.

It does not describe A0/A1/A2/A3 as a complete action-chunking study, train on arbitrary
episodes, automatically choose a recent checkpoint, or claim that offline prediction
measures task success.

## Canonical File and Structure

Create `reports/act_a_sanity.md` with these sections:

1. experiment purpose and non-goals;
2. fixed task, dataset, observation, action, policy, and training contract;
3. formal pass/fail gates;
4. minimal startup preparation;
5. B0 checkpoint creation;
6. B0 offline and closed-loop evaluation;
7. 20,000-step training from the same B0;
8. trained-checkpoint offline and closed-loop evaluation;
9. B0-versus-trained comparison procedure and table;
10. output paths and visualization commands;
11. failure handling and rerun rules;
12. a dated experiment-record template and appended formal runs;
13. links to implementation and deeper technical designs.

The guide may link to older design documents for implementation detail, but it is the
canonical source for current commands. Historical documents are not rewritten merely
because their command examples predate the generic LIBERO evaluator.

## Fixed Experiment Contract

The guide records the following immutable A-Sanity definition:

| Field | Value |
|---|---|
| Task suite | LIBERO Object |
| Simulator task ID | 8 |
| Language | `pick up the chocolate pudding and place it in the basket` |
| Dataset | `lerobot/libero` |
| Local dataset root | `datasets/lerobot/lerobot/libero` |
| Dataset task index | 29 |
| Episodes | `823, 826, 834, 840, 841` |
| Dataset size | 5 episodes, 859 frames |
| State | 8D end-effector position, axis-angle, and gripper qpos |
| Cameras | external agent view and wrist view, 256 x 256 RGB |
| Action | 7D relative translation, relative rotation, and gripper command |
| Control frequency | 10 Hz |
| ACT chunk size | 20 |
| Executed action steps | 5 |
| VAE | disabled |
| Batch size | 8 |
| Seed | 1000 |
| Training updates | 20,000 |
| W&B and Hub upload | disabled |

B0 is described precisely as a zero-optimizer-update initialization checkpoint. Its
ResNet18 visual backbone uses ImageNet weights while the remaining ACT layers use their
normal seeded initialization. It is not called a fully random policy.

## Executable Workflow

Every command is written from the repository root and uses the repository-owned shell
launchers. The guide uses explicit variables so checkpoint lineage is visible:

```bash
b0_run="outputs/act/a_sanity/<B0_TIMESTAMP>"
b0_policy="${b0_run}/checkpoints/000000/pretrained_model"

trained_run="outputs/act/a_sanity/<TRAIN_TIMESTAMP>"
trained_policy="${trained_run}/checkpoints/020000/pretrained_model"
```

The B0 operation is:

```bash
scripts/experiments/act/a_sanity.sh --steps=0
```

The positive-step operation must name the exact B0 checkpoint:

```bash
scripts/experiments/act/a_sanity.sh \
  --steps=20000 \
  --policy-path="${b0_policy}"
```

Omitting `--policy-path` is documented as a separate fresh initialization and therefore
not a valid paired B0-versus-trained experiment.

Both offline evaluations explicitly pass the five dataset-native episode IDs because the
generic evaluator now defaults to all demonstrations for a task:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --episodes 823 826 834 840 841 \
  --num-visualized-episodes=1
```

Both closed-loop evaluations use the same formal settings:

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000
```

Formal closed-loop commands omit `--max-steps`. Offline commands omit `--max-frames`.
The guide explains that visualization volume does not change which frames or episodes
contribute to metrics.

These five IDs are combined-dataset-global IDs, not task-cache-local IDs. A formal run
always passes the fixed `--dataset-root`; it must fail instead of falling back to downloaded
or generated task-local data.

## Minimal Startup Preparation

At the user's request, the canonical startup section contains only what later commands
actually depend on:

- enter the repository root and activate `robot-practice` while preserving
  `PYTHONNOUSERSITE=1`;
- define the fixed dataset root and fail early if `meta/info.json` is absent;
- create one timestamped ignored record/log directory;
- define the Bash `run_logged` helper that preserves the launcher's exit status across
  `tee`.

The canonical path does not run `doctor.sh` or `quickstart_info.sh`, capture Git status or
diffs, recompute dataset task/frame mappings, or hash metadata, Parquet, and video files.
Those checks add audit detail but are not required to execute A-Sanity. The fixed task,
dataset root, and episode selection remain explicit in the training and evaluation
commands, whose loaders reject missing or incompatible data.

None of the excluded checks is moved to an appendix or another preparatory subsection.

## B0 and Paired-Training Provenance

After creating B0, the operator validates `initialization_manifest.json`: status is
`complete`, `initialization_step=0`, `optimizer_updates=0`, the fixed dataset/task fields
match, and the manifest's `checkpoint.model_sha256` equals a fresh SHA-256 of
`checkpoints/000000/pretrained_model/model.safetensors`.

The current LeRobot `train_config.json` does not retain the source checkpoint path. Paired
lineage is therefore established operationally by recording the verified B0 manifest and
model hash, saving the exact `--print-command` output, and retaining the training log line
`Initialization checkpoint: <canonical B0 path>`. The B0 model hash is checked again after
training to prove the named input checkpoint was not changed. A missing or contradictory
lineage artifact makes the experiment invalid rather than formal.

## Training-Log Contract

LeRobot logs a loss average every 100 updates but the current A-Sanity checkpoint does not
embed loss history. The canonical 20,000-step command therefore creates a unique ignored
path under `outputs/act/a_sanity/logs/` and invokes the launcher through a copyable Bash
wrapper with `set -o pipefail` and `2>&1 | tee "${train_log}"`. It captures
`${PIPESTATUS[0]}` immediately after the pipeline and writes that launcher status next to
the log. The guide records the exact log and status-file paths.

The step-100 `loss` is the average over updates 1 through 100. The step-20,000 `loss` is
the average over updates 19,901 through 20,000. These two logged values define the formal
loss-reduction gate. If the log is missing or either value cannot be parsed, the loss gate
is not evaluable and the experiment cannot receive `PASS`.

The guide includes a small read-only extraction command or snippet and asks the operator
to record both raw values, their source log, and the computed reduction.

## Formal Pass/Fail Gates

A formal A-Sanity run receives `PASS` if and only if all four numeric gates are evaluable
and pass:

1. `loss_at_20000 <= 0.2 * loss_at_100`;
2. `trained.first_action_mae <= 0.8 * b0.first_action_mae` on exactly the five
   training episodes;
3. `trained.delta_action_mae <= b0.delta_action_mae` on the same frames;
4. trained closed-loop `success_rate` is strictly greater than B0 success rate under
   the same `seed=1000`, 20 episodes, and `init_state_id=0..19` sequence.

If all inputs and metrics are valid but one or more numeric gates fail, the result is
`FAIL`. Missing metrics, missing B0 lineage, a mismatched fixed dataset-root/task/episode
configuration in experiment artifacts, smoke evaluation, incomplete artifacts,
checkpoint-task `mismatch`/`unknown`, or any other configuration mismatch produces
`INVALID/INCOMPLETE`; it can never be reported as `PASS`.

The guide also requires visual inspection of the saved Rerun/MP4 outputs for obvious
trajectory improvement, but this diagnostic does not substitute for a failed numeric
gate.

The closed-loop comparison is explicitly labeled a proxy. The evaluator controls an
identical standard LIBERO initial-state sequence, but the repository has not established
a one-to-one mapping between those 20 states and the five selected dataset episodes. The
guide must not describe it as exact rollout from each training demonstration's initial
state.

`mismatch` or `unknown` checkpoint-task assessment invalidates a result as formal
A-Sanity even though the generic evaluator permits inference to continue. A structural
checkpoint incompatibility is already a hard evaluator failure.

## Results and Required Fields

Training loss has its own table because B0 has no training loss. It records `loss@100`,
`loss@20000`, and reduction percentage
`(loss@100 - loss@20000) / loss@100 * 100`. A zero or missing `loss@100` makes the gate
invalid.

Checkpoint comparison tables use exact `metrics.json` aggregate keys:

- offline: `first_action_mae`, `delta_action_mae`, `teacher_forced_action_tv`,
  `teacher_forced_action_jerk`, `inference_latency_p50_ms`, and
  `inference_latency_p95_ms`;
- closed loop: `success_rate`, `action_tv`, `action_jerk`, `mean_gripper_toggles`,
  `policy_step_latency_p50_ms`, `policy_step_latency_p95_ms`,
  `model_query_latency_p50_ms`, and `model_query_latency_p95_ms`.

Each row states its units and uses the aggregate value, not a selected episode. For
lower-is-better metrics, absolute change is `trained - B0` and relative change is
`(trained - B0) / B0 * 100`. A zero or `null` B0 makes relative change `N/A` without
hiding the raw values. Success-rate change is reported in percentage points. TV, jerk,
gripper, and latency rows are diagnostics unless one of the four gates names them.

Each dated formal record contains every field required by the repository instructions:

- task suite;
- dataset size;
- observation keys;
- action space;
- action horizon;
- control frequency;
- train loss;
- open-loop action error;
- closed-loop success rate;
- latency;
- failure cases.

It additionally records the Git commit, B0 run, trained run, exact checkpoint paths,
training log, offline result directories, closed-loop result directories, commands or
parameter summaries, gate outcomes, and qualitative observations. Missing values are
written as `N/A` with a reason rather than omitted.

Every dated attempt also adds an entry to `reports/experiment_log.md` containing all of
the repository-required fields and a link to its detailed section in this guide. Failed or
invalid attempts are logged as such; they are not silently omitted.

## Output and Recovery Rules

The guide documents the timestamped training outputs and task-namespaced evaluator
outputs. It never instructs the user to select the newest directory automatically.
Operators copy the paths printed by each command into variables and the dated record.

Failed and interrupted runs are retained. The canonical `tee` wrapper preserves the
launcher's status rather than accidentally reporting only `tee`'s status. The record
captures the command, exit status,
training log, `.incomplete` evaluation path when present, and failure case. Rerunning means
starting another timestamped operation; it does not overwrite or silently resume an old
run.

The guide includes common diagnoses for missing local data, incomplete checkpoints,
checkpoint-task warnings, existing output directories, absent loss logs, smoke-only
evaluation flags, and failure to open Rerun. It distinguishes viewer-launch warnings from
evaluation failure because saved `.rrd` files can be opened separately.

## Verification

Writing the guide must not launch the 20,000-step experiment or download new task data.
Verification consists of:

1. validating every shell block syntactically where possible;
2. running `a_sanity.sh --print-command` for non-mutating training argument validation;
3. checking both evaluation launchers' `--help` output against the documented flags;
4. confirming Section 4 contains only repository/environment setup, the fixed dataset-root
   existence check, the timestamped record/log directory, and the exit-code-preserving
   Bash helper; it must contain no doctor/quickstart invocation, Git snapshot/diff,
   dataset scan/mapping verification, or dataset-file hashing anywhere, including an
   appendix;
5. checking that paired evaluation commands differ only in checkpoint/output identities;
6. checking the B0-manifest validation snippet on local inputs;
7. checking that the detailed template and `experiment_log.md` entry contain every
   required experiment field;
8. checking Markdown links and repository-relative paths;
9. confirming that no generated output, data, checkpoint, video, or recording is staged.

Existing local outputs may be used to verify path shapes, but they are not registered as
formal results unless their provenance, complete loss history, paired configuration, and
all required metrics satisfy this guide.

## Acceptance Criteria

- `reports/act_a_sanity.md` exists and is understandable without reading historical specs.
- It states the fixed task/data/policy contract and all four formal gates.
- It contains copyable commands for B0, both B0 evaluations, paired 20k training, and both
  trained evaluations.
- Every offline A-Sanity command explicitly selects episodes 823, 826, 834, 840, and 841.
- Every formal offline command pins `datasets/lerobot/lerobot/libero` and cannot fall back
  to generated task-local episode namespaces.
- Every formal closed-loop command uses identical 20-episode, seed-1000 settings and no
  smoke bound.
- It requires and explains persistent training-log capture.
- It preserves the launcher exit status across `tee` and treats missing gate data as
  invalid, never passed.
- Its startup preparation contains only the four approved items and does not move
  doctor/quickstart, Git snapshot/diff, dataset scan/mapping, or dataset-file hashing into
  an appendix.
- It pins the fixed dataset root and episode selection and validates exact B0 lineage.
- It provides exact-key comparison tables, a dated detailed record template, and the
  required corresponding `reports/experiment_log.md` entry.
- Its caveats prevent offline error from being presented as success rate or standard
  LIBERO initial states from being presented as proven training-episode initial states.
- All non-mutating verification passes without adding generated artifacts to Git.
