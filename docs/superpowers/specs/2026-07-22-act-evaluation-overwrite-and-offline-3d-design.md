# ACT Evaluation Overwrite and Offline 3D Comparison Design

## Status

Approved in conversation on 2026-07-22. This document extends the existing ACT evaluation
designs. It changes output-directory handling for both evaluators and the Rerun presentation
for offline evaluation only.

## Goal

Make repeated evaluations convenient without weakening the default output safety, and make
the offline recording visibly distinguish recorded demonstrations from policy predictions.

The offline recording will show recorded camera frames and the recorded end-effector path
alongside a local 3D preview of the predicted action chunk. The preview is explicitly not a
simulator rollout and must never be labelled as actual execution.

## Command Interface

Both evaluators add an optional boolean flag:

```text
--overwrite    Replace an existing evaluator output directory.
```

Without `--overwrite`, an existing `--output-dir` continues to raise `FileExistsError`.
With `--overwrite`, the evaluator may replace the exact resolved output directory only when
it is recognizably an evaluator result: it is empty, contains `.incomplete`, or contains
`run_config.json`. A regular file, symbolic link, or unrelated non-empty directory is
rejected. Automatically timestamped output paths remain exclusive and do not need this flag.

`--overwrite` requires an explicit `--output-dir`. Before resolving the path, the evaluator
uses `lstat` semantics to reject both valid and dangling symbolic links at the requested
leaf. An `.incomplete` marker is accepted only when its contents exactly match the
evaluator's marker format. A `run_config.json` is accepted only after strict JSON parsing and
validation of its evaluator schema version and mode; the mode must match the new run.
The marker and run-config sentinels are inspected with `lstat` and must themselves be regular,
non-symbolic-link files; valid and dangling sentinel links are never followed.

Every run using an explicit output path holds an exclusive advisory lock in the parent
directory for its complete evaluation lifetime. The stable sibling lock path is
`.<output-name>.evaluation.lock`, so deleting and recreating the result directory cannot
change the locked inode. An overwrite attempt fails while another evaluator owns that lock.
The lock file may remain as harmless coordination metadata after release.

The shell launchers document and forward the flag unchanged.

## Offline Visualization Semantics

Offline evaluation is teacher-forced. Every model query receives an observation from the
demonstration dataset; predicted actions are not applied to LIBERO. Therefore:

- camera images are labelled `Demonstration - recorded images`;
- the blue 3D path is labelled `Demonstration (recorded)`;
- the orange path is labelled `Prediction (not executed)`;
- a visible note explains that the orange path is a command-space preview rather than a
  physics or simulator trajectory.

The demonstration path comes from the first three values of `observation.state`, which are
the absolute LIBERO end-effector XYZ position in metres. It grows through the current frame
while the recording is played.

The prediction preview is rebuilt at every displayed frame. It starts at the current
demonstration end-effector position and cumulatively integrates the valid XYZ translation
commands in the predicted action chunk:

```text
p[0] = observation.state[:3]
p[k + 1] = p[k] + 0.05 * clip(predicted_chunk[k, :3], -1, 1)
```

The state is taken from the dataset batch before policy preprocessing. The action chunk is
taken after the checkpoint postprocessor restores the dataset/controller action space. XYZ
commands are clipped to the OSC controller input range before applying the configured
`0.05 m` maximum translation. The preview communicates controller targets, not realized
motion: contacts, dynamics, controller convergence, workspace clipping, and later
observations are not simulated.

The dataset exposes end-effector pose and gripper state rather than robot joint angles, so
this change does not invent or render a full robot-arm linkage.

## Rerun Layout and Entity Contract

The offline blueprint contains:

1. recorded demonstration camera views;
2. one 3D view comparing the growing recorded path with the current prediction preview;
3. action, action-error, latency, and summary views already produced by the evaluator;
4. a short semantics/legend view.

Entity paths use explicit namespaces:

```text
demonstration/cameras/<camera>
demonstration/trajectory/eef_recorded
demonstration/trajectory/eef_current
demonstration/context/episode_index
demonstration/context/frame_index
demonstration/actions/<dimension>
prediction/not_executed/chunk_preview
prediction/not_executed/chunk_endpoint
prediction/not_executed/actions/<dimension>
comparison/action_error/<dimension>
runtime/batch_id
runtime/latency/model_query_ms
documentation/offline_semantics
summary/metrics
```

The recorded path is blue and the prediction preview is orange. Point labels and panel names
repeat the distinction so the meaning is still clear when one part of the layout is hidden.
Only offline entities are renamed. Closed-loop camera, executed-action, rollout, latency, and
summary paths keep their existing names and blueprint contract.

Only selected visualization episodes log images and 3D entities, matching the existing
`--num-visualized-episodes` behavior. Trajectory history resets at each episode boundary.
The action preview respects `action_is_pad` and stops at the last valid chunk step.

## Data Flow

`offline_act.py` passes the current state, complete target/predicted chunks, and valid-horizon
mask to the visualization helper. The helper owns shape validation, command integration,
colors, labels, and Rerun entity logging. Evaluation metrics and prediction semantics are
unchanged.

The existing predictions Parquet schema remains stable; this visualization does not require
new persisted columns because recorded state is already recoverable from the selected
dataset and the recording contains the rendered geometry.

## Failure and Completion Behavior

An approved existing output directory is removed immediately before the new result directory
and `.incomplete` marker are created. If evaluation later fails, the replacement remains
marked incomplete, preserving the existing atomic-completion convention. No attempt is made
to merge old and new files.

## Verification

Tests cover:

- existing outputs still fail without `--overwrite`;
- recognized evaluator outputs and empty directories can be replaced with `--overwrite`;
- files, symbolic links, and unrelated non-empty directories cannot be overwritten;
- damaged or forged `run_config.json` files and incorrect marker contents are rejected;
- valid and dangling symbolic links used as marker or run-config sentinels are rejected;
- active evaluator locks prevent concurrent overwrite, and `--overwrite` without an explicit
  output directory is rejected;
- locks are released after both successful completion and evaluation exceptions;
- a run config whose mode does not match the requested evaluator is rejected;
- both command parsers expose the flag;
- missing overwrite targets are created normally;
- predicted XYZ chunks are clipped, integrated with the 0.05 m scale, and truncated by the
  valid mask;
- all-padding chunks, invalid shapes, and non-finite state/actions are handled explicitly;
- batch sizes greater than one preserve ordering and episode trajectory history resets
  correctly;
- offline code passes raw dataset state and postprocessed predicted actions to visualization;
- offline Rerun entities and panel names explicitly distinguish demonstration from
  prediction;
- tests enumerate the complete offline entity-path contract and assert that all existing
  closed-loop entity paths remain unchanged;
- a small real Rerun recording is non-empty and readable;
- the existing lightweight evaluation integration tests still pass.

## Non-Goals

This change does not run predicted actions in LIBERO, calculate closed-loop success from an
offline recording, render robot joints that are absent from the dataset, or alter ACT
training, metrics, checkpoints, and dataset caching.
