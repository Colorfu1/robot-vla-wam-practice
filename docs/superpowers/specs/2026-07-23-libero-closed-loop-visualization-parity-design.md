# LIBERO Closed-Loop Visualization Parity Design

**Date:** 2026-07-23
**Status:** Reviewed; awaiting final user review

## 1. Goal

Bring the LIBERO closed-loop Rerun result up to the useful visualization level of
the existing ACT offline evaluator while preserving the different meaning of the two
evaluation modes.

The closed-loop recording must show:

- the two simulator observations used as policy input at time `t`;
- the simulator agentview returned after executing the action, at time `t+1`;
- the current Panda skeleton produced from measured simulator joint positions;
- the measured end-effector history;
- the command executed at the current step;
- in standard queue mode, the remaining executable queue and predicted tail that will be
  discarded;
- in Temporal Ensemble mode, the newly predicted chunk without presenting it as an
  execution queue;
- the measured end-effector response after the command;
- selectable action and simulator-state curves;
- reward, success, latency, and aggregate metrics.

The implementation must not introduce an additional model query for visualization.

## 2. Evaluation Semantics

Offline and closed-loop evaluation answer different questions:

- Offline evaluation compares expert and predicted actions under the same recorded
  observation. It measures imitation quality under teacher forcing.
- Closed-loop evaluation measures what happens after policy commands change the
  simulator state. It measures success, robustness, motion quality, and latency.

A demonstration action is not valid frame-aligned ground truth after a closed-loop rollout
diverges from the expert state. The closed-loop viewer therefore does not show a synthetic
GT/Pred comparison and does not load a demonstration by default.

An expert trajectory could be added later as an explicitly unaligned spatial reference,
but that is outside this design.

## 3. Per-Step Time Contract

One Rerun `global_step` represents a complete simulator control transition:

```text
observation[t]
  ├─ policy agentview and eye-in-hand images
  ├─ 8D simulator state
  └─ current EEF and measured-joint Panda
          ↓
ACT select_action()
  ├─ action[t] sent to the controller
  └─ the real predicted chunk, when a model query occurs
          ↓
env.step(action[t])
          ↓
observation[t+1]
  ├─ post-action agentview
  ├─ next 8D simulator state
  ├─ measured EEF response
  └─ reward, success, and done
```

The post-action observation must be logged even when the transition terminates the
episode. This preserves the final success or failure image.

The rollout loop should preprocess each simulator observation once and cache the prepared
observation for the next control step. This avoids duplicate preprocessing and keeps the
model input, visualization, and recorded state consistent.

## 4. Action Plan Capture

### 4.1 No extra inference

An `ActionPlanTracker` extends or composes the existing model-query timer. It wraps the
real `ACTPolicy.predict_action_chunk()` call already made by `select_action()`, records its
returned tensor, and forwards the same tensor unchanged.

Visualization must never call `predict_action_chunk()` independently.

### 4.2 Action domains

`predict_action_chunk()` returns policy-space values. `select_action()` also returns one
policy-space action, after which the checkpoint postprocessor and environment postprocessor
convert it to the controller-space command sent to `env.step()`.

The tracker keeps these domains explicit:

- plan progression and queue consistency are checked in policy space;
- the captured tensor forwarded back to the policy is never modified;
- a detached copy of the full chunk is passed through the same checkpoint and environment
  action processing as executed actions;
- OSC geometry, Rerun action curves, and `rollouts.parquet` use controller-space values.

If the processors only accept a two-dimensional action batch, chunk processing flattens
`[batch, horizon, action_dim]` to `[batch * horizon, action_dim]`, applies the processors
once, and restores the original shape. Tests must prove that processing a chunk entry and
processing the same entry as a selected action produce equal controller-space values.

### 4.3 Standard action-queue mode

For a policy with `chunk_size=20` and `n_action_steps=5`:

- action 0 is sent during the query step;
- actions 1 through 4 are the remaining executable queue;
- actions 5 through 19 are predicted but discarded when the queue is exhausted and the
  policy replans.

The tracker records in policy space:

- whether the current step performed a model query;
- the full captured chunk on query steps;
- the current index within the active plan;
- the current executed action;
- the remaining executable queue;
- the non-executable predicted tail.

The tracker advances exactly once per successful `select_action()` call and resets at every
episode boundary. The corresponding controller-space partitions are derived only after the
queue state has been validated.

### 4.4 Temporal Ensemble mode

Temporal Ensemble queries a chunk every step and combines overlapping predictions into one
executed action. It does not have standard queued/discarded semantics.

In this mode:

- deep orange still represents the final action sent to the simulator;
- the captured chunk is labeled `new prediction chunk`, not `ensemble candidates`;
- only the first entry of the new chunk contributes to the current ensembled action, while
  horizon-aligned entries from older chunks supply the other current candidates;
- this design does not reconstruct or display the complete candidate set or ensemble
  weights;
- `plan_step_index` is null;
- queued and discarded entities are not emitted.

The new prediction chunk is drawn as a translucent violet dashed preview, anchored at the
measured EEF in `observation[t]`. It represents the full prediction conditioned on the
current observation. It is not an executable path: only its first entry participates in the
current ensemble, and its later entries can contribute to later ensembled actions.

## 5. Simulator State and Kinematics

The 8D state contract is:

```text
[eef_x, eef_y, eef_z, axis_angle_x, axis_angle_y, axis_angle_z,
 gripper_left_qpos, gripper_right_qpos]
```

The processed ACT input contains only this 8D state, but the live LIBERO raw observation
also exposes seven measured values at `robot_state.joints.pos`. Closed-loop visualization
uses those measured joints, together with the measured gripper qpos, to run forward
kinematics and draw the Panda. It must label this geometry as measured simulator joints,
not IK reconstruction.

`libero_kinematics.py` should expose a measured-joint forward-kinematics path that reuses the
same verified Panda body names, display frame, and gripper geometry as the offline
reconstructor. Offline evaluation continues to use 8D-state IK and retains all existing
`IK reconstructed` labels.

## 6. Spatial Visualization

The closed-loop spatial recorder reuses the offline display frame, XYZ axes, Panda geometry,
fixed agentview calibration, camera frustum, clickable entities, and label policy.

Closed-loop entity roots are separate from offline roots:

```text
rollout/robot/**
rollout/trajectory/**
rollout/command/**
rollout/response/**
rollout/plan/queued/**
rollout/plan/discarded/**
rollout/plan/new_prediction_chunk/**
```

Color semantics are fixed:

| Visual | Meaning |
|---|---|
| Blue Panda | Current Panda from measured simulator joints |
| Deep-blue solid path | Measured EEF history through `observation[t]` |
| Deep-orange segment | Current command actually sent to `env.step()` |
| Light-orange solid preview | Remaining actions that are still in the executable queue |
| Fainter orange dashed preview | Predicted chunk tail that will not be executed |
| Translucent violet dashed preview | Temporal Ensemble's new prediction chunk, not an execution queue |
| Cyan segment | Measured EEF displacement from state `t` to state `t+1` |

The command and preview geometry use the existing LIBERO OSC display scaling. Their
endpoints are controller command visualizations, not simulated future states.

The current command begins at the measured EEF in `observation[t]`. Because
`observation[t+1]` is available on the same Rerun step, the remaining executable queue
begins at the measured post-action EEF. The discarded tail continues from the end of that
hypothetical queued preview. This keeps the current command and measured response anchored
to real states without presenting future preview endpoints as simulated outcomes.

Only X, Y, and Z axis labels remain permanently visible. Robot joints, trajectory points,
commands, and response segments retain labels for click-to-inspect behavior without
covering the viewport.

## 7. Image Views and Calibration

The top region contains three logical views:

1. **Policy inputs at t**
   - model-facing agentview;
   - model-facing eye-in-hand.
2. **Spatial rollout**
   - physically calibrated agentview overlay;
   - free orbitable 3D view with the fixed agentview camera frustum.
3. **Post-action simulator result t+1**
   - the agentview from the observation returned by `env.step(action[t])`.

The pre-action and post-action agentview images use the same image convention so their
visual difference represents the simulator transition rather than an orientation change.

The calibrated overlay uses a visualization-only physical-camera copy. It must not change
the policy input or checkpoint behavior.

## 8. Charts and Metrics

The lower region contains tabs:

### 8.1 Executed actions

Seven independently selectable series:

```text
x, y, z, roll, pitch, yaw, gripper
```

These are the postprocessed commands sent to the simulator. They are not joint positions
or measured EEF deltas.

### 8.2 Simulator state

Eight independently selectable series:

```text
eef_x, eef_y, eef_z,
axis_angle_x, axis_angle_y, axis_angle_z,
gripper_left_qpos, gripper_right_qpos
```

Action and state series remain in separate tabs because they have different units and
physical meanings.

### 8.3 Remaining tabs

- reward, success-so-far, and done;
- policy-step and model-query latency;
- aggregate closed-loop metrics;
- visualization semantics.

The metrics tab retains success rate, rewards, action TV, action jerk, gripper toggles,
and latency summaries produced by the current evaluator.

## 9. Output Schema

`rollouts.parquet` retains the existing fields and adds:

- `observation_state`: fixed-size list of 8 float32 values;
- `next_observation_state`: fixed-size list of 8 float32 values;
- `observation_joint_positions`: fixed-size list of 7 float32 values;
- `next_observation_joint_positions`: fixed-size list of 7 float32 values;
- `model_queried`: boolean;
- `plan_step_index`: nullable integer;
- `predicted_chunk`: nullable list of fixed-size 7D float32 controller-space actions,
  populated only on model-query steps.

In standard queue mode, the full chunk is stored once on the step that created the active
plan. Queue and discarded views can be reconstructed from the chunk, `n_action_steps`, and
`plan_step_index`. In Temporal Ensemble mode, each step performs a model query and stores
that step's new controller-space prediction chunk; `plan_step_index` remains null.

The Rerun recording remains the source of compressed images and visualization geometry.
Parquet must not duplicate image buffers.

## 10. Failure Handling

- Non-finite or incorrectly shaped state, action, image, or chunk data is a contract error.
  Evaluation stops and retains the `.incomplete` output marker.
- `n_action_steps` greater than the captured chunk length is a contract error.
- A policy-space mismatch between the tracked active plan and the action returned by
  `select_action()` is a contract error.
- Missing, non-finite, or incorrectly shaped measured joint positions are a closed-loop
  observation contract error. The viewer does not silently replace them with IK.
- Fixed agentview calibration must resolve before the rollout starts. A missing or invalid
  calibration stops evaluation rather than producing a misleading overlay.
- Episode changes reset EEF history and action-plan state.

## 11. Code Boundaries

The implementation should keep responsibilities narrow:

- `common.py`: capture and track real ACT model queries, action domains, and plan state;
- `libero_act.py`: prepare current and next observations, execute actions, persist rollout
  rows, and provide closed-loop frame data;
- `rerun_visualization.py`: closed-loop blueprint, charts, spatial entities, and semantics;
- `libero_kinematics.py`: measured-joint forward kinematics plus existing offline IK and
  camera calibration primitives;
- tests: plan tracking, logging contracts, layout, persistence, and integration behavior.

Offline entity paths and semantics remain unchanged.

## 12. Verification

Unit and integration coverage must verify:

- visualization adds no model queries;
- policy-space queue validation and controller-space chunk processing are not mixed;
- controller-space processing of a chunk entry matches processing of the same selected
  action;
- `chunk_size=20`, `n_action_steps=5` produces the correct current, queued, and discarded
  partitions;
- plan indices advance once per executed action and reset per episode;
- Temporal Ensemble logs only the new prediction chunk, uses no queue semantics, and leaves
  `plan_step_index` null;
- measured simulator joints produce the closed-loop Panda while offline IK behavior remains
  unchanged;
- terminal transitions include the post-action image, state, and measured joints;
- a terminal-transition integration test uses a unique terminal observation and a different
  reset observation, proving that persisted post-action data comes from the terminal
  transition rather than autoreset;
- EEF history never crosses episode boundaries;
- action and state chart names, colors, order, and click selection;
- expected spatial entity paths and color meanings;
- parquet schemas and nullable query-only chunks;
- existing offline visualization behavior remains unchanged.

Final verification consists of:

```bash
conda run --no-capture-output -n robot-practice pytest -q
conda run --no-capture-output -n robot-practice python -m compileall -q src tests
```

It also includes a real one-step LIBERO smoke evaluation using `--max-steps=1`, followed by
Rerun recording verification. This smoke run validates wiring only; the explicit
terminated/truncated integration test above owns terminal-observation correctness because a
one-step smoke ends through a synthetic time limit.

## 13. Acceptance Criteria

The feature is complete when:

- a closed-loop Rerun recording visibly contains all three image semantics;
- the current Panda, measured EEF path, command, and measured response are distinguishable
  and time-aligned;
- standard queue mode distinguishes the executable queue from the discarded chunk tail;
- Temporal Ensemble mode shows a violet new-prediction preview without queued/discarded
  claims;
- all action and state dimensions can be selected independently;
- no GT action or demonstration is presented as frame-aligned closed-loop truth;
- visualization causes no additional inference;
- terminal images are preserved;
- generated outputs remain untracked and existing offline recordings continue to pass
  regression tests.
