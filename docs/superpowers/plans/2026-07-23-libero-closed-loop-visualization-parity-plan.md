# LIBERO Closed-Loop Visualization Parity Implementation Plan

**Date:** 2026-07-23

## Goal

Implement the approved closed-loop visualization design without changing offline semantics
or adding model inference.

## Task 1: Capture the real ACT plan

Files:

- `src/robot_practice/evaluation/common.py`
- `tests/test_evaluation_common.py`

Work:

1. Extend the existing model-query timer to retain a detached policy-space chunk.
2. Add a per-episode plan tracker for standard queue and Temporal Ensemble modes.
3. Validate the selected policy-space action against the active queue.
4. Expose query status, plan index, queued actions, discarded actions, and Temporal Ensemble
   new-chunk state.
5. Test `chunk_size=20`, `n_action_steps=5`, reset, mismatch, and no-extra-query behavior.

## Task 2: Add measured-joint Panda forward kinematics

Files:

- `src/robot_practice/evaluation/libero_kinematics.py`
- `tests/test_libero_kinematics.py`

Work:

1. Reuse the verified LIBERO Panda body model and display frame.
2. Add a method that accepts seven measured arm joints and two measured gripper joints.
3. Return measured-joint geometry without IK residual or reconstruction claims.
4. Keep the existing offline IK API and behavior unchanged.

## Task 3: Record complete simulator transitions

Files:

- `src/robot_practice/evaluation/libero_act.py`
- `tests/test_evaluation_interfaces.py`
- `tests/test_evaluation_integration.py`

Work:

1. Extract current and next 8D state plus seven measured joint positions.
2. Cache prepared observations so each simulator observation is preprocessed once.
3. Postprocess captured chunks into controller space without mutating policy output.
4. Preserve the terminal transition's post-action image, state, and joints.
5. Extend Parquet schema with state, joints, query flag, plan index, and query-only chunk.
6. Keep image buffers only in Rerun/video outputs.

## Task 4: Build the closed-loop Rerun experience

Files:

- `src/robot_practice/evaluation/rerun_visualization.py`
- `tests/test_evaluation_rerun.py`

Work:

1. Add closed-loop trajectory history and measured-Panda logging.
2. Add command, measured-response, executable-queue, discarded-tail, and Temporal Ensemble
   new-chunk geometry.
3. Add policy-input, calibrated spatial, and post-action image views.
4. Add seven action series and eight simulator-state series with selectable legends.
5. Add closed-loop semantics text and retain reward, success, latency, and metrics tabs.
6. Preserve offline paths, colors, labels, and blueprints.

## Task 5: Integrate calibration and lifecycle

Files:

- `src/robot_practice/evaluation/libero_act.py`
- `src/robot_practice/evaluation/rerun_visualization.py`
- related tests

Work:

1. Resolve fixed agentview calibration before rollout.
2. Initialize closed-loop static 3D context.
3. Reset trajectory and plan state at each episode.
4. Verify terminal data is not replaced by autoreset data.
5. Retain `.incomplete` behavior on contract failures.

## Task 6: Documentation and verification

Files:

- `scripts/experiments/act/eval_libero.sh`
- `reports/act_a_sanity.md`
- `README.md`

Work:

1. Document the expanded closed-loop Rerun output and exact time semantics.
2. Run:

```bash
conda run --no-capture-output -n robot-practice pytest -q
conda run --no-capture-output -n robot-practice python -m compileall -q src tests
```

3. Run a one-step real LIBERO smoke evaluation and verify the generated RRD.
4. Confirm generated outputs remain ignored.
