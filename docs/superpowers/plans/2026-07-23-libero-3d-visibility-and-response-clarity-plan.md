# LIBERO 3D Visibility and Response Clarity Implementation Plan

**Date:** 2026-07-23

## Goal

Implement the approved native Rerun visibility controls and make the measured cyan response
distinguishable from an overlapping orange controller command.

## Task 1: Lock the blueprint and geometry contracts with tests

Files:

- `tests/test_evaluation_rerun.py`

Work:

1. Assert that the Free 3D content query explicitly retains coordinate frame, calibration,
   robot, measured trajectory, command, response, queued plan, discarded plan, and Temporal
   Ensemble plan groups.
2. Assert that `build_libero_blueprint()` requests an expanded Blueprint Panel.
3. Update the closed-loop geometry test to require a cyan arrow origin/vector and a cyan
   endpoint at the measured next EEF.
4. Assert the command and response radii encode the intended visual priority.

## Task 2: Implement visibility and response rendering

Files:

- `src/robot_practice/evaluation/rerun_visualization.py`

Work:

1. Replace the broad plan query with explicit selectable plan roots and retain the fixed
   camera calibration/frustum root.
2. Add an expanded `BlueprintPanel` to the LIBERO blueprint.
3. Reduce the orange command radius.
4. Replace the cyan response line with a thicker arrow from EEF `t` to measured EEF `t+1`.
5. Add the measured cyan response endpoint as a sibling entity.
6. Update in-view semantics to explain the eye toggles and revised response encoding.

## Task 3: Update launch documentation

Files:

- `README.md`
- `scripts/experiments/act/eval_libero.sh`

Work:

1. Describe the native Blueprint Panel visibility controls.
2. State the orange-command and cyan-response meanings without changing experiment
   semantics.

## Task 4: Verify

Run:

```bash
conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_evaluation_rerun.py
conda run --no-capture-output -n robot-practice pytest -q
conda run --no-capture-output -n robot-practice \
  python -m compileall -q src tests
```

Inspect a generated closed-loop RRD and confirm that:

- the Blueprint Panel starts expanded;
- all intended groups have independent eye toggles;
- the measured response is visible as a cyan arrow and endpoint;
- timeline playback and metrics remain unchanged.
