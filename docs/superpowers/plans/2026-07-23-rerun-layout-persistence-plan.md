# Rerun Layout Persistence Implementation Plan

**Date:** 2026-07-23

## Goal

Allow `rerun --persist-state` to restore user-edited layouts by keeping each generated
blueprint as the application default without forcing it active over persisted state.

## Task 1: Lock the blueprint activation contract with tests

Files:

- `tests/test_evaluation_rerun.py`
- `tests/test_visualize_aloha.py`

Work:

1. Update the shared offline/LIBERO recording test double to accept blueprint activation
   keyword arguments.
2. Require `make_active=False` and `make_default=True`.
3. Update the ALOHA recording test with the same contract.
4. Preserve the existing ordering assertions: initialize, configure the sink or viewer,
   then send the blueprint.
5. Run the focused tests before changing source and confirm they fail because the keyword
   arguments are absent.

## Task 2: Make generated blueprints persistence-friendly

Files:

- `src/robot_practice/evaluation/rerun_visualization.py`
- `src/robot_practice/visualize_aloha.py`

Work:

1. Send the shared offline/LIBERO blueprint with:

   ```python
   make_active=False
   make_default=True
   ```

2. Apply the same flags to the ALOHA blueprint emission.
3. Do not change application IDs, blueprint contents, recording data, timelines, or viewer
   launch behavior.

## Task 3: Verify generated recordings

Run:

```bash
conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_evaluation_rerun.py \
  tests/test_visualize_aloha.py
conda run --no-capture-output -n robot-practice pytest -q
conda run --no-capture-output -n robot-practice \
  python -m compileall -q src tests
```

Generate one small evaluation RRD and verify:

- `rerun rrd verify` succeeds;
- `rerun rrd print` contains
  `make_active: false, make_default: true`;
- the recording still contains the generated default blueprint.

Manual persistence across two viewer sessions remains a human acceptance step because it
requires editing a native Rerun layout and closing the viewer.

## Task 4: Finish

1. Mark the design document implemented.
2. Inspect the exact diff and run the sensitive-content scan.
3. Commit only the design status, plan, source, and tests.
4. Leave generated verification outputs ignored and do not push.

