# Rerun Metric Font Compatibility Implementation Plan

**Date:** 2026-07-23

## Goal

Render complete English metric meanings in Rerun 0.26.2 while leaving Chinese
`metric_definitions` unchanged in JSON and terminal output.

## Task 1: Lock the display contract with tests

Files:

- `tests/test_evaluation_rerun.py`

Work:

1. Import the canonical offline and closed-loop metric definition key sets.
2. Require nonempty ASCII Rerun descriptions for every canonical key in both modes.
3. Require offline Markdown to contain its mode-specific English meanings.
4. Require closed-loop Markdown to contain its mode-specific English meanings.
5. Require unknown ASCII descriptions to pass through.
6. Require unknown non-ASCII descriptions to be omitted without removing the key/value.
7. Prove that Markdown generation does not mutate the source payload.

## Task 2: Add mode-specific Rerun descriptions

Files:

- `src/robot_practice/evaluation/rerun_visualization.py`

Work:

1. Add separate offline and closed-loop English description mappings so shared keys such as
   `num_episodes` and `num_model_queries` retain the correct mode semantics.
2. Add one pure description-selection helper:
   - known mode/key uses the corresponding English mapping;
   - unknown key with an ASCII description retains it;
   - unknown key with a non-ASCII description returns no description.
3. Route both compact offline tables and generic closed-loop bullet output through that
   helper.
4. Do not modify the metrics payload, metric calculation, JSON serializer, or terminal
   logger.

## Task 3: Verify code and real output

Run:

```bash
conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_evaluation_rerun.py
conda run --no-capture-output -n robot-practice pytest -q
conda run --no-capture-output -n robot-practice \
  python -m compileall -q src tests
```

Generate a short closed-loop RRD using the existing trained checkpoint, then verify:

- `/summary/metrics` contains the expected English descriptions;
- the Rerun metrics Markdown is ASCII-only;
- `metrics.json` still contains the original Chinese descriptions;
- the RRD passes `rerun rrd verify`.

## Task 4: Finish

1. Mark the design document implemented.
2. Inspect the exact diff and run the repository sensitive-content check.
3. Commit only the design status, implementation plan, source, and tests.
4. Leave generated evaluation output ignored and do not push.
