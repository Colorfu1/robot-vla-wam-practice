# Rerun Metrics Tables Design

## Status

Approved in conversation on 2026-07-22. This is a presentation-only change to the existing
Rerun `summary/metrics` document.

## Goal

Make one offline ACT evaluation's aggregate results readable with one click on
`Offline signals -> Metrics`, without opening `metrics.json` or leaving Rerun.

## Presentation Contract

The existing `TextDocumentView` and `summary/metrics` entity path remain unchanged. For
`metrics["mode"] == "offline"`, the Markdown document contains two tables.

The first table is ordered exactly as follows:

| Display row | Aggregate key | Format |
|---|---|---|
| Episodes | `num_episodes` | integer |
| Frames | `num_frames` | integer |
| Model queries | `num_model_queries` | integer |
| Batch size | `batch_size` | integer |
| First-action MAE | `first_action_mae` | 4 decimal places |
| Delta-action MAE | `delta_action_mae` | 5 decimal places |
| Action TV | `teacher_forced_action_tv` | 5 decimal places |
| Action Jerk | `teacher_forced_action_jerk` | 5 decimal places |
| Inference latency p50 | `inference_latency_p50_ms` | 2 decimal places plus `ms` |
| Inference latency p95 | `inference_latency_p95_ms` | 2 decimal places plus `ms` |

The second table maps the seven values in `per_dimension_mae` to the fixed ACT action order
`x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`; each value uses 4 decimal places.

The document title identifies it as offline evaluation metrics. `chunk_mae_by_horizon`,
task metadata, dataset metadata, and other raw JSON fields are not expanded in this compact
view. They remain available in `metrics.json`.

## Compatibility and Invalid Data

`metrics_markdown()` is shared with the LIBERO closed-loop evaluator. Non-offline modes keep
the current generic aggregate rendering and are not converted to the offline-specific
tables.

The offline formatter expects the evaluator's existing metrics schema. A missing scalar or
an invalid scalar value renders as `N/A` instead of causing an otherwise successful
evaluation to fail while writing its Rerun summary. If `per_dimension_mae` is not a
seven-element numeric sequence, the per-dimension section renders one explicit
`Per-dimension MAE unavailable` message rather than pairing values with incorrect labels.

## Data Flow

No metric computation changes:

```text
compute_offline_metrics()
    -> metrics dictionary
    -> metrics.json (unchanged full data)
    -> metrics_markdown() (compact tables)
    -> summary/metrics TextDocument
    -> existing Rerun Metrics tab
```

Existing `.rrd` files are immutable artifacts. Users must rerun offline evaluation, using
`--overwrite` when reusing an output directory, before the new table appears.

## Verification

Tests cover:

- exact main-table row order and formatting;
- exact seven-dimension order and formatting;
- absence of the 20-element `chunk_mae_by_horizon` expansion;
- `N/A` rendering for a missing or invalid scalar;
- explicit unavailable text for malformed per-dimension input;
- preservation of the existing non-offline generic rendering;
- a valid Rerun recording containing the generated Markdown table document.
