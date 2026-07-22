# Offline 3D Click-to-Inspect Label Design

## Status

Approved in conversation on 2026-07-22. This is a presentation-only addendum to
`2026-07-22-offline-panda-skeleton-and-action-series-design.md`.

## Goal

Keep the offline 3D view unobstructed while retaining enough metadata to identify robot
joints, trajectories, and endpoints on demand.

## Interaction Contract

The following exact entities retain their existing `labels` component but set their
`show_labels` component to `False`:

| Entity path | Logged archetype | Retained label data |
|---|---|---|
| `display_frame/base` | `Points3D` | robot-base label |
| `demonstration/robot/skeleton` | `LineStrips3D` | normal or stale IK label |
| `demonstration/robot/joints` | `Points3D` | one Panda body/joint label per point |
| `demonstration/trajectory/eef_recorded` | `LineStrips3D` | demonstration label |
| `demonstration/trajectory/eef_current` | `Points3D` | recorded EEF label |
| `prediction/not_executed/chunk_preview` | `LineStrips3D` | not-executed prediction label |
| `prediction/not_executed/chunk_endpoint` | `Points3D` | prediction-endpoint label |

In Rerun 0.26.2, `show_labels=False` hides viewport text; it does not create an on-click
floating label. Clicking one of these points or line strips selects its batch entity and
shows the retained `labels`, `positions` or `strips`, colors, and radii components in the
Selection panel's Data section. A joint can be refined to its individual point instance to
inspect that instance's label. This design does not require a custom viewer, popup, or
duplicate transparent hit targets.

`display_frame/axes` is one `Arrows3D` entity containing three instances and is the sole
entity that remains `show_labels=True`. Its `X`, `Y Forward`, and `Z Up` labels stay visible
because they define the normalized frame used to interpret the arm pose.

`demonstration/robot/gripper` currently has no `labels` or `show_labels` component and
remains unchanged.

No geometry, coordinate transforms, IK behavior, action plots, metrics, or output schemas
change.

## Verification

Tests assert that:

- `display_frame/axes` has exactly the three expected labels and `show_labels=True`;
- each of the seven paths above has its exact label data and `show_labels=False`;
- both the normal and stale skeleton variants retain the correct IK label;
- `demonstration/robot/gripper` remains unlabeled;
- a generated Rerun recording remains valid.

A bounded manual smoke check with Rerun 0.26.2 selects one Panda joint instance and one
line-strip entity, then confirms that their retained labels appear in the Selection panel's
Data section without reintroducing persistent viewport text.
