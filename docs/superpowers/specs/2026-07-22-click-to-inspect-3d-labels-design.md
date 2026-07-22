# Offline 3D Click-to-Inspect Label Design

## Status

Approved in conversation on 2026-07-22. This is a presentation-only addendum to
`2026-07-22-offline-panda-skeleton-and-action-series-design.md`.

## Goal

Keep the offline 3D view unobstructed while retaining enough metadata to identify robot
joints, trajectories, and endpoints on demand.

## Interaction Contract

The following entities retain their existing `labels` data but set `show_labels=False`:

- robot-base point;
- IK-reconstructed Panda skeleton;
- Panda joint points;
- recorded demonstration trajectory;
- current recorded end-effector point;
- predicted, not-executed action-chunk path;
- predicted chunk endpoint.

Clicking a point or entity in Rerun exposes its retained label and other metadata in the
selection/inspection UI. This design does not require a custom viewer, popup, or duplicate
transparent hit targets.

The three display-frame arrows remain the only entities with `show_labels=True`. Their
`X`, `Y Forward`, and `Z Up` labels stay visible because they define the normalized frame
used to interpret the arm pose.

No geometry, coordinate transforms, IK behavior, action plots, metrics, or output schemas
change.

## Verification

Tests assert that:

- the three axis labels remain visible;
- every non-axis 3D entity with label metadata sets `show_labels=False`;
- joint, base, demonstration, and prediction labels are still present for inspection;
- a generated Rerun recording remains valid.
