# PushT Persistent Executed Action Trajectory Design

**Date:** 2026-07-29

## Goal

Make the complete sequence of action targets actually executed by one PushT closed-loop rollout
visible for the entire Rerun video timeline, while keeping it independently toggleable.

## Scope

This change applies to the existing complete-rollout visualization used by Development and E2
task Recordings. It does not change rollout execution, action selection, metrics, Parquet data,
or the per-policy-seed Recording directory layout.

## Design

The existing `/rollout/complete_action_targets` entity remains the canonical full action-target
trajectory. `log_rollout_to_stream` converts all executed absolute action targets to image
coordinates when simulator frames are present, connects them as one orange `LineStrips2D`, and
logs the entity as static.

Static logging makes the complete trajectory available from the first video frame through the
last frame without duplicating the same geometry at every timestamp. The existing Blueprint
continues to list `/rollout/complete_action_targets`, so users can independently show or hide it
through the Rerun Entity Tree.

The per-frame `/rollout/current_action_target` point remains temporal and unchanged.

## Compatibility

No new entity duplicates the existing action-target path. Existing consumers and Blueprint
paths remain valid. Previously generated RRD files are immutable and do not gain the new static
semantics; regenerating a rollout Recording is required to observe the change.

## Validation

- Unit tests verify that the complete action-target line is logged once as a static entity.
- Unit tests verify that the per-frame current action target remains temporal.
- A small closed-loop Recording is regenerated and checked with `rerun rrd verify`.
- The E2 Guide states that existing formal RRD artifacts predate the visualization-only change
  and must be regenerated if the persistent trajectory is required in those files.
