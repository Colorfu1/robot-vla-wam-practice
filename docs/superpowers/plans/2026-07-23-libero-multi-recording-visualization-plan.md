# LIBERO Multi-Recording Visualization Implementation Plan

**Date:** 2026-07-23
**Design:** `docs/superpowers/specs/2026-07-23-libero-multi-recording-visualization-design.md`

## Goal

Produce one closed-loop `rollout.rrd` containing one independently selectable Rerun
Recording per visualized LIBERO scene. Preserve the existing JSON, Parquet, MP4, metrics,
layout-persistence, and zero-visualization behavior.

## Tasks

1. Add focused tests for explicit `RecordingStream` creation, file sink ordering,
   recording names, static context, blueprint routing, and disconnect behavior.
2. Add tested helpers for merging temporary RRDs, verifying the merged artifact, and
   atomically publishing the final `rollout.rrd`.
3. Refactor the LIBERO evaluator so each visualized episode owns its stream and trajectory
   history, uses episode-local Rerun time, receives a final success/failure name, and gets
   the aggregate metric summary.
4. Make stream cleanup exception-safe and preserve temporary recordings on every failure;
   remove them only after successful final publication.
5. Keep a summary-only Recording when `--num-visualized-episodes=0`.
6. Update the launch-script documentation with the five-scene command and the Recording
   selector behavior.
7. Run targeted tests, the full test suite, RRD verification, and a short five-scene real
   LIBERO rollout. Inspect the merged Recording names and timelines before completion.

## Acceptance

- `--n-episodes=5 --num-visualized-episodes=5` writes five MP4s and one valid
  `rollout.rrd`.
- Rerun's left Recording list contains five readable scene names.
- Clicking a scene switches the complete synchronized visualization.
- Every scene timeline starts at zero and contains only that episode.
- Failed runs retain per-scene temporary RRDs and do not publish a final RRD.
- Existing result schemas and metric calculations remain unchanged.
