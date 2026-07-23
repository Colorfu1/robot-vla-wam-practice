# Offline Multi-Episode Rerun Recording Implementation Plan

**Date:** 2026-07-23
**Design:** `docs/superpowers/specs/2026-07-23-offline-multi-episode-recording-design.md`
**Status:** Completed
**Protected file baseline:** `src/robot_practice/evaluation/libero_act.py`
SHA-256 `8ee42b5bacece3141bd857afa81b0cdf1f61d1aafef93f99a23ada5430ca6f26`

## Tasks

1. Add failing tests for explicit offline `RecordingStream` setup, unique episode routing,
   independent timelines/state, summary-only mode, stream cleanup, and atomic RRD merge.
2. Add narrowly scoped Rerun helpers for an explicitly bound offline stream and verified
   multi-Recording publication while preserving the existing single global helpers.
3. Introduce an offline episode-recording state bundle and route every visualized batch row
   by dataset `episode_index`; keep inference, metrics, and Parquet row generation intact.
4. Flush and disconnect every registered stream before merge; preserve temporary files on
   pre-publication failure and mark any post-publication cleanup failure incomplete.
5. Update `eval_offline.sh` documentation with the five-episode command and left Recording
   selector behavior.
6. Run focused tests, the full suite, one/zero-visualization compatibility checks, and a
   complete five-episode evaluation. Verify/inspect the resulting RRD.
7. Recompute the protected `libero_act.py` hash, inspect the privacy-sensitive diff, mark
   the design implemented, and commit only the intended offline changes.
