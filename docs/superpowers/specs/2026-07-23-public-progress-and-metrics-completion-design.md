# Public Progress, Publication Safety, and Metrics Completion Design

## Status

Approved in conversation on 2026-07-23. This design covers three related repository
maintenance tasks: a sanitized daily progress report, persistent Codex publication rules,
and completion of the already-approved compact offline metrics presentation.

## Goals

- Record the 2026-07-23 ACT evaluation and visualization progress in a public-safe report.
- Make summary and publication safety rules mandatory for future Codex CLI work in this
  repository without requiring the user to repeat them.
- Complete the Rerun offline metrics tables specified in
  `2026-07-22-rerun-metrics-tables-design.md`.
- Verify that documentation and tracked changes do not contain private machine details or
  generated ML artifacts before any future upload.

## Non-Goals

- Uploading, pushing, or opening a pull request in this task.
- Publishing datasets, checkpoints, predictions, RRD recordings, videos, logs, or caches.
- Changing metric computation, metric JSON schemas, ACT inference, or evaluation results.
- Retrofitting historical design documents whose commands intentionally describe older
  interfaces.

## Daily Progress Report

Create `reports/2026-07-23-progress.md`. It records only repository-relevant, public-safe
facts:

- the A-Sanity 100-step and 20k checkpoint evaluation comparison;
- the generic offline and closed-loop evaluator workflow;
- Rerun recorded-image, Panda skeleton, command-segment, action-chart, metric-summary, and
  calibrated-agentview improvements;
- the LIBERO LeRobot/OpenVLA image-orientation finding and visualization-only correction;
- verification performed and the next experiment step.

The report uses repository-relative paths and public benchmark identifiers only. It must not
contain user names, absolute home paths, machine names, session/conversation identifiers,
personal note-vault locations, credentials, tokens, private endpoints, raw terminal dumps,
or generated artifact contents. Local output paths are described with generic placeholders
such as `outputs/act/a_sanity/<run>/...` rather than machine-specific locations.

`A-Sanity` is this repository's public experiment label and is safe to use. The report may
publish selected aggregate values derived from `metrics.json`, including dataset/frame
counts, first-action MAE, per-dimension MAE, TV, jerk, and latency. “Generated artifact
contents” excludes these aggregate statistics; it means raw observations, actions,
predictions, media, weights, logs, cache contents, and full machine-specific manifests.

The report distinguishes completed measurements from pending work. It must not claim a
closed-loop success result unless a formal closed-loop evaluation actually completed.

## Persistent Codex Rules

Add a `Summary And Publication Safety` section to the root `AGENTS.md`. Because Codex CLI
loads repository instructions from this file, the rules apply to future work automatically.

The section requires:

1. Every shareable progress report uses repository-relative paths and excludes private
   identifiers, local absolute paths, credentials, private URLs, raw logs, and cache paths.
2. Before commit, upload, or push, inspect `git status`, the exact staged diff, and tracked
   files for likely secrets or personal markers. Resolve findings before publication.
3. Never stage or publish generated ML artifacts covered by `.gitignore`, including data,
   checkpoints, model weights, predictions, RRD files, videos, logs, caches, or experiment
   output directories.
4. Stage only files in the explicitly requested scope. Existing dirty-worktree changes
   remain the user's unless they are deliberately included.
5. Commit and push are separate permissions. Do not push, upload, or create a remote change
   unless the user explicitly asks for that external action.
6. Before a push, report the target remote/branch and re-check the outgoing commit diff for
   sensitive material.
7. Summaries state completed work, verification, remaining work, and measured results without
   copying raw logs or unverified claims.

These rules complement rather than replace the existing `.gitignore` safeguards.

## Offline Metrics Presentation

Implement the existing `2026-07-22-rerun-metrics-tables-design.md` without changing metric
computation or `metrics.json`.

For `metrics["mode"] == "offline"`, `metrics_markdown()` renders:

1. an ordered aggregate table for episode/frame/query counts, batch size, first-action MAE,
   delta-action MAE, Action TV, Action Jerk, and inference latency p50/p95;
2. an ordered per-dimension MAE table for `x`, `y`, `z`, `roll`, `pitch`, `yaw`, and
   `gripper`.

Formatting and missing-data behavior remain exactly as specified in the predecessor design.
Non-offline evaluation keeps the current generic bullet presentation. The existing Rerun
entity `summary/metrics` and Metrics tab stay unchanged.

Formatting accepts only Python or NumPy real numeric values and explicitly rejects booleans,
numeric strings, `NaN`, and positive or negative infinity. Count rows additionally require a
non-negative integral value; a finite fractional count renders `N/A`. Float and latency rows
accept any finite real value and use their specified decimal format. Per-dimension MAE must
be a non-string sequence of exactly seven finite real, non-boolean values. If its container,
length, or any element is invalid, the complete per-dimension table is replaced with
`Per-dimension MAE unavailable`.

## Verification

Automated verification covers:

- exact offline table row order and numeric formatting;
- exact seven-dimension order;
- `N/A` behavior for missing or invalid scalar values;
- explicit unavailable text for malformed per-dimension values;
- preservation of generic closed-loop Markdown;
- the full test suite and Python compilation.

Publication-safety verification covers:

- no absolute home paths, session IDs, credentials, private keys, or personal-vault paths in
  the new report and `AGENTS.md` section;
- no ignored/generated artifacts staged;
- review of the exact diff limited to the intended files.

After tests pass, discover the existing local 100-step and 20k A-Sanity checkpoints under
`outputs/act/a_sanity/<run>/checkpoints/{000100,020000}/pretrained_model`. A candidate is
accepted only when its checkpoint assessment identifies `libero_object` task ID 8,
`pick_up_the_chocolate_pudding_and_place_it_in_the_basket`, and the combined-dataset episode
set `[823, 826, 834, 840, 841]`. These identifiers are the public A-Sanity experiment
contract recorded in `reports/act_a_sanity.md`, which is the authoritative reference for
discovery. Require exactly one candidate for each step; if none or multiple candidates
match, stop rather than guess. Reuse the candidate run's existing `offline_5` output
directory and rebuild with the verified `lerobot_180` visualization convention.

Each rebuilt result must have evaluator status `complete`, no `.incomplete` marker, 859
frames over the five expected episodes, and the resolved convention recorded as
`lerobot_180`. `rerun rrd verify` must succeed; RRD inspection must contain the
`summary/metrics` `TextDocument`, both recorded and calibrated agentview image entities, and
calibration components only on the calibrated entity. Unit tests lock down that the
blueprint's Metrics tab points to `summary/metrics` and that its Markdown contains the two
compact tables. Generated outputs remain ignored and are never committed.
