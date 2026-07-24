# LIBERO Closed-Loop Multi-Episode Rerun Recording Design

**Date:** 2026-07-24
**Status:** Approved for implementation

## Goal

Write one closed-loop `rollout.rrd` that contains one independent Rerun Recording for
each visualized LIBERO rollout episode. Opening the file must show readable episode
names in Rerun's Recording list and allow one-click switching between complete,
synchronized episode visualizations.

The change affects visualization organization only. Simulator rollout, policy inference,
metrics, Parquet rows, videos, seeds, initialization states, and CLI argument semantics
remain unchanged.

## Selected Approach

Use the same publication pattern as offline multi-episode visualization:

1. create one explicitly routed `rerun.RecordingStream` for every visualized episode;
2. log that episode only to its own stream;
3. flush and disconnect every stream;
4. merge the temporary RRD files into a verified staging RRD;
5. atomically publish the staging file as `rollout.rrd`.

Entity-prefix-only separation is rejected because it does not provide Rerun Recording
selection. Publishing one final RRD per episode is rejected because it breaks the
existing one-evaluation/one-primary-RRD workflow.

## Recording Identity and Timeline

Each visualized rollout receives a unique recording ID and a readable final name:

```text
Episode 000 | seed=1000 | init=0 | success
Episode 001 | seed=1001 | init=1 | failure
```

The stream may start with the name before the outcome suffix is known, then update the
recording name after the episode terminates.

Every Recording uses its episode-local control step as the `global_step` Rerun timeline,
starting at zero. The aggregate `global_step` retained in `rollouts.parquet` remains
continuous across the complete evaluation and does not change.

Each Recording owns an independent `LiberoTrajectoryHistory`. Robot pose, measured path,
command, response, action plan, images, state curves, reward, terminal flags, and latency
are therefore isolated by episode.

## Selection Semantics

Existing arguments retain their meaning:

- `--n-episodes=N` runs N independent simulator rollouts and includes all of them in
  metrics and `rollouts.parquet`;
- `--num-visualized-episodes=M` creates per-step Rerun data, video, and an independent
  Recording for the leading M rollout episodes;
- `M=0` creates one named summary Recording containing aggregate metrics but no per-step
  visualization.

The evaluator continues to reject `M > N`.

## Output and Publication

The public output layout remains:

```text
<output_dir>/
├── rollout.rrd
├── rollouts.parquet
├── metrics.json
├── eval_info.json
├── run_config.json
└── videos/
```

Temporary episode recordings live under `<output_dir>/.rerun_recordings/` while the
evaluation is incomplete. The final RRD is published only after:

1. rollout, metrics, JSON, Parquet, and video generation succeed;
2. aggregate metrics are logged to every Recording;
3. every stream flushes and disconnects;
4. `rerun rrd merge` succeeds;
5. `rerun rrd verify` succeeds.

The verified staging file is atomically renamed to `rollout.rrd`. Temporary files are
then removed and the incomplete marker is cleared.

If recording cleanup, merge, verification, or pre-publication finalization fails, the
final `rollout.rrd` is not published and temporary RRDs remain available for diagnosis.

## Implementation Boundaries

`rerun_visualization.py` gains a LIBERO-specific explicit RecordingStream initializer
that applies the existing LIBERO static context and blueprint. The existing merge and
verify helper is reused.

`libero_act.py` owns the per-episode recording lifecycle and routes each visualized step
to the corresponding stream. It must no longer initialize one module-global LIBERO
recording before the episode loop.

The existing `log_libero_step()` data semantics and entity paths remain unchanged. Its
timeline value becomes episode-local when called from closed-loop evaluation.

## Verification

Automated tests cover:

1. LIBERO RecordingStream creation with explicit ID, name, static context, and blueprint;
2. one unique Recording per visualized episode;
3. readable final names containing episode, seed, init state, and outcome;
4. episode-local timelines beginning at zero;
5. independent trajectory history;
6. aggregate metrics logged to every Recording;
7. summary-only behavior for zero visualized episodes;
8. merge, verify, atomic publication, and failure cleanup;
9. unchanged metrics, Parquet rows, videos, seeds, and init-state sequencing.

An integration smoke run uses two one-step episodes with both visualized, verifies the
merged RRD, and confirms that it contains two Recording stores with the expected names.

