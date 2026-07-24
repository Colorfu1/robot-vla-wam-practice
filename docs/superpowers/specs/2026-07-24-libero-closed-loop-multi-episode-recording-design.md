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

Each visualized rollout receives a unique recording ID and exactly one final name, sent
after the episode terminates:

```text
Episode 000 | seed=1000 | init=0 | success
Episode 001 | seed=1001 | init=1 | failure:time_limit
Episode 002 | seed=1002 | init=2 | failure:environment_done
```

The suffix is `success` when success was observed. Otherwise it is
`failure:<termination_reason>`, using the evaluator's final `time_limit` or
`environment_done` reason. The evaluator does not emit an initial provisional name, so
the merged archive contains only the selector-visible final name.

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
- `M=0` creates one summary Recording named exactly
  `LIBERO closed-loop evaluation summary`, containing aggregate metrics but no per-step
  visualization. Its name is sent once before metric logging and stream cleanup.

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
evaluation is incomplete. The lifecycle collection is updated immediately after each
stream is created. Cleanup is idempotent: it attempts every unattempted flush before
attempting every unattempted disconnect, even when an earlier stream fails.

If rollout, stream initialization, name logging, metric logging, flush, or disconnect
raises, the original evaluation exception is preserved; cleanup errors are logged
without replacing it. When no earlier exception exists, the first lifecycle error is
raised only after all streams received both cleanup attempts. No merge or publication
occurs after a lifecycle error.

After all required work succeeds:

1. finish rollout, metrics, JSON, Parquet, and video generation;
2. send each visualized episode's one final Recording name;
3. log aggregate metrics to every Recording;
4. flush and disconnect every stream without error;
5. merge the temporary RRD files into a staging file;
6. verify the staging file;
7. atomically rename the verified staging file to `rollout.rrd`;
8. write the final successful `run_config.json`;
9. remove the temporary recording directory;
10. remove `.incomplete` last.

The atomic RRD rename is the publication boundary. A failure before it leaves no final
`rollout.rrd`, retains `.incomplete`, and retains temporary RRDs and any staging file for
diagnosis. A failure after it leaves the valid published RRD in place and retains
`.incomplete`; the temporary directory may be present or partially cleaned, and
`run_config.json` may still carry its incomplete status if its final write failed. The
incomplete marker is authoritative.

The final archive is inspected with `rerun.dataframe.load_archive()`, not merely
`rerun rrd verify`. Verification requires:

- `num_recordings()` equals the requested visualization count, or one for summary-only
  mode;
- all Recording IDs are unique;
- every Recording's static `RecordingInfo:name` property exactly matches its required
  final name;
- each episode Recording contains only its own episode index;
- each episode Recording's `global_step` timeline begins at zero and contains no values
  from another episode.

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

1. LIBERO RecordingStream creation with explicit ID, static context, and blueprint;
2. one unique Recording per visualized episode;
3. one exact final name containing episode, seed, init state, and outcome;
4. episode-local timelines beginning at zero;
5. independent trajectory history;
6. aggregate metrics logged to every Recording;
7. summary-only behavior for zero visualized episodes, including the exact
   `LIBERO closed-loop evaluation summary` name;
8. merge, archive inspection, atomic publication, and failure cleanup;
9. partial stream initialization and multiple flush/disconnect failures without skipped
   cleanup attempts or exception masking;
10. unchanged metrics, Parquet rows, videos, seeds, and init-state sequencing.

An integration smoke run uses two one-step episodes with both visualized, verifies the
merged RRD, loads it with `rerun.dataframe.load_archive()`, and confirms that it contains
two uniquely identified Recording stores with the exact expected names and isolated
episode-local timelines.
