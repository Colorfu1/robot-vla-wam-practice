# LIBERO Multi-Recording Visualization Design

**Date:** 2026-07-23
**Status:** Approved for implementation

## 1. Goal

Evaluate and retain five LIBERO initial scenes, then open one `rollout.rrd` whose Rerun
Recording list lets the user switch between those scenes with a mouse click.

Each selected scene must retain the existing synchronized views:

- policy input cameras;
- post-action simulator rendering;
- measured robot skeleton and end-effector trajectory;
- executed command, measured response, and ACT plan;
- action, state, reward, success, and latency plots;
- metric documentation.

## 2. Existing Behavior

The evaluator already supports:

```bash
--n-episodes=5
--num-visualized-episodes=5
```

It saves all five rollouts to `rollouts.parquet`, per-episode and aggregate metrics to
`metrics.json`, and five MP4 files. However, all visualized episodes are currently written
into one Rerun Recording on a single `global_step` timeline. The viewer has no native
scene selector; the user must scrub across episode boundaries.

For the A-Sanity task, episodes 823, 826, 834, 840, and 841 are the first five
demonstrations of dataset task index 29. A five-episode closed-loop run uses LIBERO
`init_state_id` values 0 through 4.

## 3. User Interaction

The final output remains:

```text
<output_dir>/
├── rollout.rrd
├── rollouts.parquet
├── metrics.json
├── eval_info.json
├── run_config.json
└── videos/
    ├── episode_000.mp4
    ├── episode_001.mp4
    ├── episode_002.mp4
    ├── episode_003.mp4
    └── episode_004.mp4
```

Opening `rollout.rrd` loads one Rerun application with one Recording per visualized
episode. The left Recording list shows readable names:

```text
Scene 1 · init_state 0 · SUCCESS
Scene 2 · init_state 1 · FAILURE
...
```

Clicking a name changes every view to that scene. Each Recording has an independent
timeline starting at step zero. Layout persistence remains shared through the existing
`robot_practice_act_libero` application ID.

## 4. Recording Architecture

### 4.1 Per-episode streams

Create a distinct `rerun.RecordingStream` for each visualized episode:

- common application ID: `robot_practice_act_libero`;
- unique UUID recording ID;
- initial recording name containing the one-based scene number and zero-based
  `init_state_id`;
- one temporary `.rrd` sink per episode;
- the existing LIBERO blueprint sent as default with `make_active=False`;
- the existing static calibration and semantic context logged to every stream.

Every operation is explicitly bound to its episode stream. This includes timeline updates,
step logs, static context, recording-name updates, metric summaries, and
`send_blueprint(..., recording=stream)`. Multi-recording code must not use module-level
`rr.log`, `rr.set_time`, or an unbound `rr.send_blueprint`, because the global stream is
not a valid routing mechanism when several streams coexist.

For a positive visualization count, only visualized episodes create Rerun streams.
Non-visualized episodes still contribute to Parquet and metrics exactly as before. A zero
visualization count retains the existing summary-only RRD behavior through one recording
with no step data.

### 4.2 Per-scene timelines

Call the existing step logger with the episode-local `step_index` as the Rerun timeline
value. `rollouts.parquet` continues to store both `global_step` and `step_index`.

Episode-local trajectory history is reset at each scene boundary. No geometry, command, or
plan state may leak between Recordings.

### 4.3 Final names and metrics

At episode completion, update the Recording name with `SUCCESS` or `FAILURE`.

After aggregate metrics are computed, write the aggregate summary to every visualized
Recording so the existing Metrics tab remains available regardless of the selected scene.
The active scene identity is already visible in the Recording name and context scalars.

### 4.4 Merge

After every requested rollout and every metric summary write succeeds:

1. flush and disconnect every Recording stream;
2. merge their temporary `.rrd` files with the installed `rerun rrd merge` command;
3. write the merge to a temporary final path;
4. verify the merged file with `rerun rrd verify`;
5. atomically rename it to `<output_dir>/rollout.rrd`;
6. delete per-episode temporary files only after successful publication.

The merged file contains multiple Recording stores and their shared application blueprint.
The existing viewer launch path remains `<output_dir>/rollout.rrd`.

Stream ownership is exception-safe. Every created stream is registered immediately and is
flushed/disconnected from a `finally` path if rollout, visualization logging, metric
writing, merge, verification, or publication raises. Merge is never attempted after an
incomplete rollout or summary write. Temporary files are retained on every failure so
buffered recordings remain available for diagnosis, and no stream handle is leaked.

## 5. Edge Cases and Compatibility

- `--num-visualized-episodes=0` retains the existing summary-only `rollout.rrd`; it creates
  no per-episode step Recording or MP4.
- One visualized episode still produces a valid one-Recording `rollout.rrd`.
- `--num-visualized-episodes` may be smaller than `--n-episodes`; only the leading selected
  episodes appear in Rerun.
- `metrics.json`, `eval_info.json`, `rollouts.parquet`, MP4 naming, and aggregate metric
  calculations remain unchanged.
- Existing old `rollout.rrd` files remain readable but retain their continuous timeline.
- If Rerun CLI merge or verification fails, the evaluation remains marked incomplete and
  temporary per-episode files are retained for diagnosis.
- Failures before merge follow the same rule: all streams are closed, the final
  `rollout.rrd` is not published, and temporary episode RRDs remain.
- A missing `rerun` executable is an evaluation error when visualization was requested,
  because producing the required multi-Recording artifact depends on the merge command.
- Output overwrite safety continues to apply to the whole evaluation directory.

## 6. Verification

Automated tests cover:

1. deterministic creation of one stream per visualized episode;
2. unique recording IDs and readable names;
3. episode-local timeline values starting at zero;
4. explicit stream binding for timeline, dynamic logs, static logs, blueprint, name, and
   metric summary operations;
5. each temporary RRD contains only its own episode;
6. no trajectory or plan state leakage between scenes;
7. aggregate metrics logged to every visualized Recording;
8. preservation of summary-only RRD behavior when visualization count is zero;
9. merge-after-success and merge-before-publish ordering;
10. preservation of temporary files on rollout, logging, merge, or verification failure;
11. exception-safe flush/disconnect with no leaked stream handles;
12. cleanup after successful atomic publication;
13. unchanged Parquet, JSON, and MP4 behavior.

Integration verification runs a five-scene, short-step evaluation and requires:

- five recordings visible in `rerun rrd print`;
- distinct readable Recording names;
- a valid final `rollout.rrd`;
- five episode rows in metrics;
- five MP4 files;
- scene timelines beginning at zero.

Manual acceptance opens the merged file with:

```bash
rerun --persist-state <output_dir>/rollout.rrd
```

and confirms that clicking each left-side Recording switches the complete synchronized
visualization.
