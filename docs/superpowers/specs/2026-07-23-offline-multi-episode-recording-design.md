# Offline Multi-Episode Rerun Recording Design

**Date:** 2026-07-23
**Status:** Reviewed and implemented

## Goal

Run teacher-forced ACT inference over multiple explicitly selected LIBERO dataset episodes
and write one `prediction.rrd` in which every visualized dataset episode is an independent
Rerun Recording. The viewer's left Recording list must let the user switch between
episodes with one click.

This is strictly an offline-evaluation feature. It must not change
`src/robot_practice/evaluation/libero_act.py`, simulator initialization, or closed-loop
rollout behavior.

## Inputs and Selection

The existing interface remains:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path=<checkpoint> \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --episodes 823 826 834 840 841 \
  --num-visualized-episodes=5 \
  --agentview-image-convention=lerobot_180 \
  --output-dir=<output_dir>
```

`--episodes` selects distinct dataset demonstrations for inference.
`--num-visualized-episodes` selects the leading requested episodes that receive
per-frame Rerun data. Every frame reached by evaluation contributes to metrics and
`predictions.parquet`, whether or not it is visualized. Normally this covers every
selected episode; the existing global `--max-frames` smoke bound may stop before later
episodes and therefore remains explicitly non-formal.

## Output and Interaction

The output layout remains:

```text
<output_dir>/
├── prediction.rrd
├── predictions.parquet
├── metrics.json
└── run_config.json
```

Opening `prediction.rrd` presents readable Recording names such as:

```text
Dataset episode 823
Dataset episode 826
Dataset episode 834
Dataset episode 840
Dataset episode 841
```

Clicking a Recording switches the complete synchronized visualization:

- both recorded camera streams;
- GT and predicted first-action curves;
- reconstructed Panda skeleton;
- recorded EEF history;
- teacher-forced GT/Pred command segments and predicted future preview;
- latency and metric documentation.

Each Recording owns an independent frame timeline beginning at the episode's frame zero.
No trajectory, command, IK fallback pose, or plot history may cross episode boundaries.

## Architecture

### Inference remains shared

The existing dataset selection, ordered DataLoader, batching, ACT
`predict_action_chunk()` call, normalization, metrics, and Parquet row generation remain
unchanged. A batch may cross an episode boundary; each row is routed to the Recording
identified by that row's `episode_index`.

The feature does not perform five separate inference runs and does not execute predicted
actions. It only changes how already-produced offline rows are recorded for visualization.

### One stream and state bundle per episode

Before iteration, create one `rerun.RecordingStream` per visualized episode:

- common application ID `robot_practice_act_offline`, preserving layout persistence;
- unique UUID recording ID;
- readable dataset-episode name;
- one temporary `.rrd` sink;
- the existing offline blueprint as the default blueprint;
- the existing static camera calibration and visualization semantics.

Every Rerun operation is explicitly bound to the row's episode stream. This includes
timeline updates, static logs, frame logs, metric summaries, recording names, and
blueprint delivery.

Each visualized episode has an independent state bundle containing:

- `OfflineTrajectoryHistory`;
- `OfflineCommandHistory`;
- previous reconstructed skeleton pose used only for same-episode IK fallback;
- episode-local visualization frame count.

The shared skeleton reconstructor is reset when ordered iteration enters a new episode.
If future dataset ordering becomes interleaved, evaluation must fail clearly rather than
silently sharing state between episodes.

### Timeline

Use the dataset row's `frame_index` as the Recording timeline sequence. Selected LIBERO
episodes are required to start at frame zero and remain monotonically increasing. This
keeps image, action, skeleton, and 3D views aligned inside each Recording.

`predictions.parquet` retains its existing episode and frame fields; no schema change is
required.

### Metrics

After aggregate offline metrics are computed, log the same aggregate Metrics document to
every visualized Recording. This keeps the Metrics tab available regardless of the active
episode. `metrics.json` remains the authoritative aggregate result.

When `--num-visualized-episodes=0`, retain the existing summary-only RRD behavior through
one named Recording containing metrics but no per-frame data. This summary Recording uses
the same temporary sink, immediate registration, explicit stream routing, flush,
disconnect, merge, verification, and atomic-publication lifecycle as episode Recordings.

## Publication and Failure Handling

All streams are registered immediately after creation and cleaned up from an
exception-safe path. Cleanup attempts every registered stream even if one operation
raises; it first calls `flush()` on every stream, then calls `disconnect()` on every
stream, and reports the first failure only after all cleanup attempts finish.

After inference, metric logging, JSON, and Parquet writes succeed:

1. flush every registered Recording, continuing across failures;
2. disconnect every registered Recording, continuing across failures;
3. abort publication if any flush or disconnect failed;
4. merge the temporary RRDs with the installed `rerun rrd merge` into a staging file in
   `<output_dir>`;
5. verify the staging file with `rerun rrd verify`;
6. write the final successful `run_config.json`;
7. atomically replace `<output_dir>/prediction.rrd` with the verified staging file;
8. remove Recording temporary files;
9. remove `.incomplete` last.

If inference, logging, metric writing, stream cleanup, merge, verification, or
pre-publication finalization fails:

- no final `prediction.rrd` is published;
- the evaluator remains marked incomplete;
- every created stream is disconnected;
- per-episode temporary RRDs are retained for diagnosis.

If the atomic RRD rename succeeds but removing `.incomplete` or subsequent cleanup fails,
retain the published RRD, `.incomplete`, and any temporary episode RRDs not yet removed.
Already deleted temporary files are not recreated. The marker makes the run explicitly
incomplete rather than deleting an already verified artifact. Overwrite safety continues
to treat it as an incomplete evaluator result.

A missing Rerun CLI is an evaluation error because publication requires merge and verify.

## Compatibility

- One visualized episode produces a one-Recording `prediction.rrd`.
- A visualization count smaller than the selected episode count records only the leading
  subset while metrics and Parquet cover every frame reached by evaluation.
- Existing completed RRD files remain readable with their old continuous timeline.
- Metrics, prediction schema, checkpoint loading, dataset caching, image convention,
  action values, and model inference are unchanged.
- `src/robot_practice/evaluation/libero_act.py` remains byte-for-byte unchanged.

## Verification

Automated tests cover:

1. one explicitly routed stream per visualized dataset episode;
2. unique IDs and readable names;
3. row-to-stream routing even when a batch crosses an episode boundary;
4. per-episode timelines beginning at zero and increasing monotonically;
5. independent trajectory, command, and IK fallback state;
6. aggregate metrics logged to every visualized Recording;
7. summary-only behavior for zero visualized episodes;
8. merge, verify, atomic publication, and success cleanup;
9. temporary-file retention and stream cleanup on every failure path;
10. unchanged metrics and Parquet behavior.

Integration verification runs the complete episodes 823, 826, 834, 840, and 841 without
the global `--max-frames` truncation, verifies the merged RRD, inspects five Recording
names, and confirms that
`src/robot_practice/evaluation/libero_act.py` has no implementation diff.

Manual acceptance opens:

```bash
rerun --persist-state <output_dir>/prediction.rrd
```

and confirms that the left Recording list switches all synchronized views between the five
dataset demonstrations.
