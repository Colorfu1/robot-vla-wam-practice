# PushT Paired Rerun Recordings Design

**Date:** 2026-07-28
**Status:** Approved for implementation

## Goal

Make the selected BCChunk and Diffusion PushT development rollouts directly comparable
in two independent Rerun Viewer windows:

- one BCChunk archive and one Diffusion archive;
- twenty selector-visible task Recordings in each archive;
- the same stable task ID, `env_1000` through `env_1019`, on both sides;
- episode-local timelines beginning at zero;
- readable Recording names containing model, environment seed, and outcome.

The current remote evaluation is already producing the legacy single-Recording
`recording.rrd`. It must remain usable without rerunning either model. Future evaluation
runs must instead follow the established ACT/LIBERO lifecycle: write one temporary
Recording per visualized rollout, merge verified Recordings into the public archive,
then publish it.

This change reorganizes visualization only. It does not change checkpoint selection,
policy loading order, environment seeds, policy seed, scheduler, rollout actions,
rewards, success classification, Parquet data, videos, or metrics.

## Public Output Contract

The paired archives are:

```text
<output_dir>/
├── comparison.json
├── rollouts.parquet
├── replans.parquet
├── recordings/
│   ├── bc_chunk.rrd
│   └── diffusion.rrd
├── videos/
├── metrics.json
├── metrics.md
└── run_manifest.json
```

The already-running legacy evaluation may additionally contain:

```text
<output_dir>/recording.rrd
```

That file is the input to the compatibility splitter. It remains unchanged after
successful splitting so the original artifact is preserved. Future evaluations write
the two model archives directly and do not need the legacy combined file.

## Recording Identity

Each model archive contains exactly twenty Rerun Recordings. Corresponding BCChunk and
Diffusion tasks use the same Recording ID:

```text
env_1000
env_1001
...
env_1019
```

The archives use different application IDs:

```text
robot_practice_pusht_bc_chunk
robot_practice_pusht_diffusion
```

The legacy single-Recording input has the exact application ID:

```text
robot_practice_pusht_rollouts
```

The model-specific application ID prevents two stores from colliding if both archives
are accidentally loaded into one Viewer, while the shared Recording ID remains the
cross-window comparison key.

Selector-visible names use:

```text
Task env=1000 | BCChunk | success
Task env=1000 | Diffusion | failure
```

Success is the rollout's boolean success result. `comparison.json` remains the
authoritative source for the four paired categories:

- `bc_pass_diffusion_fail`
- `diffusion_pass_bc_fail`
- `both_fail`
- `both_pass`

Each Recording logs the following canonical static task context:

```text
/task/model             TextDocument: "bc_chunk" or "diffusion"
/task/environment_seed  Scalars:      integer-valued seed
/task/policy_seed       Scalars:      integer-valued seed
/task/success           Scalars:      0 or 1
/task/max_reward        Scalars:      finite rollout maximum reward
/task/checkpoint_step   Scalars:      integer-valued training step
/task/checkpoint_id     TextDocument: lowercase SHA256 digest
```

These paths, component types, and value representations are identical for compatibility
splitting and future direct generation.

## Timeline and Entity Contract

Every task Recording uses its own `global_frame` sequence starting at zero. Its dynamic
entities are limited to exactly one subtree for its model and environment seed:

```text
/rollouts/development_checkpoint_comparison/<model>/env_<seed>/policy_00/**
```

The only entities allowed outside that rollout subtree are the canonical static
`/task/*` entities defined above. The entity paths and components under the rollout
subtree remain unchanged so the compatibility splitter and future direct writer produce
the same visualization schema. Incidental SDK timelines such as `log_tick` and
`log_time` are not copied by the compatibility splitter; `global_frame` is the sole
playback timeline.

Images remain Rerun `EncodedImage` payloads. Compatibility splitting must copy the
existing encoded payloads through the Rerun DataFrame API and must not reconstruct data
from MP4 files or re-encode simulator frames.

## Compatibility Splitter for the Current Result

A standalone Python entry point and Bash launcher accept an existing completed
development-comparison output directory. The default operation reads:

```text
<output_dir>/recording.rrd
<output_dir>/comparison.json
```

and atomically creates:

```text
<output_dir>/recordings/bc_chunk.rrd
<output_dir>/recordings/diffusion.rrd
```

The splitter:

1. rejects an output directory carrying `.incomplete`;
2. validates `comparison.json` contains exactly seeds `1000..1019`, both model results,
   and one of the four registered categories for each seed;
3. loads the source archive and requires exactly one legacy Recording with application
   ID `robot_practice_pusht_rollouts`;
4. selects one model/seed entity subtree at a time;
5. requires non-empty data and a finite, monotonically increasing `global_frame`;
6. removes incidental timelines and rebases `global_frame` to zero while retaining its
   Rerun index metadata;
7. writes a temporary task Recording with the registered application ID, stable
   Recording ID, readable name, and static task context;
8. compacts and merges the twenty temporary task files into one model archive;
9. runs `rerun rrd verify` and semantic archive validation;
10. publishes the complete `recordings/` directory through
    `AtomicOutputDirectory`.

The source `recording.rrd` and all other evaluation outputs are read-only inputs.
Existing `recordings/` causes an error unless the user explicitly passes `--overwrite`.
An interrupted split leaves an incomplete staging directory and no falsely complete
public archive.

The splitter is CPU-only and can run either on the remote machine before transfer or on
the local machine after the complete result directory has been synchronized.

## Future Direct Generation

Future development comparisons follow the ACT/LIBERO multi-Recording pattern rather
than writing one aggregate Recording and splitting it afterward.

For every completed rollout:

1. create one explicitly routed temporary `RecordingStream`;
2. assign the model-specific application ID and stable `env_<seed>` Recording ID;
3. send the final Recording name once;
4. log static task context;
5. log the rollout with episode-local `global_frame=0..N`;
6. flush and disconnect the stream;
7. retain its temporary path for final model-level publication.

BCChunk remains evaluated completely before Diffusion, so the selected policies are
never resident on the GPU simultaneously. Temporary Recording creation does not change
this loading order and does not retain rendered frames after the corresponding video
and task RRD have been written.

After all rollouts and comparison outputs succeed, temporary files are compacted and
merged by model, semantically verified, and published as
`recordings/bc_chunk.rrd` and `recordings/diffusion.rrd`. Temporary task files are removed
only after both public archives pass validation.

The compatibility splitter and direct writer share identity, naming, task-context,
archive-validation, and merge helpers. They are two ingestion paths into one public
archive contract, not two independently evolving formats.

## Archive Verification

CLI verification alone is insufficient. Each final archive is also loaded with
`rerun.dataframe.load_archive()` and must satisfy:

- exactly twenty Recordings;
- application ID equals the registered model-specific ID;
- Recording IDs are exactly `env_1000..env_1019` with no duplicates;
- every Recording has exactly one expected selector-visible name;
- every Recording contains exactly its expected model/seed rollout subtree plus the
  canonical `/task/*` static entities, with no other entity paths;
- `global_frame` exists, starts at zero, is monotonic, and contains no values from
  another task;
- simulator images and measured/action/reward components are present;
- every canonical `/task/*` static component exists with the required type and agrees
  with `comparison.json`;
- corresponding BCChunk and Diffusion archives expose the same ordered ID set.

Any failed invariant leaves the archive unpublished and retains recoverable temporary
files under the incomplete staging directory.

## Viewer Workflow

The compatibility command is documented as a copy-paste command against the synchronized
output directory. The two model archives are opened in independent Viewer processes and
must use different ports:

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9876 \
  outputs/pusht_diffusion/pilot/development_comparison/recordings/bc_chunk.rrd

conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9877 \
  outputs/pusht_diffusion/pilot/development_comparison/recordings/diffusion.rrd
```

Selecting the same `env_<seed>` Recording in both windows compares the same deterministic
PushT reset. Passing both paths to one command is not the supported comparison workflow.

The Python arguments, Bash header and usage text, development-comparison guide, benchmark
guide, and PushT guide index must remain synchronized.

## Alternatives Rejected

- **Reconstruct from MP4 and Parquet:** introduces video decode/re-encode, can change
  frames, and duplicates a representation already present in the source RRD.
- **Rerun the current model comparison:** wastes completed simulator inference and is
  unnecessary because the legacy archive contains every required component.
- **Keep one Recording with model/seed entity prefixes:** does not expose task entries in
  Rerun's Recording selector.
- **Open the same combined archive twice and hide entities manually:** depends on
  per-window UI state and does not provide one stable task list per model.
- **Use identical application and Recording IDs in both model archives:** risks store
  collision if both files are loaded into one Viewer.
- **Maintain postprocessing as the permanent generation path:** adds avoidable I/O and
  diverges from the proven ACT/LIBERO per-episode Recording lifecycle.

## Verification

Automated tests cover:

1. stable identity and name generation for both models and all development seeds;
2. a synthetic legacy source with two models and multiple seeds split into two archives;
3. exact encoded-image and trajectory/action/reward component preservation;
4. task-local timeline rebasing without losing Rerun index metadata;
5. semantic archive validation, including matching task ID sets;
6. direct per-rollout task Recording generation using the same public contract;
7. compaction, merge, CLI verification, and atomic publication;
8. rejection of incomplete input, malformed comparison JSON, missing task subtrees,
   duplicate IDs, unexpected application IDs, and existing output without overwrite;
9. merge, verification, flush, disconnect, and cleanup failures without publishing
   partial archives;
10. Bash argument forwarding, duplicate/unknown argument rejection, and documented
    copy-paste commands;
11. unchanged comparison categories, metrics, Parquet rows, videos, model loading order,
    seeds, policy seed, and DDPM100 configuration.

A final integration test creates a small legacy aggregate RRD, performs compatibility
splitting, verifies both archives with the installed Rerun CLI, loads them through the
DataFrame API, and confirms selector-visible names, paired IDs, isolated entity trees,
and zero-based timelines.
