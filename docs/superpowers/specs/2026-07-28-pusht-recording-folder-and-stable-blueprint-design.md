# PushT Recording Folder and Stable Blueprint Design

**Date:** 2026-07-28
**Status:** Approved for implementation

## Goal

Make the selected BCChunk and Diffusion PushT development rollouts reliably switchable
inside two Rerun Viewer windows. Each environment seed is stored as one physical RRD,
while every Recording uses the same entity schema and the same Blueprint definition.

The final workflow must provide:

- one Recording directory for BCChunk and one for Diffusion;
- one independently transferable `env_<seed>.rrd` per model/seed pair;
- twenty selector-visible Recordings per Viewer when the directory is opened;
- identical `env_1000..env_1019` comparison keys in both windows;
- no black simulator view after switching between Recordings;
- a launcher that accepts a directory instead of requiring twenty paths to be typed.

This change only reorganizes Rerun output. It does not alter checkpoint selection,
checkpoint loading order, environment or policy seeds, scheduler settings, rollout
actions, rewards, success classification, Parquet rows, videos, metrics, or the
deterministic pairing between models.

## Diagnosed Failure

The current direct generator writes one Recording per seed and merges the task files by
model, but every Recording has a seed-specific dynamic root:

```text
/rollouts/development_checkpoint_comparison/bc_chunk/env_1000/policy_00/**
/rollouts/development_checkpoint_comparison/bc_chunk/env_1001/policy_00/**
```

The Viewer keeps the active Blueprint while the selected Recording changes. A view
bound to the first path therefore asks the second Recording for an entity that does not
exist and renders black. Inspection of the affected archive confirmed that the selected
Recording still contained all 301 simulator frames; the failure was path lookup, not
missing image data or an invalid timeline.

Physical RRD separation alone does not solve this because all files are loaded into one
Viewer and share an application-level layout. The Recording ID is already the data
namespace, so the seed must not also be encoded into the path used by the shared view.

## Public Output Contract

The completed comparison output is:

```text
<output_dir>/
├── comparison.json
├── rollouts.parquet
├── replans.parquet
├── recordings/
│   ├── bc_chunk/
│   │   ├── env_1000.rrd
│   │   ├── env_1001.rrd
│   │   └── ... env_1019.rrd
│   └── diffusion/
│       ├── env_1000.rrd
│       ├── env_1001.rrd
│       └── ... env_1019.rrd
├── videos/
├── metrics.json
├── metrics.md
└── run_manifest.json
```

There are exactly forty public RRD files. Every file contains exactly one data
Recording. The previous merged files, `recordings/bc_chunk.rrd` and
`recordings/diffusion.rrd`, are no longer generated.

The comparison's existing outer `AtomicOutputDirectory` remains the publication
boundary. Per-seed files are written and validated inside the incomplete output tree;
the complete directory becomes public only after every model/seed result and every
non-Rerun artifact succeeds. A failed run must not leave a partially complete final
output directory.

## Recording and Application Identity

Corresponding BCChunk and Diffusion files use the same Recording ID:

```text
env_1000
env_1001
...
env_1019
```

The existing model-specific application IDs remain unchanged:

```text
robot_practice_pusht_bc_chunk
robot_practice_pusht_diffusion
```

No `_v2` application ID is introduced. Application identity distinguishes the two
model windows; Recording identity distinguishes seeds inside each window.

Selector-visible Recording names remain:

```text
Task env=1000 | BCChunk | success
Task env=1000 | Diffusion | failure
```

The existing canonical `/task/*` static metadata is retained:

```text
/task/model
/task/environment_seed
/task/policy_seed
/task/success
/task/max_reward
/task/checkpoint_step
/task/checkpoint_id
```

## Stable Entity Contract

Every development task Recording uses the exact same dynamic entity root:

```text
/rollout
```

The complete dynamic schema is:

```text
/rollout/simulator
/rollout/complete_agent_path
/rollout/complete_block_path
/rollout/complete_action_targets
/rollout/current_agent
/rollout/current_block
/rollout/current_action_target
/rollout/reward
/rollout/success
/rollout/rollout_index
/rollout/metadata
```

Missing optional action entities are allowed only when the source rollout has no action
samples. Seed, model, condition, and policy identity belong in the Recording ID,
application ID, and `/task/*` metadata rather than in the view-facing entity root.

The `global_frame` timeline is local to each Recording, starts at zero, and is strictly
increasing. Rerun's SDK-managed `/__properties` and Blueprint stores are internal
metadata and are excluded from the application entity-path restriction.

Other PushT utilities that intentionally write multiple rollouts into one aggregate
Recording retain their existing unique path behavior. The fixed `/rollout/*` contract
applies to independently namespaced development task Recordings.

## Shared Blueprint

The view composition in this section is superseded by
[`2026-07-28-pusht-compact-rerun-layout-design.md`](2026-07-28-pusht-compact-rerun-layout-design.md).
The stable `/rollout/*` path, application identity, and Blueprint persistence contracts
remain unchanged.

Add one PushT development Blueprint builder and send the same Blueprint definition to
every per-seed Recording after its file sink is configured. Its view queries refer only
to the canonical `/rollout/*` and `/task/*` paths; no model, seed, condition, policy, or
filesystem path appears in the Blueprint.

The Blueprint preserves the existing PushT views:

- simulator image with the measured agent, block, action target, and complete paths;
- reward over the local timeline;
- rollout metadata and task context.

It is embedded with:

```python
make_active=False
make_default=True
```

This matches the repository's existing layout-persistence contract: a new Viewer gets
the project layout, while a user-edited persisted layout is not overwritten on every
open.

Because the application IDs stay unchanged, a machine that persisted the broken
seed-specific PushT layout may need one **Reset Blueprint** action after opening the
regenerated files. The embedded default then uses `/rollout/*`, and subsequent
Recording switches remain stable. This is a one-time local Viewer-state reset, not an
RRD migration. Automated and manual clean-state verification use an isolated Rerun
state directory so stale local UI state cannot hide generator defects.

## Generation Lifecycle

The development comparison writes each rollout directly to its public relative path
inside the outer incomplete output:

```text
recordings/<model>/env_<seed>.rrd
```

For each completed rollout:

1. construct its validated task metadata;
2. create an explicitly routed `RecordingStream` using the existing model application
   ID and `env_<seed>` Recording ID;
3. configure the file sink;
4. send the selector-visible Recording name and canonical static task context;
5. send the shared default Blueprint;
6. log the rollout under `/rollout/*` with local `global_frame=0..N`;
7. flush and disconnect the stream;
8. verify that the nonempty file contains exactly one valid Recording;
9. release the rollout's rendered frames before evaluating the next case.

BCChunk is still evaluated completely before Diffusion, so the two selected policies
are never resident on the GPU simultaneously. No model inference is repeated for
visualization postprocessing, and no merge, compact, copy, or cross-device rename step
is required for the RRD files.

After all forty rollouts complete, paired validation requires both model directories to
contain the same ordered seed set before the outer output directory can be committed.

## Metadata Contract

Each model result in `comparison.json` records:

```text
rerun_recording_path: recordings/<model>/env_<seed>.rrd
rerun_entity_path: /rollout
```

The relative path is the authoritative link from a comparison case to its visualization
file. `run_manifest.json` replaces the merged-archive map with:

```json
{
  "rerun_recording_directories": {
    "bc_chunk": "recordings/bc_chunk",
    "diffusion": "recordings/diffusion"
  },
  "rerun_recordings_per_model": 20
}
```

All paths remain relative to the comparison output. No username, absolute path, session
ID, cache location, or machine-specific identifier is written.

## Directory Viewer Launcher

Rerun accepts multiple `.rrd` positional paths but does not treat a directory argument
as “load every RRD below this directory.” Add:

```text
scripts/experiments/pusht/view_development_recordings.sh
```

Its public interface is:

```text
--recordings-dir=DIR
--port=PORT
--print-command
```

The launcher:

1. resolves a repository-relative directory safely;
2. requires exactly the expected `env_1000.rrd..env_1019.rrd` files and rejects
   unexpected RRD filenames;
3. expands them into a lexicographically sorted Bash array;
4. invokes `robot-rerun --persist-state --detach-process --port=<port>` with all twenty
   individual paths;
5. preserves each argument as a separate shell word;
6. supports `--print-command` for inspection without opening a Viewer;
7. rejects duplicate, empty, malformed, and unknown arguments.

The documented two-window workflow is:

```bash
bash scripts/experiments/pusht/view_development_recordings.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/development_comparison/recordings/bc_chunk \
  --port=9876

bash scripts/experiments/pusht/view_development_recordings.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/development_comparison/recordings/diffusion \
  --port=9877
```

The global `robot-rerun` wrapper is not changed to recursively interpret arbitrary
directories. Directory expansion stays scoped to this fixed experiment contract.

## Legacy Output and Cleanup

The current generated comparison output may be deleted and regenerated after the code
change. No backup, conversion, or migration path is required.

Remove the legacy development splitter launcher and the splitter-only DataFrame,
compaction, merge, and publication paths. Preserve reusable metadata creation,
per-task writer, stream cleanup, and semantic validation code. The removed merged
format must not remain documented as a supported output.

Generated data under `outputs/` remains untracked and is not committed.

## Verification

Automated tests cover:

1. stable application ID, Recording ID, Recording name, filename, and relative-path
   generation for both models and all development seeds;
2. direct per-rollout writing under canonical `/rollout/*`;
3. exactly one data Recording per physical file;
4. identical entity schemas and Blueprint contents across different seeds and models;
5. zero-based, strictly increasing, task-local `global_frame`;
6. presence and schema of every required `/task/*` component;
7. encoded simulator images plus measured path, action target, reward, and metadata;
8. semantic rejection of wrong application/Recording IDs, dynamic seed-specific paths,
   unrelated entities, missing simulator frames, malformed timelines, and mismatched
   task metadata;
9. paired-directory validation for missing, duplicate, extra, or asymmetric seeds;
10. comparison JSON and run-manifest relative-path contracts;
11. directory launcher ordering, quoting, argument validation, exact file-set checks,
    ports, and `--print-command`;
12. absence of the legacy merged archives and aggregate `recording.rrd`;
13. unchanged comparison categories, metrics, Parquet rows, videos, model loading order,
    seeds, policy seed, and DDPM100 configuration;
14. the full PushT test suite.

A bounded real-RRD integration test writes at least two different seed Recordings,
verifies each with `rerun rrd verify`, loads each through the Rerun DataFrame API, and
confirms:

- one data Recording per file;
- distinct Recording IDs;
- identical canonical dynamic entity paths;
- identical embedded Blueprint view queries;
- nonempty simulator frames and task-local timelines.

Manual acceptance regenerates the comparison output, opens all BC files in one Viewer
and all Diffusion files in another, selects at least `env_1000`, `env_1001`, and a later
seed in both windows, and confirms that the simulator, trajectories, reward, and metadata
all update without a black view. On a machine carrying the previous persisted layout,
the test first performs one Reset Blueprint and then repeats the switches.

## Alternatives Rejected

- **Keep seed-specific entity paths:** reproduces the confirmed Blueprint lookup failure.
- **Give every seed a different Blueprint:** makes layout persistence and cross-task
  comparison depend on dynamic UI state.
- **Introduce `_v2` application IDs:** hides incompatible local state by fragmenting
  application identity instead of fixing the data/view contract.
- **Keep one merged RRD per model:** remains technically possible, but makes individual
  retry, synchronization, inspection, and physical task isolation harder than the
  approved directory contract.
- **Teach the global launcher to recurse through directories:** broadens behavior outside
  this experiment and makes accidental loading of unrelated RRDs more likely.
- **Retain postprocessing for the newly generated result:** duplicates I/O and preserves
  two output formats after rerunning has been explicitly accepted.
