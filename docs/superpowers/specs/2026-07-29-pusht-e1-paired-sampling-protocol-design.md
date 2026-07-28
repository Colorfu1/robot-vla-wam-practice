# PushT E1 Paired Sampling Protocol Design

**Date:** 2026-07-29
**Status:** Approved for implementation

## Purpose

PushT E1 currently changes the explicit initial noise while holding the DDPM reverse
generator seed fixed. The resulting action chunks are different but visually and
numerically almost identical. A controlled probe showed that changing the reverse seed
produces much larger output diversity.

E1 must therefore report two distinct questions without conflating them:

1. **E1a — initial-noise-only:** how sensitive is the trained policy to the initial
   diffusion prior when the reverse stochastic path is fixed?
2. **E1b — full-sampling-seed:** how diverse is normal stochastic DDPM sampling when both
   the initial noise and reverse stochastic path vary?

One evaluator invocation produces both protocols from the same frozen observations,
BCChunk predictions, and explicit initial noise tensors.

## Sampling Protocol

Each candidate uses three deterministic BCChunk calls and twenty Diffusion samples in a
formal run.

### E1a: Initial Noise Only

Seed derivation uses NumPy `SeedSequence(...).generate_state(1, dtype=uint32)[0]` and
therefore returns an integer in `[0, 2**32 - 1]`. The fixed seed namespace and streams are:

```text
PROTOCOL_NAMESPACE = 20260727
INITIAL_NOISE_STREAM = 0
E1A_REVERSE_STREAM = 1
E1B_REVERSE_STREAM = 2
```

The stable candidate key is `(episode_index, frame_index)`. Candidate list ordering and
`candidate_ordinal` do not participate in seed derivation. The manifest records the
namespace, stream identifiers, seed-derivation version, and every candidate key.

For sample ordinal `s`, E1a uses:

```text
initial_noise_seed = SeedSequence([
    20260727, episode_index, frame_index, s, 0
]).generate_state(1, uint32)[0]

reverse_noise_seed = SeedSequence([
    20260727, episode_index, frame_index, 0, 1
]).generate_state(1, uint32)[0]
```

All twenty samples use different initial noise tensors and the same candidate-level reverse
seed.

### E1b: Full Sampling Seed

For sample ordinal `s`, E1b uses:

```text
initial_noise_seed = SeedSequence([
    20260727, episode_index, frame_index, s, 0
]).generate_state(1, uint32)[0]

reverse_noise_seed = SeedSequence([
    20260727, episode_index, frame_index, s, 2
]).generate_state(1, uint32)[0]
```

E1b reuses the exact E1a initial noise tensor at the same sample ordinal, but uses a unique
reverse seed per sample. This makes E1a/E1b paired by candidate and sample ordinal. Repeating
the same candidate, sample ordinal, protocol, checkpoint, and configuration must reproduce
the same action chunk.

Seed derivation must use explicit deterministic helpers. Do not use Python's process-random
`hash()` function. Changing candidate ordering must not change any seed.

Every Diffusion sampling call constructs a fresh `torch.Generator`, seeds it with the row's
reverse seed, hashes its state before sampling, and passes that new generator to the
scheduler. A generator is never reused across samples or reproducibility reruns.

## Shared Computation

The evaluator performs the following work once per candidate:

- load and validate the frozen two-frame observation history;
- preprocess the observation for BCChunk and Diffusion;
- run and retain all three BCChunk calls;
- generate and save twenty explicit initial noise tensors;
- compute observation and processor hashes.

It then performs the E1a and E1b Diffusion sampling loops using the shared tensors. BCChunk,
observation preprocessing, and initial noise generation must not be repeated for the second
protocol.

Each generated initial-noise tensor is an immutable canonical tensor owned by the evaluator.
Every E1a, E1b, and reproducibility call receives a fresh `.clone()`. After all sampling, the
canonical tensor hash must still match its pre-sampling hash. This protects the E1a/E1b pair
from in-place sampler mutation.

Smoke mode retains the same structure with a configurable reduced Diffusion sample count,
currently three. Formal `label=Offline` requires exactly twenty samples.

## Tabular Data Contract

`action_chunks.parquet` gains a required `sampling_protocol` column:

- `bc_deterministic` for the three shared BCChunk calls;
- `initial_noise_only` for E1a;
- `full_sampling_seed` for E1b.

Diffusion condition identifiers are:

```text
diffusion_ddpm100_initial_noise_only
diffusion_ddpm100_full_sampling_seed
```

BC rows keep `condition_id=bc_chunk`. Diffusion rows retain:

- `schema_version=2`;
- `sample_ordinal`;
- `initial_noise_seed`;
- `initial_noise_sha256`;
- `initial_noise_file_sha256`;
- `reverse_noise_seed`;
- `reverse_generator_state_sha256`;
- `num_inference_steps`;
- normalized and denormalized actions.

For a formal candidate:

```text
3 BC calls × 8 steps
+ 20 E1a samples × 8 steps
+ 20 E1b samples × 8 steps
= 344 rows
```

Ten formal candidates produce 3,440 rows.

`diversity_metrics.parquet` contains one row per candidate and Diffusion protocol. It gains
`schema_version=2`, `sampling_protocol`, and `repeat_max_error`, and keeps the normalized and
denormalized trajectory and endpoint metrics. BC determinism remains an
aggregate/reference metric rather than a diversity row. Reproducibility reruns do not create
additional action rows, diversity rows, or noise files.

`metrics.json` and `metrics.md` report:

- the mean across candidates for E1a and E1b normalized/denormalized trajectory RMS and
  endpoint distance;
- four E1b/E1a ratios, each defined as `aggregate E1b mean / aggregate E1a mean`, for
  normalized trajectory RMS, normalized endpoint distance, denormalized trajectory RMS, and
  denormalized endpoint distance;
- maximum reproducibility error across candidates for each protocol;
- BC repeated-call error;
- `num_rollouts=0`.

Ratios are not means of candidate-level ratios. `diversity_metrics.parquet` does not contain
ratio rows or columns. If an E1a aggregate denominator is zero, the corresponding JSON value
is `null` and Markdown renders `N/A`; infinity and NaN are forbidden.

## Noise Artifact Contract

Initial noise tensors are shared and saved once:

```text
noises/<candidate_id>_sample_00.npy
...
noises/<candidate_id>_sample_19.npy
```

`initial_noise_sha256` is the canonical tensor hash produced by `sha256_tensor`: UTF-8
`dtype|shape|` header followed by contiguous CPU tensor bytes. `initial_noise_file_sha256`
is the SHA-256 of the exact saved `.npy` file bytes. The corresponding E1a and E1b Parquet
rows must contain the same seed and both hashes for a given candidate and sample ordinal.
The evaluator reloads each `.npy` file and verifies that its canonical tensor hash matches
`initial_noise_sha256`. Reverse seeds and pre-sampling generator-state hashes remain
protocol-specific row fields; no duplicate initial-noise files are created.

## Visualization Contract

The evaluator writes two recordings:

```text
recording_initial_noise_only.rrd
recording_full_sampling_seed.rrd
```

Both recordings keep the approved synchronized two-panel layout:

1. current observation, agent, and three BCChunk repeated predictions;
2. the same observation and agent plus the selected protocol's Diffusion predictions.

The Diffusion panel title states either:

```text
E1a — initial-noise-only
E1b — full-sampling-seed
```

Each `pred_XX` parent remains independently selectable, every plan draws:

```text
agent → action[0] → ... → action[7]
```

and candidate scenes remain synchronized frames on the `candidate` timeline.

The two recordings use different stable Rerun application IDs so `--persist-state` remembers
their layouts independently. Remove the ambiguous generic `recording.rrd`; the guide must
show one open command for each protocol.

## Schema And Compatibility

This paired protocol is E1 artifact schema version 2:

- `run_manifest.json` uses `schema_version=2` and
  `experiment_schema="pusht_e1_paired_sampling_v2"`;
- both Parquet files contain a required `schema_version=2` column;
- action readers require `sampling_protocol`;
- recording readers require one of the two protocol-specific filenames.

Schema-1 outputs are not silently upgraded because they do not contain E1b or enough
information to reconstruct its reverse random stream. Readers encountering a legacy
`recording.rrd` or Parquet without `sampling_protocol` must report that the E1 command needs
to be rerun with `--overwrite`. No automatic migration or compatibility alias is created.

## Validation And Failure Behavior

Before committing the atomic output directory, verify:

- formal runs contain exactly twenty Diffusion samples per protocol and candidate;
- Smoke uses the requested reduced count in both protocols;
- all E1a reverse seeds for one candidate are identical;
- all E1b reverse seeds for one candidate are unique;
- paired E1a/E1b rows use identical initial noise seeds and hashes;
- each sampling call uses a newly created generator whose recorded hash is its
  pre-sampling state;
- canonical initial-noise hashes remain unchanged after both protocols;
- saved `.npy` file bytes and reloaded tensor hashes agree with recorded fields;
- sample ordinals are contiguous and unique within each candidate and protocol;
- BCChunk is called exactly three times;
- BC, E1a, and E1b chunks have shape `[8,2]` and contain only finite values;
- repeated sample zero for each protocol has maximum action error at or below `1e-6`;
- reproducibility reruns create no extra rows or files;
- each formal candidate contributes exactly 344 action rows;
- both RRD files are written before the atomic output is accepted.

Errors must name the candidate, protocol, sample ordinal, and violated contract where
applicable.

## Tests

Automated tests cover:

- deterministic and collision-free reverse-seed derivation;
- seed stability under candidate reordering and exact uint32 derivation vectors;
- E1a same-reverse and E1b unique-reverse contracts;
- paired initial-noise hashes;
- canonical tensor immutability, cloned sampler inputs, and `.npy` file/hash agreement;
- exact `344` rows per formal candidate and `3,440` rows for ten candidates;
- two diversity rows per candidate;
- E1a and E1b reproducibility checks;
- no row/artifact inflation from reproducibility reruns;
- Smoke row counts of `(3 + 2M) × 8` per candidate;
- exactly two output RRD files with protocol-specific application identities;
- E1a RRD containing only E1a Diffusion data and E1b RRD containing only E1b Diffusion
  data, while both contain the shared BC data;
- each RRD retaining two `Spatial2DView` panels, stable candidate paths, and independently
  selectable predictions;
- preprocessing, three BC calls, and each initial-noise generation occurring exactly once
  per candidate/sample rather than once per protocol;
- atomic-output rejection for seed/hash/row/RRD validation failures;
- explicit validation failures for non-finite actions, wrong action shapes, duplicate or
  missing sample ordinals, and protocol seed-contract violations;
- ratio-of-means calculations plus JSON `null` and Markdown `N/A` zero-denominator behavior;
- schema-1 reader rejection with the rerun/overwrite instruction;
- guide commands and artifact names.

A focused integration test uses a small mocked policy or controlled sampler seam so it can
verify both protocols without running two full DDPM-100 loops in the unit-test suite.

## Guide Update

`reports/guides/pusht_diffusion/e1_fixed_observation.md` must:

- explain the scientific distinction between E1a and E1b;
- state that one command generates both;
- list the shared and protocol-specific artifacts;
- provide separate commands for the two RRD files;
- explain that E1a is an initial-prior sensitivity test while E1b reflects normal stochastic
  DDPM sampling;
- prohibit using either offline diversity result as evidence that the sampled actions are
  successful; validity still requires E2 closed-loop execution.

## Non-Goals

- Do not add online action selection from twenty samples.
- Do not rank or choose a best sample.
- Do not add E2 rollout execution to E1.
- Do not replace DDPM with DDIM in this change.
- Do not put forty Diffusion trajectories into one RRD panel.
