# PushT E5 Multi-GPU Held-Out Audit Design

## Status

Approved by the user for implementation on 2026-07-30. The first independent
specification review found three blocking ambiguities; the document now includes their
resolutions and passed independent re-review.

## Problem

The E5 held-out audit currently fails on its first open-loop batch when CUDA is used:

```text
RuntimeError: Expected all tensors to be on the same device, but got mask is on cuda:0,
different from other tensors on cpu
```

The policy preprocessor and normalized policy outputs are on the policy device. LeRobot's
Diffusion postprocessor intentionally unnormalizes actions and then moves them to CPU. The
audit constructs one padding mask on the policy device and reuses it for both normalized
and denormalized errors. The normalized `masked_select` is valid, while the denormalized
`masked_select` combines CPU errors with a CUDA mask.

The audit also runs both selected models and all held-out episodes sequentially on one
device. The remote evaluation host has multiple GPUs, and the user wants a single command
that can distribute the formal audit over an explicit number of GPUs without changing its
evaluation protocol or output contract.

## Scope

This change will:

- fix the open-loop normalized and denormalized mask device handling;
- extend the existing E5 audit entry point and shell launcher with `--num-gpus`;
- distribute closed-loop audit episodes over persistent, model-specific GPU workers;
- retain exactly one full open-loop evaluation per selected model;
- aggregate all worker results into the existing atomic audit output layout;
- record the worker assignment and GPU count in audit provenance;
- preserve the existing single-device execution path.

This change will not:

- change the selected checkpoints, prototype assignment, audit split, policy seed,
  Diffusion scheduler, number of inference steps, initial-state reconstruction, success
  criteria, or rollout classification;
- introduce distributed model inference for one episode;
- write one formal result per worker or require a separate merge command;
- resume a partial audit;
- execute the formal audit as part of implementation or verification.

## Considered Approaches

### A. Parent aggregation with persistent model-specific workers

A parent process owns validation, the atomic output directory, result aggregation, and
final output. Each GPU worker is assigned one model and a deterministic subset of audit
episodes. The worker loads its assigned policy once, evaluates its closed-loop shard, and
writes only uniquely named recordings. One designated worker per model also runs the full
open-loop evaluation.

This is the selected approach. It supports arbitrary useful GPU counts, avoids transferring
frames through multiprocessing queues, preserves one canonical result directory, and avoids
reloading a model for every episode.

### B. Independent partial runs followed by a merge command

Each GPU would write a separate partial output directory, and a later command would validate
and merge them. This gives strong process isolation but creates more remote artifacts and
requires a second failure-prone workflow for matching manifests and preventing incomplete or
duplicate merges.

### C. One GPU per model

BC and Diffusion would run concurrently on two GPUs. This is simple but cannot use more than
two GPUs and leaves most of an eight-GPU host idle while the slow Diffusion rollouts remain
serial.

## Device-Correct Open-Loop Error

The open-loop evaluator will derive two masks from the same valid-step tensor:

- the normalized mask is moved to `normalized_squared.device`;
- the denormalized mask is moved to `denormalized_squared.device`.

Each mask is expanded to the shape of the corresponding squared-error tensor. Per-batch and
per-episode reductions use the matching mask. The implementation must not assume that the
postprocessor preserves the policy device. It must not move denormalized actions back to
CUDA, because returning postprocessed actions on CPU is intentional LeRobot behavior.

The normalized and denormalized valid-element counts must agree. A disagreement is a
programming error and fails the evaluation rather than silently producing metrics with
different denominators.

## CLI Contract

The Python entry point and `scripts/experiments/pusht/eval_e5_oracle_audit.sh` accept:

```text
--num-gpus=N
```

`N` must be a positive integer.

- `N=1` preserves the current sequential path and honors `--device`.
- `N>=2` requires CUDA, uses visible devices `cuda:0` through `cuda:N-1`, and rejects an
  explicit non-default `--device` because worker devices are assigned by rank.
- `N` cannot exceed the visible CUDA device count.
- `N` cannot exceed 42, the number of model/episode closed-loop tasks.

The launcher defaults to `--num-gpus=1`, validates the argument before starting Python, and
forwards it in `--print-command` output.

For `N=1`, `--device` may still be `cpu` for development and test compatibility. In that
case `num_gpus` represents the requested single execution slot rather than a claim that a
CUDA device was used. Runtime provenance records the actual device explicitly.

## Worker Allocation

For `N>=2`:

- BC receives `floor(N / 2)` workers;
- Diffusion receives `N - floor(N / 2)` workers, so the slower Diffusion model receives the
  extra worker for odd `N`;
- each model therefore has at least one worker;
- each worker is permanently assigned one model and loads that selected policy once;
- within a model, worker `i` of `W` receives
  `audit_episodes[i::W]`.

For eight GPUs this produces four BC workers and four Diffusion workers. The allocation is
deterministic and every worker receives at least one episode for valid `N`.

The first worker assigned to each model is the open-loop leader. It evaluates all 21 audit
episodes with the existing full-open-loop function before or alongside its own closed-loop
shard. All other workers begin their closed-loop shards without repeating open-loop work.

## Parent and Worker Responsibilities

### Parent

Before spawning workers, the parent:

1. resolves both selected checkpoints and verifies their SHA256 identities;
2. verifies the prototype manifest SHA256 against checkpoint selection;
3. loads the committed split and prototype sidecar;
4. loads all 21 audit episode descriptors;
5. completes initial-state reconstruction preflight for every audit episode;
6. validates the requested GPU count and constructs the complete worker plan;
7. opens the one atomic output directory.

The parent starts workers with the multiprocessing `spawn` context. It consumes structured
ready, progress, result, and error events. It never receives rendered frames or policy
objects.

The parent also owns a shared start event. Every worker must load and validate its assigned
policy, report `ready`, and wait without inference. The parent sets the start event only
after all workers have reported ready. If any worker reports an error or exits before the
barrier completes, the parent terminates the remaining workers and no worker begins
open-loop or closed-loop inference.

After all workers exit successfully, the parent validates and sorts the result set, writes
all canonical outputs, and atomically publishes the final directory.

### Worker

Each worker:

1. binds to its assigned CUDA device and limits CPU thread fan-out;
2. loads its one selected policy and processors;
3. validates model type and prototype capacity;
4. applies the fixed DDPM scheduler protocol for Mode-Diffusion;
5. reports `ready` and waits for the parent start event;
6. if designated leader, computes the complete model open-loop result;
7. runs its assigned closed-loop episodes;
8. classifies every rollout and writes each recording directly to its unique model/episode
   path under the parent's temporary output directory;
9. sends metric rows and progress events to the parent;
10. releases the policy and CUDA cache before exit.

## Determinism

Episode scheduling must not affect policy inputs:

- every closed-loop episode retains the existing dataset episode index, reconstructed
  initial state, oracle prototype ID, and fixed policy seed;
- every worker creates a fresh environment and resets the policy for each episode through
  the existing rollout function;
- the designated open-loop leader processes the same complete ordered episode list as the
  single-device implementation, preserving batch ordinals and Diffusion noise seeds;
- parent output ordering is canonicalized by model order and episode index.

The multi-GPU and single-GPU paths are expected to produce the same deterministic policy
outputs subject to the existing CUDA determinism limits.

## Result Validation and Aggregation

Before writing a complete manifest, the parent requires:

- exactly one closed-loop row for every pair in
  `{mode_bc_chunk, mode_diffusion} × audit_episodes`;
- no unknown model or episode;
- exactly one full open-loop summary per model;
- exactly one open-loop per-episode row for every model/episode pair;
- exactly 21 successful reconstruction preflight rows;
- a recording file for every closed-loop row.

Missing, duplicate, or unexpected records fail closed.

The final output layout remains:

```text
e5_oracle_audit/
  run_manifest.json
  selected_checkpoints.json
  rng_manifest.json
  metrics.json
  metrics.md
  closed_loop_rollouts.parquet
  open_loop_metrics.parquet
  open_loop_episode_metrics.parquet
  reconstruction_metrics.parquet
  recordings/
    mode_bc_chunk/
    mode_diffusion/
```

Metric computation reuses the current aggregation functions after records have been
validated and sorted.

## Provenance

The run manifest keeps the selected checkpoint path-resolution provenance and adds:

- `execution.parallelism`: `single_device_sequential` or
  `multi_gpu_model_episode_shards`;
- `execution.num_gpus`;
- `execution.device`: the requested device for the single-device path, otherwise `null`;
- `execution.worker_plan`: for multi-GPU execution, one record per GPU containing GPU
  index, model, worker index within that model, whether it is the open-loop leader, and
  assigned audit episode indices; for the single-device sequential path this is an empty
  list because no worker process exists;
- the requested open-loop batch size.

Worker failures are reported with GPU index, model, phase, and episode index when available.

## Failure and Atomicity Policy

- Parent preflight validation failures occur before opening the atomic staging directory
  and therefore create no audit output.
- Multi-GPU model validation is protected by the ready barrier, so all workers validate
  successfully before any worker starts policy inference.
- Any worker exception is serialized with its traceback to the parent.
- On the first worker error or unexpected worker exit, the parent terminates and joins the
  remaining workers.
- A failure after the atomic staging directory opens retains the `.incomplete` directory
  and marker. It never publishes a complete run manifest or final audit directory.
- A new run does not resume a partial audit. The existing overwrite behavior remains
  explicit.
- Recordings are written only to deterministic, non-overlapping paths owned by a single
  worker.

## Tests

Regression and orchestration coverage will include:

1. normalized and denormalized masks follow their corresponding squared-error tensor
   devices;
2. a postprocessor that returns CPU tensors does not reuse a CUDA/fake-CUDA mask;
3. `N=1` retains the sequential path and requested device;
4. two GPUs allocate one worker per model;
5. odd GPU counts give the extra worker to Diffusion;
6. eight GPUs allocate four workers per model and cover every model/episode pair exactly
   once;
7. zero, excessive, unavailable, and conflicting GPU requests fail before worker spawn;
8. aggregation rejects missing, duplicate, and unexpected closed-loop or open-loop rows;
9. workers cannot pass the ready barrier until every worker validates;
10. worker validation errors, runtime errors, and unexpected exits propagate to the parent
    and prevent publication;
11. single-device provenance uses the actual `device` and an empty worker plan;
12. launcher `--num-gpus` parsing, forwarding, duplicate rejection, and
    `--print-command` behavior;
13. existing checkpoint relocation, prototype SHA256, single-device audit, open-loop metric,
    and launcher tests continue to pass.

Verification uses unit tests, mocked workers, static shell checks, and command rendering.
It does not execute the formal held-out rollout.

## Acceptance

Implementation is ready for the user's remote audit when:

- the reported CUDA/CPU mismatch has a failing-before/passing-after regression test;
- all focused E5 audit and launcher tests pass;
- an eight-GPU plan contains 42 unique closed-loop model/episode assignments;
- static checks and `--print-command --num-gpus=8` succeed from outside the repository;
- no formal rollout was started during implementation.

The remote formal audit is complete only when its atomic output contains the complete
manifest, 42 closed-loop rows, two complete open-loop summaries, 42 open-loop per-episode
rows, 21 reconstruction rows, and no `.incomplete` marker.
