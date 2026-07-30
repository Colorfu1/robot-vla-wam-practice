# PushT E5 Audit Checkpoint Path Relocation Design

## Status

Approved by the user for implementation on 2026-07-30.

## Problem

E5 development checkpoint selection records each selected checkpoint as an absolute path.
The completed selection was produced in a remote workspace, then copied to the current
repository. Its recorded checkpoint paths no longer exist here, even though the identical
checkpoint artifacts are available under this repository's normal E5 training output tree.

`e5_oracle_audit._selected_rows()` currently accepts only the recorded absolute path and
therefore fails before audit evaluation with `FileNotFoundError`.

## Scope

This change makes completed E5 selection outputs relocatable between copies of the same
repository. It does not:

- modify the original `selected_checkpoints.json`;
- search arbitrary filesystem locations;
- accept a checkpoint whose weights do not match the selection SHA256;
- change checkpoint ranking, oracle mode assignment, simulator reconstruction, open-loop
  evaluation, or closed-loop audit protocol.

After the fix and regression tests pass, run the already specified held-out audit:

- selected Mode-BCChunk on all 21 audit episodes;
- selected Mode-Diffusion on all 21 audit episodes;
- complete audit open-loop evaluation for both models;
- single-device sequential closed-loop evaluation.

## Resolution Algorithm

Resolve each selected model independently.

1. Read `model`, `step`, `checkpoint_path`, and `checkpoint_id` from
   `selected_checkpoints.json`.
   - The top-level key and row `model` must match and be exactly one of
     `mode_bc_chunk` or `mode_diffusion`.
   - `step` must be a non-boolean positive integer and is rendered as a six-digit numeric
     checkpoint directory.
   - `checkpoint_id` must be a lowercase 64-character SHA256 digest.
2. Construct two bounded candidates in order:
   - `recorded`: the original `checkpoint_path`;
   - `repository_relative`: the current repository's
     `outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/<model>/checkpoints/<step>/pretrained_model`.
3. Remove duplicate candidates after path resolution.
4. For each candidate:
   - require `model.safetensors` to be a regular file;
   - calculate its SHA256;
   - accept it only when the digest equals `checkpoint_id`.
5. Use the first hash-matching candidate.
6. If no candidate matches, fail before loading either policy. The error must distinguish
   missing candidates from hash mismatches and list only the bounded candidate paths.

The repository-relative candidate is derived from validated `model` and integer `step`;
no suffix from the untrusted recorded absolute path is copied.

"Current repository" means the path returned by the existing `config.practice_root()`
helper, which derives the root from the installed module's `__file__`. Resolution must not
depend on the process working directory.

## Output Provenance

The in-memory selected row passed to audit contains:

- `recorded_checkpoint_path`: the path stored by selection;
- `checkpoint_path`: the path actually used by audit;
- `checkpoint_path_resolution`: `recorded` or `repository_relative`;
- the original `checkpoint_id`.

Audit's copied `selected_checkpoints.json` writes this resolved provenance. The original
selection directory remains byte-for-byte unchanged.

The audit run manifest records a per-model path-resolution summary using the same fields,
so a relocated run can be interpreted without inspecting logs.

## Failure Policy

- A present candidate with the wrong SHA256 is never loaded.
- A valid repository-relative checkpoint may be used even if the recorded candidate is
  present but has the wrong hash, because the accepted artifact still exactly matches the
  immutable selection digest. The mismatch is included in the failure diagnostics only if
  no valid candidate is found.
- Missing or invalid selection fields fail closed.
- Both selected models must resolve successfully before dataset loading or model inference.
- The existing prototype-manifest SHA256 check remains mandatory.

## Tests

Add regression coverage at the real selection-loading seam:

1. Recorded path exists and matches: use it and report `recorded`.
2. Recorded path is absent, repository-relative path matches: use fallback and report
   `repository_relative`.
3. Recorded path is absent and fallback hash mismatches: fail.
4. Neither candidate exists: fail.
5. Recorded path exists with the wrong hash while fallback matches: use the fallback and
   report `repository_relative`.
6. Invalid model names, mismatched top-level/row model names, non-integer steps, and invalid
   digest formats fail before candidate construction.
7. Launching from a working directory outside the repository still resolves through
   `practice_root()`.
8. Both models must be present and resolved.
9. Existing completed/incomplete selection validation continues to pass.

Re-run the original completed selection directory through `_selected_rows()` and verify that
both models resolve to current repository paths with their recorded SHA256 values.

## Audit Acceptance

The formal audit is complete only when the atomic output directory contains a complete run
manifest, both selected models, 42 closed-loop rollout rows, full open-loop metrics for both
models, 21 reconstruction rows, and no `.incomplete` marker.

If execution fails or is interrupted, preserve the atomic incomplete marker and report the
failure; do not summarize a partial run as a formal result.
