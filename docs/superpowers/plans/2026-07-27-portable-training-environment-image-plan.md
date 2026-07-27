# Portable Training Environment Image Implementation Plan

## Compatibility Boundary

- Keep repository code, datasets, checkpoints, outputs, caches, Git history, and
  credentials outside the image.
- Preserve the Conda environment name `robot-practice` and
  `PYTHONNOUSERSITE=1`.
- Preserve existing single-GPU and multi-GPU Bash entry points.
- Do not assume a repository mount path.
- Do not hard-code the user-supplied registry destination in tracked files.
- Report the built image as `remote verification pending` until it passes on the
  actual eight-H20 worker.

## Implementation Steps

1. Add a lock generator that separates Conda-owned and PyPI-owned packages from
   `conda list --json`, rejects private/local/VCS sources, and generates:
   - an explicit Linux x86-64 Conda artifact lock;
   - a public-PyPI, hash-pinned artifact lock using wheels for all available
     packages and an explicit allowlist for the six reviewed source-only
     dependencies;
   - a machine-readable environment manifest.
2. Add a pinned multi-stage CUDA 12.8/cuDNN 9/Ubuntu 22.04 Dockerfile. Build the
   six source-only dependencies with a locked builder toolchain, then install
   only wheels into the final stage without an index or dependency resolution.
3. Add a non-root runtime user, writable Conda environment group, headless
   graphics defaults, OCI environment labels, and a generic entrypoint.
4. Add `robot-practice-init` to validate an arbitrary repository path and perform
   an offline PEP 660 editable install with locked build tooling.
5. Add an allowlist `.dockerignore` and audit it so no repository source or
   generated artifact enters the build context.
6. Add build and verification scripts with explicit `--image`/`IMAGE_REF`
   handling. Verify versions, imports, CUDA, PushT rendering, MuJoCo EGL,
   LIBERO registry initialization, ManiSkill rendering, arbitrary-path editable
   installation, and training launcher dry runs.
7. Add an eight-process NCCL probe and remote verification mode with H20 count,
   driver, shared-memory, and runtime-group diagnostics.
8. Add concise usage documentation covering build, local verification, registry
   tagging/push, arbitrary-path project initialization, Kubernetes prerequisites,
   and remote-ready status.
9. Run formatting/static checks, lock sanitization, build-context audit, Docker
   build, no-GPU checks, local RTX 3090 checks, arbitrary-path checks, image-layer
   audit, and available repository smoke tests.

## Acceptance

- Locks reconstruct the expected Python 3.10, PyTorch CUDA 12.8, LeRobot,
  Diffusers, PushT, LIBERO, and ManiSkill environment without local/VCS/private
  package sources.
- Docker build context and all image layers contain no project source, data,
  outputs, Git history, credentials, or committed target registry.
- `conda run -n robot-practice` remains usable.
- `robot-practice-init` works for normal, spaced, symlinked, read-only, and stale
  editable-install paths.
- Local CUDA and required headless simulator checks pass.
- The exact remote verification command is delivered; `remote-ready` is claimed
  only after the actual eight-H20 CUDA/NCCL/headless checks pass.
