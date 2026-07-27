# Portable Training Environment Image Design

## Goal

Build a path-independent, environment-only container image for this repository's
LeRobot, PushT, LIBERO, ManiSkill, single-GPU, and multi-GPU workflows.

The image must reproduce the current `robot-practice` Python environment closely,
run on the local CUDA workstation, and be suitable for a Kubernetes worker with
eight NVIDIA H20 GPUs. The repository may be mounted or copied to any directory.

## Scope

The image contains:

- Ubuntu 22.04 CUDA 12.8 and cuDNN 9 runtime libraries;
- a Conda environment named `robot-practice` on Python 3.10;
- the current environment's pinned Conda and PyPI dependencies;
- ffmpeg and headless EGL/OpenGL runtime libraries required by the simulators;
- small environment initialization and verification commands.

The image does not contain:

- repository source code or Git history;
- datasets, checkpoints, model initialization files, experiment outputs, videos,
  caches, or credentials;
- a Docker daemon or Kubernetes orchestration logic;
- an NVIDIA host driver.

The registry destination is supplied at build or push time. It is not hard-coded
in tracked files.

## Image Architecture

Use an NVIDIA CUDA 12.8, cuDNN 9, Ubuntu 22.04 runtime image pinned by immutable
digest as the base. Record the digest and the resolved CUDA and cuDNN versions in
the image metadata. Install Conda under `/opt/conda` and create
`/opt/conda/envs/robot-practice`.

The image exports:

```text
PYTHONNOUSERSITE=1
PATH=/opt/conda/envs/robot-practice/bin:/opt/conda/bin:$PATH
```

Putting the environment executable directory on `PATH` makes the environment
usable immediately. Keeping the Conda executable available preserves existing
commands that use:

```bash
conda run -n robot-practice ...
```

The default working directory is only a convenience and is not part of the
runtime contract.

## Dependency Locking

Generate locks from the currently working local `robot-practice` environment:

1. An explicit Linux x86-64 Conda package lock, including immutable public
   artifact URLs, for entries that `conda list --json` identifies as Conda-owned.
2. A hash-locked Linux x86-64 PyPI artifact lock for entries that
   `conda list --json` identifies as `pypi`.

The PyPI installation uses only `https://pypi.org/simple`, `--require-hashes`,
`--only-binary=:all:`, and `--no-deps`. The lock names and hashes the exact
Linux x86-64 wheels for `torch==2.10.0`, `torchvision==0.25.0`,
`triton==3.6.0`, and the complete `nvidia-*-cu12` family currently installed in
the working environment. The resulting manifest must assert
`torch.version.cuda == "12.8"` and the expected cuDNN version. This prevents pip
from replacing Conda-owned packages implicitly or accepting an unintended
PyTorch artifact.

Source distributions are not allowed in the primary image. If any current
dependency has no compatible public PyPI wheel, lock generation stops and names
that package. Supporting it requires a separate design change for a fully locked
wheel-building stage; it is not silently built with network-resolved build
dependencies.

Before committing either lock:

- remove the editable `robot-practice` installation;
- reject local filesystem paths, VCS project references, credentials, and
  authenticated package URLs;
- retain only dependency data needed to reconstruct the environment;
- allow only approved public Conda channels and the public PyPI index;
- verify the expected key versions, including Python 3.10, PyTorch with CUDA
  12.8 support, LeRobot 0.4.4, Diffusers 0.35.2, and gym-pusht 0.1.x.

Lock generation must fail rather than silently dropping any additional local,
VCS, or unapproved-index dependency. The repository itself is the only excluded
entry and is installed after it is made available at runtime.

The image also records the lock checksums, Python architecture, Conda channels,
PyPI index, and key package versions as OCI labels and in a machine-readable
environment manifest.

## Runtime User And Permissions

Create a non-root `robot-practice` user and group with configurable build-time
UID and GID, defaulting to 1000. The Conda environment and the user's cache
directory are owned by that user and group. Environment directories that need to
receive editable-install metadata are group-writable and setgid.

The default container user is non-root. Kubernetes configurations that override
the UID must also provide the image's runtime GID, for example with
`runAsGroup`/`fsGroup`, or deliberately run as root. The initialization command
checks environment writability before invoking pip and reports the required
runtime-group setting when it is unavailable.

Repository source only needs to be readable. The implementation must verify that
PEP 660 editable installation works from a read-only repository mount. Runtime
initialization uses `--no-build-isolation`, the already locked setuptools and
wheel packages, and a writable `TMPDIR`; it must never resolve build requirements
from the network. If the selected backend still attempts to write into the
source tree, verification fails and the helper is not released.

## Path-Independent Repository Setup

Install a small command in the image:

```bash
robot-practice-init [REPOSITORY_PATH]
```

The path defaults to the current directory. The command:

1. resolves the supplied path;
2. verifies that it contains `pyproject.toml` and `src/robot_practice`;
3. runs the environment's Python interpreter with:

   ```bash
   python -m pip install -e REPOSITORY_PATH \
     --no-deps --no-build-isolation --force-reinstall
   ```

4. verifies that `robot_practice` imports from that repository.

No fixed mount path is assumed. The editable installation must be repeated when
the repository is mounted at a different path in a new container.

Initialization tests cover a normal path, a path containing spaces, a symlinked
path, a read-only source mount, and replacement of editable metadata pointing to
a previous mount path.

Commands containing relative paths such as
`bash scripts/experiments/pusht/train_diffusion.sh` still need to be launched
from the repository root. That is a property of the command, not of the image.

## Build Boundary

Use a repository-root `.dockerignore` with an allowlist for the Dockerfile,
dependency locks, and container helper scripts. This prevents the repository's
datasets, outputs, caches, source tree, and Git metadata from entering the build
context accidentally.

Provide:

```text
docker/training/
  Dockerfile
  conda-linux-64.lock
  requirements-linux-64.lock
  entrypoint.sh
  robot-practice-init
scripts/docker/
  build_training_image.sh
  verify_training_image.sh
```

The build command accepts the target image name as an argument or environment
variable. A release is tagged with both an immutable version tag and an optional
`latest` alias. Registry login and push remain explicit user actions.

Tracked files must not contain the user-supplied registry destination. A build
accepts it only through `--image` or `IMAGE_REF`, and validation uses `git grep`
to enforce that rule.

## GPU Compatibility

The initial image keeps the working local PyTorch and CUDA 12.8 software stack.
It must be tested on:

- the local RTX 3090 workstation and its installed NVIDIA driver;
- the remote H20 worker with its host-provided NVIDIA driver.

CUDA 12.x minor-version compatibility permits a CUDA 12.x runtime with an R535
driver, but newer PTX or driver features can still require a compatibility
package or driver upgrade. Therefore the base image does not force
`cuda-compat-12-8` into `LD_LIBRARY_PATH`.

If the H20 verification reports a CUDA 803/804, PTX, or driver compatibility
failure, create a separately tagged H20 compatibility variant. Do not silently
change the primary image or downgrade PyTorch.

## Verification

### Build-time checks

- the Conda environment can resolve and `pip check` succeeds;
- key packages import without repository source code;
- lock files contain no rejected path, VCS, credential, or private-registry
  entries;
- image history and filesystem do not contain repository data or credentials.
- the pinned base digest, lock checksums, package sources, and CUDA wheel family
  match the generated environment manifest.

### Local runtime checks

- report Python, PyTorch, CUDA, cuDNN, LeRobot, simulator, and ffmpeg versions;
- perform an NVIDIA CUDA tensor operation;
- mount or copy the repository at a non-default path;
- run `robot-practice-init` and verify the imported project path;
- run `scripts/doctor.sh`;
- with `DISPLAY` unset, reset and render
  `gym_pusht/PushT-v0` using `render_mode="rgb_array"` and assert a nonempty RGB
  array;
- with `MUJOCO_GL=egl`, construct a minimal MuJoCo model, create an offscreen
  `mujoco.Renderer`, and assert a nonempty RGB array;
- import `lerobot`, `gym_pusht`, `mujoco`, `libero`, `mani_skill`, and `sapien`;
- construct the LIBERO benchmark registry and assert that its task-suite mapping
  is nonempty;
- create, reset, render, and close ManiSkill `PushCube-v1` in headless
  `rgb_array` mode and assert a nonempty RGB observation or rendered frame;
- after `robot-practice-init`, run these exact single-GPU dry runs and require
  exit status zero plus the expected model entry and model-config module in the
  printed JSON:

  ```bash
  bash scripts/experiments/pusht/train_bc_chunk.sh \
    --mode=smoke --device=cuda --print-command
  bash scripts/experiments/pusht/train_diffusion.sh \
    --mode=smoke --device=cuda --print-command
  ```

- run a minimal distributed/NCCL probe when multiple local GPUs are available.

### Remote H20 checks

- confirm all eight H20 devices are visible;
- run CUDA tensor allocation and computation on every device;
- run an eight-process NCCL all-reduce;
- provide enough shared memory for PyTorch workers through the platform's
  shared-memory setting or a memory-backed `/dev/shm` volume;
- initialize the repository from its chosen path;
- run the same headless graphics and launcher dry-run checks used locally;
- run these exact eight-GPU launcher dry runs and require exit status zero,
  `world_size: 8`, and the expected model entry/config modules:

  ```bash
  bash scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh \
    --num-gpus=8 --mode=smoke --device=cuda --print-command
  bash scripts/experiments/pusht/train_diffusion_multi_gpu.sh \
    --num-gpus=8 --mode=smoke --device=cuda --print-command
  ```

- run a minimal training smoke test only after the required dataset and
  initialization assets are made available separately.

The image may be built and delivered before remote access is available, but its
status must then be reported as `remote verification pending`. It is not
considered remote-ready until the CUDA, headless runtime, and eight-process NCCL
checks pass on the actual H20 worker with the recorded R535 driver build. If a
compatibility variant is required, that exact variant must pass the same checks.

## Error Handling

- `robot-practice-init` fails with a clear message for an invalid repository
  directory or failed import.
- Build scripts stop on dependency-lock validation failures.
- Verification scripts print the detected host driver, device list, and software
  versions before running GPU tests.
- Missing GPUs produce a direct failure for GPU verification, not a silent CPU
  fallback.
- Missing NVIDIA device-plugin/runtime injection, an undersized `/dev/shm`, or a
  runtime user outside the writable environment group produces a specific
  remediation message.
- A driver compatibility failure identifies the optional H20 compatibility
  image as the next action.

Secret and layer checks inspect the allowlisted build context, image
configuration, `docker history --no-trunc`, an exported final filesystem, and
every saved image layer. They search for repository-only paths, `.git`, common
credential/token patterns, and the supplied registry destination. The external
repository tag stored in the saved-image manifest is excluded from the
registry-string assertion because the tag necessarily names the requested
destination. Deleted files found in earlier layers fail the build audit.

## Acceptance Criteria

The work is complete when:

1. the image builds from the sanitized, allowlisted build context;
2. no project source, data, outputs, Git history, credentials, or private
   registry destination is stored in the image;
3. the environment exposes the expected pinned core package versions;
4. existing `conda run -n robot-practice` commands remain valid;
5. a repository placed at an arbitrary path can be initialized and imported;
6. the checked-out source tree contains no committed user-supplied registry
   destination;
7. local CUDA, headless PushT/MuJoCo/runtime-component, and launcher dry-run
   verification succeeds;
8. initialization succeeds as the non-root runtime user from normal, spaced,
   symlinked, read-only, and replacement repository paths;
9. the image remains explicitly `remote verification pending` until CUDA,
   headless runtime, and eight-process NCCL checks pass on all eight H20 GPUs
   with the recorded R535 driver;
10. any required H20 compatibility variant is separately tagged and passes the
    same remote tests;
11. build, initialization, verification, Kubernetes runtime prerequisites,
    tagging, and push instructions are documented without embedding credentials.
