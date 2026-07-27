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

Use an NVIDIA CUDA 12.8, cuDNN 9, Ubuntu 22.04 runtime image as the base. Install
Conda under `/opt/conda` and create `/opt/conda/envs/robot-practice`.

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

1. An explicit Linux x86-64 Conda package lock for packages owned by Conda.
2. A version-pinned PyPI lock for packages installed by pip.

Before committing either lock:

- remove the editable `robot-practice` installation;
- reject local filesystem paths, VCS project references, credentials, and
  authenticated package URLs;
- retain only dependency data needed to reconstruct the environment;
- verify the expected key versions, including Python 3.10, PyTorch with CUDA
  12.8 support, LeRobot 0.4.4, Diffusers 0.35.2, and gym-pusht 0.1.x.

The repository itself is installed only after it is made available at runtime.

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
   python -m pip install -e REPOSITORY_PATH --no-deps
   ```

4. verifies that `robot_practice` imports from that repository.

No fixed mount path is assumed. The editable installation must be repeated when
the repository is mounted at a different path in a new container.

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

### Local runtime checks

- report Python, PyTorch, CUDA, cuDNN, LeRobot, simulator, and ffmpeg versions;
- perform an NVIDIA CUDA tensor operation;
- mount or copy the repository at a non-default path;
- run `robot-practice-init` and verify the imported project path;
- run `scripts/doctor.sh`;
- verify the existing training entry points can parse their configurations;
- run a minimal distributed/NCCL probe when multiple local GPUs are available.

### Remote H20 checks

- confirm all eight H20 devices are visible;
- run CUDA tensor allocation and computation on every device;
- run an eight-process NCCL all-reduce;
- initialize the repository from its chosen path;
- parse the existing single-GPU and multi-GPU PushT training commands;
- run a minimal training smoke test only after the required dataset and
  initialization assets are made available separately.

The image is not considered remote-ready until the H20 CUDA and NCCL checks pass.

## Error Handling

- `robot-practice-init` fails with a clear message for an invalid repository
  directory or failed import.
- Build scripts stop on dependency-lock validation failures.
- Verification scripts print the detected host driver, device list, and software
  versions before running GPU tests.
- Missing GPUs produce a direct failure for GPU verification, not a silent CPU
  fallback.
- A driver compatibility failure identifies the optional H20 compatibility
  image as the next action.

## Acceptance Criteria

The work is complete when:

1. the image builds from the sanitized, allowlisted build context;
2. no project source, data, outputs, Git history, credentials, or private
   registry destination is stored in the image;
3. the environment exposes the expected pinned core package versions;
4. existing `conda run -n robot-practice` commands remain valid;
5. a repository placed at an arbitrary path can be initialized and imported;
6. local CUDA verification succeeds;
7. the delivered remote verification command can validate eight H20 GPUs and
   NCCL before training;
8. build, initialization, verification, tagging, and push instructions are
   documented without embedding credentials.
