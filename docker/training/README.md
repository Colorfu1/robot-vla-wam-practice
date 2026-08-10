# Portable Robot Practice Training Environment

This image contains the CUDA, Conda, Python, LeRobot, PushT, LIBERO, SmolVLA,
PEFT, and ManiSkill environment. It does not contain this repository, datasets,
checkpoints, outputs, Hugging Face caches, or credentials. It does contain the
fixed SAPIEN PhysX GPU runtime required by offline `physx_cuda` evaluation.

The bundled runtime is SAPIEN PhysX `105.1-physx-5.3.1.patch0`, installed at
`/opt/robot-practice/sapien-cache/physx/<version>/`. The base image exposes it
through `/home/robot-practice/.sapien`; the Volc MLP derivative exposes the same
cache through `/root/.sapien`. The release zip is SHA256-verified during build.

## Generate locks

Run this only after intentionally changing the local `robot-practice`
environment:

```bash
bash scripts/docker/generate_environment_locks.sh
```

The generator excludes this repository's editable package and rejects local,
VCS, credential-bearing, and unapproved package sources.

## Build the portable environment base

Choose the registry/repository at invocation time:

```bash
IMAGE_REPOSITORY='<registry>/<namespace>/practice'
BASE_TAG='env-YYYYMMDD-<lock-hash>'

bash scripts/docker/build_training_image.sh \
  --image="${IMAGE_REPOSITORY}:${BASE_TAG}" \
  --version-tag="$BASE_TAG" \
  --no-latest
```

No registry destination is committed to the repository.

If Docker Hub is unavailable, `--base-image` may point to a trusted mirror of the
same pinned CUDA amd64 digest. Changing or omitting the digest is not allowed for
a release build. Dependency downloads are direct by default. Use
`--anaconda-proxy` when only Miniconda and Conda require the shell's
`HTTP_PROXY`/`HTTPS_PROXY`; APT and PyPI remain direct. `--use-proxy` is the
explicit all-download fallback. Docker does not persist these predefined proxy
build arguments in the resulting image.

## Build the Volc MLP runtime image

The remote `practice:latest` tag is the Volc MLP derivative, not the portable
environment base. It preserves the same locked environment while switching to
the platform-required root user, `/workspace`, and empty entrypoint:

```bash
VOLC_TAG='volc-YYYYMMDD-<purpose>'

docker build \
  --file docker/training/Dockerfile.volc-mlp \
  --build-arg "BASE_IMAGE=${IMAGE_REPOSITORY}:${BASE_TAG}" \
  --tag "${IMAGE_REPOSITORY}:${VOLC_TAG}" \
  --tag "${IMAGE_REPOSITORY}:latest" \
  .
```

## Verify locally

```bash
IMAGE_REF="${IMAGE_REPOSITORY}:${VOLC_TAG}"

bash scripts/docker/verify_training_image.sh \
  --image="$IMAGE_REF" \
  --repo="$PWD" \
  --require-gpus=1 \
  --require-bf16

bash scripts/docker/audit_training_image.sh \
  --image="$IMAGE_REF" \
  --registry-string="${IMAGE_REF%%/*}" \
  --save-layers
```

The verification mounts the repository read-only at a non-default path containing
spaces, changes directly into the repository, and checks that local imports,
TorchCodec video decoding, CUDA, and headless simulation work without initialization.

With `--require-gpus`, verification creates a GPU-backed ManiSkill environment
and checks that the prebundled PhysX library loads without network access. A
CPU-only verification intentionally continues to use `physx_cpu`.

## Push

Registry authentication remains an explicit user action:

```bash
docker login '<registry>'
docker push "${IMAGE_REPOSITORY}:${BASE_TAG}"
docker push "${IMAGE_REPOSITORY}:${VOLC_TAG}"
docker push "${IMAGE_REPOSITORY}:latest"
```

Push both immutable tags before updating `latest`.

## Use the repository at any path

After starting a worker from the image, put the repository anywhere and change to
its root:

```bash
cd /any/path/to/practice
```

Existing commands continue to work:

```bash
watch -n 2 nvidia-smi

bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=pilot

python -m robot_practice.experiments.pusht_diffusion.train_diffusion \
  --help
```

No editable install or `robot-practice-init` call is required. The image intentionally
uses the relative `PYTHONPATH=./src`, so commands must be launched from the repository
root. `watch` transparently isolates the system procps binary from Conda ncurses while
leaving the training process library path unchanged. `robot-practice-init` remains
available only as a compatibility tool.

## Verify an eight-H20 worker

The platform must inject the NVIDIA runtime/device plugin, expose exactly eight
GPUs, provide the image runtime group or a writable root context, and allocate
at least 1 GiB to `/dev/shm`.

```bash
cd /any/path/to/practice
robot-practice-verify \
  --repo=. \
  --require-gpus=8 \
  --require-bf16 \
  --nccl-world-size=8 \
  --headless
```

Until this succeeds on the actual H20 worker and its recorded R535 driver, the
image status is **remote verification pending**, not remote-ready.
