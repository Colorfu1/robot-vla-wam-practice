# Portable Robot Practice Training Environment

This image contains the CUDA, Conda, Python, LeRobot, PushT, LIBERO, and
ManiSkill environment only. It does not contain this repository, datasets,
checkpoints, outputs, caches, or credentials.

## Generate locks

Run this only after intentionally changing the local `robot-practice`
environment:

```bash
bash scripts/docker/generate_environment_locks.sh
```

The generator excludes this repository's editable package and rejects local,
VCS, credential-bearing, and unapproved package sources.

## Build

Choose the registry/repository at invocation time:

```bash
IMAGE_REF='<registry>/<namespace>/practice:latest'

bash scripts/docker/build_training_image.sh \
  --image="$IMAGE_REF"
```

The script also creates an immutable `env-YYYYMMDD-<lock hash>` tag. No registry
destination is committed to the repository.

If Docker Hub is unavailable, `--base-image` may point to a trusted mirror of the
same pinned CUDA amd64 digest. Changing or omitting the digest is not allowed for
a release build. Dependency downloads are direct by default. Use
`--anaconda-proxy` when only Miniconda and Conda require the shell's
`HTTP_PROXY`/`HTTPS_PROXY`; APT and PyPI remain direct. `--use-proxy` is the
explicit all-download fallback. Docker does not persist these predefined proxy
build arguments in the resulting image.

## Verify locally

```bash
bash scripts/docker/verify_training_image.sh \
  --image="$IMAGE_REF" \
  --repo="$PWD" \
  --require-gpus=1

bash scripts/docker/audit_training_image.sh \
  --image="$IMAGE_REF" \
  --registry-string="${IMAGE_REF%%/*}" \
  --save-layers
```

The verification mounts the repository read-only at non-default paths, replaces
a stale editable installation, and checks CUDA and headless simulation.

## Push

Registry authentication remains an explicit user action:

```bash
docker login '<registry>'
docker push "$IMAGE_REF"
```

Push the immutable tag printed by the build script as well.

## Use the repository at any path

After starting a worker from the image, put the repository anywhere and run:

```bash
robot-practice-init /any/path/to/practice
cd /any/path/to/practice
```

Existing commands continue to work:

```bash
bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=pilot
```

The editable installation must be repeated if a new container mounts the
repository at a different path.

## Verify an eight-H20 worker

The platform must inject the NVIDIA runtime/device plugin, expose exactly eight
GPUs, provide the image runtime group or a writable root context, and allocate
at least 1 GiB to `/dev/shm`.

```bash
robot-practice-init /any/path/to/practice

robot-practice-verify \
  --repo=/any/path/to/practice \
  --require-gpus=8 \
  --nccl-world-size=8 \
  --headless
```

Until this succeeds on the actual H20 worker and its recorded R535 driver, the
image status is **remote verification pending**, not remote-ready.
