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

## Thin runtime-fix layers

Two small layers sit on top of the already-built W0 image. They exist so a runtime
library fix does not rebuild the large FastWAM environments and vendor checkouts.
Both must be reapplied when the W0 image itself is rebuilt, or the fixes vanish
silently.

| Dockerfile | What it adds | Why |
| --- | --- | --- |
| `Dockerfile.w0-fastwam-runtime-patch` | `libglu1-mesa` | robosuite's EGL cleanup path imports libGLU through PyOpenGL |
| `Dockerfile.w0-fastwam-torchcodec-fix` | FFmpeg 4.4 runtime libraries | makes `torchcodec` loadable, which stops the training DataLoader from exhausting host RAM |
| `Dockerfile.w0-fastwam-torchcodec-fix` | `gcc` + `libc6-dev` | triton compiles its CUDA driver shim at `import deepspeed`; without a compiler the launcher dies once the cache is cold |

The torchcodec layer fixes a failure that is invisible at runtime and fatal hours
later. `lerobot` selects its decode backend with
`importlib.util.find_spec("torchcodec")`, which succeeds because the Python package
is installed, but `from torchcodec.decoders import VideoDecoder` then fails with
`Could not load libtorchcodec ... We support versions 4, 5, 6 and 7` because the
image only carries FFmpeg 8 (`libavutil.so.60`) inside the conda env.
`decode_video_frames` catches that and falls back to pyav *from inside the except
block*, so every decode leaves an exception whose traceback pins the failed import's
frames. Those frames form reference cycles that also hold the pyav objects and the
decoded tensors, so nothing is freed until a full gc pass.

Measured, with `scripts/experiments/literature/probe_w0_dataset_memory.py`:

| | before | after |
| --- | ---: | ---: |
| large tensors held in cycles (40 samples) | 4 | 0 |
| `frame` / `traceback` objects in cycles | 50 / 12 | 0 / 0 |
| pyav objects in cycles (`VideoFormat`, ...) | 112 | 0 |
| objects gc must free per report (150 samples) | 48-96 | 0 |
| host RSS growth | ~29 MiB/sample | 0.0 MiB/sample |

Across 64 DataLoader workers that was about 5 GiB per training step, and four 8-GPU
runs died at steps 265, 270, 560 and 598 on reaching the 1788.9 GiB cgroup limit:
the OOM killer took a worker, its rank blocked in `queue.get` and never enqueued its
gradient allreduce, and the other seven spun in DeepSpeed's `allreduce_and_copy`
until the 600 s NCCL watchdog killed the job.

FFmpeg 4.4 from jammy is used because it is the least invasive version torchcodec
accepts. It adds only FFmpeg 4-era sonames, which cannot collide with the conda
env's FFmpeg 8, so no conda package is touched and `LD_LIBRARY_PATH` is unchanged.
Installing FFmpeg 6/7 would have required either resolving conda-forge into the
pinned training env or prepending a foreign prefix to `LD_LIBRARY_PATH`, both of
which risk shadowing `libstdc++`/`zlib` for torch. The LIBERO dataset is AV1, so
AV1 decoding is mandatory; jammy's `libavcodec58` carries libdav1d and the layer
fails the build if the import does not work.

### Why the compiler is in the same layer

`import deepspeed` reaches
`deepspeed/model_implementations/transformers/ds_transformer.py`, which imports `TritonMLP`
whenever `get_accelerator().is_triton_supported()` is true -- that is, on any device with
compute capability >= 8, so H20 (9.0) qualifies. The `@triton.autotune` decorator then
resolves `driver.active`, whose `CudaUtils` constructor compiles
`triton/backends/nvidia/driver.c` and caches the result under `~/.triton/cache`.

This is a pre-existing latent failure, not a consequence of the FFmpeg change: the previous
image fails identically three times out of three from a cold cache. It stayed hidden because
a long-lived container keeps the compiled `cuda_utils.so`, and because a process where NVML
fails to initialise reports `torch.cuda.is_available() == False`, which makes
`is_triton_supported()` false and skips the import entirely. Recreating the container
discarded the cache and the failure surfaced as
`RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`

`libc6-dev` must be named explicitly: it is only a *Recommends* of `gcc-11`, so
`--no-install-recommends` omits it and the compile dies on
`fatal error: stdlib.h: No such file or directory`.

Training numerics are unaffected. `TritonMLP` and `TritonSelfAttention` are inference-path
classes that ZeRO-1 training never instantiates, so only the import changes.

```bash
FIX_TAG="w0-fastwam-$(date +%Y%m%d)-torchcodec-gcc"

docker build \
  --file docker/training/Dockerfile.w0-fastwam-torchcodec-fix \
  --build-arg "BASE_IMAGE=${IMAGE_REPOSITORY}:latest" \
  --tag "${IMAGE_REPOSITORY}:${FIX_TAG}" \
  docker/training
```

Verify against a real AV1 file before pushing; the build-time check only proves the
import, not decoding:

```bash
docker run --rm -v "${PWD}:/repo:ro" -w /repo \
  --entrypoint /opt/conda/envs/fastwam-upstream/bin/python \
  "${IMAGE_REPOSITORY}:${FIX_TAG}" \
  scripts/experiments/literature/probe_w0_dataset_memory.py \
    --stage lerobot --samples 150 --work-dir /tmp/probe
```

Expect `gc freed 0` on every report and `0.0 MiB/sample`.

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
