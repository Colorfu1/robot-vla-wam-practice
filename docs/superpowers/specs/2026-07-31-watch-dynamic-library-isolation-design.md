# Docker watch dynamic-library isolation design

## Goal

Make `watch -n 2 nvidia-smi` work directly in the training images while preserving
the global Conda-first `LD_LIBRARY_PATH` currently required by TorchCodec on the
remote H20 runtime.

## Context

The base and Volc wrapper images export the Conda environment library directory
ahead of system libraries. This lets TorchCodec load the compatible Conda C++
runtime, but it also makes the system `/usr/bin/watch` load Conda's ncurses and
tinfo libraries. That mixed runtime causes `watch` to segfault. Running the same
binary with `LD_LIBRARY_PATH` and `LD_PRELOAD` removed succeeds.

## Design

Add a small executable wrapper at `docker/training/watch` and install it as
`/usr/local/bin/watch` in the base image. The wrapper removes only
`LD_LIBRARY_PATH` and `LD_PRELOAD`, then executes `/usr/bin/watch` with the
original arguments and exit behavior.

Both the normal image and `Dockerfile.volc-mlp` inherit `/usr/local/bin/watch`.
No Conda, CUDA, TorchCodec, or training environment variable is changed for
other processes. The wrapper is intentionally limited to `watch`; broader
dynamic-library isolation requires separate H20 compatibility testing.

## Verification

Extend the image verifier with two checks:

1. `command -v watch` resolves to `/usr/local/bin/watch`;
2. a short pseudo-terminal `watch -n 0.1 true` probe runs until an external
   timeout instead of terminating with a segmentation fault.

Retain all existing TorchCodec, CUDA, NCCL, LIBERO, and ManiSkill checks. The
change is accepted only when the new `watch` probe and the existing image
verification suite pass.

