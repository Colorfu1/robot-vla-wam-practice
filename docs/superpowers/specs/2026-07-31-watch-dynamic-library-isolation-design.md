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
`LD_LIBRARY_PATH` and `LD_PRELOAD`, then uses
`exec /usr/bin/watch "$@"`. Installing it with mode `0755` and replacing the
wrapper process preserves the original PID, signal delivery, arguments, and
signal-derived exit status.

The Volc image also copies the same wrapper explicitly. Its `BASE_IMAGE` must
either be the newly built base image or an existing base whose three lock-hash
labels exactly match the current lock files. This permits a wrapper-only Volc
rebuild when the pinned CUDA registry is temporarily unavailable without
silently changing the Python environment.
No Conda, CUDA, TorchCodec, or training environment variable is changed for
other processes. The wrapper is intentionally limited to `watch`; broader
dynamic-library isolation requires separate H20 compatibility testing.

## Verification

Extend the image verifier with two checks:

1. `command -v watch` resolves to `/usr/local/bin/watch`;
2. `watch --help` succeeds and an invalid option fails, covering normal error
   and exit-code forwarding;
3. under a pseudo-terminal with fixed `TERM=xterm`, run `watch -n 0.1 true`
   behind a bounded timeout with a kill grace period. The process must remain
   alive until the deadline, and the timeout status is the expected result. Any
   early exit, including status 139, fails.

Retain all existing TorchCodec, CUDA, NCCL, LIBERO, and ManiSkill checks. The
change is accepted only when the new `watch` probes and the existing image
verification suite pass independently for both the base and Volc image variants.
