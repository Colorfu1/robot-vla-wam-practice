# Zero-init arbitrary-path training image design

## Goal

After entering the container, a user can change to the root of a
`robot-practice` checkout at any filesystem path and immediately run repository
Bash scripts, local Python modules, and multi-GPU launchers. No
`robot-practice-init` call or editable installation is required.

## Design

The image exports `PYTHONPATH=./src`. Python resolves this relative entry from
the process working directory, so a command launched from any repository root
imports that checkout's `robot_practice` package. This avoids fixed mount paths,
entrypoint-dependent setup, and stale editable-install metadata. Running from a
repository subdirectory is intentionally unsupported.

The image also exports
`LD_LIBRARY_PATH` with the Conda environment first, followed by NVIDIA driver
and CUDA library directories. Prioritizing the Conda C++ runtime fixes
TorchCodec loading on the Volc H20 worker without hiding injected NVIDIA
driver libraries.

The Volc wrapper remains root-based, has an empty entrypoint, and uses
`/workspace` only as its initial directory. `robot-practice-init` remains in the
image for compatibility but is not part of the normal workflow.

## Verification

The repository is mounted read-only at a non-default path containing spaces.
The verifier changes directly to that root and, without initialization:

1. imports `robot_practice` and verifies it came from the mounted checkout;
2. overrides the image entrypoint to prove startup hooks are unnecessary;
3. loads TorchCodec and decodes a generated fixture plus one real PushT frame when present;
4. launches one Python worker per visible GPU and verifies every worker imports this checkout;
5. validates single- and multi-GPU launcher command construction;
6. retains CUDA, NCCL, PushT, MuJoCo, LIBERO, and ManiSkill checks.

Success means the same image works at arbitrary repository roots on local Docker
and on Volc MLP, even when the platform overrides the image entrypoint.
