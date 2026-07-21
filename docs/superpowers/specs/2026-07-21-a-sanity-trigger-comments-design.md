# A-Sanity Launcher Header Reference Design

## Goal

Make the supported A-Sanity commands, every accepted CLI argument, and the fixed dataset
selection visible immediately when `scripts/experiments/act/a_sanity.sh` is opened,
without changing how the launcher parses arguments or runs experiments.

## Change

Keep the shebang as the first line, then add a compact shell-comment block before
`set -euo pipefail`. The block says that commands are run from the repository root and
shows three copyable workflows with these exact semantics:

1. create the zero-update B0 checkpoint with `--steps=0`;
2. assign the B0 `pretrained_model` path to a quoted `b0` shell variable and start 20,000
   updates from that exact checkpoint with `--policy-path="${b0}"`;
3. inspect that same B0-initialized 20,000-update command by adding `--print-command`,
   without launching it.

The B0 path remains a placeholder under
`outputs/act/a_sanity/<timestamp>/checkpoints/000000/pretrained_model`. The comment above
the assignment explicitly tells the reader to replace `<timestamp>` with the directory
printed by the preceding B0 command. Every invocation is written relative to the repository
root as `scripts/experiments/act/a_sanity.sh`.

After the trigger examples, an `Accepted arguments` block documents the complete public
CLI surface:

- `--steps=N`, including its default, canonical-decimal range, and step-zero meaning;
- `--policy-path=PATH`, including its positive-step restriction, checkpoint requirement,
  and caller-relative path resolution;
- `--print-command`, including its validation-only behavior;
- `-h` and `--help`.

It also states that the operational options are order-independent, may be specified only
once, and that unknown options are rejected.

A separate `Fixed A-Sanity dataset selection` block makes clear that the dataset repo,
task index, and five episode indices are experiment constants rather than accepted CLI
arguments. Episode selection remains immutable so A-Sanity runs continue to use the same
859 frames. The comments use the existing public interface and introduce no new arguments,
defaults, files, or runtime behavior.

The exact block is:

```bash
# Trigger examples (run from the repository root):
# Create the zero-update B0 checkpoint:
#   scripts/experiments/act/a_sanity.sh --steps=0
# Replace <timestamp> with the directory printed by the B0 command:
#   b0="outputs/act/a_sanity/<timestamp>/checkpoints/000000/pretrained_model"
# Train for 20,000 updates from that exact B0 checkpoint:
#   scripts/experiments/act/a_sanity.sh --steps=20000 --policy-path="${b0}"
# Print that same training command without launching it:
#   scripts/experiments/act/a_sanity.sh --steps=20000 --policy-path="${b0}" --print-command
#
# Accepted arguments:
#   --steps=N           Number of optimizer updates (default: 20000).
#                       N must be canonical decimal in [0, 2147483647].
#                       N=0 creates the B0 checkpoint without training.
#   --policy-path=PATH  Use a complete compatible ACT checkpoint for steps > 0.
#                       Relative paths resolve from the caller's working directory.
#   --print-command     Validate and print the command without creating output.
#   -h, --help          Print usage and exit.
# Operational options are order-independent and may each appear only once.
# Unknown options are rejected.
#
# Fixed A-Sanity dataset selection (not CLI arguments):
#   repo_id=lerobot/libero
#   task_index=29
#   episodes=[823,826,834,840,841]  (5 episodes, 859 frames total)
```

## Validation

- visually confirm the complete comment block is at the top of `a_sanity.sh` after the
  shebang and lists every accepted parser branch;
- compare the fixed dataset values against both the shell training command and the Python
  step-zero constants;
- run `bash -n scripts/experiments/act/a_sanity.sh`;
- run `pytest -q tests/test_act_a_sanity.py` in the `robot-practice` environment;
- verify the script remains executable and the Git diff contains no generated artifacts.
