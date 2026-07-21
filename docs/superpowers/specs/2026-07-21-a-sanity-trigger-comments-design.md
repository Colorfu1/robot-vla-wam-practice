# A-Sanity Launcher Trigger Comments Design

## Goal

Make the supported A-Sanity commands visible immediately when
`scripts/experiments/act/a_sanity.sh` is opened, without changing how the launcher parses
arguments or runs experiments.

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
root as `scripts/experiments/act/a_sanity.sh`. The comments use the existing public
interface and introduce no new arguments, defaults, files, or runtime behavior.

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
```

## Validation

- visually confirm the comment block is at the top of `a_sanity.sh` after the shebang;
- run `bash -n scripts/experiments/act/a_sanity.sh`;
- run `pytest -q tests/test_act_a_sanity.py` in the `robot-practice` environment;
- verify the script remains executable and the Git diff contains no generated artifacts.
