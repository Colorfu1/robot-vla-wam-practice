# LeRobot VS Code Debug Design

## Goal

Provide a local VS Code debug profile that makes the installed LeRobot ACT training
pipeline easy to inspect without starting a full production run or uploading artifacts.

## Launch Strategy

Launch `lerobot.scripts.lerobot_train` as a Python module with the
`robot-practice` interpreter. Module launch avoids the extra console-script wrapper while
following the same `main()` and `train()` path as the `lerobot-train` command.

The ignored local launch profile directly selects the `robot-practice` interpreter at
`<conda_env>/bin/python`. This is required even though the workspace also supplies
`python.defaultInterpreterPath`, because VS Code may have persisted a previously selected
interpreter. Each machine supplies its own concrete Conda environment path.

The debugger must set `justMyCode` to `false`, because the code under inspection lives in
the Conda environment's `site-packages` directory rather than this repository. Subprocess
debugging remains enabled for compatibility with Accelerate or data-loading child
processes, although the debug profile uses zero data-loader workers for deterministic
single-process stepping.

The profile uses the debugpy fields `"type": "debugpy"`, `"request": "launch"`,
`"module": "lerobot.scripts.lerobot_train"`, `"cwd": "${workspaceFolder}"`,
`"justMyCode": false`, and the case-sensitive `"subProcess": true`. Its environment sets
`PYTHONNOUSERSITE=1` to preserve package isolation.

## Debug Run

The profile trains ACT against `lerobot/aloha_mobile_cabinet` with three optimization
steps, batch size two, and zero data-loader workers. It disables Hub upload, checkpoint
saving, evaluation, and Weights & Biases. Logging occurs every step. LeRobot chooses a
timestamped output directory so repeated debug sessions do not collide.

The exact arguments are:

```text
--dataset.repo_id=lerobot/aloha_mobile_cabinet
--policy.type=act
--steps=3
--batch_size=2
--num_workers=0
--policy.push_to_hub=false
--save_checkpoint=false
--eval_freq=0
--wandb.enable=false
--log_freq=1
```

The first run may download the remote dataset into the Hugging Face cache. Later sessions
reuse that cache.

## VS Code Files

- `.vscode/settings.json` selects `<conda_env>/bin/python` for the local machine.
- `.vscode/launch.json` defines the short ACT training launch profile and explicitly
  selects the same interpreter for that launch.

Both files are local workstation configuration and remain ignored by the repository's
existing `.gitignore` rule for `.vscode/`.

## Breakpoint Path

The first breakpoint belongs at `cfg.validate()` in
`lerobot/scripts/lerobot_train.py`. At this point CLI parsing has completed, `cfg` is
inspectable, and the main training orchestration is beginning. Useful subsequent
breakpoints are dataset construction, policy construction, batch preprocessing, and
`update_policy()`.

## Verification

Validate both files as JSON and confirm the selected interpreter can import the training
module and `debugpy`. Verify the CLI arguments without launching the training module:
import `lerobot.scripts.lerobot_train` so that policy subclasses such as ACT are
registered, parse the arguments into `TrainPipelineConfig` with `draccus.parse`, and call
`cfg.validate()` only. Do not execute the module during this check, because execution
continues immediately into dataset construction after validation.
