# Robot VLA WAM Practice Agent Notes

## Robomimic EGL Conversion Diagnostic Record

For Square image conversion and exit-time `EGL_NOT_INITIALIZED`, read
`reports/daily/2026-09-23-Square图像转换EGL退出清理修复.md` before changing dependencies.
Use `scripts/experiments/rl/run_square_states_to_obs.sh` for the managed converter.
The cleanup-order mechanism was reproduced; the remote managed two-trajectory
conversion completed with cleanup errors=0 and exit code 0. The new output still
needs HDF5 and visual checks. Do not treat this smoke as rollout validation.

## Project Identity

This is a LeRobot-first practice workspace for robot manipulation, VLA policy
experiments, ManiSkill simulation work, and future LIBERO language-conditioned
manipulation experiments. It is not an upstream LeRobot source checkout. Treat
LeRobot as an installed Python dependency and keep local experiments isolated in
this repository.

## Repository Layout

- `environment.yml` defines the `robot-practice` conda environment on Python 3.10.
- `requirements.txt` installs `lerobot`, `torch`, `mani_skill`, and lightweight
  notebook/plotting tools.
- `scripts/doctor.sh` checks Python, ffmpeg, torch, lerobot, and the LeRobot CLI.
- `scripts/quickstart_info.sh` prints LeRobot CLI help and the installed package path.
- `src/robot_practice/` is for reusable local practice code.
- `envs/maniskill/` is for ManiSkill benchmark environment notes/configs.
- `envs/libero/` is for LIBERO benchmark environment notes/configs.
- `datasets/raw/` is for source demonstrations or simulator exports.
- `datasets/lerobot/` is for converted LeRobotDataset-compatible artifacts.
- `policies/` is organized by policy family: `bc`, `act`, `diffusion_policy`,
  `smolvla_or_pi0`, and `wam_tiny`.
- `eval/` holds rollout, metric, and perturbation utilities.
- `plans/` is the single entry point for work that still needs to be executed. It
  replaces the former `docs/superpowers/plans/`; see `plans/README.md` for the
  naming, status vocabulary, and required sections. Plans describe what to run and
  how to judge completion; they do not carry results or report prose.
- `reports/experiment_log.md` is the required experiment record.
- `data/` remains for compatibility with the earlier workspace, but new datasets
  should use `datasets/`.
- `outputs/` is for generated logs, videos, checkpoints, and training outputs.
- `docs/superpowers/specs/2026-06-29-lerobot-practice-workspace-design.md`
  records the original workspace design.

## Setup And Verification

Use the project conda environment:

```bash
conda env create -f environment.yml
conda activate robot-practice
```

After dependency changes:

```bash
pip install -r requirements.txt
```

Run the smoke checks before considering environment or dependency work complete:

```bash
./scripts/doctor.sh
./scripts/quickstart_info.sh
```

## Development Guidance

- Keep practice code under `src/robot_practice/`.
- Prefer small, reproducible scripts or notebooks over modifying installed LeRobot
  package files.
- Use LeRobot as the main workflow layer for dataset conversion, policy training,
  evaluation, and future hardware interfaces.
- Use ManiSkill for simulation training and fast closed-loop evaluation.
- Use LIBERO for language-conditioned manipulation and VLA benchmark suites.
- Do not commit generated artifacts from `data/`, `datasets/raw/`,
  `datasets/lerobot/`, `outputs/`, `wandb/`, `runs/`, `checkpoints/`, videos, or
  model files.
- The project targets Python 3.10 and Ruff line length 100.
- `PYTHONNOUSERSITE=1` is part of the environment design; preserve it so user-site
  packages do not leak into this workspace.
- Every long-running training, evaluation, or rollout entrypoint must expose
  observable progress while it runs. At minimum, report the completed/total
  work units (training steps, evaluation episodes/tasks, or rollout episodes)
  periodically to stdout and to a persistent log under the experiment output.
  A `tqdm` bar is preferred for foreground runs; for `nohup`, multi-process, or
  multi-GPU runs, structured `STATUS`/`DONE` lines with counts are sufficient.
  A final JSON written only at the end is not an acceptable progress signal.

## Required Experiment Fields

Every experiment entry must record:

- task suite
- dataset size
- observation keys
- action space
- action horizon
- control frequency
- train loss
- open-loop action error
- closed-loop success rate
- latency
- failure cases

## Summary And Publication Safety

These rules are mandatory whenever Codex writes a progress summary or prepares a commit,
upload, or push from this repository:

- Write shareable summaries with repository-relative paths. Do not include user names,
  absolute home paths, machine names, session or conversation identifiers, personal vault
  locations, credentials, tokens, private endpoints, raw terminal dumps, or cache paths.
- Summaries may include public benchmark identifiers and selected aggregate experiment
  metrics, but must not reproduce raw observations, actions, predictions, media, weights,
  logs, cache contents, or machine-specific manifests.
- Before any commit, upload, or push, inspect `git status`, review the exact staged diff, and
  scan tracked content for likely secrets and personal markers. Resolve findings before
  publication.
- Never stage or publish generated ML artifacts covered by `.gitignore`, including datasets,
  checkpoints, model weights, predictions, Rerun recordings, videos, logs, caches, or
  experiment output directories.
- Stage only files in the explicitly requested scope. Preserve unrelated dirty-worktree
  changes as user-owned work.
- Commit and push are separate permissions. Never push, upload, create a pull request, or
  otherwise change a remote unless the user explicitly requests that external action.
- Before a requested push, state the target remote and branch, inspect the outgoing commit
  range, and repeat the sensitive-content check.
- Progress reports must distinguish completed work, verification, measured results, and
  remaining work. Do not turn planned or smoke-test behavior into a claimed formal result.

## Progress Log Writing And Vault Sync

- Treat dated progress reports under `reports/` as the canonical source. Name each one
  `YYYY-MM-DD-<clear Chinese topic>.md`; the H1 must contain the same date and clear Chinese
  topic.
- Immediately after the H1, add `## 重要进展` with four to six concise bullets covering the
  most important completed outcomes and any crucial limitation.
- Keep completed work, measured results, limitations, and next steps distinct. Never present
  a planned, smoke, teacher-forced, or unmeasured result as a formal experiment result.
- Use repository-relative links and obey all rules in `Summary And Publication Safety`.
- Sync reports to a personal note vault only when the user explicitly requests it and gives
  the target directory. Do not persist that absolute vault path in tracked files.
- Before replacing an older vault note, write and verify the new canonical copies first.
  Compare each destination byte-for-byte with its repository source, then remove only the
  explicitly mapped superseded note. Leave unrelated vault notes untouched.

## Remote Experiment Failure Diagnosis

For remote training or evaluation failures, follow this loop and wait for the user to
paste the remote logs before making another diagnosis:

1. Separate launcher errors from worker errors. Record the exact phase, variant, GPU
   assignment, output root, worker return code, and the command that was actually run.
2. Inspect the persistent output in this order: top-level `run.log`, the affected
   `task_logs/*.log`, and any nested logs named by the worker. Capture the final log
   section plus targeted matches for `Traceback`, `Fatal`, `Aborted`, `CUDA`, `EGL`,
   `GL_`, `mjr_`, `Hydra`, `out of memory`, and `FileNotFound`.
3. Verify cheap prerequisites before theorizing: expected dataset metadata, model and
   tokenizer files, output/cache contents, active Python module paths, and source/vendor
   checksums when a manually synchronized vendor tree is involved.
4. State three to five ranked, falsifiable hypotheses. Run one minimal experiment that
   changes one variable and distinguishes the leading hypotheses. Do not present a
   workaround or a passing control as proof of root cause.
5. For a code fix, modify the local repository first, inspect the exact diff, provide
   the local-to-remote `rsync` command, then provide the remote vendor-copy command if
   the relevant source is ignored or vendored. Provide one exact remote validation/run
   command and wait for its logs.
6. Keep confirmed cause, workaround, measured result, and remaining uncertainty
   separate. Never silently edit the remote checkout or claim a remote run succeeded
   without its pasted log/result.

When a failure is in repository source, a Docker rebuild is normally unnecessary: sync
the changed repository file to the remote checkout. Rebuild the image only when the
Dockerfile, installed dependencies, base image, or image-layer files changed. For
ignored `third_party` sources, manually synchronize the source into the runtime vendor
tree and verify both checksums before rerunning.
