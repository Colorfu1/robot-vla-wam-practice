# Robot VLA WAM Practice Agent Notes

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
