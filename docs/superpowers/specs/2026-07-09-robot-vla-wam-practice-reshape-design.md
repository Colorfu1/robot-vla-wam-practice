# Robot VLA WAM Practice Reshape Design

## Goal

Reshape the existing LeRobot practice workspace in place into a LeRobot-first
robot manipulation and VLA practice repository. The first deliverable is the repo
structure and experiment-recording contract, not model quality.

## Approach

Keep the existing repository and `robot_practice` Python package. Use LeRobot as
the primary workflow layer for datasets, policies, training, evaluation, and future
hardware interfaces. Add benchmark-specific areas for ManiSkill and LIBERO without
pretending either integration is complete.

Use a Python 3.10 conda environment named `robot-practice`, matching the initial
practice workflow:

```bash
conda create -n robot-practice python=3.10 -y
conda activate robot-practice
pip install lerobot
lerobot-info
pip install --upgrade mani_skill
pip install torch
```

The repo keeps the equivalent reproducible setup in `environment.yml` and
`requirements.txt`.

## Workspace Shape

- `envs/libero/` for LIBERO benchmark notes/configs.
- `envs/maniskill/` for ManiSkill benchmark notes/configs.
- `datasets/raw/` for raw demonstrations, simulator exports, and source data.
- `datasets/lerobot/` for converted LeRobotDataset-compatible artifacts.
- `policies/bc/`, `policies/act/`, `policies/diffusion_policy/`,
  `policies/smolvla_or_pi0/`, and `policies/wam_tiny/` for policy-family work.
- `eval/rollout.py`, `eval/metrics.py`, and `eval/perturbations.py` for evaluation
  entry points.
- `reports/experiment_log.md` for run records.
- `src/robot_practice/` remains the home for reusable local Python helpers.

## Experiment Record

Every experiment must record task suite, dataset size, observation keys, action
space, action horizon, control frequency, train loss, open-loop action error,
closed-loop success rate, latency, and failure cases.

## Artifact Handling

Generated datasets and training artifacts should stay out of git. `datasets/raw/`,
`datasets/lerobot/`, `data/`, and `outputs/` keep `.gitkeep` placeholders but ignore
generated contents.

## Verification

The initial verification is structural plus environment smoke testing:

- Repository skeleton exists.
- `conda env create -f environment.yml` can create `robot-practice`.
- `./scripts/doctor.sh` imports `torch` and `lerobot`.
- `lerobot-info` is available in the active environment.

