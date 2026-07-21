# LeRobot Practice Workspace Design

## Goal

Create a simulation-first practice workspace for learning LeRobot without cloning or modifying LeRobot itself.

## Approach

Use LeRobot as an installed Python dependency in an isolated Python 3.12 conda environment. Keep local practice code, notebooks, configs, generated datasets, and experiment outputs in this repository.

This gives a small, easy-to-reset workspace for tutorials and simulation experiments. Hardware-specific dependencies can be added later when a robot target is known.

## Workspace Shape

- `environment.yml` defines the conda environment.
- `requirements.txt` installs core LeRobot and lightweight practice tools.
- `scripts/doctor.sh` checks system and Python dependencies.
- `scripts/quickstart_info.sh` runs basic LeRobot discovery commands.
- `src/robot_practice/` holds local practice code.
- `configs/`, `notebooks/`, `data/`, and `outputs/` provide places for future experiments.

## Error Handling

Setup scripts use strict shell flags and print actionable checks. Large generated artifacts are ignored by git so failed runs do not dirty the repo with datasets, videos, logs, or checkpoints.

## Testing

The initial verification is a smoke test:

- Python version is available.
- `ffmpeg` is available.
- `torch` imports.
- `lerobot` imports.
- `lerobot-info` is on `PATH` when installed by the active environment.
