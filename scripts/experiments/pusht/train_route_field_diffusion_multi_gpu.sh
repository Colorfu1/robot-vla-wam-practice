#!/usr/bin/env bash
set -euo pipefail

exec conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.train_route_field_diffusion_multi_gpu "$@"
