#!/usr/bin/env bash
# Purpose: launch the fixed A0 single-step ACT baseline on the 20-episode task.
#
# Arguments:
#   --steps=N           Optimizer updates; default 20000. N=0 saves random step-0 weights.
#   --policy-path=PATH  Compatible A0 checkpoint used to initialize a positive-step run.
#   --print-command     Print the resolved command without running it.
#   -h, --help          Show usage.
#
# Examples (run from the repository root):
#   scripts/experiments/act/a0_single_step.sh --steps=0
#   scripts/experiments/act/a0_single_step.sh --steps=20000
#   scripts/experiments/act/a0_single_step.sh --steps=20000 \
#     --policy-path=outputs/act/a0_single_step/<timestamp>/checkpoints/000000/pretrained_model
#   scripts/experiments/act/a0_single_step.sh --steps=20000 --print-command
#
# Fixed experiment contract:
#   LIBERO object task 8; 20 episodes / 3233 frames; seed 1000; batch 8;
#   ACT chunk_size=1, n_action_steps=1, use_vae=false.

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/action_chunking.sh" --variant=a0 "$@"
