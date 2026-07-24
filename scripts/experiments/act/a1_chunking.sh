#!/usr/bin/env bash
# Purpose: launch the fixed A1 action-chunking ACT run on the 20-episode task.
#
# Arguments:
#   --steps=N           Optimizer updates; default 20000. N=0 saves random step-0 weights.
#   --policy-path=PATH  Compatible A1 checkpoint used to initialize a positive-step run.
#   --print-command     Print the resolved command without running it.
#   -h, --help          Show usage.
#
# Examples (run from the repository root):
#   scripts/experiments/act/a1_chunking.sh --steps=0
#   scripts/experiments/act/a1_chunking.sh --steps=20000
#   scripts/experiments/act/a1_chunking.sh --steps=20000 \
#     --policy-path=outputs/act/a1_chunking/<timestamp>/checkpoints/000000/pretrained_model
#   scripts/experiments/act/a1_chunking.sh --steps=20000 --print-command
#
# Fixed experiment contract:
#   LIBERO object task 8; 20 episodes / 3233 frames; seed 1000; batch 8;
#   ACT chunk_size=20, n_action_steps=5, use_vae=false.
#   chunk_size=20 is the supervised training horizon. n_action_steps=5 is saved
#   as the checkpoint inference default and does not limit training loss to 5 actions.

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/action_chunking.sh" --variant=a1 "$@"
