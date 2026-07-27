#!/usr/bin/env bash
set -euo pipefail

umask 0002

mkdir -p \
  "${HF_HOME:-$HOME/.cache/huggingface}" \
  "${MPLCONFIGDIR:-$HOME/.cache/matplotlib}" \
  "${XDG_RUNTIME_DIR:-/tmp/runtime-robot-practice}"

if [[ "$#" -eq 0 ]]; then
  set -- bash
fi

exec "$@"
