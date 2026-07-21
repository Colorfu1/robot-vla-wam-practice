#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

echo "== lerobot-info --help =="
lerobot-info --help

echo
echo "== installed package location =="
python - <<'PY'
import lerobot

print(lerobot.__file__)
PY
