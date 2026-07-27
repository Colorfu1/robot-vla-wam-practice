#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

exec conda run --no-capture-output -n robot-practice \
  python scripts/docker/generate_environment_locks.py "$@"
