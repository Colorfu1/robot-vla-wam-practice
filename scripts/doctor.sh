#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

echo "== System =="
command -v python
python --version

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | sed -n '1p'
else
  echo "ffmpeg: missing"
  exit 1
fi

echo
echo "== Python imports =="
python - <<'PY'
import importlib.util
import sys

print(f"python_executable={sys.executable}")

for package in ["torch", "lerobot"]:
    spec = importlib.util.find_spec(package)
    if spec is None:
        raise SystemExit(f"{package}: missing")
    module = __import__(package)
    version = getattr(module, "__version__", "unknown")
    print(f"{package}={version}")
PY

echo
echo "== LeRobot CLI =="
if command -v lerobot-info >/dev/null 2>&1; then
  lerobot-info --help | sed -n '1,40p'
else
  echo "lerobot-info: missing from PATH"
  exit 1
fi

echo
echo "doctor: ok"
