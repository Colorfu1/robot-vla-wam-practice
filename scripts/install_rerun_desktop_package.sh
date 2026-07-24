#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/install_rerun_desktop_package.sh [version]

Build and install robot-practice-rerun-desktop on a Ubuntu/Debian host.

Defaults:
- version: 0.1.1 (if omitted)

Environment:
- RERUN_CLI_PATH: explicit rerun viewer binary path (optional)
- PYTHON: python executable to use (optional)
- CI_MODE: set to 1 to skip cache update prompts (optional)
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

VERSION="${1:-0.1.1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEB_PATH="$PROJECT_ROOT/dist/robot-practice-rerun-desktop_${VERSION}_$(dpkg --print-architecture).deb"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "error: cannot resolve project root from $SCRIPT_DIR"
  exit 1
fi

cd "$PROJECT_ROOT"

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "error: dpkg-deb not installed"
  exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util

if not importlib.util.find_spec("robot_practice"):
    raise SystemExit(1)
if not importlib.util.find_spec("rerun_cli"):
    raise SystemExit(2)
PY
then
  echo "Preparing project import path for local checkout."
fi

if ! (PYTHON="$PYTHON_BIN" PYTHONPATH="$PROJECT_ROOT/src" ./scripts/build_rerun_desktop_deb.sh "$VERSION"); then
  echo "error: deb build failed"
  exit 1
fi

if [[ ! -f "$DEB_PATH" ]]; then
  echo "error: expected package missing: $DEB_PATH"
  exit 1
fi

if command -v sudo >/dev/null 2>&1; then
  sudo dpkg -i "$DEB_PATH"
else
  dpkg -i "$DEB_PATH"
fi

if [[ -z "${CI_MODE:-}" ]]; then
  if command -v update-desktop-database >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo update-desktop-database /usr/share/applications || true
    else
      update-desktop-database /usr/share/applications || true
    fi
  fi
  if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo gtk-update-icon-cache -f -t /usr/share/icons/hicolor || true
    else
      gtk-update-icon-cache -f -t /usr/share/icons/hicolor || true
    fi
  fi
fi

echo "Install completed. Launch options:"
echo "  /opt/robot-practice-rerun/bin/rr_of"
echo "  /opt/robot-practice-rerun/bin/rr_li"
