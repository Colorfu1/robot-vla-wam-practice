#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/build_rerun_desktop_deb.sh [version]

Build a Debian package that installs:
- /opt/robot-practice-rerun/bin/rr_of   (Offline identity)
- /opt/robot-practice-rerun/bin/rr_li   (LIBERO identity)
- /opt/robot-practice-rerun/bin/rerun   (unknown identity fallback)
- /usr/share/applications/rr_of.desktop
- /usr/share/applications/rr_li.desktop
- /usr/share/applications/rerun.desktop
- /usr/share/icons/hicolor/256x256/apps/rr_of.png
- /usr/share/icons/hicolor/256x256/apps/rr_li.png
- /usr/share/icons/hicolor/256x256/apps/rerun.png

Prereqs:
- Ubuntu/Debian with dpkg-deb installed
- Python env with robot-practice editable install (for robot-practice.rerun_launcher)
- Rerun native viewer available (usually via `rerun` package)

The script will run:
- python installer for identity-aware desktop integration via
  robot_practice.rerun_launcher.install_desktop_integration
EOF
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
  exit 0
fi

VERSION="${1:-0.1.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "error: dpkg-deb not found"
  exit 1
fi

if ! command -v update-desktop-database >/dev/null 2>&1; then
  echo "warn: update-desktop-database not found, desktop cache update may be skipped"
fi

if ! command -v gtk-update-icon-cache >/dev/null 2>&1; then
  echo "warn: gtk-update-icon-cache not found, icon cache update may be skipped"
fi

PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

PROJECT_PYTHONPATH="${PROJECT_ROOT}/src"
if [[ -n "${PYTHONPATH-}" ]]; then
  PROJECT_PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
fi
LAUNCHER="${RERUN_DESKTOP_LAUNCHER:-}"
if [[ -z "$LAUNCHER" ]]; then
  LAUNCHER="$(command -v robot-rerun || true)"
  if [[ -z "$LAUNCHER" ]]; then
    LAUNCHER="$(dirname "$PYTHON_BIN")/robot-rerun"
  fi
fi

if ! PYTHONPATH="${PYTHONPATH-}" \
  RERUN_DESKTOP_LAUNCHER="$LAUNCHER" \
  RERUN_DESKTOP_NATIVE_BINARY="${RERUN_CLI_PATH:-}" \
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path
from robot_practice.rerun_launcher import install_desktop_integration

launcher = Path(os.environ["RERUN_DESKTOP_LAUNCHER"])
native = os.environ.get("RERUN_DESKTOP_NATIVE_BINARY", "")
if not native:
    from shutil import which

    native = which("rerun")
    if not native:
        from rerun_cli import __file__ as rerun_cli_file

        native = str(Path(rerun_cli_file).resolve().parent / "rerun")

install_desktop_integration(
    launcher=launcher,
    native_binary=Path(native),
)
PY
then
  if ! PYTHONPATH="$PROJECT_PYTHONPATH" \
    RERUN_DESKTOP_LAUNCHER="$LAUNCHER" \
    RERUN_DESKTOP_NATIVE_BINARY="${RERUN_CLI_PATH:-}" \
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from robot_practice.rerun_launcher import install_desktop_integration

launcher = Path(os.environ["RERUN_DESKTOP_LAUNCHER"])
native = os.environ.get("RERUN_DESKTOP_NATIVE_BINARY", "")
if not native:
    from shutil import which

    native = which("rerun")
    if not native:
        from rerun_cli import __file__ as rerun_cli_file

        native = str(Path(rerun_cli_file).resolve().parent / "rerun")

install_desktop_integration(
    launcher=launcher,
    native_binary=Path(native),
)
PY
  then
    echo "error: cannot run robot_practice.rerun_launcher install_desktop_integration"
    echo "Hint: activate your robot-practice env and make rerun_cli/robot_practice importable."
    echo "Tip: run from repository root or set PYTHONPATH=$PROJECT_ROOT/src."
    exit 1
  fi
fi

ARCH="$(dpkg --print-architecture)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
NATIVE_DIR="$DATA_HOME/robot-practice/rerun-native"

if [[ ! -x "$NATIVE_DIR/rr_of" || ! -x "$NATIVE_DIR/rr_li" ]]; then
  echo "error: patched identity binaries missing under $NATIVE_DIR"
  echo "Need rr_of and rr_li from install-desktop."
  exit 1
fi

if [[ -x "$NATIVE_DIR/rerun" ]]; then
  RERUN_EXEC="/opt/robot-practice-rerun/bin/rerun"
else
  RERUN_EXEC="/usr/bin/rerun"
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

PACKAGE_ROOT="$WORKDIR/robot-practice-rerun-desktop_${VERSION}_${ARCH}"
mkdir -p \
  "$PACKAGE_ROOT/DEBIAN" \
  "$PACKAGE_ROOT/opt/robot-practice-rerun/bin" \
  "$PACKAGE_ROOT/usr/share/applications" \
  "$PACKAGE_ROOT/usr/share/icons/hicolor/256x256/apps"

cat > "$PACKAGE_ROOT/DEBIAN/control" <<EOF
Package: robot-practice-rerun-desktop
Version: ${VERSION}
Section: graphics
Priority: optional
Architecture: ${ARCH}
Maintainer: Robot Practice
Description: Rerun viewer launchers for Robot Practice with identity-bound dock icons.
EOF

cat > "$PACKAGE_ROOT/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t /usr/share/icons/hicolor || true
fi
exit 0
EOF

cat > "$PACKAGE_ROOT/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t /usr/share/icons/hicolor || true
fi
exit 0
EOF

chmod 755 \
  "$PACKAGE_ROOT/DEBIAN/postinst" \
  "$PACKAGE_ROOT/DEBIAN/postrm"

install -m 755 "$NATIVE_DIR/rr_of" "$PACKAGE_ROOT/opt/robot-practice-rerun/bin/rr_of"
install -m 755 "$NATIVE_DIR/rr_li" "$PACKAGE_ROOT/opt/robot-practice-rerun/bin/rr_li"
if [[ "$RERUN_EXEC" == "/opt/robot-practice-rerun/bin/rerun" ]]; then
  install -m 755 "$NATIVE_DIR/rerun" "$PACKAGE_ROOT/opt/robot-practice-rerun/bin/rerun"
fi

install -m 644 \
  "$PROJECT_ROOT/src/robot_practice/assets/rerun_offline.png" \
  "$PACKAGE_ROOT/usr/share/icons/hicolor/256x256/apps/rr_of.png"
install -m 644 \
  "$PROJECT_ROOT/src/robot_practice/assets/rerun_libero.png" \
  "$PACKAGE_ROOT/usr/share/icons/hicolor/256x256/apps/rr_li.png"
install -m 644 \
  "$PROJECT_ROOT/src/robot_practice/assets/rerun_unknown.png" \
  "$PACKAGE_ROOT/usr/share/icons/hicolor/256x256/apps/rerun.png"

cat > "$PACKAGE_ROOT/usr/share/applications/rr_of.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Robot Rerun Offline
Comment=Rerun viewer for Robot Practice offline rollouts
Exec=/opt/robot-practice-rerun/bin/rr_of %F
Icon=rr_of
Terminal=false
NoDisplay=false
StartupNotify=true
StartupWMClass=rr_of
EOF

cat > "$PACKAGE_ROOT/usr/share/applications/rr_li.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Robot Rerun LIBERO
Comment=Rerun viewer for Robot Practice LIBERO rollouts
Exec=/opt/robot-practice-rerun/bin/rr_li %F
Icon=rr_li
Terminal=false
NoDisplay=false
StartupNotify=true
StartupWMClass=rr_li
EOF

cat > "$PACKAGE_ROOT/usr/share/applications/rerun.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Robot Rerun
Comment=Rerun viewer
Exec=$RERUN_EXEC %F
Icon=rerun
Terminal=false
NoDisplay=false
StartupNotify=true
StartupWMClass=rerun
EOF

OUTPUT_DIR="$PROJECT_ROOT/dist"
mkdir -p "$OUTPUT_DIR"
dpkg-deb --build "$PACKAGE_ROOT" "$OUTPUT_DIR/robot-practice-rerun-desktop_${VERSION}_${ARCH}.deb"

echo "Built: $OUTPUT_DIR/robot-practice-rerun-desktop_${VERSION}_${ARCH}.deb"
