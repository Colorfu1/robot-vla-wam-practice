#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
vendor_root="${W0_VENDOR_ROOT:-/opt/robot-practice/w0-vendor}"
patch_path="${repo_root}/docker/training/vendor-patches/fastwam-memory-efficient-loading.patch"

usage() {
  cat <<'EOF'
Usage:
  apply_w0_fastwam_memory_patch.sh [--vendor-root=PATH] [--patch=PATH]

The operation is idempotent:
  - an already-applied patch is reported and skipped;
  - a clean pinned checkout is patched;
  - an incompatible or partially modified checkout fails without overwriting files.
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --vendor-root=*) vendor_root="${argument#*=}" ;;
    --patch=*) patch_path="${argument#*=}" ;;
    --help|-h) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${patch_path}" != /* ]]; then
  patch_path="$(cd -- "$(dirname -- "${patch_path}")" && pwd)/$(basename -- "${patch_path}")"
fi

fastwam_root="${vendor_root}/FastWAM"
if [[ ! -d "${fastwam_root}/.git" ]]; then
  echo "error: FastWAM Git checkout not found: ${fastwam_root}" >&2
  exit 1
fi
if [[ ! -s "${patch_path}" ]]; then
  echo "error: memory-loading patch not found: ${patch_path}" >&2
  exit 1
fi

git_command=(git -c "safe.directory=${fastwam_root}" -C "${fastwam_root}")

if "${git_command[@]}" apply --reverse --check "${patch_path}" >/dev/null 2>&1; then
  echo "W0 FastWAM memory patch: already applied"
elif "${git_command[@]}" apply --check "${patch_path}" >/dev/null 2>&1; then
  "${git_command[@]}" apply "${patch_path}"
  echo "W0 FastWAM memory patch: applied"
else
  echo "error: cannot apply W0 FastWAM memory patch cleanly: ${fastwam_root}" >&2
  echo "hint: inspect 'git -C ${fastwam_root} status --short' and use the pinned checkout" >&2
  exit 1
fi

"${git_command[@]}" status --short -- \
  src/fastwam/models/wan22/helpers/loader.py \
  src/fastwam/models/wan22/helpers/io.py \
  src/fastwam/models/wan22/action_dit.py \
  src/fastwam/models/wan22/fastwam.py
