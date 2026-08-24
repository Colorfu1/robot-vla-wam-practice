#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

phase=""
config="configs/experiments/literature/w0_fastwam_reproduction.json"

usage() {
  cat <<'EOF'
Usage:
  run_w0_fastwam.sh --phase=validate-config [--config=...]
  run_w0_fastwam.sh --phase=preflight --config=... --output-root=...
  run_w0_fastwam.sh --phase=validate-environments --vendor-root=... --output-root=...
  run_w0_fastwam.sh --phase=validate-artifacts --cache-root=... --output-root=...
  run_w0_fastwam.sh --phase=eval-upstream-released --checkpoint=... --output-root=... [--overwrite]
  run_w0_fastwam.sh --phase=eval-lerobot-released --policy-path=... --output-root=...
  run_w0_fastwam.sh --phase=train-causal-pair --variant=with_video --output-root=...
  run_w0_fastwam.sh --phase=eval-causal-pair --with-video=... --no-video=...
  run_w0_fastwam.sh --phase=gate-w0 --released=... --causal=... --deployment=...

The runtime image must already provide the pinned conda environments and the
repo must already contain the pinned checkouts/artifacts.  This wrapper never
creates an environment, installs a package, clones/fetches a checkout, or
downloads/extracts/preprocesses a pretrained artifact.  All remaining
--key=value and boolean flags are forwarded to the repo-local Python wrapper.
Formal rollout/training is intentionally never started by this shell script
unless the user explicitly selects that phase.
EOF
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

for argument in "$@"; do
  case "${argument}" in
    --phase=*) phase="${argument#*=}" ;;
    --config=*) config="${argument#*=}" ;;
    --help|-h) usage; exit 0 ;;
  esac
done

if [[ -z "${phase}" ]]; then
  echo "error: --phase is required" >&2
  usage >&2
  exit 2
fi

exec conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.literature.w0_fastwam_audit \
  "$@"
