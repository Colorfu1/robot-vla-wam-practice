#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: audit_training_image.sh --image=IMAGE_REF [--registry-string=TEXT] [--save-layers]

Audit tracked container inputs, image configuration/history, the final filesystem,
and optionally every saved layer. The external RepoTags manifest is not scanned.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image_ref=""
registry_string=""
save_layers=false
artifact_pattern='(^|/)\.git(/|$)|(^|/)src/robot_practice(/|$)|(^|/)reports/experiment_log\.md$|^(\./)?(data|datasets|outputs|wandb|runs|checkpoints)(/|$)'

for argument in "$@"; do
  case "$argument" in
    --image=*) image_ref="${argument#*=}" ;;
    --registry-string=*) registry_string="${argument#*=}" ;;
    --save-layers) save_layers=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "$image_ref" ]]; then
  echo "--image=IMAGE_REF is required" >&2
  exit 64
fi

tracked_inputs="$(
  git -C "$repo_root" ls-files \
    .dockerignore \
    docker/training \
    scripts/docker
)"
if [[ -z "$tracked_inputs" ]]; then
  echo "No tracked container files found" >&2
  exit 66
fi

if [[ -n "$registry_string" ]] \
  && git -C "$repo_root" grep -F -- "$registry_string" -- $tracked_inputs; then
  echo "Target registry destination is committed in a container file" >&2
  exit 1
fi

temporary="$(mktemp -d)"
container_id=""
cleanup() {
  if [[ -n "$container_id" ]]; then
    docker rm -f "$container_id" >/dev/null 2>&1 || true
  fi
  rm -rf "$temporary"
}
trap cleanup EXIT

docker image inspect "$image_ref" \
  | jq 'map(del(.RepoTags, .RepoDigests))' \
  > "$temporary/inspect.json"
docker history --no-trunc "$image_ref" > "$temporary/history.txt"
container_id="$(docker create "$image_ref")"
docker export "$container_id" --output "$temporary/filesystem.tar"

tar -tf "$temporary/filesystem.tar" > "$temporary/filesystem.list"
if rg -n "$artifact_pattern" "$temporary/filesystem.list"; then
  echo "Generated repository artifacts found in final image filesystem" >&2
  exit 1
fi
if rg -n -i '(authorization:|bearer[[:space:]]+[[:alnum:]_.-]+|password=|token=)' \
  "$temporary/inspect.json" "$temporary/history.txt"; then
  echo "Credential-like content found in image configuration or history" >&2
  exit 1
fi
if jq -e '.[].Config.Env[]? | test("^(HTTP|HTTPS|ALL|NO)_PROXY="; "i")' \
  "$temporary/inspect.json" >/dev/null; then
  echo "Proxy environment variable found in final image configuration" >&2
  exit 1
fi
if [[ -n "$registry_string" ]] \
  && rg -F "$registry_string" "$temporary/inspect.json" "$temporary/history.txt"; then
  echo "Target registry destination found inside image configuration or history" >&2
  exit 1
fi

if [[ "$save_layers" == true ]]; then
  mkdir -p "$temporary/saved"
  docker save "$image_ref" --output "$temporary/image.tar"
  tar -xf "$temporary/image.tar" -C "$temporary/saved"
  find "$temporary/saved" -name layer.tar -print0 \
    | while IFS= read -r -d '' layer; do
        listing="$layer.list"
        tar -tf "$layer" > "$listing"
        if rg -n "$artifact_pattern" "$listing"; then
          echo "Generated repository artifacts found in image layer: $layer" >&2
          exit 1
        fi
      done
fi

echo "Image audit passed: $image_ref"
