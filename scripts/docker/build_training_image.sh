#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_training_image.sh --image=IMAGE_REF [options]

Build the environment-only robot-practice image.

Options:
  --image=IMAGE_REF   Required target repository or tag. May also use IMAGE_REF.
  --base-image=REF    Override the pinned CUDA pull location while preserving its digest.
  --version-tag=TAG   Immutable tag (default: env-YYYYMMDD-<lock hash>).
  --no-latest         Do not add the repository's :latest alias.
  --no-cache          Disable Docker build cache.
  --anaconda-proxy    Use the shell proxy only for Miniconda and Conda downloads.
  --use-proxy         Use the shell proxy for all dependency downloads.
  --progress=MODE     Docker progress mode (default: plain).
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pinned_base_digest="sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010"
image_ref="${IMAGE_REF:-}"
base_image="${CUDA_BASE_REF:-}"
version_tag=""
add_latest=true
no_cache=false
proxy_mode="none"
progress="plain"

for argument in "$@"; do
  case "$argument" in
    --image=*) image_ref="${argument#*=}" ;;
    --base-image=*) base_image="${argument#*=}" ;;
    --version-tag=*) version_tag="${argument#*=}" ;;
    --no-latest) add_latest=false ;;
    --no-cache) no_cache=true ;;
    --anaconda-proxy) proxy_mode="anaconda" ;;
    --use-proxy) proxy_mode="all" ;;
    --progress=*) progress="${argument#*=}" ;;
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
  echo "--image=IMAGE_REF or IMAGE_REF is required" >&2
  exit 64
fi
if [[ "$image_ref" == *@* ]]; then
  echo "Build destination must be a tag or repository, not a digest: $image_ref" >&2
  exit 64
fi
if [[ -n "$base_image" && "$base_image" != *"@$pinned_base_digest" ]]; then
  echo "--base-image must preserve the pinned CUDA digest: $pinned_base_digest" >&2
  exit 64
fi

last_component="${image_ref##*/}"
if [[ "$last_component" == *:* ]]; then
  repository="${image_ref%:*}"
  requested_tag="$image_ref"
else
  repository="$image_ref"
  requested_tag="$image_ref:latest"
fi

conda_lock="$repo_root/docker/training/conda-linux-64.lock"
pypi_lock="$repo_root/docker/training/requirements-linux-64.lock"
source_lock="$repo_root/docker/training/source-requirements-linux-64.lock"
for lock in "$conda_lock" "$pypi_lock" "$source_lock"; do
  if [[ ! -s "$lock" ]]; then
    echo "Missing lock file: $lock" >&2
    exit 66
  fi
done

conda_sha="$(sha256sum "$conda_lock" | awk '{print $1}')"
pypi_sha="$(sha256sum "$pypi_lock" | awk '{print $1}')"
source_sha="$(sha256sum "$source_lock" | awk '{print $1}')"
if [[ -z "$version_tag" ]]; then
  version_tag="env-$(date -u +%Y%m%d)-${pypi_sha:0:12}"
fi
immutable_ref="$repository:$version_tag"

tags=(-t "$immutable_ref")
if [[ "$requested_tag" != "$immutable_ref" ]]; then
  tags+=(-t "$requested_tag")
fi
if [[ "$add_latest" == true && "$repository:latest" != "$immutable_ref" ]]; then
  tags+=(-t "$repository:latest")
fi

build_arguments=(
  --file "$repo_root/docker/training/Dockerfile"
  --platform linux/amd64
  --progress "$progress"
  --build-arg "BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  --build-arg "CONDA_LOCK_SHA256=$conda_sha"
  --build-arg "PYPI_LOCK_SHA256=$pypi_sha"
  --build-arg "SOURCE_LOCK_SHA256=$source_sha"
)
if [[ "$proxy_mode" != "none" ]]; then
  if [[ -z "${HTTP_PROXY:-}${HTTPS_PROXY:-}" ]]; then
    echo "Selected proxy mode requires HTTP_PROXY and/or HTTPS_PROXY in the shell" >&2
    exit 64
  fi
  build_arguments+=(--network host)
  [[ -n "${HTTP_PROXY:-}" ]] && build_arguments+=(--build-arg "HTTP_PROXY=$HTTP_PROXY")
  [[ -n "${HTTPS_PROXY:-}" ]] && build_arguments+=(--build-arg "HTTPS_PROXY=$HTTPS_PROXY")
  [[ -n "${NO_PROXY:-}" ]] && build_arguments+=(--build-arg "NO_PROXY=$NO_PROXY")
fi
case "$proxy_mode" in
  none)
    build_arguments+=(--build-arg "USE_ANACONDA_PROXY=0")
    build_arguments+=(--build-arg "USE_PYPI_PROXY=0")
    ;;
  anaconda)
    build_arguments+=(--build-arg "USE_ANACONDA_PROXY=1")
    build_arguments+=(--build-arg "USE_PYPI_PROXY=0")
    ;;
  all)
    build_arguments+=(--build-arg "USE_ANACONDA_PROXY=1")
    build_arguments+=(--build-arg "USE_PYPI_PROXY=1")
    ;;
esac
if [[ -n "$base_image" ]]; then
  build_arguments+=(--build-arg "CUDA_BASE=$base_image")
fi
if [[ "$no_cache" == true ]]; then
  build_arguments+=(--no-cache)
fi

echo "Building environment-only image"
echo "Immutable tag: $immutable_ref"
echo "Conda lock: $conda_sha"
echo "PyPI lock: $pypi_sha"
echo "Source lock: $source_sha"
case "$proxy_mode" in
  none) echo "Dependency downloads: direct" ;;
  anaconda) echo "Dependency downloads: Anaconda proxy; APT and PyPI direct" ;;
  all) echo "Dependency downloads: explicit shell proxy for Anaconda and PyPI" ;;
esac

DOCKER_BUILDKIT=1 docker build \
  "${build_arguments[@]}" \
  "${tags[@]}" \
  "$repo_root"

docker image inspect "$immutable_ref" \
  --format 'Built {{.Id}} ({{.Size}} bytes)'

cat <<EOF

Build completed.
Immutable image: $immutable_ref

Registry login and push are explicit:
  docker push $immutable_ref
EOF
if [[ "$add_latest" == true ]]; then
  echo "  docker push $repository:latest"
fi
