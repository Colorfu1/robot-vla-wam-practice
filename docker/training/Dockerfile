ARG CUDA_BASE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04@sha256:59e0e4376a0f16d10b03d3a14344b80a866a1674cb4948cb318291387ac05010"

FROM ${CUDA_BASE} AS environment-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_24.11.1-0-Linux-x86_64.sh"
ARG MINICONDA_SHA256="7a3cdbb78a642b2b6dd0687c9e09824e53330368ab57406e27d88e25c4fe4e40"
ARG USE_ANACONDA_PROXY=0
ARG USE_PYPI_PROXY=0

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_ROOT=/opt/conda \
    ROBOT_PRACTICE_ENV=/opt/conda/envs/robot-practice \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    PATH=/opt/conda/envs/robot-practice/bin:/opt/conda/bin:${PATH}

RUN apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update \
    && apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        curl \
        git \
        jq \
        libegl1 \
        libgl1 \
        libglu1-mesa \
        libgles2 \
        libglib2.0-0 \
        libglvnd0 \
        libosmesa6 \
        libsm6 \
        libvulkan1 \
        libx11-6 \
        libx11-xcb1 \
        libxau6 \
        libxcb-glx0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libxcb1 \
        libxext6 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxkbcommon0 \
        libxrandr2 \
        libxrender1 \
        libxss1 \
        libxtst6 \
        mesa-utils \
        pciutils \
        procps \
        tini \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN if [[ "${USE_ANACONDA_PROXY}" != "1" ]]; then \
        unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy ALL_PROXY all_proxy; \
    fi \
    && curl --fail --location --retry 3 "${MINICONDA_URL}" --output /tmp/miniconda.sh \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum --check --strict \
    && bash /tmp/miniconda.sh -b -p "${CONDA_ROOT}" \
    && rm -f /tmp/miniconda.sh \
    && conda config --system --set auto_activate_base false

COPY docker/training/conda-linux-64.lock /opt/robot-practice/environment/
COPY docker/training/requirements-linux-64.lock /opt/robot-practice/environment/
COPY docker/training/source-requirements-linux-64.lock /opt/robot-practice/environment/
COPY docker/training/environment-manifest.json /opt/robot-practice/environment/

RUN if [[ "${USE_ANACONDA_PROXY}" != "1" ]]; then \
        unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy ALL_PROXY all_proxy; \
    fi \
    && conda create --yes --prefix "${ROBOT_PRACTICE_ENV}" \
        --file /opt/robot-practice/environment/conda-linux-64.lock \
    && conda clean --all --yes

RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "${USE_PYPI_PROXY}" != "1" ]]; then \
        unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy ALL_PROXY all_proxy; \
    fi \
    && python -m pip --isolated install \
        --require-hashes \
        --only-binary=:all: \
        --no-deps \
        --index-url https://pypi.org/simple \
        --requirement /opt/robot-practice/environment/requirements-linux-64.lock


FROM environment-base AS source-wheel-builder

ARG USE_PYPI_PROXY=0

RUN apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        update \
    && apt-get \
        -o Acquire::Retries=5 \
        -o Acquire::http::Proxy=false \
        -o Acquire::https::Proxy=false \
        install -y --no-install-recommends \
        build-essential \
        libegl1-mesa-dev \
        linux-libc-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "${USE_PYPI_PROXY}" != "1" ]]; then \
        unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy ALL_PROXY all_proxy; \
    fi \
    && mkdir -p /sdists /wheelhouse \
    && python -m pip --isolated download \
        --require-hashes \
        --no-deps \
        --no-build-isolation \
        --no-binary=:all: \
        --index-url https://pypi.org/simple \
        --dest /sdists \
        --requirement /opt/robot-practice/environment/source-requirements-linux-64.lock \
    && CMAKE_POLICY_VERSION_MINIMUM=3.5 PIP_NO_INDEX=1 python -m pip wheel \
        --no-deps \
        --no-build-isolation \
        --wheel-dir /wheelhouse \
        /sdists/*


FROM environment-base AS final

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ROBOT_PRACTICE_UID=1000
ARG ROBOT_PRACTICE_GID=1000
ARG BUILD_DATE="unknown"
ARG CONDA_LOCK_SHA256="unknown"
ARG PYPI_LOCK_SHA256="unknown"
ARG SOURCE_LOCK_SHA256="unknown"
ARG SAPIEN_PHYSX_VERSION="105.1-physx-5.3.1.patch0"
ARG SAPIEN_PHYSX_ZIP_SHA256="167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6"

ENV HOME=/home/robot-practice \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=./src \
    LD_LIBRARY_PATH=/opt/conda/envs/robot-practice/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/local/cuda/lib64 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    SDL_VIDEODRIVER=dummy \
    XDG_RUNTIME_DIR=/tmp/runtime-robot-practice \
    MPLCONFIGDIR=/home/robot-practice/.cache/matplotlib \
    HF_HOME=/home/robot-practice/.cache/huggingface \
    LIBERO_CONFIG_PATH=/opt/robot-practice/libero \
    ROBOT_PRACTICE_SAPIEN_CACHE=/opt/robot-practice/sapien-cache \
    ROBOT_PRACTICE_SAPIEN_PHYSX_VERSION=${SAPIEN_PHYSX_VERSION} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

LABEL org.opencontainers.image.title="Robot Practice Training Environment" \
      org.opencontainers.image.description="Environment-only LeRobot simulation and training image" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.robot-practice.conda-lock-sha256="${CONDA_LOCK_SHA256}" \
      io.robot-practice.pypi-lock-sha256="${PYPI_LOCK_SHA256}" \
      io.robot-practice.source-lock-sha256="${SOURCE_LOCK_SHA256}" \
      io.robot-practice.sapien-physx-version="${SAPIEN_PHYSX_VERSION}" \
      io.robot-practice.sapien-physx-zip-sha256="${SAPIEN_PHYSX_ZIP_SHA256}" \
      io.robot-practice.remote-status="verification-pending"

RUN --mount=from=source-wheel-builder,source=/wheelhouse,target=/wheelhouse,ro \
    python -m pip install --no-index --no-deps /wheelhouse/*.whl \
    && python -m pip check \
    && python - <<'PY'
import json
import torch

expected = {
    "python_cuda": "12.8",
    "torch": "2.10.0+cu128",
    "cudnn": 91002,
}
actual = {
    "python_cuda": torch.version.cuda,
    "torch": torch.__version__,
    "cudnn": torch.backends.cudnn.version(),
}
if actual != expected:
    raise SystemExit(f"Core runtime mismatch: expected={expected}, actual={actual}")
print(json.dumps(actual, sort_keys=True))
PY

RUN set -euxo pipefail \
    && cache_dir="${ROBOT_PRACTICE_SAPIEN_CACHE}/physx/${SAPIEN_PHYSX_VERSION}" \
    && mkdir -p "${cache_dir}" \
    && curl --fail --location --retry 5 --retry-all-errors \
        --connect-timeout 20 --max-time 300 \
        "https://github.com/sapien-sim/physx-precompiled/releases/download/${SAPIEN_PHYSX_VERSION}/linux-so.zip" \
        --output /tmp/sapien-physx-linux-so.zip \
    && echo "${SAPIEN_PHYSX_ZIP_SHA256}  /tmp/sapien-physx-linux-so.zip" \
        | sha256sum --check --strict \
    && unzip -q /tmp/sapien-physx-linux-so.zip -d "${cache_dir}" \
    && test -s "${cache_dir}/libPhysXGpu_64.so" \
    && rm -f /tmp/sapien-physx-linux-so.zip

COPY docker/training/entrypoint.sh /usr/local/bin/robot-practice-entrypoint
COPY docker/training/robot-practice-init /usr/local/bin/robot-practice-init
COPY docker/training/verify_environment.py /usr/local/lib/robot-practice/verify_environment.py
COPY docker/training/nccl_probe.py /usr/local/lib/robot-practice/nccl_probe.py
COPY docker/training/robot-practice-verify /usr/local/bin/robot-practice-verify
COPY docker/training/watch /usr/local/bin/watch
COPY docker/training/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/training/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN groupadd --gid "${ROBOT_PRACTICE_GID}" robot-practice \
    && useradd \
        --uid "${ROBOT_PRACTICE_UID}" \
        --gid "${ROBOT_PRACTICE_GID}" \
        --create-home \
        --shell /bin/bash \
        robot-practice \
    && mkdir -p \
        "${HOME}/.cache/huggingface" \
        "${HOME}/.cache/matplotlib" \
        /tmp/runtime-robot-practice \
    && site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && chown -R "${ROBOT_PRACTICE_UID}:${ROBOT_PRACTICE_GID}" \
        "${HOME}" \
        /tmp/runtime-robot-practice \
    && chown "${ROBOT_PRACTICE_UID}:${ROBOT_PRACTICE_GID}" \
        "${ROBOT_PRACTICE_ENV}/bin" \
        "${site_packages}" \
    && chmod 2775 \
        "${ROBOT_PRACTICE_ENV}/bin" \
        "${site_packages}" \
    && chmod 2770 /tmp/runtime-robot-practice \
    && chmod 0755 \
        /usr/local/bin/robot-practice-entrypoint \
        /usr/local/bin/robot-practice-init \
        /usr/local/bin/robot-practice-verify \
        /usr/local/bin/watch \
    && ln -s \
        "${ROBOT_PRACTICE_SAPIEN_CACHE}" \
        "${HOME}/.sapien" \
    && dpkg-query -W -f='${Package}\t${Version}\n' \
        > /opt/robot-practice/environment/system-packages.tsv

RUN python - <<'PY'
from pathlib import Path

import yaml

site_packages = Path("/opt/conda/envs/robot-practice/lib/python3.10/site-packages")
benchmark_root = site_packages / "libero" / "libero"
config_dir = Path("/opt/robot-practice/libero")
config_dir.mkdir(parents=True, exist_ok=True)
config = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str(site_packages / "libero" / "datasets"),
    "assets": str(benchmark_root / "assets"),
}
(config_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
PY

USER robot-practice
WORKDIR /home/robot-practice

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/robot-practice-entrypoint"]
CMD ["bash"]
