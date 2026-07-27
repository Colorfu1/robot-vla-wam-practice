"""Generate sanitized Conda and PyPI locks from the local robot-practice environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from packaging.utils import canonicalize_name

PUBLIC_PYPI_INDEX = "https://pypi.org/simple"
PUBLIC_CONDA_HOSTS = {"conda.anaconda.org", "repo.anaconda.com"}
PUBLIC_PYPI_FILE_HOSTS = {"files.pythonhosted.org"}
PROJECT_PACKAGE = canonicalize_name("robot-practice")
SOURCE_ONLY_PACKAGES = {
    canonicalize_name(name)
    for name in (
        "antlr4-python3-runtime",
        "bddl",
        "egl-probe",
        "evdev",
        "hf-egl-probe",
        "robomimic",
    )
}
REJECTED_SOURCE_PATTERN = re.compile(
    r"(?i)(?:git\+|file:|/home/|/Users/|token=|password=|authorization:)"
)


def _run(arguments: list[str], *, capture_stderr: bool = True) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if capture_stderr else None,
    )
    return completed.stdout


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _validate_public_url(url: str, *, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Only HTTPS package artifacts are allowed: {url}")
    if parsed.hostname not in allowed_hosts:
        raise ValueError(f"Unapproved package host {parsed.hostname!r}: {url}")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"Credentials, queries, and fragments are forbidden: {url}")
    if REJECTED_SOURCE_PATTERN.search(url):
        raise ValueError(f"Rejected package source: {url}")


def _conda_packages(environment: str) -> list[dict[str, Any]]:
    payload = _run(["conda", "list", "-n", environment, "--json"])
    packages = json.loads(payload)
    if not isinstance(packages, list):
        raise TypeError("conda list --json did not return a package list")
    return packages


def _conda_explicit_lock(environment: str) -> str:
    lock = _run(["conda", "list", "--explicit", "-n", environment])
    if "@EXPLICIT" not in lock:
        raise ValueError("Conda explicit lock is missing @EXPLICIT")
    for line in lock.splitlines():
        if not line.startswith("https://"):
            continue
        _validate_public_url(line, allowed_hosts=PUBLIC_CONDA_HOSTS)
        if "/conda-forge/" not in line:
            raise ValueError(f"Only conda-forge artifacts are approved: {line}")
    if REJECTED_SOURCE_PATTERN.search(lock):
        raise ValueError("Conda lock contains a rejected local or credential source")
    return lock.rstrip() + "\n"


def _pypi_inventory(
    conda_packages: list[dict[str, Any]],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    inventory: list[tuple[str, str]] = []
    project_entry: dict[str, Any] | None = None
    for package in conda_packages:
        if package.get("channel") != "pypi":
            continue
        name = str(package["name"])
        version = str(package["version"])
        canonical_name = canonicalize_name(name)
        if canonical_name == PROJECT_PACKAGE:
            if project_entry is not None:
                raise ValueError("Multiple robot-practice entries found in the environment")
            project_entry = package
            continue
        inventory.append((name, version))
    if project_entry is None:
        raise ValueError("The editable robot-practice package was not found for exclusion")
    inventory.sort(key=lambda item: canonicalize_name(item[0]))
    return inventory, project_entry


def _resolve_wheels(
    environment: str,
    packages: list[tuple[str, str]],
) -> dict[str, dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="robot-practice-lock-") as temporary:
        root = Path(temporary)
        requirements = root / "requirements.in"
        report = root / "pip-report.json"
        requirements.write_text(
            "\n".join(f"{name}=={version}" for name, version in packages) + "\n",
            encoding="utf-8",
        )
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "python",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-deps",
            "--only-binary=:all:",
            "--index-url",
            PUBLIC_PYPI_INDEX,
            "--report",
            str(report),
            "-r",
            str(requirements),
        ]
        subprocess.run(
            command, check=True, env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        )
        payload = json.loads(report.read_text(encoding="utf-8"))

    resolved: dict[str, dict[str, str]] = {}
    for item in payload.get("install", []):
        metadata = item["metadata"]
        name = str(metadata["name"])
        canonical_name = canonicalize_name(name)
        version = str(metadata["version"])
        download = item["download_info"]
        url = str(download["url"])
        _validate_public_url(url, allowed_hosts=PUBLIC_PYPI_FILE_HOSTS)
        if not urlparse(url).path.endswith(".whl"):
            raise ValueError(f"Wheel-only package resolved to a non-wheel artifact: {url}")
        hashes = download.get("archive_info", {}).get("hashes", {})
        sha256 = hashes.get("sha256")
        if not sha256:
            raise ValueError(f"Resolved wheel is missing a SHA256 hash: {url}")
        resolved[canonical_name] = {
            "name": name,
            "version": version,
            "artifact_type": "wheel",
            "url": url,
            "sha256": str(sha256),
        }
    return resolved


def _resolve_source_distribution(name: str, version: str) -> dict[str, str]:
    api_url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urlopen(api_url, timeout=60) as response:
        payload = json.load(response)
    candidates = [item for item in payload["urls"] if item["packagetype"] == "sdist"]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one source distribution for {name}=={version}, "
            f"found {len(candidates)}"
        )
    candidate = candidates[0]
    url = str(candidate["url"])
    _validate_public_url(url, allowed_hosts=PUBLIC_PYPI_FILE_HOSTS)
    sha256 = candidate.get("digests", {}).get("sha256")
    if not sha256:
        raise ValueError(f"Source distribution is missing a SHA256 hash: {url}")
    return {
        "name": str(payload["info"]["name"]),
        "version": str(payload["info"]["version"]),
        "artifact_type": "sdist",
        "url": url,
        "sha256": str(sha256),
    }


def _format_requirements(artifacts: list[dict[str, str]], *, title: str) -> str:
    lines = [
        f"# {title}",
        "# Generated from the local robot-practice environment; do not edit manually.",
    ]
    for artifact in sorted(artifacts, key=lambda item: canonicalize_name(item["name"])):
        lines.append(
            f"{artifact['name']}=={artifact['version']} --hash=sha256:{artifact['sha256']}"
        )
    return "\n".join(lines) + "\n"


def _runtime_facts(environment: str) -> dict[str, Any]:
    script = """
import json
import platform
import sys
import torch

print(json.dumps({
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "platform": sys.platform,
}, sort_keys=True))
"""
    return json.loads(
        _run(
            [
                "conda",
                "run",
                "-n",
                environment,
                "python",
                "-c",
                script,
            ]
        )
    )


def generate_locks(environment: str, output_directory: Path) -> dict[str, Any]:
    conda_packages = _conda_packages(environment)
    inventory, excluded_project = _pypi_inventory(conda_packages)
    by_name = {canonicalize_name(name): (name, version) for name, version in inventory}
    missing_allowlist = SOURCE_ONLY_PACKAGES - by_name.keys()
    if missing_allowlist:
        raise ValueError(
            "Expected source-only package(s) missing from the environment: "
            + ", ".join(sorted(missing_allowlist))
        )

    wheel_inventory = [
        package
        for package in inventory
        if canonicalize_name(package[0]) not in SOURCE_ONLY_PACKAGES
    ]
    wheels = _resolve_wheels(environment, wheel_inventory)
    expected_wheels = {canonicalize_name(name) for name, _ in wheel_inventory}
    if wheels.keys() != expected_wheels:
        missing = sorted(expected_wheels - wheels.keys())
        extra = sorted(wheels.keys() - expected_wheels)
        raise ValueError(f"Wheel resolution mismatch: missing={missing}, extra={extra}")

    sources = {
        canonical_name: _resolve_source_distribution(*by_name[canonical_name])
        for canonical_name in sorted(SOURCE_ONLY_PACKAGES)
    }
    artifacts = {**wheels, **sources}
    for canonical_name, (_, expected_version) in by_name.items():
        resolved = artifacts[canonical_name]
        if resolved["version"] != expected_version:
            raise ValueError(
                f"Version mismatch for {canonical_name}: "
                f"{resolved['version']} != {expected_version}"
            )

    conda_lock = _conda_explicit_lock(environment)
    wheel_lock = _format_requirements(
        list(wheels.values()),
        title="Public PyPI Linux x86-64 wheel lock",
    )
    source_lock = _format_requirements(
        list(sources.values()),
        title="Reviewed public PyPI source-distribution lock",
    )
    runtime = _runtime_facts(environment)
    if runtime["architecture"] != "x86_64":
        raise ValueError(f"Only x86_64 lock generation is supported: {runtime}")
    if runtime["torch_cuda"] != "12.8":
        raise ValueError(f"Expected the working PyTorch CUDA version to be 12.8: {runtime}")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_environment": environment,
        "platform": "linux-64",
        "public_conda_channels": ["https://conda.anaconda.org/conda-forge"],
        "public_pypi_index": PUBLIC_PYPI_INDEX,
        "runtime": runtime,
        "excluded_project": {
            "name": excluded_project["name"],
            "version": excluded_project["version"],
            "reason": "installed from the runtime repository path",
        },
        "counts": {
            "conda": sum(package.get("channel") != "pypi" for package in conda_packages),
            "pypi_wheels": len(wheels),
            "pypi_source_distributions": len(sources),
        },
        "lock_sha256": {
            "conda-linux-64.lock": _sha256_text(conda_lock),
            "requirements-linux-64.lock": _sha256_text(wheel_lock),
            "source-requirements-linux-64.lock": _sha256_text(source_lock),
        },
        "pypi_artifacts": [artifacts[name] for name in sorted(artifacts)],
    }

    _write_if_changed(output_directory / "conda-linux-64.lock", conda_lock)
    _write_if_changed(output_directory / "requirements-linux-64.lock", wheel_lock)
    _write_if_changed(
        output_directory / "source-requirements-linux-64.lock",
        source_lock,
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_if_changed(output_directory / "environment-manifest.json", manifest_text)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", default="robot-practice")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docker/training"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        manifest = generate_locks(
            arguments.environment,
            arguments.output_dir.expanduser().resolve(),
        )
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError) as exc:
        print(f"Lock generation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"Locks written to {arguments.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
