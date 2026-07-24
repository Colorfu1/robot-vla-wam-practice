"""Launch the official Rerun CLI with a recording-specific desktop identity."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Protocol

OFFLINE_APPLICATION_ID = "robot_practice_act_offline"
LIBERO_APPLICATION_ID = "robot_practice_act_libero"
NATIVE_APP_ID_INSTRUCTION = bytes.fromhex(
    "41 c6 45 04 6e "  # mov byte [r13 + 4], b'n'
    "41 c7 45 00 72 65 72 75"  # mov dword [r13], b'reru'
)
LEGACY_DESKTOP_IDS = (
    "robot-practice-rerun-offline",
    "robot-practice-rerun-libero",
    "robot-practice-rerun",
)


class RecordingKind(str, Enum):
    OFFLINE = "offline"
    LIBERO = "libero"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ViewerIdentity:
    name: str
    desktop_id: str
    native_app_id: str
    icon_filename: str


VIEWER_IDENTITIES = {
    RecordingKind.OFFLINE: ViewerIdentity(
        name="Robot Rerun Offline",
        desktop_id="rr_of",
        native_app_id="rr_of",
        icon_filename="rerun_offline.png",
    ),
    RecordingKind.LIBERO: ViewerIdentity(
        name="Robot Rerun LIBERO",
        desktop_id="rr_li",
        native_app_id="rr_li",
        icon_filename="rerun_libero.png",
    ),
    RecordingKind.UNKNOWN: ViewerIdentity(
        name="Robot Rerun",
        desktop_id="rerun",
        native_app_id="rerun",
        icon_filename="rerun_unknown.png",
    ),
}


@dataclass(frozen=True)
class ViewerInvocation:
    argv: tuple[str, ...]
    use_desktop_identity: bool
    rrd_paths: tuple[Path, ...]


class RecordingLike(Protocol):
    def application_id(self) -> str: ...


class ArchiveLike(Protocol):
    def all_recordings(self) -> Sequence[RecordingLike]: ...


ArchiveLoader = Callable[[Path], ArchiveLike]

TOP_LEVEL_SUBCOMMANDS = {
    "analytics",
    "auth",
    "help",
    "man",
    "mcap",
    "reset",
    "rrd",
    "server",
}
VALUE_OPTIONS = {
    "--bind",
    "--memory-limit",
    "--port",
    "--renderer",
    "--save",
    "--screenshot-to",
    "--server-memory-limit",
    "--threads",
    "--video-decoder",
    "--web-viewer-port",
    "--window-size",
    "-j",
}
FLAG_OPTIONS = {
    "--detach-process",
    "--expect-data-soon",
    "--hide-welcome-screen",
    "--newest-first",
    "--persist-state",
    "--profile",
}
DISABLE_DESKTOP_OPTIONS = {
    "--connect",
    "--headless",
    "--help",
    "--serve-grpc",
    "--serve-web",
    "--test-receive",
    "--version",
    "--web-viewer",
    "-h",
}


def _warn(message: str) -> None:
    print(f"robot-rerun: {message}", file=sys.stderr)


def _default_archive_loader(path: Path) -> ArchiveLike:
    from rerun.dataframe import load_archive

    return load_archive(path)


def _sidecar_kind(rrd_path: Path) -> RecordingKind | None:
    expected = {
        "prediction.rrd": RecordingKind.OFFLINE,
        "rollout.rrd": RecordingKind.LIBERO,
    }.get(rrd_path.name)
    if expected is None:
        return None

    sidecar = rrd_path.parent / "run_config.json"
    try:
        payload = json.loads(sidecar.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    if payload.get("mode") != expected.value:
        return None
    return expected


def _kind_from_application_ids(application_ids: set[str]) -> RecordingKind:
    if application_ids == {OFFLINE_APPLICATION_ID}:
        return RecordingKind.OFFLINE
    if application_ids == {LIBERO_APPLICATION_ID}:
        return RecordingKind.LIBERO
    return RecordingKind.UNKNOWN


def detect_rrd_kind(
    rrd_path: Path,
    *,
    archive_loader: ArchiveLoader = _default_archive_loader,
) -> RecordingKind:
    """Classify one local RRD, preferring its matching evaluator sidecar."""

    sidecar_kind = _sidecar_kind(rrd_path)
    if sidecar_kind is not None:
        return sidecar_kind

    try:
        archive = archive_loader(rrd_path)
        application_ids = {
            str(recording.application_id()) for recording in archive.all_recordings()
        }
    except (OSError, RuntimeError, TypeError, ValueError):
        return RecordingKind.UNKNOWN
    finally:
        if "archive" in locals():
            del archive
    return _kind_from_application_ids(application_ids)


def _option_with_inline_value(token: str) -> bool:
    return any(token.startswith(f"{option}=") for option in VALUE_OPTIONS)


def _existing_local_rrd(token: str) -> Path | None:
    if token == "-" or "://" in token:
        return None
    path = Path(token).expanduser()
    if path.suffix.lower() != ".rrd" or not path.is_file():
        return None
    return path


def classify_invocation(argv: Sequence[str]) -> ViewerInvocation:
    """Conservatively identify unambiguous native Viewer RRD inputs."""

    original = tuple(argv)
    rrd_paths: list[Path] = []
    after_double_dash = False
    index = 0
    while index < len(original):
        token = original[index]
        if after_double_dash:
            rrd = _existing_local_rrd(token)
            if rrd is not None:
                rrd_paths.append(rrd)
            index += 1
            continue
        if token == "--":
            after_double_dash = True
            index += 1
            continue
        if token in DISABLE_DESKTOP_OPTIONS:
            return ViewerInvocation(original, False, ())
        if token in TOP_LEVEL_SUBCOMMANDS and not Path(token).exists():
            return ViewerInvocation(original, False, ())
        if token in VALUE_OPTIONS:
            if index + 1 >= len(original):
                return ViewerInvocation(original, False, ())
            index += 2
            continue
        if _option_with_inline_value(token):
            index += 1
            continue
        if token in FLAG_OPTIONS:
            index += 1
            continue
        if token.startswith("-"):
            return ViewerInvocation(original, False, ())
        rrd = _existing_local_rrd(token)
        if rrd is not None:
            rrd_paths.append(rrd)
        index += 1

    return ViewerInvocation(original, bool(rrd_paths), tuple(rrd_paths))


def detect_invocation_kind(invocation: ViewerInvocation) -> RecordingKind:
    kinds = {detect_rrd_kind(path) for path in invocation.rrd_paths}
    if kinds == {RecordingKind.OFFLINE}:
        return RecordingKind.OFFLINE
    if kinds == {RecordingKind.LIBERO}:
        return RecordingKind.LIBERO
    return RecordingKind.UNKNOWN


def _data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".local" / "share"
    )


def _desktop_path(kind: RecordingKind, *, data_home: Path | None = None) -> Path:
    root = _data_home() if data_home is None else data_home
    return root / "applications" / f"{VIEWER_IDENTITIES[kind].desktop_id}.desktop"


def _replace_if_changed(path: Path, contents: bytes) -> bool:
    try:
        if path.read_bytes() == contents:
            return False
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(contents)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _desktop_entry(identity: ViewerIdentity, *, launcher: Path) -> bytes:
    launcher_command = shlex.quote(str(launcher))
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={identity.name}\n"
        "Comment=Rerun Viewer launched by robot-practice\n"
        f"Exec={launcher_command} %F\n"
        f"Icon={identity.desktop_id}\n"
        "Terminal=false\n"
        "NoDisplay=false\n"
        "StartupNotify=true\n"
        f"StartupWMClass={identity.native_app_id}\n"
    ).encode()


def _official_native_binary() -> Path:
    override = os.environ.get("RERUN_CLI_PATH")
    if override:
        path = Path(override).expanduser()
    else:
        import rerun_cli

        path = Path(rerun_cli.__file__).resolve().parent / "rerun"
    if not path.is_file():
        raise FileNotFoundError(f"official native Rerun binary not found: {path}")
    return path


def _patched_binary_path(
    kind: RecordingKind,
    *,
    data_home: Path | None = None,
) -> Path:
    root = _data_home() if data_home is None else data_home
    return (
        root
        / "robot-practice"
        / "rerun-native"
        / VIEWER_IDENTITIES[kind].native_app_id
    )


def _install_patched_binary(
    source: Path,
    target: Path,
    *,
    native_app_id: str,
) -> bool:
    replacement = native_app_id.encode()
    if len(replacement) != 5:
        raise ValueError("patched Rerun application IDs must contain exactly five bytes")

    source_stat = source.stat()
    metadata = {
        "schema_version": 2,
        "source": str(source.resolve()),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "native_app_id": native_app_id,
    }
    metadata_path = target.with_suffix(".json")
    try:
        if target.is_file() and json.loads(metadata_path.read_text()) == metadata:
            return False
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass

    source_data = source.read_bytes()
    occurrences = source_data.count(NATIVE_APP_ID_INSTRUCTION)
    if occurrences != 1:
        raise RuntimeError(
            "unsupported Rerun native binary: expected exactly one compiled "
            f"application-ID instruction, found {occurrences}"
        )
    instruction = bytearray(NATIVE_APP_ID_INSTRUCTION)
    instruction[4] = replacement[4]
    instruction[9:13] = replacement[:4]
    patched_data = source_data.replace(
        NATIVE_APP_ID_INSTRUCTION,
        bytes(instruction),
        1,
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(patched_data)
        temporary.chmod(source_stat.st_mode)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _replace_if_changed(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    return True


def install_desktop_integration(
    *,
    data_home: Path | None = None,
    launcher: Path | None = None,
    native_binary: Path | None = None,
) -> bool:
    """Install desktop applications, icons, and identity-patched Viewer binaries."""

    root = _data_home() if data_home is None else data_home
    applications = root / "applications"
    icon_directory = root / "icons" / "hicolor" / "256x256" / "apps"
    if launcher is None:
        command = shutil.which("robot-rerun")
        sibling = Path(sys.executable).with_name("robot-rerun")
        if command is not None:
            launcher = Path(command).resolve()
        elif sibling.is_file():
            launcher = sibling.resolve()
        else:
            raise FileNotFoundError("'robot-rerun' is not on PATH")
    if native_binary is None:
        native_binary = _official_native_binary()

    changed = False
    for legacy_id in LEGACY_DESKTOP_IDS:
        for legacy_path in (
            applications / f"{legacy_id}.desktop",
            icon_directory / f"{legacy_id}.png",
        ):
            if legacy_path.exists():
                legacy_path.unlink()
                changed = True
    for kind, identity in VIEWER_IDENTITIES.items():
        if kind is not RecordingKind.UNKNOWN:
            changed |= _install_patched_binary(
                native_binary,
                _patched_binary_path(kind, data_home=root),
                native_app_id=identity.native_app_id,
            )
        icon_resource = resources.files("robot_practice.assets").joinpath(
            identity.icon_filename
        )
        with resources.as_file(icon_resource) as icon_path:
            changed |= _replace_if_changed(
                icon_directory / f"{identity.desktop_id}.png",
                icon_path.read_bytes(),
            )
        changed |= _replace_if_changed(
            applications / f"{identity.desktop_id}.desktop",
            _desktop_entry(identity, launcher=launcher),
        )

    for updater, target in (
        ("update-desktop-database", applications),
        ("gtk-update-icon-cache", root / "icons" / "hicolor"),
    ):
        executable = shutil.which(updater)
        if changed and executable is not None:
            subprocess.run(  # noqa: S603
                [executable, "-f", str(target)]
                if updater == "gtk-update-icon-cache"
                else [executable, str(target)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    return changed


def install_desktop_main(argv: Sequence[str] | None = None) -> int:
    if tuple(sys.argv[1:] if argv is None else argv):
        _warn("robot-rerun-install-desktop does not accept arguments")
        return 2
    try:
        changed = install_desktop_integration()
    except (OSError, RuntimeError, ValueError) as exc:
        _warn(f"could not install desktop applications: {exc}")
        return 1
    print("Rerun desktop applications installed." if changed else "Already up to date.")
    return 0


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _wait_for_process(process: subprocess.Popen[bytes]) -> int:
    try:
        return int(process.wait())
    except KeyboardInterrupt:
        _terminate_process_group(process.pid)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 130


def _launch_direct(rerun_binary: str, args: Sequence[str]) -> int:
    try:
        process = subprocess.Popen(  # noqa: S603
            [rerun_binary, *args],
            start_new_session=True,
        )
    except OSError as exc:
        _warn(f"could not start official Rerun: {exc}")
        return 126
    return _wait_for_process(process)


def main(argv: Sequence[str] | None = None) -> int:
    args = tuple(sys.argv[1:] if argv is None else argv)
    rerun_binary = shutil.which("rerun")
    if rerun_binary is None:
        sibling = Path(sys.executable).with_name("rerun")
        if sibling.is_file():
            rerun_binary = str(sibling)
    if rerun_binary is None:
        _warn("official 'rerun' executable not found on PATH")
        return 127

    invocation = classify_invocation(args)
    if not invocation.use_desktop_identity:
        return _launch_direct(rerun_binary, args)

    kind = detect_invocation_kind(invocation)
    if kind is RecordingKind.UNKNOWN:
        return _launch_direct(rerun_binary, args)

    patched_binary = _patched_binary_path(kind)
    if not _desktop_path(kind).is_file() or not patched_binary.is_file():
        _warn(
            "desktop applications are not installed; run "
            "robot-rerun-install-desktop (using the standard Rerun identity)"
        )
        return _launch_direct(rerun_binary, args)
    return _launch_direct(str(patched_binary), args)


if __name__ == "__main__":
    raise SystemExit(main())
