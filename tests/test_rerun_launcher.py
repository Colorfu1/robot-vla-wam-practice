from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import robot_practice.rerun_launcher as launcher


class FakeRecording:
    def __init__(self, application_id: str):
        self._application_id = application_id

    def application_id(self) -> str:
        return self._application_id


class FakeArchive:
    def __init__(self, *application_ids: str):
        self._recordings = [FakeRecording(value) for value in application_ids]

    def all_recordings(self) -> list[FakeRecording]:
        return self._recordings


def _write_run_config(path: Path, *, mode: str, schema_version: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "mode": mode,
            }
        )
    )


@pytest.mark.parametrize(
    ("filename", "mode", "expected"),
    [
        ("prediction.rrd", "offline", launcher.RecordingKind.OFFLINE),
        ("rollout.rrd", "libero", launcher.RecordingKind.LIBERO),
    ],
)
def test_matching_sidecar_is_authoritative_fast_path(
    tmp_path: Path,
    filename: str,
    mode: str,
    expected: launcher.RecordingKind,
) -> None:
    rrd = tmp_path / filename
    rrd.write_bytes(b"not parsed")
    _write_run_config(tmp_path / "run_config.json", mode=mode)

    def unexpected_loader(path: Path) -> FakeArchive:
        raise AssertionError(f"archive loader should not inspect {path}")

    assert launcher.detect_rrd_kind(rrd, archive_loader=unexpected_loader) is expected


def test_renamed_rrd_uses_embedded_application_id(tmp_path: Path) -> None:
    rrd = tmp_path / "renamed.rrd"
    rrd.write_bytes(b"rrd")
    _write_run_config(tmp_path / "run_config.json", mode="libero")
    seen: list[Path] = []

    def load_archive(path: Path) -> FakeArchive:
        seen.append(path)
        return FakeArchive(launcher.OFFLINE_APPLICATION_ID)

    assert (
        launcher.detect_rrd_kind(rrd, archive_loader=load_archive)
        is launcher.RecordingKind.OFFLINE
    )
    assert seen == [rrd]


def test_multi_recording_archive_and_mixed_archive_are_classified(tmp_path: Path) -> None:
    rrd = tmp_path / "moved.rrd"
    rrd.write_bytes(b"rrd")

    assert (
        launcher.detect_rrd_kind(
            rrd,
            archive_loader=lambda _: FakeArchive(
                launcher.OFFLINE_APPLICATION_ID,
                launcher.OFFLINE_APPLICATION_ID,
            ),
        )
        is launcher.RecordingKind.OFFLINE
    )
    assert (
        launcher.detect_rrd_kind(
            rrd,
            archive_loader=lambda _: FakeArchive(
                launcher.OFFLINE_APPLICATION_ID,
                launcher.LIBERO_APPLICATION_ID,
            ),
        )
        is launcher.RecordingKind.UNKNOWN
    )


def test_detect_invocation_combines_multiple_rrds(tmp_path: Path) -> None:
    offline = tmp_path / "offline" / "prediction.rrd"
    offline.parent.mkdir()
    offline.write_bytes(b"offline")
    _write_run_config(offline.parent / "run_config.json", mode="offline")

    libero = tmp_path / "libero" / "rollout.rrd"
    libero.parent.mkdir()
    libero.write_bytes(b"libero")
    _write_run_config(libero.parent / "run_config.json", mode="libero")

    invocation = launcher.classify_invocation(
        ["--persist-state", "--port=9877", str(offline), str(libero)]
    )

    assert invocation.use_desktop_identity is True
    assert invocation.rrd_paths == (offline, libero)
    assert launcher.detect_invocation_kind(invocation) is launcher.RecordingKind.UNKNOWN


def test_cli_classifier_excludes_option_values_and_handles_double_dash(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture.rrd"
    input_rrd = tmp_path / "--input.rrd"
    input_rrd.write_bytes(b"rrd")

    invocation = launcher.classify_invocation(
        ["--save", str(output), "--port", "9877", "--", str(input_rrd)]
    )

    assert invocation.use_desktop_identity is True
    assert invocation.rrd_paths == (input_rrd,)


@pytest.mark.parametrize(
    "args",
    [
        ["rrd", "verify", "input.rrd"],
        ["--web-viewer", "input.rrd"],
        ["--connect", "input.rrd"],
        ["--help"],
        ["--future-option", "input.rrd"],
        ["https://example.com/input.rrd"],
        ["-"],
    ],
)
def test_cli_classifier_conservatively_uses_direct_cli(args: list[str]) -> None:
    invocation = launcher.classify_invocation(args)
    assert invocation.use_desktop_identity is False
    assert invocation.rrd_paths == ()


def test_main_selects_registered_offline_application(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd = tmp_path / "prediction.rrd"
    rrd.write_bytes(b"rrd")
    _write_run_config(tmp_path / "run_config.json", mode="offline")
    patched = tmp_path / "rr_of"
    patched.write_bytes(b"patched")
    desktop = tmp_path / "offline.desktop"
    desktop.write_text("[Desktop Entry]\n")
    events: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/env/bin/rerun")
    monkeypatch.setattr(launcher, "_patched_binary_path", lambda kind: patched)
    monkeypatch.setattr(launcher, "_desktop_path", lambda kind: desktop)
    monkeypatch.setattr(
        launcher,
        "_launch_direct",
        lambda rerun, args: events.append((rerun, tuple(args))) or 17,
    )

    args = ["--persist-state", "--port=9877", str(rrd)]
    assert launcher.main(args) == 17
    assert events == [(str(patched), tuple(args))]


def test_main_direct_cli_forwards_exact_argv_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    process = SimpleNamespace(pid=4321, wait=lambda: 17)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/env/bin/rerun")
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda argv, **kwargs: events.append((tuple(argv), kwargs)) or process,
    )

    args = ["rrd", "verify", "file.rrd"]
    assert launcher.main(args) == 17
    assert events == [
        (
            ("/env/bin/rerun", *args),
            {"start_new_session": True},
        )
    ]


def test_main_falls_back_without_installing_desktop_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rrd = tmp_path / "prediction.rrd"
    rrd.write_bytes(b"rrd")
    _write_run_config(tmp_path / "run_config.json", mode="offline")
    direct: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        lambda name: "/env/bin/rerun",
    )
    monkeypatch.setattr(
        launcher,
        "_desktop_path",
        lambda kind: tmp_path / "missing.desktop",
    )
    monkeypatch.setattr(
        launcher,
        "_launch_direct",
        lambda rerun, args: direct.append((rerun, tuple(args))) or 0,
    )

    assert launcher.main([str(rrd)]) == 0
    assert direct == [("/env/bin/rerun", (str(rrd),))]
    assert "robot-rerun-install-desktop" in capsys.readouterr().err


def test_direct_launch_converts_keyboard_interrupt_to_clean_130(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits = iter([KeyboardInterrupt(), 143])
    process = SimpleNamespace(
        pid=4321,
        wait=lambda **kwargs: (
            (_ for _ in ()).throw(result)
            if isinstance(result := next(waits), BaseException)
            else result
        ),
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda argv, **kwargs: process,
    )
    monkeypatch.setattr(
        launcher.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )

    assert launcher._launch_direct("/env/bin/rerun", ["--help"]) == 130
    assert signals == [(4321, launcher.signal.SIGTERM)]


def test_desktop_installer_creates_three_apps_and_patched_binaries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[list[str]] = []
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        lambda name: {
            "update-desktop-database": "/usr/bin/update-desktop-database",
            "gtk-update-icon-cache": "/usr/bin/gtk-update-icon-cache",
        }.get(name),
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda argv, **kwargs: updates.append(argv),
    )

    project_launcher = Path("/env/bin/robot-rerun")
    native = tmp_path / "official-rerun"
    native.write_bytes(
        b"prefix-" + launcher.NATIVE_APP_ID_INSTRUCTION + b"-suffix"
    )
    native.chmod(0o755)
    assert launcher.install_desktop_integration(
        data_home=tmp_path,
        launcher=project_launcher,
        native_binary=native,
    )

    applications = tmp_path / "applications"
    icons = tmp_path / "icons" / "hicolor" / "256x256" / "apps"
    for identity in launcher.VIEWER_IDENTITIES.values():
        desktop = applications / f"{identity.desktop_id}.desktop"
        contents = desktop.read_text()
        assert "NoDisplay=false" in contents
        assert "StartupNotify=true" in contents
        assert f"StartupWMClass={identity.native_app_id}" in contents
        assert f"Icon={identity.desktop_id}" in contents
        assert f"Exec={project_launcher} %F" in contents
        assert (icons / f"{identity.desktop_id}.png").is_file()
    for kind in (launcher.RecordingKind.OFFLINE, launcher.RecordingKind.LIBERO):
        identity = launcher.VIEWER_IDENTITIES[kind]
        patched = launcher._patched_binary_path(kind, data_home=tmp_path)
        patched_data = patched.read_bytes()
        assert patched_data[7 + 4] == identity.native_app_id.encode()[4]
        assert patched_data[7 + 9 : 7 + 13] == identity.native_app_id.encode()[:4]
        assert len(patched_data) == len(native.read_bytes())
        assert patched.stat().st_mode & 0o111
    assert updates == [
        ["/usr/bin/update-desktop-database", str(applications)],
        [
            "/usr/bin/gtk-update-icon-cache",
            "-f",
            str(tmp_path / "icons" / "hicolor"),
        ],
    ]

    updates.clear()
    assert not launcher.install_desktop_integration(
        data_home=tmp_path,
        launcher=project_launcher,
        native_binary=native,
    )
    assert updates == []


def test_binary_patch_rejects_unexpected_marker_count(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rerun"
    source.write_bytes(b"no matching app id")

    with pytest.raises(RuntimeError, match="expected exactly one"):
        launcher._install_patched_binary(
            source,
            tmp_path / "patched",
            native_app_id="rr_of",
        )
