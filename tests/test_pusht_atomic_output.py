from __future__ import annotations

import errno
from pathlib import Path

import pytest

from robot_practice.experiments.pusht_diffusion import output as output_module
from robot_practice.experiments.pusht_diffusion.output import (
    INCOMPLETE_TEXT,
    AtomicOutputDirectory,
)


def _raise_rename_error(error_number: int):
    def raise_error(_source: Path, _destination: Path) -> None:
        raise OSError(error_number, "injected rename failure")

    return raise_error


def test_atomic_output_falls_back_to_copy_on_exdev(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final = tmp_path / "result"
    temporary = tmp_path / ".result.incomplete"
    monkeypatch.setattr(Path, "rename", _raise_rename_error(errno.EXDEV))

    with AtomicOutputDirectory(final) as staging:
        (staging / "nested").mkdir()
        (staging / "nested" / "value.txt").write_text("complete\n", encoding="utf-8")

    assert (final / "nested" / "value.txt").read_text(encoding="utf-8") == "complete\n"
    assert not (final / ".incomplete").exists()
    assert not temporary.exists()


def test_atomic_output_keeps_staging_marker_on_non_exdev_rename_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final = tmp_path / "result"
    temporary = tmp_path / ".result.incomplete"
    monkeypatch.setattr(Path, "rename", _raise_rename_error(errno.EACCES))

    with (
        pytest.raises(OSError) as error,
        AtomicOutputDirectory(final) as staging,
    ):
        (staging / "value.txt").write_text("complete\n", encoding="utf-8")

    assert error.value.errno == errno.EACCES
    assert (temporary / ".incomplete").read_text(encoding="utf-8") == INCOMPLETE_TEXT
    assert not final.exists()


def test_atomic_output_marks_final_before_fallback_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final = tmp_path / "result"
    temporary = tmp_path / ".result.incomplete"
    monkeypatch.setattr(Path, "rename", _raise_rename_error(errno.EXDEV))

    def fail_copy(source: Path, destination: Path, **kwargs: object) -> None:
        assert source == temporary
        assert destination == final
        assert kwargs == {"dirs_exist_ok": True}
        assert (destination / ".incomplete").read_text(encoding="utf-8") == INCOMPLETE_TEXT
        (destination / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise OSError("injected copy failure")

    monkeypatch.setattr(output_module.shutil, "copytree", fail_copy)

    with (
        pytest.raises(OSError, match="injected copy failure"),
        AtomicOutputDirectory(final) as staging,
    ):
        (staging / "value.txt").write_text("complete\n", encoding="utf-8")

    assert (temporary / ".incomplete").is_file()
    assert (final / ".incomplete").is_file()
    assert (final / "partial.txt").is_file()


def test_atomic_output_keeps_both_markers_when_staging_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final = tmp_path / "result"
    temporary = tmp_path / ".result.incomplete"
    real_rmtree = output_module.shutil.rmtree
    monkeypatch.setattr(Path, "rename", _raise_rename_error(errno.EXDEV))

    def fail_staging_cleanup(path: str | Path, *args: object, **kwargs: object) -> None:
        if Path(path) == temporary:
            raise OSError("injected staging cleanup failure")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(output_module.shutil, "rmtree", fail_staging_cleanup)

    with (
        pytest.raises(OSError, match="injected staging cleanup failure"),
        AtomicOutputDirectory(final) as staging,
    ):
        (staging / "value.txt").write_text("complete\n", encoding="utf-8")

    assert (temporary / ".incomplete").is_file()
    assert (final / ".incomplete").is_file()
    assert (final / "value.txt").is_file()


def test_atomic_output_keeps_final_marker_when_marker_removal_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    final = tmp_path / "result"
    temporary = tmp_path / ".result.incomplete"
    final_marker = final / ".incomplete"
    real_unlink = Path.unlink
    monkeypatch.setattr(Path, "rename", _raise_rename_error(errno.EXDEV))

    def fail_final_marker_removal(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path == final_marker:
            raise OSError("injected marker removal failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_final_marker_removal)

    with (
        pytest.raises(OSError, match="injected marker removal failure"),
        AtomicOutputDirectory(final) as staging,
    ):
        (staging / "value.txt").write_text("complete\n", encoding="utf-8")

    assert not temporary.exists()
    assert final_marker.is_file()
    assert (final / "value.txt").is_file()
