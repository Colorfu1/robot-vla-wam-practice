"""Atomic output directories and human-readable metric summaries."""

from __future__ import annotations

import errno
import shutil
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .provenance import write_json_atomic

INCOMPLETE_TEXT = "PushT experiment did not reach its atomic completion step.\n"


class AtomicOutputDirectory(AbstractContextManager[Path]):
    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.final = Path(path).expanduser().resolve()
        self.overwrite = overwrite
        self.temporary = self.final.with_name(f".{self.final.name}.incomplete")
        self._committed = False

    def __enter__(self) -> Path:
        if self.final.exists():
            if not self.overwrite:
                raise FileExistsError(f"Output directory already exists: {self.final}")
            shutil.rmtree(self.final)
        if self.temporary.exists():
            shutil.rmtree(self.temporary)
        self.final.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir()
        (self.temporary / ".incomplete").write_text(INCOMPLETE_TEXT, encoding="utf-8")
        return self.temporary

    def commit(self) -> Path:
        if not self.temporary.is_dir():
            raise RuntimeError("Atomic output directory is not active")
        try:
            self.temporary.rename(self.final)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            self._commit_by_copy()
        else:
            (self.final / ".incomplete").unlink()
        self._committed = True
        return self.final

    def _commit_by_copy(self) -> None:
        self.final.mkdir()
        final_marker = self.final / ".incomplete"
        try:
            final_marker.write_text(INCOMPLETE_TEXT, encoding="utf-8")
        except OSError:
            try:
                final_marker.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                self.final.rmdir()
            except OSError:
                pass
            raise
        shutil.copytree(self.temporary, self.final, dirs_exist_ok=True)
        shutil.rmtree(self.temporary)
        final_marker.unlink()

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None and not self._committed:
            self.commit()
        return False


def write_records_parquet(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> Path:
    destination = Path(path)
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot write an empty Parquet record collection")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, destination)
    return destination


def write_metrics(
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    *,
    comments: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    json_path = write_json_atomic(root / "metrics.json", dict(metrics))
    comment_map = dict(comments or {})
    lines = [
        "# Metrics",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
    ]
    for key, value in metrics.items():
        if value is None:
            rendered = "N/A"
        elif isinstance(value, float):
            rendered = f"{value:.6g}"
        else:
            rendered = str(value)
        meaning = comment_map.get(key, "")
        lines.append(f"| `{key}` | {rendered} | {meaning} |")
    markdown_path = root / "metrics.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
