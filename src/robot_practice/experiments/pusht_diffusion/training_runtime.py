"""Small runtime adapters around the unmodified LeRobot training pipeline."""

from __future__ import annotations

import logging
import sys
import threading
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TextIO

from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.scripts import lerobot_train


class SubsetEpisodeAwareSampler(EpisodeAwareSampler):
    """Emit relative row indices when LeRobot loads only selected episodes."""

    def __init__(
        self,
        dataset_from_indices: Sequence[int],
        dataset_to_indices: Sequence[int],
        episode_indices_to_use: Sequence[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ) -> None:
        if episode_indices_to_use is None:
            super().__init__(
                list(dataset_from_indices),
                list(dataset_to_indices),
                episode_indices_to_use=None,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                shuffle=shuffle,
            )
            return

        selected = tuple(int(index) for index in episode_indices_to_use)
        selected_set = set(selected)
        if len(selected_set) != len(selected):
            raise ValueError("Episode subset contains duplicate indices")
        num_episodes = len(dataset_from_indices)
        unknown = sorted(index for index in selected_set if not 0 <= index < num_episodes)
        if unknown:
            raise ValueError(f"Episode subset contains out-of-range indices: {unknown}")

        indices: list[int] = []
        relative_start = 0
        for episode_index, (absolute_start, absolute_end) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_index not in selected_set:
                continue
            episode_length = int(absolute_end) - int(absolute_start)
            indices.extend(
                range(
                    relative_start + drop_n_first_frames,
                    relative_start + episode_length - drop_n_last_frames,
                )
            )
            relative_start += episode_length

        self.indices = indices
        self.shuffle = shuffle


def install_lerobot_runtime_adapters() -> None:
    """Install the local subset-index fix without editing installed LeRobot."""

    lerobot_train.EpisodeAwareSampler = SubsetEpisodeAwareSampler


class _TeeStream:
    """Write a text stream to the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log: TextIO, lock: threading.Lock) -> None:
        self._terminal = terminal
        self._log = log
        self._lock = lock

    def write(self, value: str) -> int:
        with self._lock:
            written = self._terminal.write(value)
            self._log.write(value)
            self._log.flush()
        return written

    def flush(self) -> None:
        with self._lock:
            self._terminal.flush()
            self._log.flush()

    def __getattr__(self, name: str) -> object:
        return getattr(self._terminal, name)


@contextmanager
def persistent_training_log(
    output_dir: Path,
    *,
    enabled: bool = True,
) -> Iterator[Path | None]:
    """Tee main-process output and move the completed log into the LeRobot run."""

    if not enabled:
        yield None
        return

    temporary = output_dir.with_name(f"{output_dir.name}.train.log")
    if temporary.exists():
        raise FileExistsError(
            f"Training log already exists: {temporary}. Remove it or choose another output directory."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    completed = False
    lock = threading.Lock()
    with temporary.open("x", encoding="utf-8", buffering=1) as log:
        stdout = _TeeStream(sys.stdout, log, lock)
        stderr = _TeeStream(sys.stderr, log, lock)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                yield temporary
            completed = True
        except BaseException:
            traceback.print_exc(file=log)
            log.flush()
            print(f"Training failed; retained log: {temporary}", file=sys.stderr)
            raise
        finally:
            logging.shutdown()
            logging.getLogger().handlers.clear()

    if completed:
        if not output_dir.is_dir():
            raise RuntimeError(
                f"LeRobot completed without creating its output directory: {output_dir}"
            )
        destination = output_dir / "train.log"
        if destination.exists():
            raise FileExistsError(f"LeRobot output already contains a training log: {destination}")
        temporary.replace(destination)


__all__ = [
    "SubsetEpisodeAwareSampler",
    "install_lerobot_runtime_adapters",
    "persistent_training_log",
]
