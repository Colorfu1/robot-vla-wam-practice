"""Shared LIBERO suite and single-task identity resolution."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from typing import Any

from libero.libero import benchmark

LOGGER = logging.getLogger(__name__)

UPSTREAM_DATASET_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
UPSTREAM_DATASET_LICENSE = "CC BY 4.0"
UPSTREAM_DATASET_URL = (
    "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
)

SUITE_EPISODE_LENGTHS: dict[str, int] = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SUPPORTED_SUITES = tuple(SUITE_EPISODE_LENGTHS)


@dataclass(frozen=True)
class LiberoTaskIdentity:
    """Unambiguous zero-based identity for one LIBERO benchmark task."""

    suite: str
    task_id: int
    task_name: str
    language: str
    problem_folder: str
    bddl_file: str
    source_relative_path: str
    episode_length: int

    @property
    def cache_name(self) -> str:
        return f"task_{self.task_id:03d}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_task_language(value: str) -> str:
    """Normalize only insignificant surrounding/repeated whitespace."""

    return " ".join(value.split())


def resolve_libero_task(suite: str, task_id: int) -> LiberoTaskIdentity:
    """Resolve and validate one task from the installed LIBERO benchmark registry."""

    if suite == "libero_100":
        raise ValueError(
            "--suite=libero_100 is ambiguous; choose --suite=libero_10 or "
            "--suite=libero_90"
        )
    if suite not in SUITE_EPISODE_LENGTHS:
        raise ValueError(
            f"Unknown LIBERO suite {suite!r}; choose one of {', '.join(SUPPORTED_SUITES)}"
        )
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise TypeError(f"--task-id must be an integer, got {type(task_id).__name__}")

    registry = benchmark.get_benchmark_dict()
    suite_type = registry.get(suite)
    if suite_type is None:
        raise RuntimeError(
            f"Installed LIBERO does not provide required suite {suite!r}; "
            f"available suites: {', '.join(sorted(registry))}"
        )

    # LIBERO prints its task ordering while constructing a benchmark. The evaluator
    # prints a clearer, stable task summary itself, so suppress this upstream noise.
    with redirect_stdout(io.StringIO()):
        task_suite = suite_type()
    task_count = len(task_suite.tasks)
    if not 0 <= task_id < task_count:
        raise ValueError(
            f"--task-id={task_id} is outside {suite}'s valid range 0..{task_count - 1}"
        )

    task = task_suite.get_task(task_id)
    language = normalize_task_language(str(task.language))
    if not language:
        raise RuntimeError(f"LIBERO {suite} task {task_id} has an empty language description")
    task_name = str(task.name)
    problem_folder = str(task.problem_folder)
    bddl_file = str(task.bddl_file)
    return LiberoTaskIdentity(
        suite=suite,
        task_id=task_id,
        task_name=task_name,
        language=language,
        problem_folder=problem_folder,
        bddl_file=bddl_file,
        source_relative_path=f"{suite}/{task_name}_demo.hdf5",
        episode_length=SUITE_EPISODE_LENGTHS[suite],
    )


def log_task_identity(task: LiberoTaskIdentity, logger: logging.Logger = LOGGER) -> None:
    logger.info("[Task] suite: %s", task.suite)
    logger.info("[Task] task_id: %d", task.task_id)
    logger.info("[Task] name: %s", task.task_name)
    logger.info("[Task] language: %s", task.language)
