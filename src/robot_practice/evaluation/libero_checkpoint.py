"""Best-effort semantic task assessment for ACT checkpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from robot_practice.evaluation.common import CheckpointManifest, practice_root, read_json
from robot_practice.evaluation.libero_dataset_cache import (
    DatasetTaskResolutionError,
    combined_dataset_root,
    inspect_lerobot_dataset,
)
from robot_practice.evaluation.libero_tasks import LiberoTaskIdentity, resolve_libero_task

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointTaskAssessment:
    status: Literal["match", "mismatch", "unknown"]
    message: str
    evidence: dict[str, Any]
    inferred_task: LiberoTaskIdentity | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["inferred_task"] = (
            None if self.inferred_task is None else self.inferred_task.to_dict()
        )
        return payload


def _run_root(policy_path: Path) -> Path | None:
    path = policy_path.resolve()
    if (
        path.name == "pretrained_model"
        and path.parent.parent.name == "checkpoints"
        and path.parent.parent.parent.is_dir()
    ):
        return path.parent.parent.parent
    return None


def _compare_identity(
    *,
    selected: LiberoTaskIdentity,
    inferred: LiberoTaskIdentity,
    evidence: dict[str, Any],
) -> CheckpointTaskAssessment:
    if (selected.suite, selected.task_id) == (inferred.suite, inferred.task_id):
        return CheckpointTaskAssessment(
            status="match",
            message=(
                f"Checkpoint task matches {selected.suite} task {selected.task_id}: "
                f"{selected.language}"
            ),
            evidence=evidence,
            inferred_task=inferred,
        )
    return CheckpointTaskAssessment(
        status="mismatch",
        message=(
            f"Checkpoint appears to target {inferred.suite} task {inferred.task_id}, "
            f"but evaluation selected {selected.suite} task {selected.task_id}. "
            "ACT has no language input; evaluation will continue, but results may be meaningless."
        ),
        evidence=evidence,
        inferred_task=inferred,
    )


def _manifest_identity(
    path: Path,
    selected: LiberoTaskIdentity,
) -> CheckpointTaskAssessment | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        return None
    suite = dataset.get("task_suite")
    task_id = dataset.get("simulator_task_id")
    if not isinstance(suite, str) or isinstance(task_id, bool) or not isinstance(task_id, int):
        return None
    try:
        inferred = resolve_libero_task(suite, task_id)
    except (TypeError, ValueError, RuntimeError):
        return None
    return _compare_identity(
        selected=selected,
        inferred=inferred,
        evidence={"source": str(path.resolve()), "dataset": dataset},
    )


def assess_checkpoint_task(
    manifest: CheckpointManifest,
    selected: LiberoTaskIdentity,
) -> CheckpointTaskAssessment:
    """Use explicit manifests first, then resolve train episodes when possible."""

    run_root = _run_root(manifest.path)
    if run_root is not None:
        result = _manifest_identity(run_root / "initialization_manifest.json", selected)
        if result is not None:
            return result

    train_config = manifest.train_config
    dataset_config = train_config.get("dataset") if isinstance(train_config, dict) else None
    if isinstance(dataset_config, dict):
        suite = dataset_config.get("task_suite")
        task_id = dataset_config.get("simulator_task_id")
        if isinstance(suite, str) and isinstance(task_id, int) and not isinstance(task_id, bool):
            try:
                inferred = resolve_libero_task(suite, task_id)
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                return _compare_identity(
                    selected=selected,
                    inferred=inferred,
                    evidence={"source": "train_config.json", "dataset": dataset_config},
                )

        episodes = dataset_config.get("episodes")
        configured_root = dataset_config.get("root")
        if (
            isinstance(episodes, list)
            and episodes
            and all(isinstance(value, int) for value in episodes)
        ):
            if configured_root is None:
                root = combined_dataset_root()
            else:
                root = Path(str(configured_root)).expanduser()
                if not root.is_absolute():
                    root = practice_root() / root
            if (root / "meta" / "info.json").is_file():
                try:
                    selection = inspect_lerobot_dataset(
                        root=root,
                        repo_id=str(dataset_config.get("repo_id", "checkpoint/dataset")),
                        task=selected,
                        source="checkpoint_training_dataset",
                        episode_namespace="checkpoint dataset indices",
                    )
                except DatasetTaskResolutionError:
                    return CheckpointTaskAssessment(
                        status="mismatch",
                        message=(
                            "Checkpoint training data does not contain the selected task. "
                            "ACT has no language input; evaluation will continue, but results "
                            "may be meaningless."
                        ),
                        evidence={
                            "source": "train_config.json",
                            "dataset_root": root.resolve(),
                            "episodes": episodes,
                        },
                    )
                configured = {int(value) for value in episodes}
                if configured.issubset(set(selection.episodes)):
                    return CheckpointTaskAssessment(
                        status="match",
                        message=(
                            f"Checkpoint episodes map to {selected.suite} task "
                            f"{selected.task_id}: {selected.language}"
                        ),
                        evidence={
                            "source": "train_config.json",
                            "dataset_root": root.resolve(),
                            "episodes": episodes,
                            "dataset_task_index": selection.task_index,
                        },
                        inferred_task=selected,
                    )
                return CheckpointTaskAssessment(
                    status="mismatch",
                    message=(
                        "Checkpoint episode selection does not belong entirely to the selected "
                        "task. ACT has no language input; evaluation will continue, but results "
                        "may be meaningless."
                    ),
                    evidence={
                        "source": "train_config.json",
                        "dataset_root": root.resolve(),
                        "episodes": episodes,
                        "selected_task_episodes": list(selection.episodes),
                    },
                )

    return CheckpointTaskAssessment(
        status="unknown",
        message=(
            "Cannot verify the checkpoint training task. ACT has no language input; "
            "evaluation will continue, but confirm the selected task manually."
        ),
        evidence={"source": "none"},
    )


def log_checkpoint_task_assessment(
    assessment: CheckpointTaskAssessment,
    logger: logging.Logger = LOGGER,
) -> None:
    logger.info("[Policy] checkpoint task: %s", assessment.status)
    if assessment.status == "match":
        logger.info("[Policy] %s", assessment.message)
    else:
        logger.warning("[Warning] %s", assessment.message)
