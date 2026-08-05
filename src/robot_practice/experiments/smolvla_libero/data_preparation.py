"""Prepare reproducible SmolVLA LIBERO domain-adaptation and target-task splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

SCHEMA_VERSION = 1
MAPPING_CONTRACT = "raw_action_noop_v1"
RNG_CONTRACT = "pcg64_seedsequence_v1"
BACKGROUND_RNG_NAMESPACE = 0
TARGET_DEMONSTRATION_RNG_NAMESPACE = 1
TARGET_CLOSED_LOOP_RNG_NAMESPACE = 2
DEFAULT_RAW_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DEFAULT_RAW_REVISION = "main"
DEFAULT_SUITE = "libero_goal"
DEFAULT_TASK_IDS = (1, 4, 8)
DEFAULT_SPLIT_SEED = 1000
DEFAULT_OFFLINE_VALIDATION_PER_TASK = 10
DEFAULT_CLOSED_LOOP_VALIDATION_PER_TASK = 20
DEFAULT_CLOSED_LOOP_TEST_PER_TASK = 30
TARGET_TASKS = {
    1: {
        "combined_task_index": 17,
        "canonical": "put the bowl on the stove",
        "task_name": "put_the_bowl_on_the_stove",
    },
    4: {
        "combined_task_index": 18,
        "canonical": "put the bowl on top of the cabinet",
        "task_name": "put_the_bowl_on_top_of_the_cabinet",
    },
    8: {
        "combined_task_index": 10,
        "canonical": "put the bowl on the plate",
        "task_name": "put_the_bowl_on_the_plate",
    },
}
TARGET_COMBINED_TASK_INDICES = frozenset(
    int(definition["combined_task_index"]) for definition in TARGET_TASKS.values()
)
EXPECTED_TOTAL_TASKS = 40
EXPECTED_TOTAL_EPISODES = 1693
EXPECTED_TOTAL_FRAMES = 273465
EXPECTED_OFFICIAL_INIT_STATES = 50
TRAIN_PARAPHRASE_COUNT = 8
HELD_OUT_PARAPHRASE_COUNT = 5
REPO_ROOT = Path(__file__).resolve().parents[4]


def deterministic_permutation(
    values: Iterable[int],
    *,
    seed: int,
    namespace: int,
    identity: int,
) -> list[int]:
    ordered = np.asarray(sorted(values), dtype=np.int64)
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([seed, namespace, identity]))
    )
    return [int(value) for value in generator.permutation(ordered).tolist()]


def rng_contract_payload(
    *,
    seed: int,
    namespaces: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "name": RNG_CONTRACT,
        "bit_generator": "PCG64",
        "seed": seed,
        "namespaces": dict(sorted(namespaces.items())),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {source}")
    return payload


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _relative_to_repo(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_integer_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("integer list must be non-empty and unique")
    return values


def _target_definitions(task_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for task_id in task_ids:
        if task_id not in TARGET_TASKS:
            raise ValueError(f"Unsupported target task ID: {task_id}")
        selected[task_id] = TARGET_TASKS[task_id]
    return selected


def _libero_suite(suite: str) -> Any:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite: {suite}")
    return benchmark_dict[suite]()


def _installed_init_state_records(
    suite: str,
    task_ids: Iterable[int],
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    task_suite = _libero_suite(suite)
    arrays: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        task = task_suite.get_task(task_id)
        states = np.asarray(task_suite.get_task_init_states(task_id), dtype=np.float64)
        if states.ndim != 2:
            raise ValueError(f"Task {task_id} init states must be a 2D array")
        from libero.libero import get_libero_path

        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        arrays[task_id] = states
        records.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "relative_path": f"{task.problem_folder}/{task.init_states_file}",
                "sha256": sha256_file(path),
                "shape": list(states.shape),
            }
        )
    return arrays, records


def _sibling_lfs_sha256(sibling: Any) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256")
    else:
        value = getattr(lfs, "sha256", None)
    return value if isinstance(value, str) else None


def fetch_sources(
    *,
    raw_repo_id: str,
    raw_revision: str,
    raw_data_root: str | Path,
    suite: str,
    task_ids: Sequence[int],
    source_manifest: str | Path,
) -> dict[str, Any]:
    """Download and checksum the complete selected LIBERO suite."""
    from huggingface_hub import HfApi, snapshot_download

    _target_definitions(task_ids)
    api = HfApi()
    info = api.repo_info(
        raw_repo_id,
        repo_type="dataset",
        revision=raw_revision,
        files_metadata=True,
    )
    resolved_revision = info.sha
    siblings = [
        sibling
        for sibling in info.siblings
        if sibling.rfilename.startswith(f"{suite}/") and sibling.rfilename.endswith(".hdf5")
    ]
    if not siblings:
        raise ValueError(f"No HDF5 files found for {raw_repo_id}:{suite}")

    destination = Path(raw_data_root).expanduser().resolve()
    snapshot_download(
        repo_id=raw_repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=[f"{suite}/*.hdf5"],
        local_dir=destination,
    )

    file_records: list[dict[str, Any]] = []
    for sibling in sorted(siblings, key=lambda item: item.rfilename):
        local_path = destination / sibling.rfilename
        if not local_path.is_file():
            raise FileNotFoundError(f"Downloaded source is missing: {sibling.rfilename}")
        digest = sha256_file(local_path)
        expected_digest = _sibling_lfs_sha256(sibling)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {sibling.rfilename}: "
                f"expected {expected_digest}, got {digest}"
            )
        file_records.append(
            {
                "relative_path": sibling.rfilename,
                "size_bytes": local_path.stat().st_size,
                "sha256": digest,
            }
        )

    _, init_state_records = _installed_init_state_records(suite, task_ids)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source": {
            "repo_id": raw_repo_id,
            "requested_revision": raw_revision,
            "resolved_revision": resolved_revision,
            "suite": suite,
            "files": file_records,
        },
        "target_task_ids": sorted(task_ids),
        "installed_libero": {
            "version": _package_version("hf-libero", "libero"),
            "init_state_files": init_state_records,
        },
        "mapping_contract": MAPPING_CONTRACT,
    }
    write_json_atomic(source_manifest, manifest)
    return manifest


def verify_source_manifest(
    *,
    source_manifest: str | Path,
    raw_data_root: str | Path,
    suite: str,
    task_ids: Sequence[int],
) -> dict[str, Any]:
    manifest = load_json_object(source_manifest)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported source manifest schema")
    if manifest.get("status") != "complete":
        raise ValueError("Source manifest is not complete")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("suite") != suite:
        raise ValueError("Source manifest suite mismatch")
    resolved_revision = source.get("resolved_revision")
    if not isinstance(resolved_revision, str) or resolved_revision == "main":
        raise ValueError("Source manifest must contain a resolved immutable revision")
    if manifest.get("mapping_contract") != MAPPING_CONTRACT:
        raise ValueError("Source manifest mapping contract mismatch")
    if tuple(manifest.get("target_task_ids", ())) != tuple(sorted(task_ids)):
        raise ValueError("Source manifest target task IDs mismatch")

    root = Path(raw_data_root).expanduser().resolve()
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Source manifest contains no HDF5 files")
    for record in files:
        if not isinstance(record, dict):
            raise TypeError("Invalid source file record")
        relative_path = record.get("relative_path")
        expected = record.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected, str):
            raise TypeError("Invalid source file checksum record")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Source file missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Source checksum mismatch for {relative_path}: expected {expected}, got {actual}"
            )

    _, current_init_records = _installed_init_state_records(suite, task_ids)
    installed = manifest.get("installed_libero")
    expected_init = installed.get("init_state_files") if isinstance(installed, dict) else None
    if expected_init != current_init_records:
        raise ValueError("Installed LIBERO init-state checksums do not match source manifest")
    return manifest


def scaffold_paraphrases(
    *,
    suite: str,
    task_ids: Sequence[int],
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    definitions = _target_definitions(task_ids)
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Paraphrase file already exists: {destination}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite,
        "canonical_probability": 0.25,
        "tasks": {
            str(task_id): {
                "canonical": str(definition["canonical"]),
                "train_paraphrases": [],
                "held_out_paraphrases": [],
                "semantic_review": {
                    "status": "pending",
                    "reviewed_at": None,
                    "checks": {
                        "object_preserved": False,
                        "target_preserved": False,
                        "completion_condition_preserved": False,
                        "train_test_separated": False,
                    },
                },
            }
            for task_id, definition in sorted(definitions.items())
        },
    }
    write_json_atomic(destination, payload)
    return payload


def normalize_instruction(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.rstrip(" .!?")


def audit_paraphrases(
    path: str | Path,
    *,
    suite: str,
    task_ids: Sequence[int],
) -> dict[str, Any]:
    payload = load_json_object(path)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("suite") != suite:
        raise ValueError("Paraphrase schema or suite mismatch")
    if payload.get("canonical_probability") != 0.25:
        raise ValueError("canonical_probability must be 0.25")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {str(value) for value in task_ids}:
        raise ValueError("Paraphrase task IDs do not match the requested targets")

    records: dict[str, Any] = {}
    for task_id, definition in sorted(_target_definitions(task_ids).items()):
        record = tasks[str(task_id)]
        if not isinstance(record, dict):
            raise TypeError(f"Task {task_id} paraphrase record must be an object")
        canonical = record.get("canonical")
        train = record.get("train_paraphrases")
        held_out = record.get("held_out_paraphrases")
        if canonical != definition["canonical"]:
            raise ValueError(f"Task {task_id} canonical instruction mismatch")
        if not isinstance(train, list) or len(train) != TRAIN_PARAPHRASE_COUNT:
            raise ValueError(f"Task {task_id} must have {TRAIN_PARAPHRASE_COUNT} train paraphrases")
        if not isinstance(held_out, list) or len(held_out) != HELD_OUT_PARAPHRASE_COUNT:
            raise ValueError(
                f"Task {task_id} must have {HELD_OUT_PARAPHRASE_COUNT} held-out paraphrases"
            )
        values = [canonical, *train, *held_out]
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Task {task_id} instructions must be non-empty strings")
        normalized = [normalize_instruction(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Task {task_id} contains duplicate normalized instructions")

        review = record.get("semantic_review")
        if not isinstance(review, dict) or review.get("status") != "approved":
            raise ValueError(f"Task {task_id} semantic review is not approved")
        reviewed_at = review.get("reviewed_at")
        try:
            date.fromisoformat(reviewed_at)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Task {task_id} reviewed_at must be YYYY-MM-DD") from error
        checks = review.get("checks")
        expected_checks = {
            "object_preserved",
            "target_preserved",
            "completion_condition_preserved",
            "train_test_separated",
        }
        if (
            not isinstance(checks, dict)
            or set(checks) != expected_checks
            or any(checks[key] is not True for key in expected_checks)
        ):
            raise ValueError(f"Task {task_id} semantic review checks are incomplete")
        records[str(task_id)] = {
            "canonical_count": 1,
            "train_paraphrase_count": len(train),
            "held_out_paraphrase_count": len(held_out),
            "normalized_duplicates": 0,
            "semantic_review_status": "approved",
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source_sha256": sha256_file(path),
        "tasks": records,
    }


def verify_paraphrase_binding(
    *,
    target_split: str | Path,
    paraphrases: str | Path,
) -> dict[str, Any]:
    split = load_json_object(target_split)
    if split.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported target split schema")
    suite = split.get("suite")
    tasks = split.get("tasks")
    expected_hash = split.get("paraphrases_sha256")
    if not isinstance(suite, str) or not isinstance(tasks, dict):
        raise TypeError("Target split lacks suite or tasks")
    if not isinstance(expected_hash, str):
        raise TypeError("Target split lacks paraphrases_sha256")
    try:
        task_ids = tuple(sorted(int(task_id) for task_id in tasks))
    except ValueError as error:
        raise TypeError("Target split task IDs must be integers") from error
    audit = audit_paraphrases(paraphrases, suite=suite, task_ids=task_ids)
    if audit["source_sha256"] != expected_hash:
        raise ValueError("Paraphrases checksum does not match target split")
    return audit


@dataclass(frozen=True)
class EpisodeCatalogEntry:
    episode_index: int
    task_index: int
    length: int


@dataclass(frozen=True)
class ActionEpisode:
    episode_index: int
    task_index: int
    actions: np.ndarray
    states: np.ndarray


@dataclass(frozen=True)
class RawDemo:
    name: str
    actions: np.ndarray


@dataclass(frozen=True)
class DemoMapping:
    episode_index: int
    raw_demo: str
    action_max_abs_error: float


def _tasks_table(dataset_root: Path) -> dict[int, str]:
    table = pq.read_table(dataset_root / "meta" / "tasks.parquet").to_pydict()
    indices = table.get("task_index")
    names = table.get("__index_level_0__")
    if not isinstance(indices, list) or not isinstance(names, list) or len(indices) != len(names):
        raise ValueError("Unsupported tasks.parquet schema")
    result = {int(index): str(name) for index, name in zip(indices, names, strict=True)}
    if len(result) != EXPECTED_TOTAL_TASKS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_TASKS} tasks, got {len(result)}")
    return result


def load_episode_catalog(
    dataset_root: str | Path,
) -> tuple[dict[int, EpisodeCatalogEntry], dict[int, str]]:
    root = Path(dataset_root).expanduser().resolve()
    info = load_json_object(root / "meta" / "info.json")
    if info.get("total_tasks") != EXPECTED_TOTAL_TASKS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_TASKS} combined tasks")
    if info.get("total_episodes") != EXPECTED_TOTAL_EPISODES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_EPISODES} combined episodes")
    if info.get("total_frames") != EXPECTED_TOTAL_FRAMES:
        raise ValueError(f"Expected {EXPECTED_TOTAL_FRAMES} combined frames")

    episodes_dataset = pads.dataset(root / "meta" / "episodes", format="parquet")
    episodes = episodes_dataset.to_table(columns=["episode_index", "length"]).to_pydict()
    lengths = {
        int(episode): int(length)
        for episode, length in zip(
            episodes["episode_index"],
            episodes["length"],
            strict=True,
        )
    }
    starts = pads.dataset(root / "data", format="parquet").to_table(
        columns=["episode_index", "task_index"],
        filter=pads.field("frame_index") == 0,
    )
    starts_payload = starts.to_pydict()
    task_by_episode: dict[int, int] = {}
    for episode, task in zip(
        starts_payload["episode_index"],
        starts_payload["task_index"],
        strict=True,
    ):
        episode_index = int(episode)
        if episode_index in task_by_episode:
            raise ValueError(f"Episode {episode_index} has multiple frame-zero rows")
        task_by_episode[episode_index] = int(task)
    if set(lengths) != set(task_by_episode):
        raise ValueError("Episode metadata and frame-zero task mapping disagree")

    catalog = {
        episode: EpisodeCatalogEntry(
            episode_index=episode,
            task_index=task_by_episode[episode],
            length=length,
        )
        for episode, length in lengths.items()
    }
    if sum(entry.length for entry in catalog.values()) != EXPECTED_TOTAL_FRAMES:
        raise ValueError("Episode lengths do not sum to total_frames")
    return catalog, _tasks_table(root)


def build_background_split(
    catalog: Mapping[int, EpisodeCatalogEntry],
    task_names: Mapping[int, str],
    *,
    seed: int,
    excluded_task_indices: frozenset[int] = TARGET_COMBINED_TASK_INDICES,
) -> dict[str, Any]:
    by_task: dict[int, list[EpisodeCatalogEntry]] = defaultdict(list)
    for entry in catalog.values():
        by_task[entry.task_index].append(entry)
    background_tasks = sorted(set(by_task) - set(excluded_task_indices))
    if len(background_tasks) != EXPECTED_TOTAL_TASKS - len(excluded_task_indices):
        raise ValueError("Background task count does not equal 37")

    task_payload: dict[str, Any] = {}
    total_train_episodes = 0
    total_validation_episodes = 0
    total_train_frames = 0
    total_validation_frames = 0
    for task_index in background_tasks:
        entries = sorted(by_task[task_index], key=lambda item: item.episode_index)
        entries_by_id = {entry.episode_index: entry for entry in entries}
        permutation = [
            entries_by_id[episode_index]
            for episode_index in deterministic_permutation(
                entries_by_id,
                seed=seed,
                namespace=BACKGROUND_RNG_NAMESPACE,
                identity=task_index,
            )
        ]
        validation_count = max(1, math.floor(0.1 * len(entries)))
        validation = sorted(permutation[:validation_count], key=lambda item: item.episode_index)
        train = sorted(permutation[validation_count:], key=lambda item: item.episode_index)
        train_frames = sum(item.length for item in train)
        validation_frames = sum(item.length for item in validation)
        total_train_episodes += len(train)
        total_validation_episodes += len(validation)
        total_train_frames += train_frames
        total_validation_frames += validation_frames
        task_payload[str(task_index)] = {
            "task_index": task_index,
            "canonical": task_names[task_index],
            "train_episode_indices": [item.episode_index for item in train],
            "validation_episode_indices": [item.episode_index for item in validation],
            "train_frames": train_frames,
            "validation_frames": validation_frames,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_repo_id": "lerobot/libero",
        "split_seed": seed,
        "rng_contract": rng_contract_payload(
            seed=seed,
            namespaces={"background_task": BACKGROUND_RNG_NAMESPACE},
        ),
        "excluded_target_task_indices": sorted(excluded_task_indices),
        "tasks": task_payload,
        "aggregate": {
            "num_tasks": len(background_tasks),
            "train_episodes": total_train_episodes,
            "validation_episodes": total_validation_episodes,
            "train_frames": total_train_frames,
            "validation_frames": total_validation_frames,
        },
    }


def filter_raw_noop_actions(actions: np.ndarray, *, threshold: float = 1e-4) -> np.ndarray:
    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"Expected raw actions with shape (T, 7), got {array.shape}")
    kept: list[np.ndarray] = []
    for action in array:
        previous_kept = kept[-1] if kept else None
        near_zero_motion = np.linalg.norm(action[:-1]) < threshold
        same_gripper = previous_kept is not None and action[-1] == previous_kept[-1]
        is_noop = near_zero_motion if previous_kept is None else near_zero_motion and same_gripper
        if not is_noop:
            kept.append(action)
    if not kept:
        return np.empty((0, 7), dtype="<f4")
    return np.asarray(kept, dtype="<f4", order="C")


def action_fingerprint(actions: np.ndarray) -> tuple[int, str]:
    array = np.asarray(actions, dtype="<f4", order="C")
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"Expected actions with shape (T, 7), got {array.shape}")
    return len(array), hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_target_action_episodes(
    dataset_root: str | Path,
    *,
    task_index: int,
) -> list[ActionEpisode]:
    root = Path(dataset_root).expanduser().resolve()
    table = pads.dataset(root / "data", format="parquet").to_table(
        columns=[
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "action",
            "observation.state",
        ],
        filter=pads.field("task_index") == task_index,
    )
    rows = table.to_pylist()
    rows.sort(key=lambda row: int(row["index"]))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_index"])].append(row)

    episodes: list[ActionEpisode] = []
    for episode_index, episode_rows in sorted(grouped.items()):
        frame_indices = [int(row["frame_index"]) for row in episode_rows]
        if frame_indices != list(range(len(episode_rows))):
            raise ValueError(f"Episode {episode_index} frame indices are not contiguous")
        actions = np.asarray([row["action"] for row in episode_rows], dtype="<f4")
        states = np.asarray([row["observation.state"] for row in episode_rows], dtype="<f4")
        if actions.shape != (len(episode_rows), 7):
            raise ValueError(f"Episode {episode_index} action shape mismatch: {actions.shape}")
        if states.shape != (len(episode_rows), 8) or not np.isfinite(states).all():
            raise ValueError(f"Episode {episode_index} state contract is invalid")
        episodes.append(
            ActionEpisode(
                episode_index=episode_index,
                task_index=task_index,
                actions=actions,
                states=states,
            )
        )
    return episodes


def load_raw_demos(path: str | Path) -> list[RawDemo]:
    source = Path(path)
    demos: list[RawDemo] = []
    with h5py.File(source, "r") as hdf5:
        if "data" not in hdf5:
            raise ValueError(f"Raw HDF5 lacks /data: {source.name}")

        def demo_order(name: str) -> int:
            match = re.fullmatch(r"demo_(\d+)", name)
            if match is None:
                raise ValueError(f"Unexpected raw demo key: {name}")
            return int(match.group(1))

        for name in sorted(hdf5["data"].keys(), key=demo_order):
            demo = hdf5["data"][name]
            if "actions" not in demo:
                raise ValueError(f"Raw demo {name} lacks actions")
            raw_actions = np.asarray(demo["actions"])
            demos.append(
                RawDemo(
                    name=name,
                    actions=filter_raw_noop_actions(raw_actions),
                )
            )
    return demos


def match_action_episodes(
    episodes: Sequence[ActionEpisode],
    raw_demos: Sequence[RawDemo],
    *,
    tolerance: float = 1e-6,
) -> tuple[dict[int, tuple[RawDemo, float]], list[str]]:
    by_fingerprint: dict[tuple[int, str], list[RawDemo]] = defaultdict(list)
    for demo in raw_demos:
        by_fingerprint[action_fingerprint(demo.actions)].append(demo)

    matches: dict[int, tuple[RawDemo, float]] = {}
    used_raw_names: set[str] = set()
    for episode in episodes:
        candidates = by_fingerprint.get(action_fingerprint(episode.actions), [])
        if len(candidates) != 1:
            raise ValueError(
                f"Episode {episode.episode_index} has {len(candidates)} raw action matches"
            )
        candidate = candidates[0]
        if candidate.actions.shape != episode.actions.shape:
            raise ValueError(f"Episode {episode.episode_index} matched action shape mismatch")
        error = float(np.max(np.abs(candidate.actions - episode.actions), initial=0.0))
        if error > tolerance:
            raise ValueError(
                f"Episode {episode.episode_index} action error {error} exceeds {tolerance}"
            )
        if candidate.name in used_raw_names:
            raise ValueError(f"Raw demo {candidate.name} maps to multiple LeRobot episodes")
        used_raw_names.add(candidate.name)
        matches[episode.episode_index] = (candidate, error)
    unmatched = sorted(demo.name for demo in raw_demos if demo.name not in used_raw_names)
    return matches, unmatched


def build_target_task_split(
    mappings: Sequence[DemoMapping],
    *,
    task_id: int,
    seed: int,
    official_state_count: int,
    offline_validation_count: int,
    closed_loop_validation_count: int,
    closed_loop_test_count: int,
) -> dict[str, Any]:
    if offline_validation_count <= 0:
        raise ValueError("Offline validation count must be positive")
    if closed_loop_validation_count <= 0 or closed_loop_test_count <= 0:
        raise ValueError("Closed-loop split counts must be positive")
    if closed_loop_validation_count + closed_loop_test_count != official_state_count:
        raise ValueError("Closed-loop split counts must cover every official init state")
    if len(mappings) <= offline_validation_count:
        raise ValueError(
            f"Task {task_id} requires more than {offline_validation_count} "
            f"mapped demonstrations, got {len(mappings)}"
        )

    episode_ids = [mapping.episode_index for mapping in mappings]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError(f"Task {task_id} has duplicate episode_index mappings")
    raw_demo_names = [mapping.raw_demo for mapping in mappings]
    if len(raw_demo_names) != len(set(raw_demo_names)):
        raise ValueError(f"Task {task_id} has duplicate raw_demo mappings")

    mappings_by_episode = {mapping.episode_index: mapping for mapping in mappings}
    demonstration_order = deterministic_permutation(
        mappings_by_episode,
        seed=seed,
        namespace=TARGET_DEMONSTRATION_RNG_NAMESPACE,
        identity=task_id,
    )
    offline_validation_ids = set(demonstration_order[:offline_validation_count])
    train_ids = set(demonstration_order[offline_validation_count:])

    def records_for(episode_indices: set[int]) -> list[dict[str, Any]]:
        return [
            {
                "episode_index": mapping.episode_index,
                "raw_demo": mapping.raw_demo,
                "action_max_abs_error": mapping.action_max_abs_error,
            }
            for mapping in sorted(mappings, key=lambda item: item.episode_index)
            if mapping.episode_index in episode_indices
        ]

    train_records = records_for(train_ids)
    offline_validation_records = records_for(offline_validation_ids)
    train_raw_demos = {record["raw_demo"] for record in train_records}
    offline_validation_raw_demos = {record["raw_demo"] for record in offline_validation_records}
    if train_ids & offline_validation_ids:
        raise ValueError(f"Task {task_id} demonstration episode sets overlap")
    if train_raw_demos & offline_validation_raw_demos:
        raise ValueError(f"Task {task_id} demonstration raw demo sets overlap")

    closed_loop_order = deterministic_permutation(
        range(official_state_count),
        seed=seed,
        namespace=TARGET_CLOSED_LOOP_RNG_NAMESPACE,
        identity=task_id,
    )
    closed_loop_validation = sorted(closed_loop_order[:closed_loop_validation_count])
    closed_loop_test = sorted(closed_loop_order[closed_loop_validation_count:])
    if len(closed_loop_test) != closed_loop_test_count:
        raise ValueError(
            f"Task {task_id} closed-loop test split has {len(closed_loop_test)} states"
        )
    if set(closed_loop_validation) & set(closed_loop_test):
        raise ValueError(f"Task {task_id} closed-loop state sets overlap")
    if set(closed_loop_validation) | set(closed_loop_test) != set(range(official_state_count)):
        raise ValueError(f"Task {task_id} closed-loop states do not cover all official states")

    return {
        "demonstration_split": {
            "rng_namespace": TARGET_DEMONSTRATION_RNG_NAMESPACE,
            "sampling_strategy": "uniform_task_then_uniform_episode",
            "train_demonstrations": train_records,
            "offline_validation_demonstrations": offline_validation_records,
        },
        "closed_loop_split": {
            "rng_namespace": TARGET_CLOSED_LOOP_RNG_NAMESPACE,
            "validation_init_state_ids": closed_loop_validation,
            "test_init_state_ids": closed_loop_test,
        },
    }


def _write_config(
    path: str | Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        existing = load_json_object(destination)
        if existing != payload:
            raise FileExistsError(
                f"Config exists with different content; pass --overwrite: {destination}"
            )
        return destination
    return write_json_atomic(destination, payload)


def build_splits(
    *,
    dataset_root: str | Path,
    raw_data_root: str | Path,
    source_manifest: str | Path,
    suite: str,
    task_ids: Sequence[int],
    offline_validation_per_task: int,
    closed_loop_validation_per_task: int,
    closed_loop_test_per_task: int,
    seed: int,
    paraphrases: str | Path,
    output_background_split: str | Path,
    output_target_split: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    definitions = _target_definitions(task_ids)
    source_payload = verify_source_manifest(
        source_manifest=source_manifest,
        raw_data_root=raw_data_root,
        suite=suite,
        task_ids=task_ids,
    )
    paraphrase_audit = audit_paraphrases(paraphrases, suite=suite, task_ids=task_ids)
    catalog, task_names = load_episode_catalog(dataset_root)
    background_split = build_background_split(catalog, task_names, seed=seed)
    official_by_task, _ = _installed_init_state_records(suite, task_ids)

    target_tasks: dict[str, Any] = {}
    provenance_tasks: dict[str, Any] = {}
    metrics_tasks: dict[str, Any] = {}
    root = Path(raw_data_root).expanduser().resolve()
    for task_id, definition in sorted(definitions.items()):
        combined_task_index = int(definition["combined_task_index"])
        episodes = load_target_action_episodes(
            dataset_root,
            task_index=combined_task_index,
        )
        raw_path = root / suite / f"{definition['task_name']}_demo.hdf5"
        raw_demos = load_raw_demos(raw_path)
        action_matches, unmatched_raw = match_action_episodes(episodes, raw_demos)

        mappings: list[DemoMapping] = []
        for episode in episodes:
            raw_demo, action_error = action_matches[episode.episode_index]
            mappings.append(
                DemoMapping(
                    episode_index=episode.episode_index,
                    raw_demo=raw_demo.name,
                    action_max_abs_error=action_error,
                )
            )
        split = build_target_task_split(
            mappings,
            task_id=task_id,
            seed=seed,
            official_state_count=len(official_by_task[task_id]),
            offline_validation_count=offline_validation_per_task,
            closed_loop_validation_count=closed_loop_validation_per_task,
            closed_loop_test_count=closed_loop_test_per_task,
        )
        demonstration_split = split["demonstration_split"]
        train_episode_ids = {
            int(record["episode_index"]) for record in demonstration_split["train_demonstrations"]
        }
        offline_validation_episode_ids = {
            int(record["episode_index"])
            for record in demonstration_split["offline_validation_demonstrations"]
        }
        demonstration_split["train_frames"] = sum(
            catalog[episode_index].length for episode_index in train_episode_ids
        )
        demonstration_split["offline_validation_frames"] = sum(
            catalog[episode_index].length for episode_index in offline_validation_episode_ids
        )
        target_tasks[str(task_id)] = {
            "task_id": task_id,
            "combined_task_index": combined_task_index,
            "canonical": definition["canonical"],
            **split,
        }
        mapping_records = [
            {
                "episode_index": mapping.episode_index,
                "raw_demo": mapping.raw_demo,
                "action_max_abs_error": mapping.action_max_abs_error,
            }
            for mapping in sorted(mappings, key=lambda item: item.episode_index)
        ]
        provenance_tasks[str(task_id)] = {
            "task_id": task_id,
            "combined_task_index": combined_task_index,
            "raw_file": f"{suite}/{raw_path.name}",
            "mappings": mapping_records,
            "unmatched_raw_demos": unmatched_raw,
        }
        closed_loop_split = split["closed_loop_split"]
        metrics_tasks[str(task_id)] = {
            "combined_episodes": len(episodes),
            "raw_demos": len(raw_demos),
            "mapped_episodes": len(mappings),
            "unmatched_raw_demos": len(unmatched_raw),
            "train_episodes": len(demonstration_split["train_demonstrations"]),
            "train_frames": demonstration_split["train_frames"],
            "offline_validation_episodes": len(
                demonstration_split["offline_validation_demonstrations"]
            ),
            "offline_validation_frames": demonstration_split["offline_validation_frames"],
            "closed_loop_validation_states": len(closed_loop_split["validation_init_state_ids"]),
            "closed_loop_test_states": len(closed_loop_split["test_init_state_ids"]),
            "max_action_abs_error": max(
                (mapping.action_max_abs_error for mapping in mappings),
                default=0.0,
            ),
        }

    source_manifest_hash = sha256_file(source_manifest)
    target_split = {
        "schema_version": SCHEMA_VERSION,
        "dataset_repo_id": "lerobot/libero",
        "suite": suite,
        "split_seed": seed,
        "rng_contract": rng_contract_payload(
            seed=seed,
            namespaces={
                "target_demonstration": TARGET_DEMONSTRATION_RNG_NAMESPACE,
                "target_closed_loop": TARGET_CLOSED_LOOP_RNG_NAMESPACE,
            },
        ),
        "mapping_contract": MAPPING_CONTRACT,
        "training_sampling_strategy": "uniform_task_then_uniform_episode",
        "validation_instruction": "canonical",
        "main_test_instruction": "canonical",
        "auxiliary_test_protocols": [
            "held_out_paraphrase",
            "prompt_swap",
        ],
        "source_manifest_sha256": source_manifest_hash,
        "paraphrases_sha256": paraphrase_audit["source_sha256"],
        "tasks": target_tasks,
    }
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": "Offline",
        "inputs": {
            "dataset_root": _relative_to_repo(dataset_root_path),
            "dataset_info_sha256": sha256_file(dataset_root_path / "meta" / "info.json"),
            "dataset_tasks_sha256": sha256_file(dataset_root_path / "meta" / "tasks.parquet"),
            "source_manifest": _relative_to_repo(source_manifest),
            "source_manifest_sha256": source_manifest_hash,
            "paraphrases": _relative_to_repo(paraphrases),
            "paraphrases_sha256": paraphrase_audit["source_sha256"],
        },
        "parameters": {
            "suite": suite,
            "task_ids": sorted(task_ids),
            "offline_validation_per_task": offline_validation_per_task,
            "closed_loop_validation_per_task": closed_loop_validation_per_task,
            "closed_loop_test_per_task": closed_loop_test_per_task,
            "split_seed": seed,
            "mapping_contract": MAPPING_CONTRACT,
            "rng_contract": RNG_CONTRACT,
        },
        "outputs": {
            "background_split": _relative_to_repo(output_background_split),
            "target_split": _relative_to_repo(output_target_split),
            "background_split_sha256": canonical_json_sha256(background_split),
            "target_split_sha256": canonical_json_sha256(target_split),
        },
        "source_revision": source_payload["source"]["resolved_revision"],
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "mode": "Offline",
        "status": "complete",
        "background": background_split["aggregate"],
        "targets": metrics_tasks,
        "paraphrases": paraphrase_audit["tasks"],
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "mapping_contract": MAPPING_CONTRACT,
        "tasks": provenance_tasks,
    }

    _write_config(output_background_split, background_split, overwrite=overwrite)
    _write_config(output_target_split, target_split, overwrite=overwrite)
    write_json_atomic(output_root / "target_provenance.json", provenance)
    write_json_atomic(output_root / "paraphrase_audit.json", paraphrase_audit)
    write_json_atomic(output_root / "metrics.json", metrics)
    write_json_atomic(output_root / "manifest.json", manifest)
    return {
        "background_split": background_split,
        "target_split": target_split,
        "provenance": provenance,
        "paraphrase_audit": paraphrase_audit,
        "metrics": metrics,
        "manifest": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("fetch-sources", "scaffold-paraphrases", "build-splits"),
        required=True,
    )
    parser.add_argument("--raw-repo-id", default=DEFAULT_RAW_REPO_ID)
    parser.add_argument("--raw-revision", default=DEFAULT_RAW_REVISION)
    parser.add_argument("--raw-data-root", type=Path, default=Path("datasets/raw/libero"))
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/lerobot/lerobot/libero")
    )
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--task-ids", type=parse_integer_csv, default=DEFAULT_TASK_IDS)
    parser.add_argument(
        "--offline-validation-per-task",
        type=int,
        default=DEFAULT_OFFLINE_VALIDATION_PER_TASK,
    )
    parser.add_argument(
        "--closed-loop-validation-per-task",
        type=int,
        default=DEFAULT_CLOSED_LOOP_VALIDATION_PER_TASK,
    )
    parser.add_argument(
        "--closed-loop-test-per-task",
        type=int,
        default=DEFAULT_CLOSED_LOOP_TEST_PER_TASK,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--init-states-source", choices=("installed_libero",), default="installed_libero"
    )
    parser.add_argument(
        "--paraphrases",
        "--output-paraphrases",
        dest="paraphrases",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/paraphrases.json"),
    )
    parser.add_argument(
        "--output-background-split",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/background_split.json"),
    )
    parser.add_argument(
        "--output-target-split",
        "--output-split",
        dest="output_target_split",
        type=Path,
        default=Path("configs/experiments/smolvla_libero/target_split.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smolvla_libero/data_audit"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _positive_split_counts(args: argparse.Namespace) -> None:
    counts = (
        args.offline_validation_per_task,
        args.closed_loop_validation_per_task,
        args.closed_loop_test_per_task,
    )
    if any(value <= 0 for value in counts):
        raise ValueError("Target split counts must be positive")
    if (
        args.closed_loop_validation_per_task + args.closed_loop_test_per_task
        != EXPECTED_OFFICIAL_INIT_STATES
    ):
        raise ValueError(f"Closed-loop split counts must sum to {EXPECTED_OFFICIAL_INIT_STATES}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    task_ids = tuple(args.task_ids)
    source_manifest = args.source_manifest or args.output_dir / "source_manifest.json"
    if args.stage == "fetch-sources":
        manifest = fetch_sources(
            raw_repo_id=args.raw_repo_id,
            raw_revision=args.raw_revision,
            raw_data_root=args.raw_data_root,
            suite=args.suite,
            task_ids=task_ids,
            source_manifest=source_manifest,
        )
        print(
            "SmolVLA LIBERO sources ready: "
            f"{len(manifest['source']['files'])} files, "
            f"revision={manifest['source']['resolved_revision']}"
        )
        print(f"manifest: {source_manifest}")
        return
    if args.stage == "scaffold-paraphrases":
        scaffold_paraphrases(
            suite=args.suite,
            task_ids=task_ids,
            output_path=args.paraphrases,
            overwrite=args.overwrite,
        )
        print(f"Paraphrase review template created: {args.paraphrases}")
        return

    _positive_split_counts(args)
    result = build_splits(
        dataset_root=args.dataset_root,
        raw_data_root=args.raw_data_root,
        source_manifest=source_manifest,
        suite=args.suite,
        task_ids=task_ids,
        offline_validation_per_task=args.offline_validation_per_task,
        closed_loop_validation_per_task=args.closed_loop_validation_per_task,
        closed_loop_test_per_task=args.closed_loop_test_per_task,
        seed=args.seed,
        paraphrases=args.paraphrases,
        output_background_split=args.output_background_split,
        output_target_split=args.output_target_split,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        "SmolVLA LIBERO splits ready: "
        f"background_tasks={result['metrics']['background']['num_tasks']}, "
        f"target_tasks={len(result['metrics']['targets'])}"
    )
    print(f"background split: {args.output_background_split}")
    print(f"target split: {args.output_target_split}")


if __name__ == "__main__":
    main()
