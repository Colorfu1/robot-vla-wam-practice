"""Closed-loop SmolVLA evaluation with synchronized LIBERO Rerun recording."""

from __future__ import annotations

import argparse
import copy
import logging
import re
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.random_utils import set_seed
from typing_extensions import Self

from robot_practice.evaluation.common import (
    CheckpointManifest,
    local_timestamp,
    read_json,
    synchronize_device,
    timed_call,
    validate_processor_state_files,
)
from robot_practice.evaluation.libero_act import (
    PERTURBATION_MODES,
    LiberoEvaluationConfig,
    LiberoEvaluationResult,
    LiberoPolicyRuntime,
    LiberoPolicyStep,
    _parse_init_state_ids,
    _postprocess_policy_action_tensor,
    evaluate_libero,
    resolve_n_action_steps,
)
from robot_practice.evaluation.libero_checkpoint import CheckpointTaskAssessment
from robot_practice.evaluation.libero_tasks import SUPPORTED_SUITES, LiberoTaskIdentity
from robot_practice.experiments.smolvla_libero.b0_training import (
    LOCAL_VLM_ASSETS_REFERENCE,
    OFFLINE_VLM_ASSETS_DIRNAME,
    REQUIRED_OFFLINE_VLM_ASSET_FILES,
    portable_path,
    resolve_repository_path,
)

LOGGER = logging.getLogger(__name__)
_SAFE_POLICY_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SmolVLAEvaluationConfig:
    policy_path: str
    suite: str
    task_id: int
    n_episodes: int = 1
    num_visualized_episodes: int = 1
    n_action_steps: int | None = None
    perturbation: str = "none"
    proprio_noise_std: float = 0.01
    seed: int = 1000
    policy_seed: int | None = None
    init_state_ids: tuple[int, ...] | None = None
    libero_assets_dir: Path | None = None
    output_dir: Path | None = None
    overwrite: bool = False
    max_steps: int | None = None
    open_viewer: bool = True


@dataclass(frozen=True)
class LoadedSmolVLA:
    manifest: CheckpointManifest
    config: SmolVLAConfig
    policy: torch.nn.Module
    preprocessor: Any
    postprocessor: Any
    checkpoint_n_action_steps: int


def _autocast_context(device: str, use_amp: bool) -> Any:
    resolved = torch.device(device)
    if use_amp:
        return torch.autocast(device_type=resolved.type)
    return nullcontext()


def _resolved_policy_reference(policy_path: str) -> tuple[str, Path | None]:
    if not policy_path.strip():
        raise ValueError("--policy-path must not be empty")
    candidate = resolve_repository_path(policy_path)
    if candidate.is_dir():
        return portable_path(candidate), candidate
    if candidate.exists():
        raise ValueError(f"--policy-path must be a policy directory or Hub repo ID: {candidate}")
    return policy_path, None


def _safe_policy_name(policy_reference: str, local_path: Path | None) -> str:
    source_name = local_path.name if local_path is not None else policy_reference.rsplit("/", 1)[-1]
    safe = _SAFE_POLICY_COMPONENT.sub("_", source_name).strip("._-")
    return safe or "policy"


def _default_output_dir(
    *,
    policy_reference: str,
    local_path: Path | None,
    suite: str,
    task_id: int,
) -> Path | None:
    if (
        local_path is not None
        and local_path.name == "pretrained_model"
        and local_path.parent.parent.name == "checkpoints"
    ):
        return None
    return (
        Path("outputs")
        / "smolvla_libero"
        / "eval"
        / _safe_policy_name(policy_reference, local_path)
        / suite
        / f"task_{task_id:03d}"
        / local_timestamp()
    )


def _portable_recorded_path(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return value
    return portable_path(resolve_repository_path(value))


def _portable_local_train_config(
    payload: dict[str, Any] | None,
    *,
    policy_reference: str,
    vlm_assets_reference: str,
) -> dict[str, Any] | None:
    if payload is None:
        return None
    result = copy.deepcopy(payload)
    if "output_dir" in result:
        result["output_dir"] = _portable_recorded_path(result["output_dir"])
    dataset = result.get("dataset")
    if isinstance(dataset, dict) and "root" in dataset:
        dataset["root"] = _portable_recorded_path(dataset["root"])
    policy = result.get("policy")
    if isinstance(policy, dict):
        policy["pretrained_path"] = policy_reference
        policy["vlm_model_name"] = vlm_assets_reference
    return result


def _local_manifest(
    path: Path,
    *,
    policy_reference: str | None = None,
    vlm_assets_reference: str | None = None,
) -> CheckpointManifest:
    required = (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete SmolVLA checkpoint. Missing: {', '.join(missing)}")
    policy_config = read_json(path / "config.json")
    if policy_config.get("type") != "smolvla":
        raise ValueError(f"Expected a SmolVLA checkpoint, got {policy_config.get('type')!r}")
    if (path / "adapter_config.json").is_file():
        if not (path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"PEFT checkpoint is missing adapter_model.safetensors: {path}")
    elif not (path / "model.safetensors").is_file():
        raise FileNotFoundError(
            "SmolVLA checkpoint must contain model.safetensors or a complete "
            "adapter_config.json/adapter_model.safetensors pair"
        )
    train_path = path / "train_config.json"
    train_config = read_json(train_path) if train_path.is_file() else None
    manifest_path = path
    if policy_reference is not None:
        manifest_path = Path(policy_reference)
        policy_config = copy.deepcopy(policy_config)
        policy_config["pretrained_path"] = policy_reference
    if vlm_assets_reference is not None:
        policy_config["vlm_model_name"] = vlm_assets_reference
    if policy_reference is not None and vlm_assets_reference is not None:
        train_config = _portable_local_train_config(
            train_config,
            policy_reference=policy_reference,
            vlm_assets_reference=vlm_assets_reference,
        )
    return CheckpointManifest(
        path=manifest_path,
        policy_config=policy_config,
        train_config=train_config,
        processor_state_files=validate_processor_state_files(path),
    )


def _complete_vlm_assets(path: Path) -> bool:
    return path.is_dir() and all(
        (path / filename).is_file()
        for filename in REQUIRED_OFFLINE_VLM_ASSET_FILES
    )


def _resolve_local_vlm_assets(
    checkpoint_path: Path,
    saved_reference: str,
) -> tuple[Path, str]:
    candidates = [
        checkpoint_path / OFFLINE_VLM_ASSETS_DIRNAME,
        resolve_repository_path(saved_reference),
        resolve_repository_path(LOCAL_VLM_ASSETS_REFERENCE),
    ]
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if _complete_vlm_assets(candidate):
            return candidate, portable_path(candidate)
    formatted = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "SmolVLA inference requires complete local VLM tokenizer/config assets. "
        f"Checked: {formatted}"
    )


def _checkpoint_rename_map(path: Path | None) -> dict[str, str] | None:
    """Recover the feature rename map saved with a local LeRobot processor."""

    if path is None:
        return None
    payload = read_json(path / "policy_preprocessor.json")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise TypeError(f"Invalid policy_preprocessor.json steps: {path}")
    for step in steps:
        if not isinstance(step, dict):
            raise TypeError(f"Invalid policy_preprocessor.json step: {path}")
        if step.get("registry_name") != "rename_observations_processor":
            continue
        step_config = step.get("config")
        rename_map = step_config.get("rename_map") if isinstance(step_config, dict) else None
        if not isinstance(rename_map, dict) or not all(
            isinstance(source, str) and isinstance(target, str)
            for source, target in rename_map.items()
        ):
            raise ValueError(f"Invalid saved observation rename map: {path}")
        return dict(rename_map) or None
    return None


def _hub_manifest(
    policy_reference: str,
    config: SmolVLAConfig,
) -> CheckpointManifest:
    return CheckpointManifest(
        path=Path(policy_reference),
        policy_config=asdict(config),
        train_config=None,
        processor_state_files=(),
    )


def _validate_loaded_policy(
    policy: torch.nn.Module,
    config: SmolVLAConfig,
) -> None:
    from peft import PeftModel

    if isinstance(policy, SmolVLAPolicy):
        return
    if isinstance(policy, PeftModel):
        base_policy = policy.get_base_model()
        if isinstance(base_policy, SmolVLAPolicy):
            return
        raise TypeError(
            f"PEFT policy base model must be SmolVLAPolicy, got {type(base_policy).__name__}"
        )
    raise TypeError(
        "Loaded policy must be SmolVLAPolicy or a PEFT wrapper around it, got "
        f"{type(policy).__name__}"
    )


def _load_smolvla(
    *,
    policy_reference: str,
    local_path: Path | None,
    env_config: LiberoEnv,
    n_action_steps_override: int | None,
) -> LoadedSmolVLA:
    config = PreTrainedConfig.from_pretrained(
        local_path if local_path is not None else policy_reference,
        local_files_only=local_path is not None,
    )
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    checkpoint_n_action_steps = config.n_action_steps
    config.n_action_steps = resolve_n_action_steps(
        checkpoint_n_action_steps=checkpoint_n_action_steps,
        override=n_action_steps_override,
        chunk_size=config.chunk_size,
        temporal_ensemble_coeff=None,
    )
    action_feature = config.action_feature
    if action_feature is None or tuple(action_feature.shape) != (7,):
        shape = None if action_feature is None else tuple(action_feature.shape)
        raise ValueError(f"SmolVLA LIBERO action shape must be (7,), got {shape}")

    manifest = None
    processor_overrides = {
        "device_processor": {"device": str(config.device)},
    }
    pretrained_reference = local_path if local_path is not None else Path(policy_reference)
    if local_path is not None:
        vlm_assets_path, vlm_assets_reference = _resolve_local_vlm_assets(
            local_path,
            config.vlm_model_name,
        )
        LOGGER.info(
            "[Policy] rebased local VLM assets to %s",
            vlm_assets_reference,
        )
        # Runtime consumers receive absolute paths; persisted evaluation
        # metadata remains repository-relative through the manifest below.
        config.vlm_model_name = str(vlm_assets_path)
        processor_overrides["tokenizer_processor"] = {
            "tokenizer_name": str(vlm_assets_path),
        }
        manifest = _local_manifest(
            local_path,
            policy_reference=policy_reference,
            vlm_assets_reference=vlm_assets_reference,
        )
    config.pretrained_path = pretrained_reference
    policy = make_policy(
        cfg=config,
        env_cfg=env_config,
        rename_map=_checkpoint_rename_map(local_path),
    )
    _validate_loaded_policy(policy, config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained_reference),
        preprocessor_overrides=processor_overrides,
    )
    if manifest is None:
        manifest = _hub_manifest(policy_reference, config)
    return LoadedSmolVLA(
        manifest=manifest,
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        checkpoint_n_action_steps=checkpoint_n_action_steps,
    )


class SmolVLALiberoPolicyDriver:
    """Execute and visualize the same stochastic SmolVLA action chunk."""

    def __init__(self, loaded: LoadedSmolVLA, env_postprocessor: Any) -> None:
        self.loaded = loaded
        self.env_postprocessor = env_postprocessor
        self._active_chunk: np.ndarray | None = None
        self._next_index = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def reset_episode(self) -> None:
        self.loaded.policy.reset()
        self._active_chunk = None
        self._next_index = 0

    def select_step(self, policy_input: dict[str, Any]) -> LiberoPolicyStep:
        synchronize_device(self.loaded.config.device)
        policy_started = time.perf_counter_ns()
        queried_chunk: np.ndarray | None = None
        model_latency_ms: float | None = None

        if self._active_chunk is None:
            with (
                torch.inference_mode(),
                _autocast_context(
                    str(self.loaded.config.device),
                    self.loaded.config.use_amp,
                ),
            ):
                chunk_policy, model_latency_ms = timed_call(
                    self.loaded.policy.predict_action_chunk,
                    policy_input,
                    device=self.loaded.config.device,
                )
            chunk_controller = _postprocess_policy_action_tensor(
                chunk_policy,
                checkpoint_postprocessor=self.loaded.postprocessor,
                env_postprocessor=self.env_postprocessor,
            )
            chunk_numpy = chunk_controller.detach().cpu().float().numpy()
            if (
                chunk_numpy.ndim != 3
                or chunk_numpy.shape[0] != 1
                or chunk_numpy.shape[2] != 7
                or chunk_numpy.shape[1] < self.loaded.config.n_action_steps
                or not np.isfinite(chunk_numpy).all()
            ):
                raise ValueError(
                    "Postprocessed SmolVLA chunk must have finite shape "
                    "[1,horizon,7] with horizon >= n_action_steps, got "
                    f"{chunk_numpy.shape}"
                )
            self._active_chunk = chunk_numpy[0].copy()
            self._next_index = 0
            queried_chunk = self._active_chunk.copy()

        if self._active_chunk is None:
            raise RuntimeError("SmolVLA execution queue has no active chunk")
        index = self._next_index
        n_action_steps = self.loaded.config.n_action_steps
        executed_action = self._active_chunk[index].copy()
        queued_actions = self._active_chunk[index + 1 : n_action_steps].copy()
        discarded_actions = self._active_chunk[n_action_steps:].copy()
        self._next_index += 1
        if self._next_index == n_action_steps:
            self._active_chunk = None
            self._next_index = 0

        synchronize_device(self.loaded.config.device)
        policy_latency_ms = (time.perf_counter_ns() - policy_started) / 1_000_000
        return LiberoPolicyStep(
            executed_action=executed_action,
            queried_chunk=queried_chunk,
            queued_actions=queued_actions,
            discarded_actions=discarded_actions,
            new_prediction_chunk=None,
            model_queried=queried_chunk is not None,
            plan_step_index=index,
            policy_latency_ms=policy_latency_ms,
            model_query_latency_ms=model_latency_ms,
        )


def evaluate_smolvla_libero(
    config: SmolVLAEvaluationConfig,
) -> LiberoEvaluationResult:
    """Evaluate a Hub or local SmolVLA policy through the shared LIBERO engine."""

    policy_reference, local_path = _resolved_policy_reference(config.policy_path)
    if config.overwrite and config.output_dir is None:
        raise ValueError("--overwrite requires an explicit --output-dir")
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = _default_output_dir(
            policy_reference=policy_reference,
            local_path=local_path,
            suite=config.suite,
            task_id=config.task_id,
        )
    base_config = LiberoEvaluationConfig(
        policy_path=Path(policy_reference),
        suite=config.suite,
        task_id=config.task_id,
        n_episodes=config.n_episodes,
        num_visualized_episodes=config.num_visualized_episodes,
        n_action_steps=config.n_action_steps,
        temporal_ensemble_coeff=None,
        perturbation=config.perturbation,
        proprio_noise_std=config.proprio_noise_std,
        seed=config.seed,
        policy_seed=config.policy_seed,
        init_state_ids=config.init_state_ids,
        libero_assets_dir=config.libero_assets_dir,
        output_dir=output_dir,
        overwrite=config.overwrite,
        max_steps=config.max_steps,
        open_viewer=config.open_viewer,
    )

    def load_runtime(
        env_config: LiberoEnv,
        task: LiberoTaskIdentity,
    ) -> LiberoPolicyRuntime:
        loaded = _load_smolvla(
            policy_reference=policy_reference,
            local_path=local_path,
            env_config=env_config,
            n_action_steps_override=config.n_action_steps,
        )
        assessment = CheckpointTaskAssessment(
            status="unknown",
            message=(
                "SmolVLA is language-conditioned; the selected LIBERO task instruction "
                "is supplied at runtime rather than inferred from one training task."
            ),
            evidence={
                "source_kind": "local" if local_path is not None else "huggingface_hub",
                "language": task.language,
            },
        )
        LOGGER.info("[Policy] %s", assessment.message)
        policy_execution = {
            "family": "smolvla",
            "chunk_size": loaded.config.chunk_size,
            "checkpoint_n_action_steps": loaded.checkpoint_n_action_steps,
            "effective_n_action_steps": loaded.config.n_action_steps,
            "n_action_steps_overridden": config.n_action_steps is not None,
            "flow_matching_steps": loaded.config.num_steps,
            "policy_seed": (
                config.seed if config.policy_seed is None else config.policy_seed
            ),
            "single_query_drives_execution_and_visualization": True,
        }
        return LiberoPolicyRuntime(
            loaded=loaded,
            checkpoint_task_assessment=assessment,
            policy_execution=policy_execution,
            application_id="robot_practice_smolvla_libero",
            make_driver=lambda env_postprocessor: SmolVLALiberoPolicyDriver(
                loaded,
                env_postprocessor,
            ),
            reset_policy_rng=set_seed,
        )

    set_seed(config.seed)
    return evaluate_libero(
        base_config,
        policy_runtime_loader=load_runtime,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Hugging Face policy repo ID or local pretrained_model directory",
    )
    parser.add_argument(
        "--suite",
        required=True,
        help=f"LIBERO suite ({', '.join(SUPPORTED_SUITES)})",
    )
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--num-visualized-episodes", type=int, default=1)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        help="Evaluation execution horizon; defaults to checkpoint n_action_steps",
    )
    parser.add_argument(
        "--perturbation",
        choices=PERTURBATION_MODES,
        default="none",
    )
    parser.add_argument("--proprio-noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--policy-seed",
        type=int,
        help="Reset all policy RNGs to this seed before every rollout (default: --seed)",
    )
    parser.add_argument(
        "--init-state-ids",
        type=_parse_init_state_ids,
        help=(
            "Comma-separated official LIBERO init-state IDs; the number of IDs must "
            "equal --n-episodes"
        ),
    )
    parser.add_argument(
        "--libero-assets-dir",
        type=Path,
        help="Validated offline LIBERO assets tree; prevents Hugging Face fallback",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--no-open-viewer",
        action="store_false",
        dest="open_viewer",
    )
    parser.set_defaults(open_viewer=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_argument_parser().parse_args(argv)
    evaluate_smolvla_libero(SmolVLAEvaluationConfig(**vars(args)))


if __name__ == "__main__":
    main()
