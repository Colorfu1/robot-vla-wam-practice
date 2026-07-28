"""Small external pretrained PushT evaluator compatibility benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from .config import (
    DATASET_REVISION,
    PRETRAINED_POLICY_REPO_ID,
    PRETRAINED_POLICY_REVISION,
    default_output_root,
    practice_root,
)
from .dataset_split import load_split
from .output import AtomicOutputDirectory, write_metrics, write_records_parquet
from .policy_loading import load_policy_with_processors, replace_scheduler
from .provenance import collect_runtime_provenance, state_dict_hashes, write_json_atomic
from .rerun_visualization import RolloutRecording
from .rollout import run_rollout, save_rollout_video
from .statistics import wilson_interval


def _fingerprint(policy: DiffusionPolicy) -> str:
    encoded = json.dumps(
        state_dict_hashes(policy.state_dict()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def benchmark_pretrained(
    *,
    policy_source: str,
    revision: str | None,
    num_episodes: int,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    split = load_split()
    if not 1 <= num_episodes <= len(split.development_environment_seeds):
        raise ValueError(f"num_episodes must be in [1,{len(split.development_environment_seeds)}]")
    seeds = split.development_environment_seeds[:num_episodes]
    loaded = load_policy_with_processors(
        policy_source,
        revision=revision,
        local_files_only=False,
    )
    if not isinstance(loaded.policy, DiffusionPolicy):
        raise TypeError("The external reference must be a DiffusionPolicy")
    inference_steps = min(100, loaded.config.num_train_timesteps)
    replace_scheduler(
        loaded.policy,
        scheduler_name="DDPM",
        num_inference_steps=inference_steps,
    )
    checkpoint_id = _fingerprint(loaded.policy)
    runtime = collect_runtime_provenance(practice_root())
    code_commit = runtime["repository"]["commit"]
    results = []
    rollout_rows = []
    replan_rows = []
    with AtomicOutputDirectory(output_dir, overwrite=overwrite) as temporary:
        (temporary / "videos").mkdir()
        recorder = RolloutRecording(temporary / "recording.rrd")
        try:
            for environment_seed in seeds:
                result = run_rollout(
                    policy=loaded.policy,
                    preprocessor=loaded.preprocessor,
                    postprocessor=loaded.postprocessor,
                    environment_seed=environment_seed,
                    policy_seed=0,
                    record_frames=True,
                    condition_id="external_pretrained_ddpm",
                )
                recorder.add(result)
                save_rollout_video(
                    result,
                    temporary / "videos" / f"env_{environment_seed:04d}.mp4",
                )
                rollout_rows.append(
                    {
                        "experiment_id": "external_pretrained_reference",
                        "condition_id": result.condition_id,
                        "model": "diffusion",
                        "checkpoint_id": checkpoint_id,
                        "train_seed": None,
                        "environment_seed": environment_seed,
                        "policy_seed": 0,
                        "replan_index": None,
                        "initial_noise_seed": None,
                        "initial_noise_sha256": None,
                        "reverse_noise_seed": None,
                        "reverse_generator_state_sha256": None,
                        "num_inference_steps": inference_steps,
                        "num_action_samples": 1,
                        "num_rollouts": num_episodes,
                        "dataset_revision": DATASET_REVISION,
                        "code_commit": code_commit,
                        "success": result.success,
                        "max_reward": result.max_reward,
                        "final_reward": result.final_reward,
                        "episode_length": result.episode_length,
                        "agent_path": result.agent_path.tolist(),
                        "block_path": result.block_path.tolist(),
                        "action_targets": result.action_targets.tolist(),
                    }
                )
                replan_rows.extend(
                    {
                        "experiment_id": "external_pretrained_reference",
                        "condition_id": result.condition_id,
                        "model": "diffusion",
                        "checkpoint_id": checkpoint_id,
                        "train_seed": None,
                        "environment_seed": environment_seed,
                        "policy_seed": 0,
                        "replan_index": replan.replan_index,
                        "initial_noise_seed": replan.initial_noise_seed,
                        "initial_noise_sha256": replan.initial_noise_sha256,
                        "reverse_noise_seed": replan.reverse_noise_seed,
                        "reverse_generator_state_sha256": (replan.reverse_generator_state_sha256),
                        "num_inference_steps": inference_steps,
                        "num_action_samples": 1,
                        "num_rollouts": num_episodes,
                        "dataset_revision": DATASET_REVISION,
                        "code_commit": code_commit,
                        "chunk_latency_ms": replan.latency_ms,
                    }
                    for replan in result.replans
                )
                results.append(replace(result, frames=()))
        finally:
            recorder.close()
        write_records_parquet(temporary / "rollouts.parquet", rollout_rows)
        write_records_parquet(temporary / "replans.parquet", replan_rows)
        interval = wilson_interval(sum(result.success for result in results), len(results))
        write_metrics(
            temporary,
            {
                "success_rate": interval.estimate,
                "success_ci95_lower": interval.lower,
                "success_ci95_upper": interval.upper,
                "mean_max_reward": sum(result.max_reward for result in results) / len(results),
                "num_rollouts": len(results),
            },
            comments={
                "success_rate": "External compatibility reference; not split-comparable",
                "success_ci95_lower": "Wilson 95% lower bound",
                "success_ci95_upper": "Wilson 95% upper bound",
                "mean_max_reward": "Mean best coverage reward reached per rollout",
                "num_rollouts": "Independent PushT environment executions",
            },
        )
        write_json_atomic(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "label": "external pretrained reference — not split-comparable",
                "experiment": "external_pretrained_reference",
                "policy_source": policy_source,
                "revision": revision,
                "checkpoint_id": checkpoint_id,
                "scheduler": "DDPM",
                "num_inference_steps": inference_steps,
                "num_action_samples": 1,
                "num_rollouts": len(results),
                "environment_seeds": list(seeds),
                "runtime": runtime,
            },
        )
    return output_dir.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=PRETRAINED_POLICY_REPO_ID)
    parser.add_argument("--revision", default=PRETRAINED_POLICY_REVISION)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "benchmark_pretrained",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = benchmark_pretrained(
        policy_source=args.policy,
        revision=args.revision,
        num_episodes=args.n_episodes,
        output_dir=args.output_dir.expanduser().resolve(),
        overwrite=args.overwrite,
    )
    print(f"External pretrained benchmark: {result}")


if __name__ == "__main__":
    main()
