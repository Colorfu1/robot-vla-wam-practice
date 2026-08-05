from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import robot_practice.experiments.smolvla_libero.target_protocol_evaluation as target_eval
from robot_practice.evaluation import libero_act, libero_smolvla
from robot_practice.evaluation.libero_assets import REQUIRED_ASSET_DIRECTORIES
from robot_practice.evaluation.libero_goal_classifier import (
    GOAL_CLASSIFIER_NAME,
    GOAL_PREDICATES,
    classify_bowl_goal,
    disable_reference_goal_termination,
)

PRACTICE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_split.json"
PARAPHRASES = PRACTICE_ROOT / "configs/experiments/smolvla_libero/paraphrases.json"
PROTOCOL_CONFIG = PRACTICE_ROOT / "configs/experiments/smolvla_libero/target_eval_protocol.json"


def _assets(root: Path) -> Path:
    for directory in REQUIRED_ASSET_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "scenes/libero_tabletop_base_style.xml").write_text("<mujoco/>", encoding="utf-8")
    return root


def _policy(root: Path, *, adapter: bool = False) -> Path:
    root.mkdir(parents=True)
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (root / name).write_text("{}", encoding="utf-8")
    if adapter:
        (root / "adapter_config.json").write_text("{}", encoding="utf-8")
        (root / "adapter_model.safetensors").write_bytes(b"adapter")
    else:
        (root / "model.safetensors").write_bytes(b"model")
    return root.resolve()


def _arguments(tmp_path: Path) -> target_eval.TargetEvaluationArguments:
    return target_eval.TargetEvaluationArguments(
        policy_path=_policy(tmp_path / "policy"),
        model_label="b0_domain_adapted_zero_shot",
        split_file=SPLIT_FILE,
        paraphrases=PARAPHRASES,
        protocol_config=PROTOCOL_CONFIG,
        protocols=target_eval.PROTOCOLS,
        num_gpus=8,
        env_seed=1000,
        policy_seed=2000,
        n_action_steps=10,
        num_visualized_episodes_per_job=0,
        max_steps=None,
        libero_assets_dir=_assets(tmp_path / "assets"),
        output_dir=(tmp_path / "result").resolve(),
        resume=False,
        overwrite=False,
        print_command=True,
    )


def test_build_jobs_freezes_21_jobs_and_210_paired_rollouts(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    spec = target_eval.load_evaluation_spec(arguments)
    jobs = target_eval.build_jobs(arguments, spec)

    assert len(jobs) == 21
    assert sum(job.rollout_count for job in jobs) == 210
    assert sum(job.rollout_count for job in jobs if job.protocol == "canonical") == 90
    assert sum(job.rollout_count for job in jobs if job.protocol == "held_out_paraphrase") == 90
    assert sum(job.rollout_count for job in jobs if job.protocol == "prompt_swap") == 30

    canonical = {
        (job.task_id, state): seed
        for job in jobs
        if job.protocol == "canonical"
        for state, seed in zip(job.init_state_ids, job.episode_seeds, strict=True)
    }
    paraphrase = {
        (job.task_id, state): seed
        for job in jobs
        if job.protocol == "held_out_paraphrase"
        for state, seed in zip(job.init_state_ids, job.episode_seeds, strict=True)
    }
    assert canonical == paraphrase

    prompt_swap = [job for job in jobs if job.protocol == "prompt_swap"]
    assert {job.expected_goal for job in prompt_swap} == {"stove", "cabinet", "plate"}
    assert len({job.init_state_ids for job in prompt_swap}) == 1
    assert len({job.episode_seeds for job in prompt_swap}) == 1


def test_prompt_swap_command_disables_reference_scoring_via_classifier(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)
    spec = target_eval.load_evaluation_spec(arguments)
    job = next(
        job
        for job in target_eval.build_jobs(arguments, spec)
        if job.protocol == "prompt_swap" and job.expected_goal == "cabinet"
    )

    command = target_eval.build_evaluation_command(job, arguments)

    assert f"--goal-classifier={GOAL_CLASSIFIER_NAME}" in command
    assert "--expected-goal=cabinet" in command
    assert "--instruction=put the bowl on top of the cabinet" in command
    assert any(value.startswith("--episode-seeds=") for value in command)
    assert "--no-open-viewer" in command


def test_print_command_reports_complete_workload_without_creating_output(
    tmp_path: Path,
) -> None:
    arguments = _arguments(tmp_path)

    payload = target_eval.evaluate_target_protocols(arguments)

    assert payload["num_jobs"] == 21
    assert payload["total_rollouts"] == 210
    assert len(payload["commands"]) == 21
    assert not arguments.output_dir.exists()


def test_summary_reports_paired_gap_and_prompt_confusion(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    records = []
    for task_id in target_eval.EXPECTED_TASK_IDS:
        for state in range(30):
            records.append(
                {
                    "protocol": "canonical",
                    "task_id": task_id,
                    "init_state_id": state,
                    "success": state < 24,
                }
            )
            records.append(
                {
                    "protocol": "held_out_paraphrase",
                    "task_id": task_id,
                    "init_state_id": state,
                    "success": state < 21,
                }
            )
    for goal in ("stove", "cabinet", "plate"):
        for index in range(10):
            records.append(
                {
                    "protocol": "prompt_swap",
                    "task_id": 1,
                    "init_state_id": index,
                    "expected_goal": goal,
                    "predicted_goal": goal if index < 8 else "failed",
                    "success": index < 8,
                }
            )

    result = target_eval.summarize_results(arguments, records)

    assert result["canonical"]["macro_success_rate"] == pytest.approx(0.8)
    assert result["held_out_paraphrase"]["macro_success_rate"] == pytest.approx(0.7)
    assert result["para_gap"]["point_estimate"] == pytest.approx(0.1)
    assert result["para_gap"]["num_paired_cases"] == 90
    assert result["prompt_swap"]["prompt_compliance"] == pytest.approx(0.8)
    assert result["prompt_swap"]["confusion_matrix"]["stove"] == {
        "stove": 8,
        "cabinet": 0,
        "plate": 0,
        "failed": 2,
    }


class _FakeRawEnvironment:
    def __init__(self, matches: dict[tuple[str, str, str], bool]) -> None:
        self.matches = matches
        self.object_states_dict = {
            value: object() for predicate in GOAL_PREDICATES.values() for value in predicate[1:]
        }

    def check_success(self) -> bool:
        return True

    def _eval_predicate(self, predicate: list[str]) -> bool:
        return self.matches.get(tuple(predicate), False)


def test_three_goal_classifier_is_independent_of_reference_success() -> None:
    raw = _FakeRawEnvironment({GOAL_PREDICATES["plate"]: True})
    env = SimpleNamespace(num_envs=1, envs=[SimpleNamespace(_env=raw)])

    disable_reference_goal_termination(env)
    result = classify_bowl_goal(env)

    assert raw.check_success() is False
    assert result["predicted_goal"] == "plate"
    assert result["failure_reason"] is None


def test_policy_validation_accepts_full_and_lora_checkpoints(tmp_path: Path) -> None:
    assert target_eval._validate_policy(_policy(tmp_path / "full")) == (tmp_path / "full").resolve()
    assert (
        target_eval._validate_policy(_policy(tmp_path / "lora", adapter=True))
        == (tmp_path / "lora").resolve()
    )


def test_protocol_config_hashes_match_tracked_inputs() -> None:
    protocol = json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))

    assert protocol["target_split_sha256"] == target_eval.sha256_file(SPLIT_FILE)
    assert protocol["paraphrases_sha256"] == target_eval.sha256_file(PARAPHRASES)


def test_smolvla_cli_accepts_paired_seeds_and_prompt_swap_scoring() -> None:
    parsed = libero_smolvla.build_argument_parser().parse_args(
        [
            "--policy-path=checkpoint",
            "--suite=libero_goal",
            "--task-id=1",
            "--n-episodes=2",
            "--init-state-ids=4,9",
            "--episode-seeds=1003,1011",
            "--instruction=put the bowl on the plate",
            f"--goal-classifier={GOAL_CLASSIFIER_NAME}",
            "--expected-goal=plate",
        ]
    )

    assert parsed.episode_seeds == (1003, 1011)
    assert parsed.instruction == "put the bowl on the plate"
    assert parsed.expected_goal == "plate"


def test_goal_classifier_contract_requires_instruction_and_expected_goal() -> None:
    task = SimpleNamespace(episode_length=300, suite="libero_goal")
    base = {
        "policy_path": Path("checkpoint"),
        "suite": "libero_goal",
        "task_id": 1,
        "n_episodes": 1,
        "init_state_ids": (1,),
        "episode_seeds": (1000,),
        "goal_classifier": GOAL_CLASSIFIER_NAME,
    }
    with pytest.raises(ValueError, match="expected-goal"):
        libero_act.validate_config(libero_act.LiberoEvaluationConfig(**base), task=task)

    base["expected_goal"] = "stove"
    with pytest.raises(ValueError, match="instruction"):
        libero_act.validate_config(libero_act.LiberoEvaluationConfig(**base), task=task)

    base["instruction"] = "put the bowl on the stove"
    libero_act.validate_config(libero_act.LiberoEvaluationConfig(**base), task=task)
