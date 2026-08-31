from __future__ import annotations

import pytest

from robot_practice.experiments.rl import policy_decorator_eval


def test_formal_cli_dispatches_to_pinned_policy_adapter(monkeypatch, tmp_path) -> None:
    called = {}

    def fake_formal_evaluation(arguments):
        called["arguments"] = arguments
        return {}

    monkeypatch.setattr(policy_decorator_eval, "run_formal_evaluation", fake_formal_evaluation)

    result = policy_decorator_eval.main(
        [
            "--reference-root",
            str(tmp_path / "reference_repo"),
            "--env-id",
            "PegInsertionSide-v2",
            "--base-policy-ckpt",
            str(tmp_path / "best.pt"),
            "--arm",
            "base",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    assert called["arguments"].dummy is False
    assert called["arguments"].base_policy_ckpt.endswith("best.pt")


def test_output_guard_allows_empty_tee_placeholder(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run.log").touch()

    policy_decorator_eval._prepare_output(output)


def test_output_guard_rejects_previous_run(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        policy_decorator_eval._prepare_output(output)


def test_formal_manifest_uses_disjoint_reset_seed_ranges() -> None:
    arguments = type(
        "Arguments",
        (),
        {
            "manifest": None,
            "episodes_per_seed": 3,
            "env_id": "PegInsertionSide-v2",
            "act_horizon": 4,
        },
    )()

    rows = policy_decorator_eval._formal_manifest(arguments, [1, 2, 3])

    assert [row["reset_seed"] for row in rows] == [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
    assert len({row["reset_seed"] for row in rows}) == len(rows)
    assert [row["policy_seed"] for row in rows] == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_input_manifest_rejects_wrong_action_contract(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"episode_key":"s1-1-0","policy_seed":1,"reset_seed":1001,'
        '"reset_state_id":"state","requested_env_id":"PegInsertionSide-v2",'
        '"action_dim":6,"n_obs_steps":2,"action_horizon":4,'
        '"execution_action_steps":4}\n',
        encoding="utf-8",
    )
    arguments = type(
        "Arguments",
        (),
        {
            "manifest": str(manifest),
            "episodes_per_seed": 1,
            "env_id": "PegInsertionSide-v2",
            "act_horizon": 4,
        },
    )()

    with pytest.raises(ValueError, match="action_dim"):
        policy_decorator_eval._formal_manifest(arguments, [1])


def test_sampling_seed_includes_policy_seed() -> None:
    first = policy_decorator_eval._sampling_seed(1, 1001, 0)
    second = policy_decorator_eval._sampling_seed(2, 1001, 0)

    assert first == policy_decorator_eval._sampling_seed(1, 1001, 0)
    assert first != second
