from __future__ import annotations

import json

from robot_practice.experiments.rl import policy_decorator_official_base_eval as evaluator


def test_official_base_eval_contract_defaults_to_five_envs() -> None:
    arguments = evaluator.build_parser().parse_args(
        [
            "--reference-root",
            "reference",
            "--base-policy-ckpt",
            "best.pt",
            "--seed",
            "1",
            "--output-dir",
            "output",
        ]
    )

    assert arguments.episodes == 50
    assert arguments.num_eval_envs == 5
    assert arguments.ddim_steps == 4
    assert arguments.act_horizon == 4


def test_load_online_args_preserves_frozen_model_contract(tmp_path) -> None:
    checkpoint = tmp_path / "model" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint.parent.parent / "args.json").write_text(
        json.dumps(
            {
                "env_id": "PegInsertionSide-v2",
                "obs_horizon": 2,
                "pred_horizon": 16,
                "diffusion_step_embed_dim": 64,
                "unet_dims": [64, 128, 256],
                "n_groups": 8,
                "control_mode": "pd_ee_delta_pose",
            }
        ),
        encoding="utf-8",
    )

    args = evaluator._load_online_args(
        checkpoint,
        env_id="PegInsertionSide-v2",
        seed=7,
        episodes=50,
        num_eval_envs=5,
        ddim_steps=4,
        act_horizon=4,
    )

    assert args.seed == 7
    assert args.num_envs == 16
    assert args.num_eval_envs == 5
    assert args.num_eval_episodes == 50
    assert args.res_scale == 0.0
    assert args.control_mode == "pd_ee_delta_pose"
