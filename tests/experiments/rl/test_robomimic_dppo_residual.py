import json
import os
import subprocess
from pathlib import Path

import torch
import yaml

from robot_practice.experiments.rl.robomimic_dppo_residual import (
    FrozenDPPOBase,
    _check_path_groups,
    _merge_official_base_eval,
    assert_paired_manifests,
    build_episode_manifest,
    compose_bounded_residual,
    create_optimizers,
    load_episode_manifest,
    main,
    pixel_state_features,
    resolve_reward_shaping,
    summarize,
    unnormalize_action,
    write_episode_manifest,
)


def test_zero_residual_and_bound():
    base = torch.tensor([[[-0.8, 0.1, 0.0]]])
    residual = torch.tensor([[[10.0, -10.0, 0.2]]])
    bound = torch.tensor([[[0.05, 0.05, 0.1]]])
    out = compose_bounded_residual(base, residual, bound)
    assert torch.allclose(out, torch.tensor([[[-0.75, 0.05, 0.1]]]))
    assert torch.equal(compose_bounded_residual(base, torch.zeros_like(base), bound), base)


def test_manifest_is_disjoint_and_pairing_is_strict(tmp_path: Path):
    manifest = build_episode_manifest([1, 2, 3], 50)
    assert len(manifest) == 150
    assert len({row["reset_seed"] for row in manifest}) == 150
    path = tmp_path / "manifest.jsonl"
    write_episode_manifest(path, manifest)
    assert len(load_episode_manifest(path)) == 150
    duplicate = tmp_path / "duplicate.jsonl"
    write_episode_manifest(duplicate, [manifest[0], manifest[0]])
    try:
        load_episode_manifest(duplicate)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate manifest was accepted")
    shorter = tmp_path / "shorter.jsonl"
    write_episode_manifest(shorter, manifest[:-1])
    try:
        assert_paired_manifests(path, shorter)
    except ValueError as exc:
        assert "paired" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched manifest was accepted")
    altered = tmp_path / "altered.jsonl"
    altered_rows = [dict(row) for row in manifest]
    altered_rows[0]["reset_seed"] += 10000
    write_episode_manifest(altered, altered_rows)
    try:
        assert_paired_manifests(path, altered)
    except ValueError as exc:
        assert "paired" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("altered reset seed was accepted")


def test_normalized_action_mapping_and_explicit_normalized_bounds():
    cfg = yaml.safe_load(Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text())
    assert cfg["train"]["residual_bound"] == [0.1] * 7
    assert cfg["ft_denoising_steps"] == 0
    assert "residual_scale" not in cfg["train"]
    out = unnormalize_action(torch.tensor([[-1.0, 0.0, 1.0]]), torch.tensor([[-2.0] * 3]), torch.tensor([[2.0] * 3]))
    assert torch.allclose(out, torch.tensor([[-2.0, 0.0, 2.0]]))


def test_reward_mode_and_split_optimizers():
    cfg = yaml.safe_load(Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text())
    assert cfg["env"]["reward_shaping"] is False
    assert resolve_reward_shaping(cfg, anchor=True) is False
    cfg["env"]["reward_shaping"] = True
    assert resolve_reward_shaping(cfg, anchor=True) is False
    assert resolve_reward_shaping(cfg) is True
    actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    actor_opt, critic_opt = create_optimizers(actor, critic, cfg)
    assert actor_opt is not critic_opt
    assert actor_opt.param_groups[0]["lr"] == cfg["train"]["actor_lr"]
    assert critic_opt.param_groups[0]["lr"] == cfg["train"]["critic_lr"]


def test_path_groups_reject_unused_pretrain_assets():
    cfg = yaml.safe_load(Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text())
    cfg["paths"]["pretrain"] = {"train_dataset": "unused.npz"}
    try:
        _check_path_groups(cfg)
    except ValueError as exc:
        assert "must not require pretrain data" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unused pretrain assets were accepted")


def test_residual_actor_mean_receives_policy_gradient():
    from robot_practice.experiments.rl.robomimic_dppo_residual import ResidualActor

    actor = ResidualActor(4, 2, 3, [0.1, 0.1, 0.1])
    features = torch.randn(8, 4)
    with torch.no_grad():
        _, old_log_prob, latent = actor.sample(features)
    new_log_prob, _ = actor.log_prob(features, latent)
    ratio = (new_log_prob - old_log_prob).exp()
    loss = -(ratio * torch.linspace(-1, 1, 8)).mean()
    loss.backward()
    assert actor.net[-1].weight.grad is not None
    assert actor.net[-1].weight.grad.abs().sum() > 0


def test_pixel_features_accept_dppo_time_axis():
    obs = {
        "rgb": torch.zeros(2, 1, 3, 96, 96),
        "state": torch.zeros(2, 1, 9),
    }
    features = pixel_state_features(obs, image_feature_hw=12)
    assert features.shape == (2, 9 + 3 * 12 * 12)


def test_residual_eval_without_base_manifest_raises(monkeypatch, tmp_path: Path):
    """F1 pairing must be enforced, not silently degraded.

    A missing base episode_manifest.jsonl must raise before the env is touched.
    Falling back to a freshly generated manifest would let base and residual
    evaluate different opening states without any signal.
    """

    import robot_practice.experiments.rl.robomimic_dppo_residual as module

    monkeypatch.setattr(module, "_load_real_dppo_base", lambda cfg, dppo_root, device: torch.nn.Linear(1, 1))

    class _UnreachableEnv:
        def reset(self):  # pragma: no cover
            raise AssertionError("manifest check must fire before env.reset()")

    monkeypatch.setattr(module, "_make_real_env", lambda *args, **kwargs: _UnreachableEnv())

    cfg = {
        "dppo_root": ".", "env": {"n_envs": 1}, "observation": {"state_dim": 9},
        "horizon_steps": 4, "action_dim": 7,
        "train": {
            "seeds": [1], "actor_lr": 1e-5, "critic_lr": 1e-3, "n_train_itr": 1,
            "image_feature_hw": 12, "residual_bound": [0.1] * 7, "initial_log_std": -1.0,
        },
        "evaluation": {"episodes": 50, "random_seed_base": 1001},
    }
    output = tmp_path / "residual_eval"
    output.mkdir(parents=True)
    try:
        module._run_real_residual(cfg, output, "eval", "zero", None, False, 1)
    except FileNotFoundError as exc:
        assert "episode_manifest.jsonl" in str(exc)
        return
    raise AssertionError("expected FileNotFoundError when base manifest is missing")


def test_real_eval_uses_paired_rng_seed_and_complete_progress(monkeypatch, tmp_path: Path):
    import robot_practice.experiments.rl.robomimic_dppo_residual as module

    class _ConstantPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, *, cond, deterministic=True):
            del deterministic
            batch = cond["state"].shape[0]
            return torch.zeros(batch, 4, 7, device=cond["state"].device)

    class _TinyVectorEnv:
        def __init__(self):
            self.seed_value = None
            self.steps = 0

        def seed(self, value):
            self.seed_value = value

        def _obs(self):
            return {
                "rgb": torch.zeros(1, 1, 3, 96, 96).numpy(),
                "state": torch.zeros(1, 1, 9).numpy(),
            }

        def reset(self):
            return self._obs()

        def step(self, action):
            del action
            self.steps += 1
            return (
                self._obs(),
                torch.zeros(1).numpy(),
                torch.zeros(1, dtype=torch.bool).numpy(),
                torch.tensor([self.steps == 2]).numpy(),
                {},
            )

        def close(self):
            pass

    env = _TinyVectorEnv()
    monkeypatch.setattr(module, "_load_real_dppo_base", lambda *args: _ConstantPolicy())
    monkeypatch.setattr(module, "_make_real_env", lambda *args, **kwargs: env)
    cfg = yaml.safe_load(
        Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text()
    )
    cfg["dppo_root"] = "."
    cfg["env"]["n_envs"] = 1
    cfg["env"]["max_episode_steps"] = 8
    cfg["train"]["seeds"] = [1]
    manifest = build_episode_manifest([1], 1, 1001)
    manifest_path = tmp_path / "base_manifest.jsonl"
    write_episode_manifest(manifest_path, manifest)

    output = tmp_path / "eval"
    module._run_real_residual(
        cfg, output, "eval", "zero", None, False, 1, manifest_path=manifest_path
    )

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["policy_rng_seed"] == 1002
    assert env.seed_value == [1002]
    assert "DONE completed=2 total=2" in (output / "run.log").read_text()


def test_frozen_base_hash_does_not_change():
    base = FrozenDPPOBase(torch.nn.Linear(4, 3))
    before = base.parameter_hash
    x = {"state": torch.zeros(1, 1, 4)}
    _ = base(x)
    base.assert_unchanged()
    assert base.parameter_hash == before


def test_pixel_whitelist_rejects_object_observation():
    obs = {"rgb": torch.zeros(1, 3, 96, 96), "state": torch.zeros(1, 9), "object": torch.zeros(1, 1)}
    try:
        pixel_state_features(obs)
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("object observation was accepted")


def test_dummy_train_pipeline(tmp_path: Path):
    config = Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml")
    assert main(["--phase=train", "--dummy", "--smoke", f"--config={config}", f"--output-root={tmp_path}"]) == 0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "progress.jsonl").exists()
    assert (tmp_path / "checkpoint.pt").exists()
    text = (tmp_path / "run.log").read_text()
    assert "STATUS completed=1 total=1" in text
    assert "DONE completed=1 total=1" in text


def test_preflight_requires_only_frozen_base_assets(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text())
    normalization = tmp_path / "normalization.npz"
    checkpoint = tmp_path / "state_4000.pt"
    import numpy as np

    np.savez(
        normalization,
        obs_min=np.zeros(9),
        obs_max=np.ones(9),
        action_min=np.zeros(7),
        action_max=np.ones(7),
    )
    torch.save({"model": {"network.weight": torch.zeros(1)}}, checkpoint)
    cfg["paths"]["anchor"] = {
        "normalization": str(normalization),
        "base_checkpoint": str(checkpoint),
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "preflight"
    assert main(["--phase=preflight", f"--config={config}", f"--output-root={output}"]) == 0
    gates = json.loads((output / "gates.json").read_text())
    assert gates["missing_assets"] == []
    assert gates["asset_errors"] == []
    assert set(gates["assets"]) == {"anchor_normalization", "anchor_base_checkpoint"}


def test_summary_requires_complete_paired_formal_runs(tmp_path: Path):
    cfg = yaml.safe_load(Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text())
    cfg["evaluation"]["episodes"] = 2
    manifest = build_episode_manifest(cfg["train"]["seeds"], 2)
    base_root = tmp_path / "base"
    write_episode_manifest(base_root / "episode_manifest.jsonl", manifest)
    base_rates = {1: 0.2, 2: 0.4, 3: 0.6}
    (base_root / "metrics.json").write_text(json.dumps({
        "episodes": 6,
        "success_rate": 0.4,
        "per_seed": [
            {"seed": seed, "policy_rng_seed": 1001 + seed, "success_rate": rate}
            for seed, rate in base_rates.items()
        ],
    }))
    group_rates = {
        "zero_eval": base_rates,
        "residual_eval": {1: 0.4, 2: 0.6, 3: 0.8},
        "random_residual": base_rates,
    }
    for group, rates in group_rates.items():
        for seed, rate in rates.items():
            run_root = tmp_path / group / f"seed_{seed}"
            write_episode_manifest(run_root / "episode_manifest.jsonl", manifest)
            (run_root / "metrics.json").write_text(json.dumps({
                "seed": seed,
                "policy_rng_seed": 1001 + seed,
                "episodes": 2,
                "success_rate": rate,
                "saturation_fraction": 0.1 if group == "residual_eval" else 0.0,
                "base_hash_unchanged": True,
            }))
    for seed in cfg["train"]["seeds"]:
        run_root = tmp_path / "residual" / f"seed_{seed}"
        run_root.mkdir(parents=True)
        (run_root / "checkpoint.pt").touch()
        (run_root / "metrics.json").write_text(json.dumps({
            "seed": seed,
            "iterations": cfg["train"]["n_train_itr"],
            "base_hash_unchanged": True,
        }))

    summarize(tmp_path, cfg)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["experiment_valid"] is True
    assert summary["hypothesis_supported"] is True


def test_merge_multi_gpu_base_eval_outputs(tmp_path: Path):
    cfg = yaml.safe_load(
        Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml").read_text()
    )
    cfg["evaluation"]["episodes"] = 2
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base-checkpoint")
    cfg["paths"]["anchor"]["base_checkpoint"] = str(checkpoint)
    manifest = build_episode_manifest(cfg["train"]["seeds"], 2)

    for seed_index, seed in enumerate(cfg["train"]["seeds"]):
        run_root = tmp_path / f"seed_{seed}"
        partial_manifest = [
            row for row in manifest if int(row["seed_index"]) == seed_index
        ]
        write_episode_manifest(run_root / "episode_manifest.jsonl", partial_manifest)
        (run_root / "metrics.json").write_text(json.dumps({
            "per_seed": [{
                "seed": seed,
                "policy_rng_seed": partial_manifest[0]["reset_seed"],
                "reset_seed_start": partial_manifest[0]["reset_seed"],
                "episodes": 2,
                "success_rate": 0.5,
                "avg_best_reward": 0.5,
                "elapsed_seconds": 1.0,
                "result_path": f"seed_{seed}/result.npz",
            }],
        }))

    _merge_official_base_eval(cfg, tmp_path)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["episodes"] == 6
    assert metrics["success_rate"] == 0.5
    assert [row["seed"] for row in metrics["per_seed"]] == [1, 2, 3]
    assert load_episode_manifest(tmp_path / "episode_manifest.jsonl") == manifest


def test_multi_gpu_launcher_schedules_three_seeds_on_selected_cards(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    fake_docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n')
    fake_docker.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(docker_log)

    result = subprocess.run(
        [
            "bash",
            "scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh",
            "--stage=train",
            "--gpus=0,1",
            "--seeds=1,2,3",
            "--launch-mode=docker",
            f"--experiment-root={tmp_path / 'experiment'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    launches = docker_log.read_text().splitlines()
    assert len(launches) == 3
    assert sum("--gpus device=0" in line for line in launches) == 2
    assert sum("--gpus device=1" in line for line in launches) == 1
    assert "DONE stage=train completed=3 total=3 failures=0" in result.stdout


def test_multi_gpu_launcher_direct_mode_uses_selected_visible_devices(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_conda = bin_dir / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == env && "$2" == list ]]; then
  printf 'dppo /opt/conda/envs/dppo\\n'
  exit 0
fi
printf 'CUDA_VISIBLE_DEVICES=%s %s\\n' "${CUDA_VISIBLE_DEVICES:-}" "$*" >> "$FAKE_CONDA_LOG"
exit 0
"""
    )
    fake_conda.chmod(0o755)
    conda_log = tmp_path / "conda.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_CONDA_LOG"] = str(conda_log)

    result = subprocess.run(
        [
            "bash",
            "scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh",
            "--stage=train",
            "--gpus=0,1",
            "--seeds=1,2,3",
            "--launch-mode=direct",
            f"--experiment-root={tmp_path / 'experiment'}",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    launches = conda_log.read_text().splitlines()
    assert len(launches) == 3
    assert sum("CUDA_VISIBLE_DEVICES=0" in line for line in launches) == 2
    assert sum("CUDA_VISIBLE_DEVICES=1" in line for line in launches) == 1
    assert all("run --no-capture-output -n dppo" in line for line in launches)
    assert "DONE stage=train completed=3 total=3 failures=0" in result.stdout


def test_docker_launcher_accepts_one_gpu_and_rejects_multiple(tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_docker = bin_dir / "docker"
    fake_docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FAKE_DOCKER_LOG"\n')
    fake_docker.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(docker_log)
    launcher = "scripts/experiments/rl/run_residual_robomimic_dppo_docker.sh"

    env["ROBOT_PRACTICE_DPPO_GPU"] = "4"
    single = subprocess.run(
        ["bash", launcher, "--phase=preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert single.returncode == 0, single.stderr
    assert "--gpus device=4" in docker_log.read_text()

    docker_log.unlink()
    env["ROBOT_PRACTICE_DPPO_GPU"] = "0,1"
    multiple = subprocess.run(
        ["bash", launcher, "--phase=preflight"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert multiple.returncode == 2
    assert "exactly one GPU" in multiple.stderr
    assert not docker_log.exists()
