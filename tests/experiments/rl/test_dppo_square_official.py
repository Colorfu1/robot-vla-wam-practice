import logging

from omegaconf import OmegaConf

from robot_practice.experiments.rl.dppo_square_official import Progress, configure


def test_training_preserves_schedule_and_smoke_only_shortens_run(tmp_path):
    directory = tmp_path / "cfg/robomimic/pretrain/square"
    directory.mkdir(parents=True)
    original = OmegaConf.create(
        {
            "train": {
                "n_epochs": 2000,
                "batch_size": 256,
                "lr_scheduler": {"first_cycle_steps": 2000, "warmup_steps": 100},
            },
            "seed": 42,
            "wandb": {},
            "train_dataset_path": "unused",
            "logdir": "unused",
        }
    )
    OmegaConf.save(original, directory / "pre_diffusion_mlp_img.yaml")
    cfg = configure(tmp_path, tmp_path / "out", tmp_path / "data", "train")
    assert cfg.train == original.train
    smoke = configure(tmp_path, tmp_path / "out", tmp_path / "data", "train", smoke=True)
    assert smoke.train.n_epochs == 1
    assert smoke.train.lr_scheduler == original.train.lr_scheduler
    assert smoke.train.batch_size == 256


def test_progress_filter_does_not_duplicate_for_two_handlers():
    record = logging.LogRecord("test", logging.INFO, "", 1, "500: train loss 0.01", (), None)
    progress = Progress(2000)
    progress.filter(record)
    progress.filter(record)
    assert record.getMessage().count("STATUS") == 1
    assert "completed=500 total=2000" in record.getMessage()


def test_legacy_training_does_not_modernize_ema_or_schedule(tmp_path):
    directory = tmp_path / "cfg/robomimic/pretrain/square"
    directory.mkdir(parents=True)
    original = OmegaConf.create(
        {
            "train": {
                "n_epochs": 4000,
                "batch_size": 256,
                "update_ema_freq": 10,
                "epoch_start_ema": 20,
                "lr_scheduler": {"first_cycle_steps": 8000},
            },
            "train_dataset": {"max_n_episodes": 100},
        }
    )
    OmegaConf.save(original, directory / "pre_diffusion_mlp_img.yaml")
    cfg = configure(tmp_path, tmp_path / "out", tmp_path / "data", "train")
    assert cfg.train == original.train
    assert cfg.train_dataset.max_n_episodes == 100


def test_eval_preserves_official_budget_and_only_fixes_import(tmp_path):
    directory = tmp_path / "cfg/robomimic/eval/square"
    directory.mkdir(parents=True)
    original = OmegaConf.create(
        {
            "seed": 42,
            "n_steps": 400,
            "env": {"n_envs": 20},
            "model": {"_target_": "model.diffusion.diffusion_eval_ft.DiffusionEval"},
        }
    )
    OmegaConf.save(original, directory / "eval_diffusion_mlp_img.yaml")
    cfg = configure(tmp_path, tmp_path / "out", tmp_path / "data", "eval", tmp_path / "x.pt")
    assert cfg.seed == 42 and cfg.n_steps == 400 and cfg.env.n_envs == 20
    assert cfg.model._target_ == "model.diffusion.diffusion_eval.DiffusionEval"


def test_rl_eval_overrides_resolve_into_model(tmp_path):
    directory = tmp_path / "cfg/robomimic/eval/square"
    directory.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "seed": 42, "n_steps": 400, "env": {"n_envs": 20},
        "ft_denoising_steps": 0,
        "model": {"ft_denoising_steps": "${ft_denoising_steps}"},
    }), directory / "eval_diffusion_mlp_img.yaml")
    cfg = configure(tmp_path, tmp_path / "out", tmp_path / "data", "eval",
                    tmp_path / "rl.pt", eval_overrides={
                        "ft_denoising_steps": 5, "seed": 2,
                        "env.n_envs": 50, "n_steps": 100,
                    })
    assert cfg.model.ft_denoising_steps == 5
    assert (cfg.seed, cfg.env.n_envs, cfg.n_steps) == (2, 50, 100)
