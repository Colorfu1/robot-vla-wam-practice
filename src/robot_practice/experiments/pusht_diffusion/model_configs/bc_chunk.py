"""The complete BCChunk model configuration used by the PushT experiment."""

from lerobot.configs.types import NormalizationMode

from robot_practice.policies.bc_chunk import BCChunkConfig


def make_bc_chunk_config(*, device: str) -> BCChunkConfig:
    """Build the deterministic clean-action regression baseline."""

    # LeRobot fills input_features/output_features from the selected dataset metadata.
    return BCChunkConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        drop_n_last_frames=7,
        vision_backbone="resnet18",
        resize_shape=None,
        crop_ratio=1.0,
        crop_shape=(84, 84),
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        use_separate_rgb_encoder_per_camera=False,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        beta_start=1e-4,
        beta_end=2e-2,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        num_inference_steps=None,
        compile_model=False,
        do_mask_loss_for_padding=False,
        optimizer_lr=1e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
        fixed_timestep=0,
        device=device,
        use_amp=False,
        use_peft=False,
        push_to_hub=False,
    )


__all__ = ["make_bc_chunk_config"]
