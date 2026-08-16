ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)
from robot_practice.experiments.maniskill.ddp_device import (
    rank_render_backend,
    rank_sim_backend,
)


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    dataset_format: str = "official"
    """Dataset contract: official, M1-W 7D, or the matched 18D/4D ablations."""
    episode_ids_file: Optional[str] = None
    """Optional explicit trajectory ID list used by the M1-W lazy adapter."""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 64  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    eval_seed: int = 20000
    """base environment seed for deterministic distributed training evaluation"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None
    resume_from: Optional[str] = None
    """Resume from a checkpoint written by this training entry point."""
    # torchrun 2.0+ may pass this flag; rank is read from the environment below.
    local_rank: Optional[int] = None


def _distributed_context(args):
    """Initialize optional torchrun context without changing the single-GPU path."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if not distributed:
        return False, 0, 0, 1

    if not args.cuda or not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA; do not launch torchrun with --no-cuda")
    if args.num_dataload_workers < 0:
        raise ValueError("num_dataload_workers must be non-negative")
    if args.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch_size={args.batch_size} must be divisible by world_size={world_size}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, local_rank, world_size


def _is_main_process(distributed, rank):
    return (not distributed) or rank == 0


def _partition_count(total: int, world_size: int, rank: int) -> int:
    """Return the deterministic episode shard owned by one policy rank."""

    if total <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("total, world_size, and rank must describe a valid shard")
    if total < world_size:
        raise ValueError(
            f"num_eval_episodes={total} must be at least world_size={world_size} "
            "for distributed multi-GPU evaluation"
        )
    base, remainder = divmod(total, world_size)
    return base + int(rank < remainder)


def _partition_offset(total: int, world_size: int, rank: int) -> int:
    """Return the first global episode index owned by one policy rank."""

    if not 0 <= rank <= world_size:
        raise ValueError("rank must be within [0, world_size]")
    return sum(
        _partition_count(total, world_size, previous_rank)
        for previous_rank in range(rank)
    )


def _merge_eval_metrics(shards):
    """Merge per-rank CPU metric arrays without changing episode semantics."""

    if not shards:
        raise ValueError("distributed evaluation produced no metric shards")
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards[1:]):
        raise ValueError("distributed evaluation metric keys differ across ranks")
    return {
        key: np.concatenate([np.asarray(shard[key]) for shard in shards], axis=0)
        for key in sorted(keys)
    }


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


class SmallDemoDataset_DiffusionPolicy(Dataset):  # Load everything into memory
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, device, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of dict, each dict is a traj, with keys in obs_space, values with length L+1
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, obs_space
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(
                    _obs_traj_dict["depth"].astype(np.float32)
                ).to(device=device, dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(
                    device
                )  # still uint8
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(
                device
            )
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(
                device=device
            )
        print(
            "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if (
            "delta_pos" in args.control_mode
            or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        ):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
            self.pad_action_arm = torch.zeros(
                (trajectories["actions"][0].shape[1] - 1,), device=device
            )
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        else:
            # NOTE for absolute joint pos control probably should pad with the final joint position action.
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = (
            args.obs_horizon,
            args.pred_horizon,
        )
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
        if start < 0:  # pad before the trajectory
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            gripper_action = act_seq[-1, -1]  # assume gripper is with pos controller
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "observations": obs_seq,
            "actions": act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert (
            len(env.single_observation_space["state"].shape) == 2
        )  # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1  # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (
            env.single_action_space.low == -1
        ).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim,  # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=self.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # has big impact on performance, try not to change
            clip_sample=True,  # clip output to [-1,1] to improve stability
            prediction_type="epsilon",  # predict noise (instead of denoised action)
        )

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3*k, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1*k, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)  # (B, obs_horizon, C, H, W), C=4*k
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # (B*obs_horizon, C, H, W)
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)  # (B*obs_horizon, C, H, W)
        visual_feature = self.visual_encoder(img_seq)  # (B*obs_horizon, D)
        visual_feature = visual_feature.reshape(
            batch_size, self.obs_horizon, visual_feature.shape[1]
        )  # (B, obs_horizon, D)
        feature = torch.cat(
            (visual_feature, obs_seq["state"]), dim=-1
        )  # (B, obs_horizon, D+obs_state_dim)
        return feature.flatten(start_dim=1)  # (B, obs_horizon * (D+obs_state_dim))

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]

        # observation as FiLM conditioning
        obs_cond = self.encode_obs(
            obs_seq, eval_mode=False
        )  # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond
        )

        return F.mse_loss(noise_pred, noise)

    def forward(self, obs_seq, action_seq):
        """Expose the loss through DDP while preserving the original API."""
        return self.compute_loss(obs_seq, action_seq)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq['state']: (B, obs_horizon, obs_state_dim)
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)

            obs_cond = self.encode_obs(
                obs_seq, eval_mode=True
            )  # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn(
                (B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device
            )

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]  # (B, act_horizon, act_dim)


def save_ckpt(run_name, tag):
    if not is_main:
        return
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save(
        {
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": int(tag) if str(tag).isdigit() else None,
            "best_eval_metrics": dict(best_eval_metrics),
        },
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    distributed, rank, local_rank, world_size = _distributed_context(args)
    is_main = _is_main_process(distributed, rank)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    custom_ablation_formats = {
        "official_rgb_18d_4d",
        "m1w_projected_rgb_4d",
    }
    if args.dataset_format not in {"official", "m1w_live_rgb", *custom_ablation_formats}:
        raise ValueError(
            f"unknown dataset_format={args.dataset_format!r}; "
            "expected 'official', 'm1w_live_rgb', 'official_rgb_18d_4d', "
            "or 'm1w_projected_rgb_4d'"
        )
    is_m1w = args.dataset_format == "m1w_live_rgb"
    is_ablation = args.dataset_format in custom_ablation_formats
    is_custom_rgb = is_m1w or is_ablation
    if is_m1w:
        if args.episode_ids_file is None:
            raise ValueError("m1w_live_rgb requires --episode-ids-file")
        if args.env_id != "StackCube-v1":
            raise ValueError("m1w_live_rgb currently supports StackCube-v1 only")
        if args.control_mode != "pd_ee_delta_pose":
            raise ValueError("m1w_live_rgb requires --control-mode=pd_ee_delta_pose")
        if args.obs_mode != "rgb":
            raise ValueError("m1w_live_rgb requires --obs-mode=rgb")
        from robot_practice.experiments.maniskill.official_m1w_adapter import (
            M1WFlattenRGBObservationWrapper,
            M1WLiveRGBDataset,
        )
    elif is_ablation:
        if args.episode_ids_file is None:
            raise ValueError(f"{args.dataset_format} requires --episode-ids-file")
        if args.env_id != "StackCube-v1":
            raise ValueError(f"{args.dataset_format} currently supports StackCube-v1 only")
        if args.control_mode != "pd_ee_delta_pos":
            raise ValueError(f"{args.dataset_format} requires --control-mode=pd_ee_delta_pos")
        if args.obs_mode != "rgb":
            raise ValueError(f"{args.dataset_format} requires --obs-mode=rgb")
        from robot_practice.experiments.maniskill.official_stackcube_18d4_adapter import (
            M1WFlattenRGBObservationWrapper,
            StackCube18D4DDataset,
        )
    elif args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1
    if args.num_eval_episodes <= 0 or args.num_eval_envs <= 0:
        raise ValueError("num_eval_episodes and num_eval_envs must be positive")
    if distributed and args.num_eval_episodes < world_size:
        raise ValueError(
            "distributed official evaluation requires num_eval_episodes >= "
            f"world_size ({args.num_eval_episodes} < {world_size})"
        )

    rank_seed = args.seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    # ``torch.cuda.set_device(local_rank)`` only changes PyTorch's current
    # device.  ManiSkill's backend parser defaults an unspecified GPU backend
    # to cuda:0, so every DDP rank must receive an explicit device suffix.
    eval_sim_backend = rank_sim_backend(args.sim_backend, local_rank)
    eval_render_backend = rank_render_backend(args.sim_backend, local_rank)
    if eval_render_backend is not None:
        env_kwargs["render_backend"] = eval_render_backend
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    local_eval_episodes = (
        _partition_count(args.num_eval_episodes, world_size, rank)
        if distributed
        else args.num_eval_episodes
    )
    local_eval_seed = args.eval_seed + _partition_offset(
        args.num_eval_episodes, world_size, rank
    )
    # Treat num_eval_envs as a global cap.  Each policy rank owns a fair
    # fraction of the CPU vector workers while still having at least one.
    global_env_cap_per_rank = max(
        1, (args.num_eval_envs + world_size - 1) // world_size
    )
    eval_num_envs = min(local_eval_episodes, global_env_cap_per_rank)
    eval_wrappers = [M1WFlattenRGBObservationWrapper] if is_custom_rgb else [FlattenRGBDObservationWrapper]
    envs = make_eval_envs(
        args.env_id,
        eval_num_envs,
        eval_sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video and is_main else None,
        wrappers=eval_wrappers,
    )

    if args.track and is_main:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"],
        )
    writer = SummaryWriter(f"runs/{run_name}") if is_main else None
    if is_main:
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    if not is_custom_rgb:
        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(
                np.transpose, axes=(0, 3, 1, 2)
            ),  # (B, H, W, C) -> (B, C, H, W)
            state_obs_extractor=build_state_obs_extractor(args.env_id),
            depth="rgbd" in args.demo_path,
        )

    # Create a temporary environment to get the original observation space as
    # AsyncVectorEnv (CPU parallelization) does not expose it.  Keep this
    # introspection environment on the same rank-bound rollout backend;
    # otherwise a seemingly harmless ``gym.make`` here can still initialize a
    # renderer on the wrong GPU once per DDP rank.
    tmp_env_kwargs = dict(env_kwargs, sim_backend=eval_sim_backend)
    if eval_sim_backend == "physx_cpu":
        tmp_env_kwargs["render_backend"] = "sapien_cpu"
    tmp_env = gym.make(args.env_id, **tmp_env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    # determine whether the env will return rgb and/or depth data
    include_rgb = True if is_custom_rgb else tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = False if is_custom_rgb else tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    if is_m1w:
        dataset = M1WLiveRGBDataset(
            data_path=args.demo_path,
            episode_ids_file=args.episode_ids_file,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            control_mode=args.control_mode,
            expected_action_dim=7,
        )
    elif is_ablation:
        dataset = StackCube18D4DDataset(
            data_path=args.demo_path,
            episode_ids_file=args.episode_ids_file,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            dataset_format=args.dataset_format,
        )
    else:
        dataset = SmallDemoDataset_DiffusionPolicy(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=orignal_obs_space,
            include_rgb=include_rgb,
            include_depth=include_depth,
            device=device,
            num_traj=args.num_demos,
        )
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        local_batch_size = args.batch_size // world_size
    else:
        sampler = RandomSampler(dataset, replacement=False)
        local_batch_size = args.batch_size
    batch_sampler = BatchSampler(sampler, batch_size=local_batch_size, drop_last=True)
    resume_path = None
    resume_step = 0
    if args.resume_from is not None:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_from))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        basename = os.path.basename(resume_path)
        stem = os.path.splitext(basename)[0]
        if stem.isdigit():
            resume_step = int(stem)
        if resume_step >= args.total_iters:
            raise ValueError(
                f"resume checkpoint step {resume_step} must be smaller than total_iters={args.total_iters}"
            )
    batch_sampler = IterationBasedBatchSampler(
        batch_sampler, args.total_iters, start_iter=resume_step
    )
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
        pin_memory=(device.type == "cuda"),
        **({"prefetch_factor": 2} if args.num_dataload_workers > 0 else {}),
    )

    agent = Agent(envs, args).to(device)
    ddp_agent = (
        DistributedDataParallel(
            agent,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        if distributed
        else agent
    )

    optimizer = optim.AdamW(
        params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location=device)
        if not isinstance(resume_payload, dict):
            raise TypeError(f"resume checkpoint must be a dictionary: {resume_path}")
        agent_state = resume_payload.get("agent")
        ema_agent_state = resume_payload.get("ema_agent", agent_state)
        if agent_state is None or ema_agent_state is None:
            raise KeyError("resume checkpoint must contain agent or ema_agent weights")
        agent.load_state_dict(agent_state)
        ema_agent.load_state_dict(ema_agent_state)
        if resume_payload.get("optimizer") is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
        elif is_main:
            print(
                "WARNING: legacy checkpoint has no optimizer state; resuming with a fresh optimizer",
                flush=True,
            )
        if resume_payload.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(resume_payload["lr_scheduler"])
        if resume_payload.get("ema_state") is not None:
            ema.load_state_dict(resume_payload["ema_state"])
        else:
            # Legacy checkpoints contain the materialized EMA model but not
            # EMAModel's shadow/step state. Start the EMA shadow from it.
            ema_state = ema.state_dict()
            ema_state["shadow_params"] = [
                parameter.detach().clone() for parameter in ema_agent.parameters()
            ]
            ema.load_state_dict(ema_state)
        saved_best = resume_payload.get("best_eval_metrics")
        if isinstance(saved_best, dict):
            best_eval_metrics.update(
                {str(key): float(value) for key, value in saved_best.items()}
            )
        if is_main:
            print(
                f"Resumed checkpoint={resume_path} at step={resume_step}; "
                f"target_total_iters={args.total_iters}",
                flush=True,
            )

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        # Avoid a full evaluation before the first optimizer update.  Every
        # DDP rank performs policy inference on its own GPU and uses the
        # configured rank-bound rollout backend; the external evaluator
        # repeats the same protocol after training for checkpoint selection.
        if args.eval_freq <= 0 or iteration <= 0 or iteration % args.eval_freq != 0:
            return
        if distributed:
            dist.barrier()
        last_tick = time.time()
        # Every DDP rank performs policy inference on its own CUDA device.  For
        # M1-W, the simulator is also on that rank's CUDA device; for the
        # official data route, evaluate() transfers actions to CPU PhysX.
        ema.copy_to(ema_agent.parameters())
        local_eval_metrics = evaluate(
            local_eval_episodes,
            ema_agent,
            envs,
            device,
            eval_sim_backend,
            progress_bar=is_main,
            seed=local_eval_seed,
        )
        if distributed:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(
                gathered,
                {key: np.asarray(value) for key, value in local_eval_metrics.items()},
            )
            eval_metrics = _merge_eval_metrics(gathered) if is_main else None
        else:
            eval_metrics = local_eval_metrics
        timings["eval"] += time.time() - last_tick

        if is_main:
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
        if distributed:
            dist.barrier()

    def log_metrics(iteration):
        if is_main and iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    def move_batch_to_device(batch):
        """Move both lazy CPU batches and legacy eager GPU batches to the rank device."""

        return {
            "observations": {
                key: value.to(device, non_blocking=True)
                for key, value in batch["observations"].items()
            },
            "actions": batch["actions"].to(device, non_blocking=True),
        }

    ddp_agent.train()
    pbar = tqdm(total=args.total_iters, disable=not is_main)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader, start=resume_step):
        step = iteration + 1
        timings["data_loading"] += time.time() - last_tick
        data_batch = move_batch_to_device(data_batch)

        # forward and compute loss
        last_tick = time.time()
        total_loss = ddp_agent(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(step)
        log_metrics(step)

        # Checkpoint
        if args.save_freq is not None and step % args.save_freq == 0:
            save_ckpt(run_name, str(step))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)
    if args.save_freq is not None:
        # The loop's zero-based counter saves 0, save_freq, ... before the
        # final iteration.  Also persist the actual terminal step so a runner
        # can resume/evaluate the requested budget without guessing.
        save_ckpt(run_name, str(args.total_iters))

    envs.close()
    if writer is not None:
        writer.close()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
