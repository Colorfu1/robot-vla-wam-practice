from collections import defaultdict

import numpy as np
import torch
from mani_skill.utils import common
from tqdm import tqdm


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _mask(value, fallback):
    if value is None:
        return fallback
    return fallback & np.asarray(_to_numpy(value), dtype=bool).reshape(-1)


def _append_episode_metrics(eval_metrics, info, truncated, *, limit=None):
    """Read both Gymnasium final_info schemas used by ManiSkill releases.

    Older Gymnasium vector environments expose completed episode data through
    ``info["final_info"]``.  Gymnasium 1.x with the current ManiSkill CPU
    wrapper exposes the same data directly through ``info["episode"]`` and
    marks which sub-environments reported each metric with a sibling
    ``"_<metric>"`` boolean mask inside the same dict.  The metric values are
    unchanged; this only adapts the container format.

    The ``"_<metric>"`` entries are Gymnasium's presence masks, not metrics, so
    they are used for masking and never recorded as metrics themselves.
    """
    # Gymnasium/ManiSkill versions differ on whether termination metadata is
    # returned as NumPy or torch tensors.  The official CPU rollout contract
    # ends at this boundary; never let a CUDA tensor reach NumPy masking.
    truncated_mask = np.asarray(_to_numpy(truncated), dtype=bool).reshape(-1)

    def append_vector_metrics(episode, completed_mask):
        """Append one scalar per completed environment and return its count."""
        completed_count = int(completed_mask.sum())
        if limit is not None:
            completed_count = min(completed_count, int(limit))
        if completed_count == 0:
            return 0

        for key, value in episode.items():
            if key.startswith("_"):
                continue
            values = _to_numpy(value)
            if values.ndim == 0:
                selected = np.repeat(values[None], completed_count, axis=0)
            elif values.shape[0] == completed_mask.shape[0]:
                selected = values[completed_mask][:completed_count]
            elif values.shape[0] == int(completed_mask.sum()):
                selected = values[:completed_count]
            else:
                raise ValueError(
                    f"final episode metric {key!r} has shape {values.shape}, "
                    f"expected leading dimension {completed_mask.shape[0]} or "
                    f"{int(completed_mask.sum())}"
                )
            for item in selected:
                eval_metrics[key].append(item)
        return completed_count

    final_info = info.get("final_info")
    if final_info is not None:
        if isinstance(final_info, dict):
            # ManiSkillVectorEnv stores full vector tensors in final_info and
            # puts the completed-environment mask next to it as _final_info.
            # The old implementation appended each vector as one item, so a
            # 100-episode evaluation with 64 envs became two "episodes" whose
            # values were actually 64-element arrays.
            completed_mask = _mask(info.get("_final_info"), truncated_mask)
            return append_vector_metrics(final_info.get("episode", {}), completed_mask)
        else:
            count = 0
            for item in final_info:
                if item is None:
                    continue
                if limit is not None and count >= int(limit):
                    break
                episode = item.get("episode", {})
                for key, value in episode.items():
                    if key.startswith("_"):
                        continue
                    eval_metrics[key].append(_to_numpy(value))
                count += 1
            return count

    episode = info.get("episode")
    if episode is None:
        raise KeyError("evaluation info has neither 'final_info' nor 'episode'")

    completed_mask = _mask(info.get("_episode"), truncated_mask)
    # Gymnasium 1.x exposes per-key presence masks in the episode dict.
    # Materialize the selected rows first so the same limit semantics apply
    # to both info schemas.
    selected_episode = {}
    for key, value in episode.items():
        if key.startswith("_"):
            continue
        values = _to_numpy(value)
        key_mask = _mask(episode.get(f"_{key}"), completed_mask)
        if values.ndim == 0:
            if key_mask.any():
                selected_episode[key] = values
            continue
        if values.shape[0] != key_mask.shape[0]:
            raise ValueError(
                f"episode metric {key!r} has shape {values.shape}, "
                f"expected leading dimension {key_mask.shape[0]}"
            )
        selected_episode[key] = values[key_mask]
    return append_vector_metrics(
        selected_episode,
        np.ones(int(completed_mask.sum()), dtype=bool),
    )


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
    seed: int | None = None,
):
    """Roll out ``agent`` for ``n`` episodes and return per-episode metrics.

    Every evaluated episode starts from an explicit ``reset()``.  This matters
    because Gymnasium >= 1.0 vector environments default to
    ``AutoresetMode.NEXT_STEP``: the ``step()`` that truncates an episode
    returns the FINAL observation, and the following ``step()`` is swallowed by
    the autoreset (its action is discarded).  Re-using that final observation to
    compute the next action chunk would apply ``act_horizon - 1`` actions
    planned from the previous episode to the freshly reset episode, corrupting
    every episode after the first ``num_envs``.  Resetting explicitly is correct
    under both the old same-step and the new next-step autoreset semantics.

    Args:
        seed: optional base seed making the evaluated initial states
            reproducible.  Round ``r`` is reset with ``seed + r * num_envs``, so
            episode ``i`` always uses seed ``seed + i``.  NOTE: ManiSkill
            motionplanning demonstration ``i`` was itself recorded with
            ``episode_seed == i``, so a small base seed evaluates the policy on
            the very initial states it was trained on.  Leave as ``None`` (the
            default) to sample unseen initial states.
    """
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset(seed=seed)
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                action = action_seq[:, i]
                if sim_backend == "physx_cpu":
                    # Model inference may be on any CUDA rank.  CPU
                    # simulation must receive a host action, not a CUDA
                    # tensor.
                    action = _to_numpy(action)
                obs, _rew, _terminated, truncated, info = eval_envs.step(action)
                # Normalize the termination flag immediately after the
                # environment boundary.  This also covers GPU-backed
                # wrappers used by older ManiSkill releases.
                truncated = np.asarray(_to_numpy(truncated), dtype=bool).reshape(-1)
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                remaining = n - eps_count
                added = _append_episode_metrics(
                    eval_metrics, info, truncated, limit=remaining
                )
                if added <= 0:
                    raise RuntimeError(
                        "evaluation reported a truncated step without any "
                        "completed episode metrics"
                    )
                eps_count += added
                if progress_bar:
                    pbar.update(added)
                if eps_count < n:
                    obs, info = eval_envs.reset(
                        seed=None if seed is None else seed + eps_count
                    )
    agent.train()
    for k in eval_metrics:
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
