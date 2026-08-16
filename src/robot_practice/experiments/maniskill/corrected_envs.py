"""Repo-local ManiSkill environments used only after the M1 predicate audit."""

from __future__ import annotations

from .success_predicates import PEG_INSERTION_LOWER_X

CORRECTED_PEG_ENV_ID = "RobotPracticePegInsertionSideStrict-v1"


def register_corrected_peg_env() -> str:
    """Register and return the strict PegInsertionSide environment ID.

    Imports are lazy so pure protocol tests do not require a simulator process.
    The installed ManiSkill package is never modified.
    """

    import gymnasium as gym
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.utils.registration import register_env

    try:
        if gym.spec(CORRECTED_PEG_ENV_ID) is not None:
            return CORRECTED_PEG_ENV_ID
    except gym.error.NameNotFound:
        pass

    @register_env(CORRECTED_PEG_ENV_ID, max_episode_steps=100)
    class RobotPracticePegInsertionSideStrictEnv(PegInsertionSideEnv):
        def has_peg_inserted(self):
            peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
            x_flag = (peg_head_pos_at_hole[:, 0] >= PEG_INSERTION_LOWER_X) & (
                peg_head_pos_at_hole[:, 0] <= self.peg_half_sizes[:, 0]
            )
            y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
                peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
            )
            z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
                peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
            )
            return x_flag & y_flag & z_flag, peg_head_pos_at_hole

        def evaluate(self):
            success, peg_head_pos_at_hole = self.has_peg_inserted()
            return {
                "success": success.bool(),
                "peg_head_pos_at_hole": peg_head_pos_at_hole,
                "strict_success": success.bool(),
            }

    return CORRECTED_PEG_ENV_ID
