# Temporal Ensemble Result Guide Update

## Goal

Append the completed clean formal closed-loop result to
`reports/guides/act/act_temporal_ensemble.md` without overstating causality or generalization.

## Result structure

Add one final section containing:

1. The evaluated task, source checkpoint family, 20 episodes, clean perturbation,
   seeds 1000–1019, and initialization IDs 0–19.
   Include a directly runnable formal reproduction command using:
   `outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model`.
2. A comparison table with:
   - A0 single-step as an auxiliary baseline;
   - A1 Replan as the source-matched control;
   - A1 Temporal Ensemble as the evaluated configuration.
3. Success rate, Action TV, Action Jerk, mean gripper toggles, and p95 policy latency.
4. Changes for A1 Ensemble versus A1 Replan, reporting success as
   `+70 percentage points (20% to 90%)` and other relative changes in percent.
5. The two observed failure cases:
   - episode 2 nearly grasped the chocolate pudding but did not complete the grasp;
   - episode 16 placed the object at the basket edge and did not satisfy success.
6. A conclusion and an explicit limitations paragraph.

## Interpretation rules

- Treat A1 Replan versus A1 Ensemble as the primary causal comparison because they
  use the same checkpoint, seeds, initialization IDs, and per-step replanning.
- Treat A0 only as an auxiliary baseline because it was trained as a different policy.
- State that lower TV and Jerk are better.
- Describe gripper toggles without claiming that lower is always better.
- Characterize episode 2 as a visually observed grasp-stage precision failure and
  episode 16 as a visually observed placement-stage precision failure. Do not present
  either description as a proven root cause.
- Limit the conclusion to one task and 20 fixed clean initializations. Noisy and
  cross-task robustness remain unverified.
- Use only repository-relative paths and include the `Formal` and `Closed-loop`
  result labels.
