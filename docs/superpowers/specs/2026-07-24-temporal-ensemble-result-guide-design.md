# Temporal Ensemble Result Guide Update

## Goal

Append the completed clean formal closed-loop result to
`reports/act_temporal_ensemble.md` without overstating causality or generalization.

## Result structure

Add one final section containing:

1. The evaluated task, source checkpoint family, 20 episodes, clean perturbation,
   seeds 1000–1019, and initialization IDs 0–19.
2. A comparison table with:
   - A0 single-step as an auxiliary baseline;
   - A1 Replan as the source-matched control;
   - A1 Temporal Ensemble as the evaluated configuration.
3. Success rate, Action TV, Action Jerk, mean gripper toggles, and p95 policy latency.
4. Relative changes for A1 Ensemble versus A1 Replan.
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
- Characterize the two failures as end-stage precision failures based on visual
  inspection, not as proven root causes.
- Limit the conclusion to one task and 20 fixed clean initializations. Noisy and
  cross-task robustness remain unverified.
- Use only repository-relative paths and include the `Formal` and `Closed-loop`
  result labels.
