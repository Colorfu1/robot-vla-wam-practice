# ACT Temporal Ensemble Closed-loop Eval Override

## Goal

Allow the existing LIBERO closed-loop evaluator to enable ACT temporal ensembling from
the command line without modifying the saved checkpoint:

```bash
--n-action-steps=1 --temporal-ensemble-coeff=0.01
```

The evaluator must construct `ACTPolicy` with the effective temporal-ensemble setting,
because LeRobot creates `ACTTemporalEnsembler` during policy construction.

## Interface

Add this optional argument to `robot_practice.evaluation.libero_act` and document it in
`scripts/experiments/act/eval_libero.sh`:

```text
--temporal-ensemble-coeff=C
```

When omitted, preserve the checkpoint value. When provided, use it only for the current
evaluation. The user must also explicitly set `--n-action-steps=1`; the evaluator must
reject incompatible values rather than silently changing the execution horizon.

## Loading and validation

Resolve both runtime overrides before calling LeRobot's policy constructor:

1. Read the checkpoint ACT config.
2. Retain the checkpoint values for provenance.
3. Apply optional `n_action_steps` and `temporal_ensemble_coeff` overrides to the
   in-memory config.
4. Validate:
   - `1 <= n_action_steps <= chunk_size`;
   - temporal-ensemble coefficient is finite when present;
   - the resulting LeRobot float32 exponential weights are finite;
   - temporal ensembling requires `n_action_steps == 1`;
   - a command-line coefficient override requires an explicit
     `--n-action-steps=1`.
5. Construct the policy and processors from the effective in-memory config.

The new loader overrides default to absent, preserving the existing offline evaluator
and every other caller. Do not rewrite any checkpoint file. Reset the policy at every
episode boundary so temporal-ensemble history cannot leak between rollouts.

## Result provenance

Record these fields under `input.policy_execution` in `run_config.json`:

```json
{
  "checkpoint_n_action_steps": 5,
  "effective_n_action_steps": 1,
  "n_action_steps_overridden": true,
  "checkpoint_temporal_ensemble_coeff": null,
  "effective_temporal_ensemble_coeff": 0.01,
  "temporal_ensemble_coeff_overridden": true
}
```

Log the checkpoint and effective values at evaluator startup.

## Visualization and metrics

The existing temporal mode in `ActionPlanTracker` remains authoritative:

- the model is queried every control step;
- the executed action is LeRobot's ensembled action;
- the current full ACT prediction chunk remains available to Rerun;
- model-query counts and latency therefore reflect per-step replanning.

No new visualization schema is required.

## Tests

Cover:

- parser default and explicit coefficient;
- non-finite coefficients;
- explicit coefficient without explicit `--n-action-steps=1`;
- ensemble with an effective horizon other than 1;
- horizon larger than `chunk_size`;
- coefficient values whose float32 exponential weights overflow;
- construction of a real policy with an ensembler, including a fresh chunk query on
  consecutive `select_action()` calls and cleared history after `reset()`;
- unchanged checkpoint bytes and reset ensemble history across episode boundaries;
- unchanged offline loading when no override is provided;
- checkpoint/effective provenance in `run_config.json`;
- a bounded real LIBERO smoke evaluation.

Update the temporal-ensemble experiment guide after implementation so its command is
directly runnable, its unavailable statement is removed, and its implementation-status
section does not contradict the completed behavior.
