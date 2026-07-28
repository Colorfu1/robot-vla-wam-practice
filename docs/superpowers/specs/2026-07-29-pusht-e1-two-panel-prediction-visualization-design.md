# PushT E1 Two-Panel Prediction Visualization Design

**Date:** 2026-07-29
**Status:** Approved for implementation

## Purpose

PushT E1 compares action chunks produced from the same frozen two-frame observation
history. For every candidate scene, BCChunk is called three times to verify deterministic
behavior and Diffusion Policy is called twenty times with independent initial noises.

The Viewer must present these twenty-three predictions without creating twenty-three image
views. It must support direct BC-versus-Diffusion comparison, individual prediction
visibility controls, and frame-by-frame switching between candidate scenes. E1 remains an
offline experiment: none of the displayed actions are executed.

## Data Contract

For each candidate, save all three BCChunk calls instead of retaining only the first output.
Each call produces one eight-step public action chunk.

`action_chunks.parquet` uses:

- `model=bc_chunk`, `sample_ordinal=0,1,2` for the three BC calls;
- `model=diffusion`, `sample_ordinal=0..19` for the twenty noise samples;
- `step=0..7` for every chunk;
- the existing normalized and denormalized action fields.

Formal E1 (`label=Offline`) requires exactly twenty Diffusion samples. Each candidate
contributes:

```text
(3 BC predictions + 20 Diffusion predictions) × 8 steps = 184 rows
```

Ten candidates therefore produce 1,840 rows. `bc_repeat_max_error` remains the maximum
normalized-action absolute difference between BC call zero and calls one and two, reduced
across both eight steps and both action coordinates. It continues to enforce the configured
determinism tolerance. Diffusion noise artifacts, reproducibility fields, diversity metrics,
and the `num_rollouts=0` contract remain unchanged.

The visualization input changes from one `bc_actions` array with shape `[8, 2]` to
`bc_actions` with shape `[3, 8, 2]`. `diffusion_actions` remains `[20, 8, 2]`.

Smoke E1 (`label=Smoke`) retains its configurable small Diffusion sample count, currently
three. Its visualization shape is `[M,8,2]` and its per-candidate row count is
`(3 + M) × 8`. Smoke does not change the formal E1 requirement.

## Artifact Contract

Keep the authoritative E1 artifacts:

- frozen observation tensors and input hashes;
- saved Diffusion noise tensors;
- `action_chunks.parquet`;
- `diversity_metrics.parquet`;
- `metrics.json` and `metrics.md`;
- `run_manifest.json`;
- `recording.rrd`.

Remove `fixed_observation_samples.png`. Its non-interactive layout cannot represent three
BC and twenty Diffusion predictions with independent visibility, and its current
single-BC-array consumer would become incorrect after the shape change. Update the E1 guide
and any output-contract tests accordingly.

## Viewer Layout

The embedded default Blueprint contains exactly two equal-purpose `Spatial2DView` panels:

1. **BC predictions** — the current observation image, current agent position, and three BC
   prediction overlays;
2. **Diffusion predictions** — the same current observation image and agent position, plus
   twenty Diffusion prediction overlays.

Do not add previous-observation, residual, metrics, dataframe, or auto-generated views to
the main viewport. Metrics remain available in `metrics.md`, `metrics.json`, and Parquet.

Each panel receives its own image and agent entity so its visibility controls are
independent:

```text
/e1/bc/image
/e1/bc/agent
/e1/bc/pred_00/{path,steps,first_action,endpoint}
/e1/bc/pred_01/{path,steps,first_action,endpoint}
/e1/bc/pred_02/{path,steps,first_action,endpoint}

/e1/dp/image
/e1/dp/agent
/e1/dp/pred_00/{path,steps,first_action,endpoint}
...
/e1/dp/pred_19/{path,steps,first_action,endpoint}
```

Every `pred_xx` parent is independently selectable in the Blueprint tree. Hiding the parent
hides all four child entities for that prediction.

## Prediction Drawing

Every prediction uses the previously agreed E1 representation:

```text
current agent → action[0] → action[1] → ... → action[7]
```

- `path` draws the complete connected plan;
- `steps` draws all eight action targets;
- `first_action` emphasizes step zero;
- `endpoint` emphasizes step seven;
- every BC and Diffusion prediction receives its own stable color;
- labels state that the plan is predicted and not executed.

Coordinates are denormalized PushT absolute targets mapped from the 512-unit workspace to
the current observation dimensions. The visualization does not integrate actions and does
not alter saved model outputs.

## Candidate Navigation

Candidate scenes are temporal frames on one stable `candidate` sequence timeline. Order is
defined by validated `candidate_ordinal`, not incidental JSON/list order:

```text
candidate #0 = candidate scene 0
candidate #1 = candidate scene 1
...
candidate #9 = candidate scene 9
```

All entities use stable paths and receive one temporal row per candidate. The user switches
scenes with Rerun's previous-frame/next-frame controls, the left/right arrow keys, or the
timeline. Both BC and Diffusion panels update together. Candidate IDs must not appear in
entity paths because unique roots would reintroduce latest-at accumulation.

Candidate ordinals must be unique. Formal candidates must form the expected ordered sequence
for the selected candidate set before recording begins.

The E1 application and recording identities remain stable so `--persist-state` can remember
user visibility choices and panel sizing. The embedded Blueprint is the active/default
layout for newly generated E1 recordings.

## Validation And Errors

The evaluator fails before writing a completed output when:

- the configured BC repeat count is not three for this formal protocol;
- `label=Offline` is requested with a Diffusion sample count other than twenty;
- BC or Diffusion chunks are not `[8, 2]`;
- the expected three BC or requested Diffusion samples are missing;
- BC repeated-call error exceeds the configured tolerance;
- a prediction contains non-finite action targets.

Automated tests verify:

- three BC calls are persisted with ordinals `0,1,2`;
- twenty Diffusion samples retain ordinals `0..19`;
- each formal candidate contributes 184 action rows;
- visualization arrays have shapes `[3,8,2]` and `[20,8,2]`;
- the Blueprint contains exactly two `Spatial2DView` panels and no other viewport;
- each panel includes only its model-specific image, agent, and prediction hierarchy;
- all candidate frames reuse stable entity paths;
- moving from one candidate frame to the next updates both panels without stale entities;
- each prediction path contains the agent plus eight targets;
- all three BC and all twenty formal Diffusion prediction parents contain `path`, `steps`,
  `first_action`, and `endpoint` on every candidate frame;
- prediction colors remain stable by model and sample ordinal across candidate frames;
- metrics and deterministic-reproduction checks remain unchanged.

Manual acceptance opens a newly generated RRD with `--persist-state`, switches across at
least three candidates using previous/next frame controls, and independently toggles one BC
prediction and multiple Diffusion predictions in their respective panels.

The implementation also updates
`reports/guides/pusht_diffusion/e1_fixed_observation.md`: it must describe the two-panel
layout, three saved BC calls, twenty formal Diffusion samples, candidate frame navigation,
individual prediction controls, and the removal of the static PNG. No previous-observation,
summary, or residual panel may remain in the guide's Viewer instructions.

## Rejected Alternatives

- **Twenty-three image panels:** makes one candidate unreadable and prevents useful model
  comparison.
- **One shared BC/DP panel:** maximizes image size but prevents side-by-side comparison and
  mixes visibility controls.
- **BC and Diffusion tabs:** avoids crowding but requires repeated tab switching.
- **Candidate-specific entity roots:** creates many tabs and causes stale latest-at data to
  accumulate when moving through candidate frames.
