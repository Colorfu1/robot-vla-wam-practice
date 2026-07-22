# Concise ACT A-Sanity Experiment Guide Design

## Status

User-directed rewrite approved in conversation on 2026-07-22. This replaces the earlier
audit-oriented guide design for `reports/act_a_sanity.md`.

## Goal

Rewrite the guide as a short experiment manual that answers only:

1. What experiment is being run?
2. Why is this comparison meaningful?
3. Which commands execute it from start to finish?

The guide targets a reader who wants to run A-Sanity, not audit its historical design.
The final guide should stay below roughly 220 lines and contain no long embedded Python
validators.

## Required Guide Structure

After the title and a one-sentence introduction, the guide has exactly three H2 sections.

### 1. Experiment Description

State the experiment question: can ACT overfit five fixed LIBERO demonstrations, and does
the trained checkpoint improve offline action prediction and closed-loop task success over
its own zero-update initialization?

Include one compact fixed-configuration table:

- LIBERO Object task 8;
- language `pick up the chocolate pudding and place it in the basket`;
- LeRobot dataset `lerobot/libero` at `datasets/lerobot/lerobot/libero`;
- dataset task index 29 and episodes 823, 826, 834, 840, 841;
- 5 episodes / 859 frames / 10 Hz;
- two 256x256 RGB cameras, 8D state, and 7D relative action;
- ACT `chunk_size=20`, `n_action_steps=5`, `use_vae=false`;
- batch size 8, seed 1000, and 20,000 optimizer updates.

Explain in one sentence that LeRobot supplies the dataset format, ACT training, and
evaluation tooling, while LIBERO supplies the task and simulator success signal.

### 2. Experiment Idea

Describe one paired comparison:

1. create B0 with zero optimizer updates;
2. evaluate B0 on the five training episodes and 20 LIBERO rollouts;
3. train 20,000 updates from that exact B0 checkpoint;
4. repeat the same evaluations for the trained checkpoint;
5. compare B0 with trained.

Define the two evaluation modes:

- offline: teacher-forced prediction on all 859 demonstration frames; it measures action
  error and does not measure task success;
- closed loop: execute the policy in LIBERO with seed 1000 and 20 episodes; it measures
  simulator success and actual rollout smoothness.

Retain only four concise expected outcomes:

- `loss@20000 <= 0.2 * loss@100`;
- trained `first_action_mae` is at least 20% lower than B0;
- trained `delta_action_mae` does not exceed B0;
- trained closed-loop `success_rate` is higher than B0.

### 3. Concrete Operating Steps

Provide one linear, copyable command sequence from the repository root:

1. activate `robot-practice` and define one `experiment_id`, log directory, evaluation
   root, and fixed dataset root;
2. run `a_sanity.sh --steps=0` through `tee`, parse its printed output directory, and set
   `b0_policy`;
3. run B0 offline evaluation with the explicit dataset root and all five episode IDs;
4. run B0 closed-loop evaluation with 20 episodes, seed 1000, no perturbation, and no
   smoke step bound;
5. run 20,000-step training with `--policy-path="${b0_policy}"` through `tee`, parse the
   printed output directory, and set `trained_policy`;
6. repeat the same offline and closed-loop commands for `trained_policy`;
7. show the four exact `metrics.json` paths, the training-log command for loss lines, and
   the four Rerun/MP4 paths used to inspect results;
8. add one sentence reminding the operator to record the resulting required experiment
   fields in `reports/experiment_log.md`, without including a template.

Use explicit unique evaluation result directories beneath
`outputs/act/a_sanity/eval/${experiment_id}/`. The evaluator output directory itself must
not exist before launch.

## Content to Remove

The rewrite removes all of the following from the guide:

- Git commit/status/diff capture;
- doctor and quickstart checks;
- dataset metadata scans, task-mapping validation, and data/video hashes;
- B0 manifest/hash and trained-lineage Python validators;
- `--print-command` preflight;
- automatic loss extraction and automatic gate-comparison Python programs;
- formal PASS/FAIL/INVALID taxonomy;
- large result tables;
- troubleshooting matrices;
- detailed experiment-record templates;
- historical design references and appended-run sections.

The launchers and evaluators already reject missing datasets, incomplete checkpoints, and
incompatible task/config inputs. The guide does not duplicate their implementation-level
validation.

## Verification

Before completion:

1. every Bash block passes `bash -n`;
2. the B0 and trained offline commands use identical arguments except checkpoint/output;
3. both offline commands pin the fixed dataset root and five episodes;
4. both closed-loop commands use 20 episodes, seed 1000, and `perturbation=none` without
   `--max-steps`;
5. the positive-step training command names the parsed B0 checkpoint;
6. all documented output filenames match the current evaluator implementation;
7. the guide contains exactly the three required H2 sections and no prohibited content.

## Acceptance Criteria

- The guide is immediately understandable as one experiment.
- A reader can copy the commands in order without reconstructing paths from old docs.
- The distinction between offline prediction and LIBERO closed-loop success is explicit.
- The fixed data, model, and evaluation settings are visible without prose archaeology.
- No audit machinery or historical material remains.
