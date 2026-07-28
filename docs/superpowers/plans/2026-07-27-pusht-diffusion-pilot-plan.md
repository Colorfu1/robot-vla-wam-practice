# PushT Diffusion Pilot Implementation Plan

**Date:** 2026-07-27

## Goal

Implement the approved PushT Pilot infrastructure without running formal training or
implementing optional E3 state restoration.

## Task 1: Dependency, split, and configuration foundation

Files:

- `requirements.txt`
- `configs/experiments/pusht_diffusion/*.json`
- `src/robot_practice/experiments/pusht_diffusion/config.py`
- `src/robot_practice/experiments/pusht_diffusion/dataset_split.py`
- `src/robot_practice/experiments/pusht_diffusion/provenance.py`

Work:

1. Add the LeRobot PushT extra and clustering dependency.
2. Commit the deterministic 165/20/21 episode split.
3. Validate split disjointness, dataset identity, and smoke-subset rules.
4. Record public, repository-relative experiment metadata and package versions.

## Task 2: Reproducible sampling and common initialization

Files:

- `src/robot_practice/experiments/pusht_diffusion/noise.py`
- `src/robot_practice/experiments/pusht_diffusion/policy_loading.py`
- `src/robot_practice/experiments/pusht_diffusion/initialization.py`

Work:

1. Implement stable initial-noise and DDPM reverse-generator seeds.
2. Implement a local controlled Diffusion sampler using the installed inner
   `conditional_sample()` interface.
3. Reconstruct DDPM/DDIM schedulers without changing model weights.
4. Create paired step-zero BCChunk/Diffusion checkpoints with identical common tensors.

## Task 3: Strict BCChunk policy

Files:

- `src/robot_practice/policies/bc_chunk/configuration_bc_chunk.py`
- `src/robot_practice/policies/bc_chunk/modeling_bc_chunk.py`
- `src/robot_practice/policies/bc_chunk/processor_bc_chunk.py`

Work:

1. Register `policy.type=bc_chunk`.
2. Reuse the Diffusion RGB encoder and temporal U-Net.
3. Feed a zero trajectory and integer timestep zero.
4. Regress the complete 16-step clean action target with MSE.
5. Preserve LeRobot queue behavior and expose the `[1:9]` action chunk.

## Task 4: Metrics, outputs, and environment rollout

Files:

- `src/robot_practice/experiments/pusht_diffusion/diversity_metrics.py`
- `src/robot_practice/experiments/pusht_diffusion/statistics.py`
- `src/robot_practice/experiments/pusht_diffusion/output.py`
- `src/robot_practice/experiments/pusht_diffusion/rollout.py`
- `src/robot_practice/experiments/pusht_diffusion/latency.py`

Work:

1. Implement E1 action diversity and E2 success-conditioned path diversity.
2. Implement the pre-registered trajectory-mode distance and clustering threshold.
3. Implement Wilson intervals and environment-cluster bootstrap.
4. Record complete per-rollout/replan seed and noise metadata.
5. Measure synchronized model-only and end-to-end latency.

## Task 5: E0, E1, E2, E4 entrypoints and visualization

Files:

- `src/robot_practice/experiments/pusht_diffusion/data_inspection.py`
- `src/robot_practice/experiments/pusht_diffusion/fixed_observation_eval.py`
- `src/robot_practice/experiments/pusht_diffusion/rollout_eval.py`
- `src/robot_practice/experiments/pusht_diffusion/sampling_steps_eval.py`
- `src/robot_practice/experiments/pusht_diffusion/rerun_visualization.py`
- `src/robot_practice/experiments/pusht_diffusion/report.py`

Work:

1. Add qualitative demonstration inspection and candidate export.
2. Freeze complete E1 inputs and sample 20 explicit noises.
3. Execute independent same-seed E2 rollouts.
4. Execute the paired DDIM/DDPM E4 matrix and latency benchmark.
5. Produce Rerun 2D recordings, static figures, JSON, Markdown, and Parquet outputs.

## Task 6: Direct launchers and experiment guide

Files:

- `scripts/experiments/pusht/*.sh`
- `reports/guides/pusht_diffusion/`
- `reports/experiment_log.md`

Work:

1. Keep evaluation launchers direct and make training Bash files thin environment forwarders
   to model-specific Python modules.
2. Build official LeRobot configs in Python and support documented parameters, strict
   duplicate/unknown-argument rejection, and side-effect-free `--print-command`.
3. Keep smoke and Pilot output roots separate.
4. Document copy-ready commands, output locations, labels, and completion criteria.

## Task 7: Verification

Files:

- `tests/test_pusht_*.py`

Work:

1. Test split, seeds, controlled sampling, BC loss, shared hashes, metrics, output schemas,
   and launchers.
2. Run focused tests and Ruff over changed Python files.
3. Run shell syntax checks and every launcher in `--print-command` mode.
4. Run environment/interface smoke only when `gym-pusht` and the dataset are available.
5. Confirm no generated dataset, checkpoint, video, RRD, or machine manifest is tracked.
