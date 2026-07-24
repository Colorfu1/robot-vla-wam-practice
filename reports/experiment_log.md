# 实验日志（Experiment Log）

Use one section per run. Record incomplete or failed runs too.
每条记录必须至少包含：

- 模式：`Offline` / `Closed-loop` / `Smoke` / `Formal`
- 结果落点（`metrics.json` / `eval_info.json` / `rollouts.parquet` / `prediction.rrd`）

## Template

- Date:
- Task suite:
- Dataset size:
- Observation keys:
- Action space:
- Action horizon:
- Control frequency:
- Policy:
- Train loss:
- Open-loop action error:
- Closed-loop success rate:
- Latency:
- Failure cases:
- Notes:
- Result artifacts:
- Modes:

## 2026-07-21 — ACT Evaluation Tooling Smoke

- Date: 2026-07-21
- Task suite: `lerobot/libero` task index 29 offline; `libero_object` task ID 8 closed-loop
- Dataset size: one frame from episode 823 for bounded offline verification
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: seven-dimensional LIBERO relative action; six arm dimensions plus gripper
- Action horizon: `chunk_size=20`, `n_action_steps=5`
- Control frequency: 10 Hz result/video timeline
- Policy: reduced randomly initialized ACT fixture; pipeline verification only
- Train loss: N/A; no training was launched
- Open-loop action error: metric computation and schema verified; random-policy value is not
  treated as an experiment result
- Closed-loop success rate: N/A; the one-step bounded smoke intentionally serializes formal
  `success_rate` as null
- Latency: synchronized policy-step and real model-query p50/p95 paths verified; smoke timing
  is not retained as a performance result
- Failure cases: none in the bounded smoke; meaningful evaluation remains blocked on the
  trained A-Sanity checkpoint
- Notes: 1-frame offline prediction and two serial 1-step real LIBERO rollouts wrote strict
  JSON, Parquet, Rerun, and (for LIBERO) MP4 outputs successfully. Repository verification:
  71 tests passed with 3 heavy tests skipped by default; the 3 heavy integration tests
  passed when explicitly enabled.
- Result artifacts:
  - `run_config.json`
  - `metrics.json`
  - `rollouts.parquet`
  - `rollout.rrd`
- Modes: Offline / Closed-loop / Smoke（`run_config` 标记为 smoke）

## 2026-07-23 — ACT A-Sanity 20k

- Date: 2026-07-23
- Task suite: `libero_object`, task ID 8
- Dataset size: 5 demonstrations, 859 frames
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative EEF action
- Action horizon: `chunk_size=20`, `n_action_steps=5`
- Control frequency: 10 Hz
- Policy: ACT, `use_vae=false`, 20,000 training updates
- Train loss: unavailable; W&B and retained terminal log were both absent
- Open-loop action error: first-action MAE 0.02076; delta-action MAE 0.00972
- Closed-loop success rate: 18/20, 90%
- Latency: model-query p50/p95 18.28/24.47 ms; policy-step p50/p95 0.77/22.94 ms
- Failure cases: 2/20 closed-loop rollouts failed; failure types not yet annotated
- Notes: Compared with the available 100-step reference, first-action MAE decreased 87.8%
  and delta-action MAE decreased 58.9%. A zero-update B0 result was not retained, so this
  is not a strict B0-versus-20k comparison.
- Result artifacts:
  - `run_config.json`
  - `metrics.json`
  - `eval_info.json`（closed-loop）
  - `rollouts.parquet`
  - `rollout.rrd`
- Modes: Offline / Closed-loop / Formal

## 2026-07-25 — ACT A2 Delta-action Loss 20k

- Date: 2026-07-25
- Task suite: `libero_object`, task ID 8
- Dataset size: 20 demonstrations, 3233 frames
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative EEF action; gripper GT is `-1/+1`
- Action horizon: `chunk_size=20`; Formal Clean Closed-loop uses `n_action_steps=5`
- Control frequency: 10 Hz
- Policy: `act_delta`, `use_vae=false`, `delta_loss_weight=0.1`, 20,000 updates,
  seed 1000
- Train loss: final 100-update window mean — L1 0.06815, delta-action 0.47619,
  weighted delta-action 0.04762, total 0.11577
- Open-loop action error: first-action MAE 0.02111; delta-action MAE 0.01250;
  teacher-forced Action TV 0.01908; teacher-forced Action Jerk 0.01370
- Closed-loop success rate: 14/20, 70% under clean Formal evaluation
- Latency: model-query p50/p95 7.07/14.16 ms; policy-step p50/p95 0.46/8.67 ms
- Failure cases: init states `0, 1, 4, 5, 8, 14` reached the 280-step time limit;
  failure stage classification remains pending
- Notes: Under the same `n_action_steps=5`, seed and init states, A1 reached 3/20 success
  with Action Jerk 0.03017; A2 reached 14/20 with Action Jerk 0.02403. A2 therefore
  improved success by 55 percentage points and reduced jerk by 20.3%. This run did not
  retain a tensor-verified A1/A2 shared step-0 checkpoint, and only one training seed has
  been evaluated. Mean gripper command sign toggles remained high at 14.65; this raw
  zero-crossing metric has no deadband and must not be read as physical full open-close
  cycles.
- Result artifacts:
  - `outputs/act/a2_delta_loss/run_20000_01/train_loss_components.jsonl`
  - `outputs/act/a2_delta_loss/run_20000_01/offline_20/metrics.json`
  - `outputs/act/a2_delta_loss/run_20000_01/offline_20/predictions.parquet`
  - `outputs/act/a2_delta_loss/run_20000_01/offline_20/prediction.rrd`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_n5/metrics.json`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_n5/eval_info.json`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_n5/rollouts.parquet`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_n5/rollout.rrd`
- Modes: Offline / Closed-loop / Formal

## 2026-07-25 — ACT A2 + Temporal Ensemble Clean

- Date: 2026-07-25
- Task suite: `libero_object`, task ID 8
- Dataset size: source policy trained on 20 demonstrations, 3233 frames; evaluation uses
  20 simulator rollouts
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative EEF action
- Action horizon: `chunk_size=20`; effective `n_action_steps=1`
- Control frequency: 10 Hz
- Policy: existing A2 `act_delta` 20k checkpoint with inference-only
  `temporal_ensemble_coeff=0.01`
- Train loss: N/A for this inference-only configuration; source A2 training result is
  recorded in the preceding entry
- Open-loop action error: unchanged from the source A2 checkpoint; Temporal Ensemble is a
  closed-loop history mechanism and does not create a new Offline MAE result
- Closed-loop success rate: 18/20, 90% under clean Formal evaluation
- Latency: model-query p50/p95 6.80/10.80 ms; policy-step p50/p95 7.46/11.63 ms
- Failure cases: init states `16, 17` reached the 280-step time limit; failure stages remain
  pending
- Notes: Compared with A2 Standard `n_action_steps=5`, success increased from 70% to 90%,
  Action TV decreased 35.3%, Action Jerk decreased 82.6%, and mean raw gripper command
  sign toggles decreased from 14.65 to 2.30. Standard failures `0, 1, 4, 5, 8, 14`
  recovered, while init states `16, 17` regressed. A strict A2
  `n_action_steps=1, temporal_ensemble_coeff=None` Replan control has not been run, so this
  comparison establishes the combined deployment configuration improvement but does not
  isolate the causal contribution of Temporal Ensemble.
- Result artifacts:
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/metrics.json`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/eval_info.json`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/rollouts.parquet`
  - `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/rollout.rrd`
- Modes: Closed-loop / Formal
