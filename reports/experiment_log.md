# Experiment Log

Use one section per run. Record incomplete or failed runs too.

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
