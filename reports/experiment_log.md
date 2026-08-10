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

## 2026-07-27 — ACT A1 Observation-equal Loss 20k

- Date: 2026-07-27
- Task suite: `libero_object`, task ID 8
- Dataset size: 20 demonstrations, 3233 frames
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative EEF action
- Action horizon: `chunk_size=20`; Formal Clean Closed-loop uses `n_action_steps=5`
- Control frequency: 10 Hz
- Policy: `act_obs_equal`, `use_vae=false`, 20,000 updates, seed 1000; valid action
  errors are averaged per observation before the batch mean
- Train loss: unavailable; the retained checkpoint does not include a local training-loss
  series
- Open-loop action error: first-action MAE 0.02488; gripper MAE 0.04537;
  delta-action MAE 0.01392; teacher-forced Action Jerk 0.01353
- Closed-loop success rate: 9/20, 45% under clean Formal evaluation
- Latency: model-query p50/p95 7.00/14.06 ms; policy-step p50/p95 0.45/8.64 ms
- Failure cases: init states `0, 1, 4, 5, 6, 8, 9, 12, 15, 16, 17` reached the
  280-step time limit; failure stages are not yet structurally annotated
- Notes: Compared with A1, success increased by 30 percentage points, Action Jerk
  decreased 14.1%, and mean gripper toggles decreased 8.8%. This single-seed result
  supports the observation-level loss-weighting hypothesis but is not a cross-seed
  conclusion.
- Result artifacts:
  - `outputs/act/a1_observation_equal/run_20000_01/offline_20/metrics.json`
  - `outputs/act/a1_observation_equal/run_20000_01/offline_20/predictions.parquet`
  - `outputs/act/a1_observation_equal/run_20000_01/offline_20/prediction.rrd`
  - `outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/metrics.json`
  - `outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/eval_info.json`
  - `outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/rollouts.parquet`
  - `outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/rollout.rrd`
- Modes: Offline / Closed-loop / Formal

## 2026-07-27 — ACT A1 VAE 20k

- Date: 2026-07-27
- Task suite: `libero_object`, task ID 8
- Dataset size: 20 demonstrations, 3233 frames
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative EEF action
- Action horizon: `chunk_size=20`; Formal Clean Closed-loop uses `n_action_steps=5`
- Control frequency: 10 Hz
- Policy: standard ACT, `use_vae=true`, `latent_dim=32`, `kl_weight=10.0`,
  20,000 updates, seed 1000
- Train loss: unavailable; the retained checkpoint does not include a local training-loss
  or KLD series
- Open-loop action error: unavailable; A1-VAE Offline evaluation has not been run
- Closed-loop success rate: 19/20, 95% under clean Formal evaluation
- Latency: model-query p50/p95 7.13/12.39 ms; policy-step p50/p95 0.56/8.67 ms
- Failure cases: init state `17` reached the 280-step time limit and produced 34 raw
  gripper command sign toggles
- Notes: Compared with A1, success increased by 80 percentage points, Action Jerk
  decreased 26.9%, and mean gripper toggles decreased 23.0%. A1-VAE is the strongest
  tested A1-family configuration on this task, subject to the single-seed and
  single-task evidence boundary.
- Result artifacts:
  - `outputs/act/a1_vae/run_20000_01/closed_clean_n5/metrics.json`
  - `outputs/act/a1_vae/run_20000_01/closed_clean_n5/eval_info.json`
  - `outputs/act/a1_vae/run_20000_01/closed_clean_n5/rollouts.parquet`
  - `outputs/act/a1_vae/run_20000_01/closed_clean_n5/rollout.rrd`
- Modes: Closed-loop / Formal

## 2026-07-27 — PushT Diffusion Pilot 准备（计划，未运行）

- Date: 2026-07-27
- Task suite: PushT
- Dataset size: planned fixed split of 206 episodes — 165 train / 20 development /
  21 held-out audit; Smoke uses the first 8 train episodes only
- Observation keys: `observation.image`, `observation.state`; two-step history
- Action space: 2D absolute target position
- Action horizon: internal horizon 16; public execution slice `[1:9]`, 8 actions
- Control frequency: 10 Hz
- Policy: strict deterministic BCChunk baseline and local Diffusion Policy, paired
  step-zero initialization, train seed 1000
- Train loss: not available; training has not started
- Open-loop action error: not available; E1 has not run
- Closed-loop success rate: not available; E2/E3 have not run
- Latency: not available; E3 has not run
- Failure cases: no measured rollout failures yet
- Notes: Infrastructure and copy-ready commands are documented under
  `reports/guides/pusht_diffusion/`. This entry records planned work only and must not
  be interpreted as a Smoke or Formal result.
- Result artifacts: none yet
- Modes: Planned（未运行）

## 2026-07-28 — PushT BCChunk 与 Diffusion Formal 200k

- Date: 2026-07-28
- Task suite: PushT
- Dataset size: 165 train episodes, 20,610 frames；完整固定数据为 206 episodes
- Observation keys: `observation.image`, `observation.state`；两步 history
- Action space: 2D absolute target position
- Action horizon: internal `horizon=16`；执行 `[1:9]`，8 actions
- Control frequency: 10 Hz
- Policy: paired-initialized BCChunk and Diffusion；各 200,000 updates，train seed 1000
- Train loss: 最后日志值 BCChunk 0.000（仅三位小数，非精确零）；
  Diffusion epsilon loss 0.004；二者语义不同
- Open-loop action error: 本训练 run 未独立测量；在后续 checkpoint selection 中测量
- Closed-loop success rate: 本训练 run 未独立测量；在后续 development selection 中测量
- Latency: 未测量；等待 E3
- Failure cases: 两个训练任务均正常结束；未发生训练中断
- Notes: 213 个共享 step-zero tensor 已验证一致。训练 loss 不能跨策略直接比较，
  也不能单独用于选模。
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/bc_chunk/train.log`
  - `outputs/pusht_diffusion/pilot/diffusion/train.log`
  - `outputs/pusht_diffusion/initialization/pilot/seed_1000/initialization_manifest.json`
- Modes: Formal

## 2026-07-28 — PushT Development Checkpoint Selection

- Date: 2026-07-28
- Task suite: PushT
- Dataset size: checkpoints trained on 165 episodes；selection uses 20 development
  episodes and 20 committed environment seeds
- Observation keys: `observation.image`, `observation.state`；两步 history
- Action space: 2D absolute target position
- Action horizon: internal `horizon=16`；执行 `[1:9]`，8 actions
- Control frequency: 10 Hz
- Policy: BCChunk checkpoints 25k–200k；Diffusion checkpoints 25k–200k，
  DDPM-100，policy seed 0
- Train loss: 继承 Formal training；不作为 selection 排序项
- Open-loop action error: selected BCChunk-150k denormalized MSE 582.31；
  selected Diffusion-175k denormalized MSE 1169.87
- Closed-loop success rate: BCChunk-150k 11/20（55%）；Diffusion-175k
  16/20（80%）
- Latency: 未测量；等待 E3
- Failure cases: selected checkpoint 逐 seed 对比中，仅 BC 成功 1、仅 Diffusion
  成功 6、两者失败 3、两者成功 10
- Notes: exact paired McNemar `p=0.125`。同一 development seed 参与了选模，
  因而结果不是独立 final 泛化估计。最低 offline MSE 与最高闭环成功率不重合。
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/checkpoint_selection/metrics.md`
  - `outputs/pusht_diffusion/pilot/checkpoint_selection/selection.parquet`
  - `outputs/pusht_diffusion/pilot/development_comparison/metrics.md`
  - `outputs/pusht_diffusion/pilot/development_comparison/rollouts.parquet`
- Modes: Closed-loop / Formal

## 2026-07-28 — PushT E0 Full Demonstration Audit

- Date: 2026-07-28
- Task suite: PushT
- Dataset size: 206 demonstrations, 25,650 frames
- Observation keys: recorded PushT image/state trajectory and reconstructed block geometry
- Action space: 2D absolute target position
- Action horizon: E0 analyzes complete demonstrations；训练公共 chunk 为 8 actions
- Control frequency: 10 Hz
- Policy: N/A；dataset audit
- Train loss: N/A；未训练
- Open-loop action error: N/A；未执行模型预测
- Closed-loop success rate: N/A；未执行 rollout
- Latency: N/A；未测量
- Failure cases: 206 episodes 均无显式正 `next.success` label；只存在 terminal 和
  dense reward 信息
- Notes: 自动识别 33 个全局策略组，block geometry coverage 为 100%。
  全局分组不证明相同 observation 下的严格条件多模态。
- Result artifacts:
  - `outputs/pusht_diffusion/data_inspection_e0_full/metrics.md`
  - `outputs/pusht_diffusion/data_inspection_e0_full/metrics.json`
- Modes: Offline / Formal

## 2026-07-29 — PushT E1 Paired Fixed-observation Sampling

- Date: 2026-07-29
- Task suite: PushT
- Dataset size: selected policies trained on 165 episodes；E1 uses 10 held-out
  observation candidates and 20 Diffusion samples per protocol
- Observation keys: fixed two-frame `observation.image` and `observation.state` history
- Action space: 2D absolute target position
- Action horizon: 8-action public chunk `[1:9]`
- Control frequency: 10 Hz action semantics；E1 不执行环境
- Policy: selected BCChunk-150k and Diffusion-175k；DDPM-100
- Train loss: 继承 Formal training；E1 不训练
- Open-loop action error: E1a denormalized trajectory RMS 0.00659 px、endpoint
  0.00750 px；E1b 为 1.94563 px、2.80300 px
- Closed-loop success rate: 未测量；E1 的 `num_rollouts=0`
- Latency: 未测量；等待 E3
- Failure cases: 未执行闭环。BCChunk、E1a、E1b 的相同条件复现误差均为 0；
  固定 observation action-mode 探索分析中 E1b 为 0/10 candidate 通过
- Notes: E1a 只改变 initial noise、固定 reverse seed；E1b 改变完整 sampling
  seed。E1b/E1a trajectory RMS 约 295 倍、endpoint distance 约 374 倍，
  说明完整 reverse sampling stream 是当前 action 多样性的主要来源。复用现有
  20 samples 的 GMM/BIC 与完整 chunk 距离聚类交叉验证得到 E1a 0/10、E1b 0/10，
  BC averaging-consistent signature 0/10；因此未获得固定 observation
  action-distribution 多峰证据，但小样本阴性结果不证明真实分布必然单峰。
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/metrics.md`
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/action_chunks.parquet`
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/diversity_metrics.parquet`
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis/metrics.json`
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis/candidate_summary.parquet`
  - `outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis/figures/`
- Modes: Offline / Formal

## 2026-07-29 — PushT E2 Dual Closed-loop Sampling Rollout

- Date: 2026-07-29
- Task suite: PushT
- Dataset size: selected policies trained on 165 episodes；E2 uses 20 committed final
  environment seeds
- Observation keys: 96×96 RGB and 2D agent position；two-step history
- Action space: 2D absolute target position
- Action horizon: internal horizon 16；每次闭环执行 `[1:9]` 共 8 actions
- Control frequency: 10 Hz
- Policy: selected BCChunk-150k and Diffusion-175k；DDPM-100
- Train loss: 继承 Formal training；最后日志值 BCChunk 0.000（三位小数）、
  Diffusion epsilon loss 0.004
- Open-loop action error: 继承 checkpoint selection；BCChunk denormalized MSE
  582.31，Diffusion 1169.87
- Closed-loop success rate: E2b BCChunk 5/20（25%）；Diffusion `policy_seed=0`
  11/20（55%）；全部 Diffusion samples 105/200（52.5%）；any@10 为 18/20（90%，
  oracle coverage）
- Latency: E2b per-chunk median/p95：BCChunk 9.35/10.98 ms；Diffusion
  585.79/601.55 ms。E2 不作最终 speed benchmark，正式速度结论由 E3 给出
- Failure cases: E2b 有 2 个环境的 10 条 Diffusion rollout 全部失败；失败轨迹
  diversity 0.18217，高于成功轨迹 0.11576
- Notes: E2a 使用 2×5、只改变 initial noise；env 2000 为 0/5、轨迹被明显放大，
  env 2001 为 5/5、轨迹几乎重合，因此没有 success 翻转或多成功 mode 证据。E2b
  使用 20×10 完整 sampling stream；主阈值 0.05 下 16/20 个环境至少有 2 个成功
  mode，在阈值 0.025 和 0.075 下仍为 16/20。该结果支持多个成功闭环行为 mode，
  但结合 E1 action-mode 阴性结果后，不能声称已经证明固定 observation 的
  \(p(a\mid o)\) 多峰或 DP 克服了 MSE mode averaging
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity/metrics.md`
  - `outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/metrics.md`
  - `outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/per_environment_metrics.parquet`
  - `outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/mode_analysis.json`
  - `outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/recordings/`
- Modes: Closed-loop / Formal

## 2026-07-29 — PushT E4 Rotation-only DP OOD Diagnostic

- Date: 2026-07-29
- Task suite: custom PushT E4 center-aligned rotation-only scene
- Dataset size: E4 dataset 0; evaluated DP-175000 was trained on 165 original PushT
  train episodes
- Observation keys: 96×96 RGB and 2D agent position; two-step history
- Action space: 2D absolute target position
- Action horizon: internal horizon 16; executes `[1:9]`, 8 actions per replan
- Control frequency: 10 Hz
- Policy: selected Diffusion-175k; DDPM-100; full sampling policy seeds 0–4
- Train loss: inherited from original PushT training; E4 did not train
- Open-loop action error: not measured
- Closed-loop success rate: 0/5; OOD diagnostic only, not an E4 performance estimate
- Latency: not aggregated in this diagnostic
- Failure cases: all five rollouts reached the 300-step time limit; per-seed max reward
  was 0.4597, 0.5515, 0.4612, 0.5858, and 0.5836
- Notes: all rollouts used the same deterministic E4 initial state and independent
  full-DDPM sampling streams. Results show that the unadapted original PushT policy did
  not directly solve E4; they do not test whether E4 training can learn CW/CCW modes.
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/metrics.json`
  - `outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/rollouts.json`
  - `outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/recordings/`
  - `outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/videos/`
- Modes: Closed-loop / OOD diagnostic

## 2026-07-29 — PushT E4 Contact-offset OOD Diagnostic

- Date: 2026-07-29
- Task suite: custom PushT E4 lower-left-to-center 180-degree scene
- Dataset size: E4 dataset 0; evaluated Diffusion-175k was trained on 165 original
  PushT train episodes; reference screening used 206 original demonstrations
- Observation keys: 96×96 RGB and 2D agent position; two-step history
- Action space: 2D absolute target position in `[0,512]`
- Action horizon: internal horizon 16; executes `[1:9]`, 8 actions per replan
- Control frequency: 10 Hz
- Policy: selected Diffusion-175k; DDPM-100; full sampling seeds 0–4
- Train loss: inherited from original PushT training; E4 did not train
- Open-loop action error: not measured
- Closed-loop success rate: left-up `0/5`, center `0/5`, right-down `0/5`; OOD
  diagnostic only
- Latency: not aggregated in this diagnostic
- Failure cases: all 15 rollouts reached the 300-step time limit. Mean max reward was
  0.1595/0.2266/0.5664 for left-up/center/right-down; right-down seeds 0/1/3 reached
  0.8357/0.8441/0.6671 but remained below the 0.95 success threshold
- Notes: moving the initial contact point changed overlap quality substantially but did
  not produce a rollout that approached from the opposite side. This supports contact
  geometry sensitivity, not fixed-observation left/right multimodality. Offline screening
  selected left references 110/132/151 and right references 171/134/90, but they are not
  paired E4 demonstrations
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/e4_contact_offset_probe/metrics.json`
  - `outputs/pusht_diffusion/pilot/e4_contact_offset_probe/reference_demonstrations.json`
  - `outputs/pusht_diffusion/pilot/e4_contact_offset_probe/*/recordings/`
  - `outputs/pusht_diffusion/pilot/e4_contact_offset_probe/*/videos/`
- Modes: Closed-loop / OOD diagnostic; reference screening is Offline / diagnostic

## 2026-07-30 — PushT E5 Prototype-ID Conditioning Formal Evaluation

- Date: 2026-07-30
- Task suite: PushT
- Dataset size: 206 episodes；165 train / 20 development / 21 audit
- Observation keys: 96×96 RGB and 2D agent position；two-step history
- Action space: 2D absolute target position
- Action horizon: internal horizon 16；closed loop executes `[1:9]`, 8 actions per replan
- Control frequency: 10 Hz
- Policy: selected Mode-BCChunk-175000 and Mode-Diffusion-125000；
  `nn.Embedding(K+1, 32)`；20% train-only NULL dropout；DP uses DDPM-100
- Train loss: selected-step logs report BC `0.000` at displayed precision；DP approximately
  `0.007–0.009`
- Open-loop action error: held-out normalized/denormalized MSE is 0.01333/804.90 for BC and
  0.01315/793.59 for DP
- Closed-loop success rate: oracle subset BC 7/21 (33.3%), DP 12/21 (57.1%). Across all
  tasks, BC P0/P1/P2/NULL is 23.8%/38.1%/38.1%/52.4%; DP is
  47.6%/52.4%/33.3%/52.4%
- Latency: closed-loop replan p95 is 10.22 ms for BC and 608.93 ms for DP DDPM-100
- Failure cases: requested and achieved prototypes frequently disagree；15/17 of BC and
  12/15 of DP unclassifiable conditional rollouts have valid sketches but fall outside all
  prototype radii；the remaining cases have ineligible sketches
- Notes: development selection and the 168-rollout held-out all-condition audit are
  complete. DP P0/P1/P2 adherence is only 14.3%/28.6%/19.0%; BC is
  19.0%/23.8%/19.0%. Fixed-noise diagnostics verify that condition IDs enter the model and
  change action chunks, but requested→achieved matrices are not diagonal. The formal
  conclusion is that current E5 preserves task ability but does not establish semantic
  prototype controllability. Training data contain one demonstration/prototype per initial
  state and no paired counterfactual supervision.
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/metrics.json`
  - `outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json`
  - `outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/leakage_audit.json`
  - `outputs/pusht_diffusion/pilot/e5_oracle_checkpoint_selection/`
  - `outputs/pusht_diffusion/pilot/e5_oracle_audit/`
- Pretrain / Provenance: random initialization with `train_seed=1000`, no external
  pretrained checkpoint (`verified=true`); dataset/prototype/weight identity verified
  across artifacts；runtime source commit unavailable (`verified=false`)
- Modes: Offline / Formal prototype audit；Closed-loop / Formal held-out evaluation；
  fixed-noise condition-response diagnostic

## 2026-07-31 — PushT E6 7×7 Route Field Training Pipeline Smoke

- Date: 2026-07-31
- Task suite: PushT
- Dataset size: 206 episodes total；RF7 train-only route mining yields 113 canonical
  macro-route families；all 165 train episodes、20,610 frames and 20,610 sampleable indices
  remain available；fixed Smoke uses all 8 configured episodes and 793 frames
- Observation keys: 96×96 RGB and 2D agent position；two-step history；additional
  `[4,7,7]` route field
- Action space: 2D absolute target position
- Action horizon: internal horizon 16；closed-loop execution contract remains `[1:9]`,
  8 actions per replan
- Control frequency: 10 Hz
- Policy: RouteField-BCChunk and RouteField-Diffusion；7×7 occupancy/progress/dx/dy
  field encoded to a 64D global condition；no NULL or condition dropout
- Train loss: one-step pipeline Smoke reports BC `0.855` and DP epsilon loss `0.916`;
  neither is a convergence result
- Open-loop action error: not measured；checkpoint selection and route-switch evaluation
  have not run
- Closed-loop success rate: not measured
- Latency: not measured；single optimizer updates took approximately 3.0 seconds in the
  local Smoke and are not inference benchmarks
- Failure cases: the first RF7 real Smoke launch exposed a stale schema-v1 assertion in
  the shared training validator；the validator now uses the RF7 schema constant and a
  regression test covers the real call path. The data remain unpaired: Route A/B are not
  demonstrations from exactly the same initial state, so controllability is unverified
- Notes: single-cell and singleton-family routes remain trainable；exact-route support
  never controls eligibility. To make the fixed field representable, immediate backtracks
  and residual closed loops are simplified only in the condition: 44/165 train episodes
  are affected and 110 route cells are removed, while no episode、frame or sampleable index
  is removed. The primary A/B diagnostic pair is `16→17→24` (support 3) versus
  `16→23→24` (support 2). Both saved Smoke checkpoints were reloaded successfully and then
  deleted as temporary generated artifacts. This is pipeline validation, not a formal
  E6 result.
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json`
  - `outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/episode_assignments.parquet`
  - `outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_fields.npz`
- Modes: Offline route mining / Smoke training pipeline

## 2026-07-30 — SmolVLA LIBERO Native LoRA Two-step Smoke

- Date: 2026-07-30
- Task suite: `libero_goal` tasks 1/4/8; one demonstration per target task
- Dataset size: 3 episodes, 315 frames
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state`
- Action space: 7D LIBERO relative end-effector action
- Action horizon: `chunk_size=50`, `n_action_steps=10`
- Control frequency: dataset 10 Hz; no closed-loop execution
- Policy: `lerobot/smolvla_base`, LoRA rank 32, BF16, batch size 1, seed 1000
- Train loss: step 1 `0.252`; step 2 `0.814`; both finite
- Open-loop action error: N/A; two-step optimizer-path Smoke only
- Closed-loop success rate: N/A; no environment rollout
- Latency: optimizer update time 0.656/0.111 seconds for steps 1/2; not a benchmark
- Failure cases: the first launch stopped before model loading because local Smoke omitted
  `--policy.push_to_hub=false`; the corrected launch completed. The saved base config retains
  a 6D state declaration while dataset samples and normalization statistics are 8D.
  SmolVLA consumes the runtime state through dynamic padding, but closed-loop processor
  compatibility remains to be checked.
- Notes: PEFT wrapped successfully with 1,485,312 trainable parameters out of 451,531,488.
  The step-2 adapter checkpoint is 18 MB including training state and reloaded successfully
  as an inference-mode `PeftModel` with a 7D action output. This is a native training Smoke
  and does not satisfy the planned instrumented VRAM/gradient audit or C0 closed-loop
  acceptance gate.
- Result artifacts:
  - `outputs/smolvla_libero/smoke/lora_20260730_010647/metrics.json`
  - `outputs/smolvla_libero/smoke/lora_20260730_010647/checkpoints/000002/pretrained_model/train_config.json`
  - `outputs/smolvla_libero/smoke/lora_20260730_010647/checkpoints/000002/pretrained_model/adapter_config.json`
  - `outputs/smolvla_libero/smoke/lora_20260730_010647/checkpoints/000002/pretrained_model/adapter_model.safetensors`
- Modes: Smoke

## 2026-07-30 — Public SmolVLA-LIBERO Closed-loop Smoke

- Date: 2026-07-30
- Task suite: `libero_goal`, task 1, `put the bowl on the stove`
- Dataset size: evaluated public checkpoint `HuggingFaceVLA/smolvla_libero`; local training
  data was not used
- Observation keys: `observation.images.image`, `observation.images.image2`,
  `observation.state` (8D)
- Action space: 7D LIBERO relative end-effector action
- Action horizon: `chunk_size=50`, `n_action_steps=1`
- Control frequency: environment configured at 30 Hz; evaluation ran as fast as possible,
  not wall-clock real time
- Policy: `HuggingFaceVLA/smolvla_libero`; 10 flow-matching inference steps; seed 1000;
  fixed LIBERO initial state
- Train loss: N/A; public pretrained checkpoint was evaluated without training
- Open-loop action error: N/A; no dataset-side evaluation in this Smoke
- Closed-loop success rate: 1/1 (100%); task completed at control step 90
- Latency: 31.12 seconds for the episode, approximately 0.346 seconds per environment
  step including policy inference, simulation, and recording; not policy-only latency
- Failure cases: none observed in the single episode
- Notes: This run verifies native compatibility between the public checkpoint and the
  current LeRobot LIBERO environment: two 256×256 image inputs, 8D state, and 7D relative
  action all matched without an adapter. A single successful episode is only a pipeline
  and visualization Smoke, not a formal success-rate estimate.
- Result artifacts:
  - `outputs/smolvla_libero/smoke/hf_libero_goal_task1_20260730_013110_http/eval_info.json`
  - `outputs/smolvla_libero/smoke/hf_libero_goal_task1_20260730_013110_http/videos/libero_goal_1/eval_episode_0.mp4`
  - `outputs/smolvla_libero/smoke/hf_libero_goal_task1_20260730_013110_http/videos/libero_goal_1/contact_sheet.png`
- Modes: Closed-loop / Smoke

## 2026-08-03 — PushT E6 RF7 Route Field Held-out Audit

- Date: 2026-08-03
- Task suite: PushT，`lerobot/pusht` revision `7628202a…`
- Dataset size: 206 episodes；165 train / 20 development / 21 audit；训练 20610 帧
- Observation keys: 96×96 RGB、二维 agent position（历史长度 2）；外加 4×7×7 route field
  （occupancy / progress / direction_x / direction_y），整条 episode 锁存不变
- Action space: 二维绝对 target position
- Action horizon: 内部 16，闭环执行 8 步（300 步 = 38 次 replan）
- Control frequency: 10 Hz
- Policy: `route_field_diffusion`，选中 step 100000（development 8 个 checkpoint 中
  成功率最高的 0.20）；DDPM full sampling 100 步；`train_seed=1000`；`policy_seed=0`；
  初始状态为 dataset 第一帧重建
- Train loss: 200k step 处 0.003（约 621 epoch）
- Open-loop action error: normalized 0.04886 / denormalized 2991.26
- Closed-loop success rate: oracle 1/21（4.8%，Wilson 95% [0.85%, 22.7%]）；
  reversed 2/19、permuted 0/5 属反号服从度读数，非性能指标
- Latency: replan 均值 597.2 ms、p95 607.9 ms
- Failure cases: 20/21 条 oracle rollout timeout；`obeyed_rate` 0/19 与 0/5；
  逆序配对中 13/19 因路线过短而结构上不可判定；permuted mean max reward 仅 0.2782
- Notes: 45/45 条闭环 rollout 完成，21/21 首帧重建通过。离线闸门显示条件确实进入模型
  （固定 observation + 固定噪声下逆序使 action chunk 移动 113.6 px、末端 54.1 px），
  因此负结论是"读到了但没当成指令"。原始产物两处指标缺陷已修正并从同一批 rollout
  重导出：`final_rotation_error_deg` 参考角由 0 rad 改为目标块角 π/4（oracle 均值
  102.192° → 89.503°），并补充服从度可判定性口径；两项修正均不改变冻结判据结论。
  引用旋转误差须使用 `_corrected` 产物。audit run 的 repository commit 为 null，
  code provenance 未验证。
- Result artifacts:
  - `outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection/metrics.json`
  - `outputs/pusht_diffusion/pilot/route_field_condition_gate/metrics.json`
  - `outputs/pusht_diffusion/pilot/route_field_held_out_eval/route_field_held_out_eval/metrics.json`
  - `outputs/pusht_diffusion/pilot/route_field_held_out_eval_corrected/metrics.json`
  - `reports/2026-08-03-RF7 路线条件遵循正式评测结论.md`
- Modes: Closed-loop / Formal（闸门与重导出为 Offline）

## 2026-07-31 — SmolVLA LIBERO B0 37-task Domain Adaptation

- Date: 2026-07-31（训练与选模）/ 2026-08-01（held-out test）
- Task suite: LIBERO-Goal 背景 37 任务（`lerobot/libero`，排除目标任务全局
  `task_index` 10/17/18，即仿真侧 libero_goal 本地 8/1/4）
- Dataset size: 1,412 train episodes / 236,470 frames；另有 138 / 23,004 作离线 validation
- Observation keys: 两路 256×256 RGB（`observation.images.image`、`image2`）+ 8 维
  `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`
- Control frequency: 10 Hz
- Policy: 从本地 materialized `smolvla_base` 出发；`freeze_vision_encoder=True`、
  `train_expert_only=True`、`train_state_proj=True`、PEFT 关闭；8 卡 × batch 8 = global 64；
  BF16；AdamW LR 1e-4 / betas (0.9,0.95) / eps 1e-8 / wd 1e-10 / grad clip 10；
  warmup 1,000 + cosine 30,000 → 2.5e-6；train seed 1000
- Train loss: 30,000 updates，首步 0.607，末 1,000 步均值 0.064；loss/grad norm/LR 全程
  有限，LR 峰值 9.990e-5 出现在 step 999，step 30000 为 2.5e-6
- Open-loop action error: 未评测（本阶段不做离线动作误差）
- Closed-loop success rate: background validation 选模 840 条 rollout，winner `025000`
  macro success 0.8643（121/140）；selected B0 在保留的 test init states 上 210 条 rollout，
  macro success 0.8714（183/210）
- Latency: test 集 macro 模型查询延迟 p50 253.2 ms、p95 348.9 ms
- Failure cases: test 失败 27 条全部为 `time_limit`；最弱任务是 libero_goal 本地 3
  “open the top drawer and put the bowl inside”，test 0.50（15/30），选模阶段亦最弱（10/20）
- Notes: 选模用 6 checkpoint × 7 任务 × 20 validation init states；每任务 20/30
  validation/test init states 互斥且合并覆盖官方 0..49。25000 与 30000 主指标并列，按注册
  顺序由成功 episode 平均步数分出 25000（120.85 < 122.21）；配对诊断 25k→30k exact
  McNemar p=1.0、净增益 0，未改动 winner。第一个 tie-breaker `mean_max_reward_desc` 在
  LIBERO 稀疏 0/1 奖励下恒等于成功率，实际不起作用。**held-out 只针对 init state**：这 7 个
  任务都在 B0 的 37 个训练任务内，结果为同任务未见初始状态表现，不是跨任务泛化。
  49 个 job 的 `task.language` 与 split `canonical` 逐条一致，确认两套 task 编号命名空间
  没有串用。
- Result artifacts:
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/run_config.json`
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/train_metrics.jsonl`
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection/checkpoint_results.json`
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection/selected_checkpoint.json`
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection/validation_paired_analysis.json`
  - `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/background_test/test_results.json`
  - `reports/guides/smolvla_libero/02_b0_domain_adaptation.md`
- Modes: Formal / Offline（训练）；Formal / Closed-loop / Validation（选模）；
  Formal / Closed-loop / Test（held-out test）

## 2026-08-04 — SmolVLA LIBERO Z0 Baseline and C1 Target Protocol Test

- Date: 2026-08-04
- Task suite: LIBERO-Goal 三个目标任务（本地编号 1 / 4 / 8；`lerobot/libero` 全局
  `task_index` 17 / 18 / 10）
- Dataset size: C1 训练 113 条 demonstration / 11,022 帧；离线 validation 30 条 / 2,969 帧。
  Z0 不训练
- Observation keys: 两路 256×256 RGB（`observation.images.image`、`image2`）+ 8 维
  `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`
- Control frequency: 10 Hz
- Policy: Z0 = selected B0 `025000` 原样零样本；C1 = 从同一 selected B0 出发训练 30,000 步后
  选出的 `030000`。C1 训练为 `data_parallel_layout=8x1`（全局 batch 8）、BF16、
  AdamW LR 1e-4 / betas (0.9,0.95) / eps 1e-8 / wd 1e-10 / clip 10、warmup 1,000 +
  cosine 30,000 → 2.5e-6、seed 1000；`freeze_vision_encoder=True`、`train_expert_only=True`、
  PEFT 关闭
- Train loss: C1 30,000 updates，首步 0.1599、末 1,000 步均值 0.01739；grad norm 峰值 9.42；
  8 个 rank 状态均为 `complete`。Z0 无训练
- Open-loop action error: C1 选中 checkpoint 归一化 0.309668 / 去归一化 0.040252（30 条离线
  validation demonstration 的 2,969 帧，固定 flow-noise）。Z0 未评测
- Closed-loop success rate: C1 validation macro 0.9833（59/60）。test 三套协议——
  canonical Z0 0/90 对 C1 87/90（0.0% 对 96.7%）；held-out paraphrase Z0 2/90 对 C1 30/90
  （2.2% 对 33.3%）；Prompt Swap 首次达成 Z0 0/30 对 C1 29/30（0.0% 对 96.7%），
  C1 终态保持 27/30（90.0%，仅作保持性诊断）
- Latency: 模型查询 p50 / p95 —— Z0 310.1 / 471.3 ms，C1 285.3 / 443.2 ms（各 21 个 job 均值）
- Failure cases: Z0 canonical 90 条全部 `time_limit`，Prompt Swap 30 条全部未首次达成；
  C1 canonical 3 条 `time_limit`、held-out paraphrase 60 条 `time_limit`；Prompt Swap 1 条
  `no_goal_reached`（stove 指令 1 条未首次达成；另有 2 条 plate 在终态保持阶段丢失目标）
- Notes: 两次评测用同一冻结协议，`contract` 中仅 `model_label` / `policy_path` /
  `policy_config_sha256` / `jobs`（只差 `output_dir` 与 `log_path`）不同，init states、
  episode seeds、指令文本完全一致，因此可比。C1 选模中三个 checkpoint 在 0.9833 并列，由第二
  判据（macro 归一化开环 MSE）分出 `030000`。`para_gap`（canonical 减 held-out paraphrase）
  点估计 +0.6333、90 个配对 case 上 bootstrap 95% CI `[0.5222, 0.7333]`，不含 0。
  **Prompt Swap 的首次达成并非均匀**：cabinet 10/10、plate 10/10、stove 9/10；终态保持为
  cabinet 10/10、stove 9/10、plate 8/10。旧的 18/30 与 stove 0/10 来自判定 reset 问题，
  不再作为当前模型结论。
  同一份 B0 权重以 `checkpoints/025000` 与 `checkpoints/selected` 两个路径各跑一次
  （sha256 同为 `811bd1a7ba06c32b…`），四个指标块逐值相同，附带验证了确定性。
  test 的 30 个初始状态与训练摆放同属约 3 cm 范围的同一分布，衡量的是同任务未被使用过的
  摆放，不是远距离泛化。未记录评测机器源码 commit。
- Result artifacts:
  - `outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection/checkpoint_results.json`
  - `outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection/selected_checkpoint.json`
  - `outputs/smolvla_libero/eval/z0_b0_zero_shot/metrics.json`
  - `outputs/smolvla_libero/eval/c1_default_non_peft/metrics.json`
  - `outputs/smolvla_libero/eval/c1_default_non_peft/prompt_goal_confusion.csv`
  - `reports/2026-08-04-Z0基线与C1目标适配正式评测结论.md`
- Modes: Formal / Offline（C1 训练）；Formal / Closed-loop / Validation（C1 选模）；
  Formal / Closed-loop / Test（Z0 与 C1 的 210 条协议评测）

## 2026-08-05 — SmolVLA LIBERO C2 Canonical LoRA Target and Background Test

- Date: 2026-08-05
- Task suite: LIBERO-Goal 三个目标任务（本地编号 1 / 4 / 8）用于目标协议；7 个背景任务
  （本地编号 0 / 2 / 3 / 5 / 6 / 7 / 9）用于回退评测
- Dataset size: 训练 113 条 demonstration / 11,022 帧；离线 validation 30 条 / 2,969 帧
- Observation keys: 两路 256×256 RGB + 8 维 `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`
- Control frequency: 10 Hz
- Policy: 从 selected B0 `025000` 出发的 LoRA 适配，rank 32 / alpha 8 / dropout 0，target 覆盖
  `lm_expert` 的 q/v 投影与 `state_proj`、`action_in_proj`、`action_out_proj`、
  `action_time_mlp_{in,out}`；`data_parallel_layout=8x1`（全局 batch 8）、BF16、30,000 步、
  每 5,000 步保存；AdamW LR 1e-4 / betas (0.9,0.95) / eps 1e-8 / wd 1e-10 / clip 10；
  warmup 1,000 + cosine 30,000 → 2.5e-6；seed 1000。选中 checkpoint 为 `020000`
- Train loss: 30,000 updates，首步 0.1568、末 1,000 步均值 0.06798（C1 同项 0.01739）；
  grad norm 峰值 0.64（C1 为 9.42）；8 个 rank 状态均为 `complete`
- Open-loop action error: 选中 checkpoint 归一化 0.371719（C1 为 0.309668）
- Closed-loop success rate: validation macro 0.9667（58/60）。目标协议 test——canonical
  87/90 = 96.7%（与 C1 相同）、held-out paraphrase 19/90 = 21.1%（C1 为 33.3%）、
  Prompt Swap 首次达成 28/30 = 93.3%（C1 为 29/30 = 96.7%），终态保持 28/30 = 93.3%。
  背景回退 test——8/210 = 3.8%
  （C1 为 0/210，B0 记录为 183/210）
- Latency: 模型查询 p50 376.8 ms、p95 541.4 ms（21 个 job 均值；C1 为 285.3 / 443.2 ms）
- Failure cases: canonical 3 条 `time_limit`；held-out paraphrase 71 条 `time_limit`；
  Prompt Swap 2 条 `no_goal_reached`（均为 plate 指令首次未达成，终态保持同为 28/30）；
  背景任务 202 条 `time_limit`，七个背景任务中四个为 0/30
- Notes: 目标协议 contract 与 C1、Z0 仅差 `model_label` / `policy_path` /
  `policy_config_sha256` / `jobs`（后者只差输出路径），其余 sha256 与 seed 全部相同，因此三者
  可比。选模中三个 checkpoint 在 0.9667 并列，由第二判据（macro 归一化开环 MSE）分出 `020000`；
  C2 的最优 step 是 20000 而非终点，且 005000 就已达 0.9667（C1 同期 0.8833），LoRA 收敛更快。
  **两条负结果**：一是语言泛化更差，`para_gap` 从 C1 的 +0.6333 升到 +0.7556，bootstrap 95% CI
  `[0.6667, 0.8444]` 与 C1 的 `[0.5222, 0.7333]` 不重叠；二是 LoRA 未防住灾难性遗忘，背景
  8/210 相对 B0 的保留率仅 4.4%，因此 C1 guide 的 TODO-2 假设验证为负。在数据、LR、步数、
  batch、seed 全部相同的对照下，遗忘程度对参数化方式不敏感，指向根因是训练数据里没有背景任务。
  C2 的推理延迟与动作抖动均高于 C1（jerk 0.0588 对 0.0285，gripper 切换 10.52 对 3.76）。
  B0 背景基线 183/210 取自文档记录，原始产物已丢失且未重跑，故只能做非配对比较。
  未记录评测机器源码 commit。
- Result artifacts:
  - `outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoint_selection/checkpoint_results.json`
  - `outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoint_selection/selected_checkpoint.json`
  - `outputs/smolvla_libero/eval/c2_canonical_lora/metrics.json`
  - `outputs/smolvla_libero/eval/c2_canonical_lora/prompt_goal_confusion.csv`
  - `outputs/smolvla_libero/eval/background_regression/c2_selected/test_results.json`
  - `reports/2026-08-05-C2 Canonical LoRA 正式评测结论.md`
- Modes: Formal / Offline（训练）；Formal / Closed-loop / Validation（选模）；
  Formal / Closed-loop / Test（目标协议 210 条与背景回退 210 条）

<!-- PROMPT_SWAP_MERGE_LOG:BEGIN -->
## 2026-08-06 — SmolVLA LIBERO Prompt Swap 判定修复后合并

- Date: 2026-08-06
- Task suite: LIBERO-Goal Prompt Swap reference task 1，stove/cabinet/plate 三目标
- Mode: Formal / Closed-loop / Test（只补跑 Prompt Swap；canonical 与 held-out paraphrase 沿用原始结果）
- 判定口径: `first_goal_reached` 为主指标，`terminal_goal` 为保持性诊断
- Results:
  - z0_b0_zero_shot: first-goal 0/30 (0.0%); terminal-goal 0/30 (0.0%); full horizon 30/30
  - c1_default_non_peft: first-goal 29/30 (96.7%); terminal-goal 27/30 (90.0%); full horizon 30/30
  - c2_canonical_lora: first-goal 28/30 (93.3%); terminal-goal 28/30 (93.3%); full horizon 30/30
- Notes: 原始 Prompt Swap 旧值因参考任务成功后 reset 而偏低；合并脚本保留原始 180 条非 Prompt Swap 数据，并将 corrected job/rollout/episodes/metrics 写回原始主题结果目录。
- Result artifacts:
  - `outputs/smolvla_libero/eval/z0_b0_zero_shot`
  - `outputs/smolvla_libero/eval/c1_default_non_peft`
  - `outputs/smolvla_libero/eval/c2_canonical_lora`
- Consolidated report: `reports/2026-08-06-PromptSwap修复后合并结果.md`
<!-- PROMPT_SWAP_MERGE_LOG:END -->

## 2026-08-06 — SmolVLA C1 学习率缩放遗忘权衡诊断

- Date: 2026-08-06
- Task suite: LIBERO-Goal 目标任务 1/4/8 的 checkpoint validation；背景任务 0/2/3/5/6/7/9 的 held-out test
- Dataset size: 目标训练 113 episodes / 11,022 frames；offline validation 30 episodes / 2,969 frames；每个背景曲线点 210 条 rollout
- Observation keys: 两路 256×256 RGB（`observation.images.image`、`observation.images.image2`）+ 8 维 `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`，每条 episode 最多 300 steps
- Control frequency: 10 Hz
- Train loss: A（LR `1e-4`）首步 0.159907、末 1,000 步均值 0.017388；B（LR `1.25e-5`）首步 0.159907、末 1,000 步均值 0.066107；两者均完成 30,000 步
- Open-loop action error: A step 30,000 normalized MSE 0.309668；B selected step 10,000 normalized MSE 0.381604
- Closed-loop success rate: A step 30,000 目标 59/60（98.3%）/ 背景 0/210；B step 10,000 目标 58/60（96.7%）/ 背景 55/210（26.2%）
- Latency: B selected 目标 validation model-query p50 299.3–316.5 ms、p95 437.9–486.4 ms；背景 test p50 271.2 ms、p95 376.6 ms
- Failure cases: B selected 目标 validation 2 条 `time_limit`；背景 test 155 条 `time_limit`，task 3 和 task 6 均为 0/30
- Conclusion: 按 batch 缩放 LR 能部分缓解灾难性遗忘；B 的最佳目标/背景权衡点为 step 10,000，但仍远低于 B0 记录的 183/210。该 checkpoint 仍是 diagnostic，不进入正式 C1 结果或正式 210 条目标协议。
- Result artifacts: `outputs/smolvla_libero/diagnostic/c1_a_same_config/`、`outputs/smolvla_libero/diagnostic/c1_b_scaled_lr/`
- Report: `reports/2026-08-06-C1学习率缩放遗忘权衡诊断结论.md`
- Modes: Diagnostic / Offline（训练）；Diagnostic / Closed-loop / Validation（target selection）；Diagnostic / Closed-loop / Test（background retention）

## 2026-08-06 — SmolVLA C1 Background Replay k=4 Diagnostic

- Date: 2026-08-06
- Task suite: LIBERO-Goal 目标任务 1/4/8 的 checkpoint validation；背景任务 0/2/3/5/6/7/9 的 held-out test
- Dataset size: 目标训练 113 episodes / 11,022 frames；background replay 1,412 episodes / 236,470 frames；混合训练 1,525 episodes / 247,492 frames；offline validation 30 episodes / 2,969 frames；background test 210 条 rollout
- Observation keys: 两路 256×256 RGB（`observation.images.image`、`observation.images.image2`）+ 8 维 `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`，每条 episode 最多 300 steps
- Control frequency: 10 Hz
- Policy: selected B0 初始化；Non-PEFT；global batch 8；BF16；seed 1000；LR `1e-4`；`background_replay_k=4`，每个 batch 50% background
- Train loss: 30,000 updates 完成；末 1,000 步 loss 均值 0.055969
- Open-loop action error: selected step 20,000 normalized MSE 0.337981 / denormalized MSE 0.043326
- Closed-loop success rate: target validation 59/60（98.3%）；background test 149/210（71.0%）；相对 B0 记录的 183/210 保留率为 81.4%
- Latency: background test model-query p50 265.8 ms、p95 375.6 ms
- Failure cases: background test 61 条 `time_limit`；task 9 为 14/30，是主要剩余短板
- Notes: `checkpoint_selection/selected_checkpoint.json` 将 selected checkpoint 固定为 step 20,000；direct-policy background runner 的 `Selected checkpoint: 000000` 是占位值。由于 replay 训练见过 background demonstration，该 background 分数是混合训练后的能力/保持结果，不是纯粹的遗忘读数。
- Conclusion: 在 target 能力保持 59/60 的同时，background 从 target-only C1 的 0/210、降 LR B 的 55/210 提升到 149/210；数据回放比单独降低 LR 更有效，但仍未恢复到 B0 的 183/210。
- Result artifacts: `outputs/smolvla_libero/diagnostic/c1_replay_k4/`
- Report: `reports/2026-08-06-C1遗忘诊断与背景数据回放结论.md`
- Modes: Diagnostic / Offline（训练）；Diagnostic / Closed-loop / Validation（目标选模）；Diagnostic / Closed-loop / Test（背景保留）

## 2026-08-07 — SmolVLA LIBERO C3 LoRA + Paraphrase 正式结论

- Date: 2026-08-07（结果汇总；训练于 2026-08-04 完成 30,000 updates，目标协议于 2026-08-07 完成）
- Task suite: LIBERO-Goal 目标任务 1 / 4 / 8；Prompt Swap reference task 1 的 stove/cabinet/plate 三目标
- Dataset size: 训练 113 条 demonstration / 11,022 帧；offline validation 30 条 / 2,969 帧；checkpoint selection 360 条 validation rollout；最终 target protocol 210 条 rollout
- Observation keys: 两路 256×256 RGB（`observation.images.image`、`observation.images.image2`）+ 8 维 `observation.state`
- Action space: 7 维相对末端动作
- Action horizon: `chunk_size=50`，执行 `n_action_steps=10`
- Control frequency: 10 Hz；正式 episode 最多 300 steps
- Policy: selected B0 `025000` 初始化；rank-32 LoRA；25% canonical + 75% train paraphrase；BF16；`data_parallel_layout=8x1`；global batch 8；train seed 1000；test policy seed 2000
- Train loss: step 1 为 0.156775；selected step 15,000 前 1,000 步均值 0.073442；最后 1,000 步均值 0.070925；step 30,000 最后记录 0.066038；grad norm 峰值 0.662
- Open-loop action error: selected `015000` normalized MSE 0.383028；denormalized MSE 0.051588
- Closed-loop success rate: target validation winner 57/60（95.0%）；test canonical 88/90（97.8%）；held-out paraphrase 67/90（74.4%）；Prompt Swap 首次达成 30/30（100.0%）；terminal hold 29/30（96.7%）
- Latency: 21 个 target protocol job 的 job-level 平均模型查询 p50 331.6 ms、p95 479.5 ms；平均 action TV 0.0392、action jerk 0.0496、gripper toggle 3.72
- Failure cases: canonical 2 条 `time_limit`；held-out paraphrase 23 条 `time_limit`，主要集中在 task 1 prompt 0 和 task 8 prompt 1；Prompt Swap 首次达成无失败，1 条 plate episode 在 terminal hold 阶段丢失目标
- Notes: C3 相对 C2 的 held-out paraphrase 从 19/90 提升到 67/90（+53.3 个百分点），三个任务均改善；canonical 仅从 87/90 变为 88/90。选模的 `015000` 与 `025000` 都是 57/60，按精确成功数和 normalized MSE 规则选出 `015000`。训练 metrics 与六个 checkpoint 完整，但各 rank runtime manifest 在 teardown 阶段记录 `Default process group has not been initialized`，因此 launcher clean-success 状态仍需修复。当前 winner 的 background retention 尚未重跑，旧的 `c3_selected 0/210` 不计入本条结论。
- Result artifacts:
  - `outputs/smolvla_libero/pilot/c3_paraphrase_lora/`
  - `outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoint_selection/`
  - `outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/selected/`
  - `outputs/smolvla_libero/eval/c3_paraphrase_lora/metrics.json`
  - `reports/daily/2026-08-07-C3改写增强LoRA正式评测结论.md`
- Modes: Formal / Offline（训练）；Formal / Closed-loop / Validation（选模）；Formal / Closed-loop / Test（目标协议）

## 2026-08-09 — ManiSkill M0 development checkpoint selection

- Date: 2026-08-09（8 卡远端 rollout 完成；selection JSON 后续纯离线补写）
- Task suite: ManiSkill `PickCube-v1`
- Dataset size: RL `pd_ee_delta_pose` 7D 主线；固定 development 200 条状态；两个 checkpoint 各评测 200 条，共 400 条候选 rollout
- Observation keys: 128×128 RGB + 21D student state（9D qpos、9D qvel、3D goal_pos）
- Action space: 7D `pd_ee_delta_pose`
- Action horizon: `n_obs_steps=2`、`horizon=16`、`n_action_steps=8`
- Control frequency: 20 Hz；最大 episode 50 steps；`physx_cuda`
- Train loss: 本条 selection 未测量；不作为选模指标
- Open-loop action error: 未测量
- Closed-loop success rate: `010000` 为 6/200（3.0%）；`020000` 为 120/200（60.0%）
- Latency: `020000` policy forward p50 约 2179.6 ms、p95 约 2367.7 ms；environment step p50 约 190.5 ms
- Failure cases: `010000` success 6 / truncated 194；`020000` success 120 / truncated 80；两者均无额外 native fail 计数
- Conclusion: 按 `success_at_end_rate` 降序、`success_once_rate` 降序、较早 step 作为 tie-breaker，冻结 `020000`。它超过 M0 development ≥50% 门槛，但尚未经过 test 400 条正式泛化评测。初版 selector 只记录 source checkpoint，未发布实体 winner copy；代码已修复，待远端用 `--select-only` 补发布。
- Result artifacts: `outputs/maniskill/m0_probe/checkpoint_selection/selected_checkpoint.json`、`outputs/maniskill/m0_probe/checkpoint_selection/checkpoint_results.json`、`outputs/maniskill/m0_probe/checkpoint_selection/selected/pretrained_model`（待远端补发布）
- Report: `reports/2026-08-09-ManiSkill M0 development checkpoint selection结论.md`
- Modes: Formal / Offline（训练）；Formal / Closed-loop / Validation（checkpoint selection）；Formal / Closed-loop / Test（待执行）
