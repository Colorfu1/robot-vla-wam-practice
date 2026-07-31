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
