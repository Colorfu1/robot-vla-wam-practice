# ACT Offline and Closed-Loop Evaluation Design

## Goal

Add two explicit ACT evaluation entry points for the current A-Sanity experiment:

1. offline, teacher-forced prediction on selected `lerobot/libero` episodes, including
   quantitative action errors and synchronized Rerun visualization;
2. closed-loop rollout in `libero_object` task 8, including simulator success, reward,
   executed-action smoothness, inference latency, video, and Rerun visualization.

Offline prediction measures imitation quality but cannot claim task success. Closed-loop
success comes only from the simulator's `is_success` signal. The two result types remain
separate even though they share checkpoint loading, metrics, result metadata, and Rerun
utilities.

## Scope

The first version supports:

- LeRobot 0.4.4 ACT checkpoints;
- the local `lerobot/libero` dataset;
- dataset task index 29, “pick up the chocolate pudding and place it in the basket”;
- `libero_object` simulator task ID 8;
- the clean evaluation condition only.

The implementation does not attempt to support arbitrary policies, datasets, tasks, Hub
checkpoints, standalone `.pth` files, or perturbations in the first version.

## Checkpoint Contract

Both entry points require an explicit `--policy-path` pointing to a complete LeRobot
`pretrained_model` directory. At minimum it must contain:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

The two processor JSON files are parsed before loading. Every relative `state_file`
referenced by any entry in their `steps` arrays is also required to exist inside the
checkpoint directory. These files normally contain normalization statistics in additional
`.safetensors` files, so validating only the four names above is insufficient.

`train_config.json` is recorded when present but is not required for inference. A single
`.pth` or `.safetensors` file is rejected with a message explaining that policy structure,
processor configuration, and every referenced processor state file are also required.

For an A-Sanity run, the expected input is:

```text
outputs/act/a_sanity/<train_timestamp>/checkpoints/020000/pretrained_model
```

## Architecture

Reusable Python code lives under `src/robot_practice/`, while shell launchers stay with
the existing experiment scripts:

```text
src/robot_practice/evaluation/
├── __init__.py
├── common.py
├── metrics.py
├── rerun_visualization.py
├── offline_act.py
└── libero_act.py

scripts/experiments/act/
├── eval_offline.sh
└── eval_libero.sh
```

`common.py` validates and loads the checkpoint, loads the saved processor pipelines,
resolves the default result directory from the checkpoint, writes run metadata, and owns
shared CUDA timing helpers. `metrics.py` contains pure NumPy metric functions.
`rerun_visualization.py` contains no inference logic. The two CLI modules own their
respective data and environment loops.

The shell launchers resolve the practice root and use:

```text
conda run --no-capture-output -n robot-practice python -m <evaluation module>
```

They forward all arguments unchanged and do not activate or mutate the caller's shell.

## Command Interfaces

Offline prediction:

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path=<pretrained_model> \
  --episodes 823 826 834 840 841
```

The episode argument is optional and defaults to the five A-Sanity training episodes.
Other same-task episode indices are accepted. Useful options are `--dataset-root`,
`--batch-size` (default 1 for meaningful per-observation latency), `--output-dir`, and
`--no-open-viewer`. `--max-frames` is an optional positive debug/smoke-test bound; its
default is unset and a bounded run is marked `smoke`, not a formal evaluation.

Closed-loop evaluation:

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path=<pretrained_model> \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none
```

The closed-loop path is fixed to `libero_object` task ID 8 and batch size 1 in the first
version. `--n-episodes` defaults to 20, `--num-visualized-episodes` defaults to 1,
`--seed` defaults to 1000, and `--output-dir` is optional. The visualization count must
be between zero and `n_episodes`. `--perturbation` is part of the stable interface but
only `none` is accepted initially; any other value fails explicitly. `--max-steps` is an
optional positive per-episode debug/smoke-test bound. It defaults to the environment's
280-step limit; lowering it marks the result `smoke` and cannot produce a formal success
rate.

## Default Output Location

When the checkpoint matches the expected LeRobot layout, the evaluation root is derived
from its training run:

```text
outputs/act/a_sanity/<train_timestamp>/
└── eval/
    ├── offline/<eval_timestamp>/
    └── libero/<eval_timestamp>/
```

If the checkpoint is elsewhere, `--output-dir` is required. An explicit output directory
must not already exist. Timestamp leaves use `YYYYMMDD_HHMMSS` and are never overwritten.

## Offline Data Flow

The offline evaluator performs the following steps:

1. validate and load the ACT policy plus its saved preprocessor and postprocessor;
2. resolve and validate the requested episodes against metadata and data Parquet files;
3. require every requested episode to use dataset task index 29;
4. call LeRobot 0.4.4's `resolve_delta_timestamps(policy.config, metadata)` and construct
   the dataset from those values. Delta timestamps are seconds, so the action timestamps
   must equal `[i / metadata.fps for i in range(chunk_size)]`, not integer frame offsets;
5. copy the raw `[B, chunk_size, 7]` target chunk before preprocessing and define
   `valid_horizon = ~batch["action_is_pad"]`;
6. build an observation-only policy input, preprocess only those observation fields, and
   call `predict_action_chunk()` under inference mode. Target actions and padding fields
   are never passed to the inference model;
7. postprocess the complete predicted tensor with shape `[B, chunk_size, 7]` back to raw
   action space before slicing horizon 0 or computing metrics;
8. compute delta metrics only where both adjacent horizons are valid;
9. write frame-level predictions and model-call metrics, then build/save/open the Rerun
   recording.

Every demonstration observation is evaluated independently. This is teacher forcing:
the predicted action does not change the next dataset observation. The evaluator must
label its continuity metrics accordingly and must not emit success or reward claims.

## Closed-Loop Data Flow

The closed-loop evaluator uses LeRobot's LIBERO environment factory and saved policy
processors but owns a local sequential rollout loop so it can record timing and executed
actions without modifying site-packages. The fixed environment configuration is:

```text
task="libero_object"
task_ids=[8]
n_envs=1, use_async_envs=false
camera_name="agentview_image,robot0_eye_in_hand_image"
obs_type="pixels_agent_pos"
control_mode="relative"
observation_height=256, observation_width=256
episode_length=280
init_states=true
```

The control/video timeline is recorded at the dataset rate of 10 Hz. The loop processes
both camera observations and the 8-dimensional agent state produced by the standard
`LiberoProcessorStep` into the keys expected by the trained ACT checkpoint.

For each episode it:

1. resets the simulator and `policy.reset()` with a deterministic seed;
2. converts and environment-processes observations using the same LIBERO processor path
   as `lerobot-eval`;
3. times every `policy.select_action()` control step with CUDA synchronization;
4. instruments ACT's `predict_action_chunk()` calls separately to record actual neural
   network query latency, distinguishing queue hits from model calls;
5. postprocesses and executes the action in the environment;
6. when terminal, first requires `info["_final_info"][0]`, then reads
   `info["final_info"]["is_success"][0]`, ORs that terminal success across steps, and
   never substitutes reward, `terminated`, or top-level post-reset info for success;
7. records the executed raw action, reward, terminal flags, cumulative success, and
   termination reason;
8. captures images only for the first `num_visualized_episodes` episodes.

LeRobot 0.4.4 constructs this LIBERO `SyncVectorEnv` with Gymnasium's default
`AutoresetMode.NEXT_STEP`; the vector wrapper must not be changed to SAME_STEP. Separately,
the underlying LeRobot `LiberoEnv.step()` calls its own `reset()` on termination and places
the completed episode's data in `final_info`. Therefore the observation returned by a
terminal step is already a reset observation even though the vector wrapper is NEXT_STEP.
The evaluator consumes vectorized `final_info` first, records the action/reward/terminal
row, and immediately breaks without consuming the returned observation or calling
`env.step()` again.

That underlying terminal reset increments LeRobot's `init_state_id`. To avoid skipping an
initial state when the evaluator explicitly resets for the next serial rollout, before
episode `e` it sets `env.envs[0].init_state_id = e`, records `init_state_id=e`, and then
calls `env.reset(seed=[seed + e])`. This makes formal episodes use initial states
`0..n_episodes-1` even though the previous terminal step performed a redundant internal
reset. An implementation may rebuild the single environment per episode instead, but it
must produce and record the same initial-state sequence. If step 280 is reached without
simulator termination, the evaluator synthesizes `done_after_step=true`, `truncated=true`,
`termination_reason="time_limit"`, and `success=false`.

All formal episodes contribute to metrics. Only the configured visualization prefix
contributes frames to MP4 and Rerun. The model-query instrumentation is local and restored
after the rollout; installed LeRobot files are never patched. A run using a reduced
`--max-steps` is explicitly excluded from formal aggregate success reporting.

## Timing Contract

Timing uses `time.perf_counter_ns()`. A timing helper synchronizes immediately before and
after the measured call only when the policy device is CUDA; it never calls CUDA APIs for
CPU/MPS. Measurements include the first call rather than hiding cold-start cost.

Offline latency is one sample per `predict_action_chunk()` batch call. If `--batch-size`
is greater than one, it is described as batch-call latency, not per-frame latency. Every
row records a `batch_id`; the batch latency appears only on `batch_row=0` and is null on
the remaining rows. Percentiles are computed only from those non-null call samples and
are never weighted by batch size.

Closed-loop instrumentation resets a step-local `model_query_latency_ms` variable to null
before every `select_action()` call. A temporary wrapper measures only actual invocations
of `predict_action_chunk()`. Queue-hit steps therefore retain null model-query latency,
while `policy_step_latency_ms` is present on every step and encloses the complete
`select_action()` call. Percentiles use NumPy's default linear percentile over the raw
call samples. If no model query exists, the model-query percentiles serialize as JSON
null rather than NaN.

## Metric Definitions

All action metrics use postprocessed raw actions. The first six LIBERO action dimensions
are treated as arm commands; the seventh gripper dimension is excluded from TV and jerk.
Padding and post-termination samples are always masked.

For TV and jerk, the absolute difference is averaged across both time and the first six
action dimensions. Episode-level values require at least two and three executed/predicted
steps respectively; otherwise they are null. Aggregate smoothness values are macro means
over episodes with a defined value. JSON never contains NaN or Infinity.

### Offline Metrics

- `first_action_mae`: mean absolute error between horizon-0 prediction and the current
  demonstration action.
- `per_dimension_mae`: horizon-0 MAE for each of the seven action dimensions.
- `chunk_mae_by_horizon`: MAE at every valid chunk horizon from 0 to `chunk_size - 1`.
- `delta_action_mae`: MAE between consecutive predicted-action differences and
  consecutive target-action differences where both adjacent horizons are valid.
- `teacher_forced_action_tv`: mean absolute first difference of the horizon-0 arm-action
  prediction along each demonstration episode.
- `teacher_forced_action_jerk`: mean absolute second difference of that prediction.
- `inference_latency_p50_ms` and `inference_latency_p95_ms`: synchronized model latency;
  these are model-call latency percentiles, and values remain comparable only when batch
  size and hardware are unchanged.

### Closed-Loop Metrics

- `success_rate`: fraction of rollouts for which LIBERO reports `is_success`.
- `avg_sum_reward` and `avg_max_reward`: reward summaries over rollouts.
- `action_tv`: per-episode mean `abs(a[t] - a[t-1])`, then macro-averaged.
- `action_jerk`: per-episode mean `abs(a[t] - 2*a[t-1] + a[t-2])`, then
  macro-averaged.
- `gripper_toggles`: transitions between negative and nonnegative gripper commands,
  reported per episode and as a macro mean.
- `policy_step_latency_p50_ms` and `policy_step_latency_p95_ms`: all calls to
  `select_action()`, including queue hits.
- `model_query_latency_p50_ms` and `model_query_latency_p95_ms`: only real
  `predict_action_chunk()` calls.

TV is a normalized total-variation proxy rather than a sum, which makes episodes of
different lengths comparable. `action_jerk` is explicitly described as a per-step action
jerk proxy, not physical joint or end-effector jerk. A low smoothness score is never
treated as success; success rate remains the guard against a stationary policy.

## Metric Annotations

The terminal summary, `metrics.json`, and Rerun summary include a short description for
every metric. Required wording includes:

```text
action_tv: 相邻执行动作的平均变化量，越低通常越平滑
action_jerk: 动作二阶差分，反映动作突变和来回抖动
success_rate: 成功完成任务的 rollout 比例
gripper_toggles: 夹爪开合状态发生切换的次数
first_action_mae: 当前步预测动作与 demonstration 动作的平均绝对误差
delta_action_mae: 预测动作变化与 demonstration 动作变化的平均绝对误差
policy_step_latency_p95_ms: 95% 控制步骤不超过的策略调用耗时
model_query_latency_p95_ms: 95% 神经网络实际推理不超过的耗时
```

Other metrics receive similarly short annotations. JSON stores these under a separate
`metric_definitions` object rather than mixing strings into numeric aggregates.

## Result Files

Offline output:

```text
run_config.json
metrics.json
predictions.parquet
prediction.rrd
```

`run_config.json` has the same exact top-level keys in both modes:

```text
schema_version: 1
status: "complete" | "smoke" | "incomplete"
mode: "offline" | "libero"
created_at: ISO-8601 string with timezone
policy: {path, type, config, train_config}
runtime: {python, lerobot, torch, rerun, device}
input: mode-specific object
arguments: complete parsed CLI argument object
```

`policy.path` is absolute, `policy.type` is `act`, `policy.config` is the parsed
`config.json`, and `policy.train_config` is the parsed object or null. Offline `input`
contains exactly `{dataset_repo_id, dataset_root, fps, task_index, episodes}`. LIBERO
`input` contains exactly `{suite, task_id, cameras, obs_type, control_mode,
observation_size, episode_length, control_frequency_hz, seeds, init_state_ids}`.

Offline `metrics.json` has this exact structure:

```text
schema_version: 1
mode: "offline"
status: "complete" | "smoke"
aggregate: {
  num_episodes, num_frames, num_model_queries, batch_size,
  first_action_mae, per_dimension_mae[7], chunk_mae_by_horizon[H],
  delta_action_mae, teacher_forced_action_tv, teacher_forced_action_jerk,
  inference_latency_p50_ms, inference_latency_p95_ms
}
per_episode: [{
  episode_index, num_frames, first_action_mae, per_dimension_mae[7],
  chunk_mae_by_horizon[H], delta_action_mae,
  teacher_forced_action_tv, teacher_forced_action_jerk
}]
metric_definitions: {metric_name: short Chinese description}
```

`H` is `policy.config.chunk_size`. Undefined scalars and horizons are JSON null. The
aggregate MAEs pool all valid elements; episode TV/jerk values are macro-averaged as
defined above.

`predictions.parquet` has one row per evaluated dataset observation and these exact Arrow
columns:

| column | Arrow type | rule |
| --- | --- | --- |
| `episode_index` | `int64` | dataset episode |
| `frame_index` | `int64` | frame within episode |
| `dataset_index` | `int64` | original global dataset index |
| `timestamp_s` | `float64` | original dataset timestamp |
| `task_index` | `int64` | always 29 |
| `batch_id` | `int64` | monotonically increasing model-call ID |
| `batch_row` | `int32` | zero-based position in that batch |
| `batch_size` | `int32` | actual size of that batch |
| `target_action` | `fixed_size_list<float32>[7]` | raw target at horizon 0 |
| `predicted_action` | `fixed_size_list<float32>[7]` | raw prediction at horizon 0 |
| `target_chunk` | `fixed_size_list<float32>[H*7]` | row-major raw `[H,7]` |
| `predicted_chunk` | `fixed_size_list<float32>[H*7]` | row-major raw `[H,7]` |
| `valid_horizon` | `fixed_size_list<bool>[H]` | `~action_is_pad` |
| `absolute_error` | `fixed_size_list<float32>[7]` | horizon-0 absolute error |
| `model_query_latency_ms` | nullable `float64` | populated only for `batch_row == 0` |

The Parquet schema metadata stores `chunk_shape=[H,7]` and `action_dimension=7` so the
flattened chunk columns are unambiguous. Latency percentiles use one non-null value per
`batch_id`.

Closed-loop output:

```text
run_config.json
metrics.json
eval_info.json
rollouts.parquet
rollout.rrd
videos/episode_000.mp4
```

Closed-loop `metrics.json` has this exact structure:

```text
schema_version: 1
mode: "libero"
status: "complete" | "smoke"
aggregate: {
  is_formal, num_episodes, num_successes, success_rate,
  avg_sum_reward, avg_max_reward, action_tv, action_jerk,
  mean_gripper_toggles, policy_step_latency_p50_ms,
  policy_step_latency_p95_ms, model_query_latency_p50_ms,
  model_query_latency_p95_ms, num_policy_steps, num_model_queries
}
per_episode: [{
  episode_index, seed, init_state_id, num_steps, success, sum_reward, max_reward,
  termination_reason, action_tv, action_jerk, gripper_toggles,
  policy_step_latency_p50_ms, policy_step_latency_p95_ms,
  model_query_latency_p50_ms, model_query_latency_p95_ms,
  num_model_queries
}]
metric_definitions: {metric_name: short Chinese description}
```

`success_rate` is a fraction in `[0,1]`; `num_successes` is the count. A normal 280-step
run has `is_formal=true`. When a reduced `--max-steps` is used, `is_formal=false` and
`success_rate` is null; reward, smoothness, and timing values remain explicitly marked as
smoke observations. Empty latency/smoothness groups serialize as null.

`eval_info.json` is deliberately LeRobot-shaped but fully specified:

```text
schema_version: 1
suite: "libero_object"
task_id: 8
is_formal: bool
per_episode: [{
  episode_ix, seed, init_state_id, num_steps, sum_reward, max_reward, success,
  termination_reason
}]
aggregated: {
  n_episodes, n_successes, avg_sum_reward, avg_max_reward,
  pc_success, eval_s, eval_ep_s
}
```

`pc_success` is percentage `[0,100]` for a formal run and null for a bounded smoke run.

`rollouts.parquet` has one row per action actually sent to the simulator:

| column | Arrow type | rule |
| --- | --- | --- |
| `episode_index` | `int64` | zero-based evaluation episode |
| `seed` | `int64` | `seed + episode_index` |
| `init_state_id` | `int64` | controlled LIBERO initial-state index; equals episode index |
| `step_index` | `int32` | zero-based within episode |
| `global_step` | `int64` | monotonic across all episodes |
| `timestamp_s` | `float64` | `step_index / 10.0` |
| `executed_action` | `fixed_size_list<float32>[7]` | postprocessed raw action sent to env |
| `reward` | `float32` | reward returned by this step |
| `terminated` | `bool` | simulator termination flag |
| `truncated` | `bool` | env flag or synthetic time-limit flag |
| `done_after_step` | `bool` | cumulative terminal state after this row |
| `success_this_step` | `bool` | terminal `final_info.is_success` from this step only |
| `success_so_far` | `bool` | cumulative OR through this step |
| `termination_reason` | nullable `string` | final row: `success`, `environment_done`, or `time_limit` |
| `policy_step_latency_ms` | `float64` | non-null on every control step |
| `model_query_latency_ms` | nullable `float64` | null on ACT queue-hit steps |
| `visualized` | `bool` | whether this episode is in MP4/Rerun |

Images remain in MP4 and Rerun rather than being duplicated into Parquet. Every JSON and
Parquet writer is schema-tested; adding or renaming a field requires a schema-version
change.

Every JSON writer converts NumPy/Torch scalars to native JSON values and nulls undefined
measurements; NaN and Infinity are rejected before the atomic final write.

## Rerun Visualization

Offline Rerun contains both camera streams, ground-truth and horizon-0 predicted action
curves for all seven dimensions, absolute-error curves, inference latency, frame and
episode identifiers, and a text summary of aggregate metrics and annotations. It supports
frame-by-frame playback on the dataset timeline.

Closed-loop Rerun contains rendered camera frames, executed action curves, reward,
success/done state, policy-step latency, model-query latency, episode/step identifiers,
and the metric summary. Only the first configured number of episodes are logged visually.

Offline uses one monotonically increasing sequence timeline named `global_frame`; closed
loop uses `global_step`. The counter never resets between episodes. At each observation or
control step, code calls Rerun's time setter exactly once, then logs both camera images and
all action/error/reward/latency scalars at that same time. Dataset `episode_index` and
`frame_index`, or rollout `episode_index` and `step_index`, are logged as scalar data on
separate `context/*` entity paths; they are not used as the primary timeline. This avoids
overwriting or interleaving frames when an episode-local counter resets and makes the
saved recording deterministically frame-steppable across episode boundaries.

For closed loop, the visual frame at a control step is the pre-action observation that
produced that step's action. The reset observation returned by LeRobot's underlying
terminal reset is never logged as part of the completed episode. MP4 frames follow the
same ordering and are encoded at 10 fps.

Both modes save `.rrd` and attempt to open the viewer by default. Viewer launch failure is
reported but does not invalidate already written metrics or recordings. `--no-open-viewer`
keeps generation headless.

## Failure Handling

The CLIs fail before expensive work when:

- the policy path is a file, lacks a required checkpoint/processor JSON, or lacks any
  processor `state_file` referenced by those JSON files;
- the policy is not ACT or its action dimension is not seven;
- an episode is absent or belongs to another task;
- the local dataset or LIBERO assets are unavailable;
- an output directory already exists;
- numeric counts are invalid or visualization count exceeds evaluation count;
- `--perturbation` is anything other than `none`.

Partial outputs receive an incomplete marker and are not reported as successful results.
Simulator and Rerun resources are closed in `finally` blocks. Ctrl-C returns a nonzero
status without deleting any completed result files.

## Verification Strategy

Verification does not launch training or the full 20-episode evaluation.

- Unit tests use known scalar and multidimensional sequences to lock down MAE, masks, TV,
  jerk, gripper toggles, macro aggregation, and percentile behavior.
- CLI tests cover defaults, custom episode lists, visualization counts, perturbation
  rejection, existing output directories, standalone-weight rejection, missing processor
  state files, and positive validation for `max_frames`/`max_steps`.
- A temporary randomly initialized ACT `pretrained_model` plus saved processors exercises
  checkpoint loading without depending on completion of A-Sanity training. Its fixture
  sets `pretrained_backbone_weights=None` so it does not download a backbone and saves
  every processor-referenced state file, not only the two processor JSON files.
- A short offline integration case evaluates a bounded number of frames and validates
  Parquet/JSON/Rerun schemas without opening a viewer.
- A real LIBERO task-8 environment smoke test executes a short bounded rollout and checks
  action, reward, vectorized `_final_info` handling, controlled consecutive
  `init_state_id` values, latency, and rendering capture. It does not claim meaningful
  policy performance.

The bounded seams are available both through the two CLI debug flags and through explicit
Python functions:

```text
evaluate_offline(config, *, max_frames: int | None = None) -> OfflineResult
evaluate_libero(config, *, max_steps: int | None = None) -> LiberoResult
```

Tests call these functions directly. Production entry points pass null unless the user
explicitly supplies a debug bound.

Formal A-Sanity results remain blocked until training produces its final checkpoint.

## Acceptance Criteria

- Both shell launchers work from outside the practice root and automatically use the
  `robot-practice` environment.
- Both CLIs require an explicit complete checkpoint directory and never patch LeRobot.
- Offline evaluation accepts an episode list, defaults to the five A-Sanity episodes, and
  rejects mixed-task inputs.
- Closed-loop evaluation defaults to 20 episodes, visualizes one, accepts a visualization
  count override, and derives success only from LIBERO.
- All approved metrics, annotations, JSON/Parquet outputs, MP4 output, and saved Rerun
  recordings follow this design.
- `--perturbation=none` is the only initially accepted perturbation mode.
- Tests and short smoke checks pass without training or running the full evaluation.
- Implementation does not initialize Git.

## Version-Control Constraint

At design time, the workspace was not a Git working tree. The design and implementation could
be written and verified locally, but could not be committed unless version control was
initialized later. This work did not initialize Git as a side effect.
