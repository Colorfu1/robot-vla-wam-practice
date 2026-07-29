# PushT E1：固定 Observation 的配对 Diffusion Sampling

状态：`Offline / Formal / Stage complete`。

## 实验目标

在完全相同的两帧 observation history 上，分别回答：

1. E1a：只改变初始高斯 noise，固定 DDPM 反向随机路径，action chunk 是否变化？
2. E1b：同时改变初始 noise 和 DDPM 反向随机路径，正常随机采样能产生多大差异？

E1 只比较模型输出，不执行环境动作，因此不能单独证明某种预测能够成功。

## 当前结论

结论先行：

1. **完整 DDPM 反向随机路径会稳定地改变 action chunk。** 同 seed 重复误差为 0，
   E1b 平均 action 轨迹 RMS 为 1.946 px，但绝对幅度不大。
2. **现有 20-sample 数据不支持固定 observation 下存在多个 action-chunk mode。**
   预注册的 GMM/BIC 与完整 chunk 距离聚类交叉验证中，E1b 为 0/10 candidate 通过；
   E1a negative control 同样为 0/10。
3. **目前不能声称已经证明 Diffusion Policy 克服了 MSE mode averaging。** E2b
   仍支持多个成功闭环行为 mode，但固定 observation 下的 action-distribution 多峰性
   没有在当前小样本分析中得到证据。

这个阴性结果不证明 \(p(a\mid o)\) 一定是单峰分布。每个 observation 只有 20 个
samples，当前结论应限定为“未获得探索性多模证据”。

## 实验协议

每个 E0 candidate 只读取和预处理一次：

- BCChunk 调用 3 次，验证 deterministic 输出；
- 生成并保存 20 个 canonical initial-noise tensors；
- E1a 和 E1b 按 sample ordinal 共用同一份 initial noise；
- E1a 的 20 个 sample 共用一个 candidate-level reverse seed；
- E1b 的 20 个 sample 分别使用不同的 reverse seed；
- 每次 Diffusion sampling 都输出完整的 8-step 公共 action chunk。

seed 由稳定的 `(episode_index, frame_index, sample_ordinal)` 派生，不依赖 candidate
在 JSON 中的排列顺序。

## 执行命令

读取 development 选择出的 checkpoint：

```bash
BC_STEP="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["bc_chunk"]["step"])')"
DIFFUSION_STEP="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["step"])')"
BC_POLICY="$(printf 'outputs/pusht_diffusion/pilot/bc_chunk/checkpoints/%06d/pretrained_model' "${BC_STEP}")"
DIFFUSION_POLICY="$(printf 'outputs/pusht_diffusion/pilot/diffusion/checkpoints/%06d/pretrained_model' "${DIFFUSION_STEP}")"
```

一条命令同时生成 E1a 和 E1b：

```bash
bash scripts/experiments/pusht/eval_fixed_observations.sh \
  --bc-policy="${BC_POLICY}" \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --candidates=outputs/pusht_diffusion/data_inspection/candidates.json \
  --num-action-samples=20 \
  --output-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation \
  --overwrite
```

## 打开可视化

E1a：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state \
  outputs/pusht_diffusion/pilot/e1_fixed_observation/recording_initial_noise_only.rrd
```

E1b：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state \
  outputs/pusht_diffusion/pilot/e1_fixed_observation/recording_full_sampling_seed.rrd
```

两套 RRD 都有两个同步面板：

- 左侧是同一 observation、agent 和 3 次 BCChunk 预测；
- 右侧是对应协议的 20 次 Diffusion 预测；
- 每个 `pred_XX` 可以独立显示或隐藏；
- 每条轨迹是 `agent → action[0] → ... → action[7]`；
- 用左右方向键或 Rerun 时间控制切换 `candidate=0..9`。

两个 RRD 使用不同 application ID，`--persist-state` 会分别保存布局。

## 结果目录

```text
outputs/pusht_diffusion/pilot/e1_fixed_observation/
├── observations/
├── noises/
├── processor_metadata/
├── action_chunks.parquet
├── diversity_metrics.parquet
├── metrics.json
├── metrics.md
├── run_manifest.json
├── recording_initial_noise_only.rrd
└── recording_full_sampling_seed.rrd
```

schema v2 不再生成含义不明确的 `recording.rrd`。如果目录中还是旧文件，使用上面的
命令加 `--overwrite` 重新执行。

## 查看指标

`metrics.md` 直接对比两种协议：

- normalized/denormalized trajectory pairwise RMS；
- normalized/denormalized endpoint pairwise distance；
- E1b aggregate mean / E1a aggregate mean；
- 每种协议相同 noise、相同 reverse seed 的复现误差；
- BCChunk 三次调用的最大误差。

当 E1a aggregate mean 为 0 时，比值在 JSON 中为 `null`，Markdown 中显示 `N/A`，
不会写入 `inf` 或 `NaN`。

## 固定 Observation Action-mode 探索分析

该分析复用现有 `action_chunks.parquet`，不会加载 checkpoint 或重新执行模型：

```bash
bash scripts/experiments/pusht/analyze_fixed_observation_action_modes.sh \
  --input-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation \
  --output-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis \
  --overwrite
```

判据同时要求：

- GMM/BIC 选择至少两个 mode，且相对单峰模型的 BIC improvement 至少为 10；
- 每个 mode 至少包含 3/20 samples；
- 200 次 80% subsample 中至少 70% 保持多模选择；
- 完整 action-chunk RMS complete-linkage 聚类与 GMM 具有足够 silhouette 和 assignment
  agreement；
- E1a negative control 不能大量误报。

当前结果：

| 指标 | 结果 |
|---|---:|
| E1a multimodal candidate | 0/10 |
| E1b multimodal candidate | 0/10 |
| negative-control failure | false |
| BC averaging-consistent signature | 0/10 |

唯一由 GMM 选择 \(K>1\) 的 `audit_ep065_frame083` 选择了 \(K=4\)，但 BIC improvement
只有 0.95，200 次 subsample 中只有 5% 支持多模，而且距离聚类出现 1-sample cluster，
因此不能计为多模 action distribution。

结果目录：

```text
outputs/pusht_diffusion/pilot/e1_fixed_observation/action_mode_analysis/
├── candidate_summary.parquet
├── mode_assignments.parquet
├── metrics.json
├── metrics.md
├── analysis_manifest.json
└── figures/
```

BC 在部分 candidate 上位于 DP samples 的低密度区域或接近 DP 均值，但由于这些
candidate 没有先通过 action multimodality 判据，不能把它解释为 mode-averaging
signature，更不能据此认定 BC 动作在闭环中无效。

## 验收标准

- 正式 E1 每个 candidate 有 3 条 BC、20 条 E1a 和 20 条 E1b 预测；
- 每条预测包含 8 个 action target，即每个 candidate 在
  `action_chunks.parquet` 中正好有 344 行；
- E1a/E1b 同 ordinal 的 initial-noise seed 和 hash 完全一致；
- E1a 的 reverse seed 相同，E1b 的 20 个 reverse seed 唯一；
- 两种协议的重复采样最大误差均不超过 `1e-6`；
- E1b 原始指标只能证明完整随机采样产生不同 action patterns；当前 action-mode
  探索分析没有进一步获得固定 observation 多模证据。
- E2b 负责验证 stochastic sampling 是否具有多个成功闭环行为 mode，但它不能反向
  把 E1 的阴性 action-mode 结果改写为固定 observation 的多峰分布证据。
