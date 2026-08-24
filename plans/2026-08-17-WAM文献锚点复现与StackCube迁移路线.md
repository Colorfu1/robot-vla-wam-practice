# 2026-08-17 WAM 文献锚点复现与 StackCube 迁移路线

## 状态

- 状态：`todo`
- 范围：用 Fast-WAM 和 Light-WAM 的官方 LIBERO paper/code/checkpoint/eval 先建立
  WAM 文献锚点，再把已复现的方法原则迁移到 StackCube tiny Flow。
- 当前只解锁：W0 Fast-WAM LIBERO reproduction/mechanism anchor。
- 不包含：Mini-WAM、Residual RL、W4 imagination RL、RoboTwin/真机正式复现或
  未经 W2 development 冻结的 400-state StackCube formal test。
- 最后更新：2026-08-17

## 目标

把 WAM 主线改成“官方 heavy WAM 机制复现 -> 官方 lightweight WAM 复现 ->
StackCube 同架构对照 -> 一次性 formal test”。最终只在文献机制和本地开发
证据都成立后，比较 Action-only、Fast-WAM-aligned world supervision 和
Light-WAM-aligned latent supervision。

## 路线决策

### 新的 W0-W3

```text
M2-W StackCube native-4D Action-only Flow (done)
        │
        │ local action-policy anchor; does not warm-start LIBERO literature models
        │
W0 Fast-WAM / official LIBERO
   released checkpoint eval
   + with-video vs no-video equal-budget causal pair
        ↓ only if artifact + mechanism + direct-deployment pass
W1 Light-WAM / official LIBERO
   released checkpoint eval
   + full-resolution vs 2x-downsampled video-latent supervision
   + efficiency/non-inferiority audit
        ↓ only if latent compression retains control benefit
W2 StackCube tiny aligned comparison / development only
   M2-W Action-only Flow
   vs Fast-WAM-aligned future-video-latent co-training
   vs Light-WAM-aligned compressed-latent co-training
        ↓ only after architecture/checkpoint/claim freeze
W3 StackCube 400-state one-shot formal evaluation
```

M2-W 对 W0/W1 是研究顺序的前序，不是 artifact 依赖：W0/W1 使用官方
LIBERO data/model/evaluator，不从 M2-W checkpoint warm-start。M2-W 只在 W2 回到 StackCube
时成为 action-only anchor。

### 为什么不再用 Mini-WAM 开局

- Mini-WAM 公开 repo 的 image decoder/training contract 不完整；
- 用 StackCube 自行设计 next-state/pixel heads 会同时改动 architecture、loss 和 benchmark，
  无法先建立文献复现锚点；
- Fast-WAM 直接提供本项目要验证的命题：training-time video co-training
  是否改善 test-time direct action。

因此 Mini-WAM W0 与更早的 latent action-conditioning W0 全部 `dropped`，不作为
新 W0-W3 的前置或备用通道。

## 文献事实与命名边界

| 方法 | 固定来源 | 本路线要复现的命题 |
| --- | --- | --- |
| Fast-WAM | [paper](https://arxiv.org/abs/2603.16666) / [code](https://github.com/yuantianyuan01/FastWAM) / [checkpoint](https://huggingface.co/yuanty/fastwam) | full future-video-latent co-training 能改善 direct action，推理不需 future generation |
| LeRobot FastWAM | [v0.6.0 guide](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/fastwam.mdx) | Fast-WAM 可进入 LeRobot dataset/policy/checkpoint/eval 标准接口 |
| Light-WAM | [paper](https://arxiv.org/abs/2606.08242) / [code](https://github.com/L1ziang/Light-WAM) / [checkpoints](https://huggingface.co/l1ziang/lightwam-checkpoints) | compact backbone + 2x-downsampled video-latent supervision + StateFusion direct action 在降本后保留 LIBERO 能力 |

Fast-WAM 和 Light-WAM 在本计划日期都是 arXiv preprints，不写成已获同行评议会议
接收。“正式锚点”指公开 paper result + official code + checkpoint + evaluator 可审计，
不表示 venue acceptance。

### 关键技术纠正

Fast-WAM 本身已经在 pretrained VAE video latent 上做 Flow Matching，不是 raw-pixel
reconstruction。Light-WAM 也不是把 pixel target 简单改成任意 abstract latent；它将
future-video latent 进一步做 2x spatial downsampling，并同时改用 compact Wan2.1 backbone、
LoRA/WAM adapters 和 StateFusionActionExpert。

所以 W1 精确问题是：

> full/heavy video-latent co-training 的收益，能否在 downsampled latent supervision 和
> lightweight direct-action architecture 中保留？

到 W2 才会用同一 StackCube tiny action architecture 将“target compression”从 backbone/action-head
差异中拆出来。

## 阶段配置

### W0 - Fast-WAM reproduction / mechanism anchor

- 状态：`todo`
- 详细计划：
  [W0 Fast-WAM LIBERO 复现与机制锚点](2026-08-17-W0-Fast-WAM-LIBERO复现与机制锚点.md)
- canonical evidence：official FastWAM code path；LeRobot 只做 integration cross-check。
- 必做：released checkpoint 2,000-trial eval，以及同初始化 20k-step
  `lambda_video=1` vs `lambda_video=0` pair。
- 解锁 W1：artifact、causal mechanism 和 no-future-generation deployment 三层通过。

### W1 - Light-WAM reproduction / latent compression anchor

- 状态：`blocked`，阻塞项是 W0 mechanism gate。
- source pin：Light-WAM commit `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae`；
  checkpoint revision `7cc8593fb95423a9cfbb93f82c95c2fa7d5357bd`。
- released-checkpoint reference：Spatial 98.2、Object 99.6、Goal 97.8、Long 93.0，
  average 97.2；这些是 paper/repo reference，不是本地结果。
- 必做复现：
  1. 四 suite released checkpoint evaluation；
  2. LIBERO-Spatial 的 1x full latent vs 2x downsampled latent 对照；
  3. success non-inferiority、training throughput、peak memory、inference latency 审计；
  4. future latent prediction qualitative/metric 只用固定 samples，不作 checkpoint selection。
- 预注册方向：2x 相对 1x success 不劣于 `-2pp`，且实测训练显存与
  throughput 都改善；精确统计单元与命令在 W0 通过后写入独立 W1 plan。

### W2 - StackCube tiny aligned comparison

- 状态：`blocked`，阻塞项是 W0/W1 都通过。
- 任务/data/eval 回到已冻结的 M1-W/M2-W StackCube native-4D 协议。
- 三组主对照：

| ID | action path | train-time auxiliary | literature alignment |
| --- | --- | --- | --- |
| `SC-A` | M2-W Flow chunk | 无 | action-only anchor |
| `SC-F` | 与 `SC-A` 相同 | higher-resolution future visual-latent Flow | Fast-WAM principle |
| `SC-L` | 与 `SC-A` 相同 | compressed/downsampled future-latent Flow | Light-WAM principle |

- 三组必须共享 action architecture、initial checkpoint、data windows、update/global-batch/
  optimizer/noise budget 和 direct-action inference path。
- W2 不照搬 6B/2B architecture，也不将 Fast-WAM 与 Light-WAM 的原始成功率
  跨 benchmark 比较。W0/W1 的作用是冻结可迁移原则，W2 负责小模型公平对照。
- W2 只用 200-state development bank；不访问 400-state formal bank。

### W3 - StackCube one-shot formal test

- 状态：`blocked`，阻塞项是 W2 已预注册选定 checkpoint、claim 和 analysis hash。
- 不训练、不调参、不重选 representation/horizon。
- 一次性打开 StackCube 400-state formal bank，比较 `SC-A/SC-F/SC-L`。
- primary 是 paired `success_at_end`；secondary 包含 `success_once`、stage/failure、
  clipping/non-finite、latency 和 active parameter count。
- W3 详细阈值不在 W0/W1 结果出来前臆造；但 formal bank access guard 现在就保持关闭。

## 路线公平性合同

1. W0/W1 是 LIBERO literature reproduction；W2/W3 是 StackCube local transfer，两类结果不并表。
2. 官方 released checkpoint 只证明 artifact；方法机制必须有等初始化/等预算对照。
3. W0/W1 每个上游 repo、paper version、HF revision/checkpoint hash、dataset/evaluator 全部固定。
4. W2 三组只改 world-supervision representation，不同时改 action head/chunk/controller/data。
5. W0-W2 不根据 formal test 选模；W3 打开后不再改 architecture/checkpoint/analysis。
6. 所有阶段记录 dataset size、observation/action contract、horizon/frequency、train loss、
   open-loop error、closed-loop success、latency 和 failure cases。

## 前置条件

- [x] M1-W StackCube data/control/state-bank protocol 已冻结。
- [x] M2-W native-4D Action-only Flow 已建立本地 action anchor。
- [ ] W0 的 Fast-WAM revisions/artifact hashes/resource gate 已通过。
- [ ] W1 开始前，W0 `artifact/mechanism/deployment` 已全通过。
- [ ] W2 开始前，W0/W1 报告已分别冻结“必须保留的机制”。
- [ ] W3 开始前，W2 method/checkpoint/analysis hashes 已预注册且 formal bank access count=0。

## 执行步骤

### 1. 当前只执行 W0 preflight

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=preflight \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/preflight
```

后续命令以
[W0 详细计划](2026-08-17-W0-Fast-WAM-LIBERO复现与机制锚点.md)为准。

### 2. W0 通过后才创建可执行 W1 计划

```bash
bash scripts/experiments/literature/run_wam_route.sh \
  --phase=unlock \
  --completed-stage=w0_fastwam \
  --gate=outputs/literature/w0_fastwam/w0_gate.json \
  --next-stage=w1_lightwam
```

### 3. W0/W1 都通过后才冻结 W2

```bash
bash scripts/experiments/literature/run_wam_route.sh \
  --phase=freeze-stackcube-transfer \
  --fastwam-gate=outputs/literature/w0_fastwam/w0_gate.json \
  --lightwam-gate=outputs/literature/w1_lightwam/w1_gate.json \
  --output=outputs/maniskill/w2_wam_aligned/preflight/frozen_protocol.json
```

`run_wam_route.sh` 是计划内待实现的 protocol guard；在其存在前，不手工跳阶段。

## 判断标准

| Gate | 通过条件 | 失败后 |
| --- | --- | --- |
| W0 artifact | 官方 Fast-WAM checkpoint 完成 2,000 LIBERO trials 并达 reproduction tolerance | 修 environment/evaluator，不训 causal pair |
| W0 mechanism | with-video 相对 no-video `>=2pp`且 paired CI 下界 `>0` | 不能声称 world co-training 改善 direct action，W1 blocked |
| W0 deployment | future generation/decode calls=0，direct action contract 通过 | 不能称 Fast-WAM-style train-only world supervision |
| W1 compression | 2x latent 相对 1x non-inferiority `>=-2pp`，且 memory/throughput 改善 | 不迁移 Light-WAM-aligned target |
| W2 development | 按 W0/W1 证据后预注册；必须同 action path/budget 三组公平对照 | formal bank 继续关闭 |
| W3 formal | 按 W2 冻结的 primary comparison 一次性执行 | 如实记录，不回开发集调参 |

W0 与 W1 的 released-checkpoint pass 都不能单独解锁 W2；必须同时有对应的
causal/compression evidence。

## 产物路径

- `plans/2026-08-17-W0-Fast-WAM-LIBERO复现与机制锚点.md`
- `outputs/literature/w0_fastwam/`
- `outputs/literature/w1_lightwam/`
- `outputs/maniskill/w2_wam_aligned/`
- `outputs/maniskill/w3_wam_aligned_formal/`
- `outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks/development.jsonl`
- `outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks/test.jsonl`

## 风险与停止条件

- Fast-WAM/Light-WAM 的 upstream reference 已高成功，ceiling 会让差异集中在
  Spatial/Long；不减少到只跑简单 suite 以制造 pass。
- Fast-WAM 官方没有发布 no-video checkpoint；无 8-GPU causal pair 就只能完成 artifact
  reproduction，不能用 paper table 代替。
- Light-WAM 同时改 backbone、adapters、action decoder 和 latent resolution；W1 的系统复现不能
  自动证明“只有 latent compression 有用”，该归因由 W2 同架构对照完成。
- LeRobot 0.6.0 需 Python >=3.12，不得为此升级现有 M2-W Python3.10/LeRobot0.4.4 环境。
- 任一阶段不满足 GPU 资源时标记为 resource-blocked，不降模型/分辨率/试验数
  后仍称官方复现。
- W2 前禁止往 M2-W 直接加自创 pixel/state head；先结束 W0/W1 文献锚点。
- W2 未通过前，400-state formal bank 继续禁止 parse/reset/rollout。

## 收尾

- [ ] W0 结果只写入独立日期报告，不回写本计划。
- [ ] W0 pass 后新建并冻结 Light-WAM W1 独立执行计划。
- [ ] W1 pass 后新建 StackCube W2 独立计划，不复活旧 horizon/shuffle 计划。
- [ ] W2 pass 后才新建 W3 formal plan 并冻结 400-state access manifest。
