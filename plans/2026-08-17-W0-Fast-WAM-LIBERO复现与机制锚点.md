# 2026-08-17 W0 Fast-WAM LIBERO 复现与机制锚点

## 状态

- 状态：`in-progress`
- 范围：在官方 LIBERO 数据/评测协议上，先复现 Fast-WAM released
  checkpoint 闭环，再用同初始化 `with-video` / `no-video` 训练对照复现
  `world/video co-training -> better direct action` 机制。
- 当前可执行：source/artifact/resource preflight、镜像环境/本地 artifact validation 和阶段 wrapper；正式 eval 需满足
  显存要求，正式 causal training 需 `8 x >=80 GiB` 级资源。
- 判断方式：本计划**不设 pass/fail 阈值**，只规定必须测什么、以及数字在什么条件下有意义；
  是否满意由使用者判断（2026-08-21 起）。
- 不包含：Mini-WAM、Light-WAM、ManiSkill/StackCube 改模型、400-state
  formal test、Fast-WAM-Joint/IDM 完整复现、RoboTwin、真机或 W4。
- 最后更新：2026-08-21
- 已实现：repo-local 配置、audit/analysis、只读 validation、阶段 runner、单测和可复制 guide；尚未运行正式大规模
  artifact eval 或 causal training。

## 目标

用可审计的官方 artifact 和等预算对照，测量三件事：Fast-WAM released policy 能否在 LIBERO 上
完成 checkpoint-to-eval 闭环；`FW-V` 相对冻结 video expert 的 VLA 形状对照 `FW-FZ` 差多少；
（可选）相对论文同款 `lambda_video=0` 消融 `FW-NV` 差多少。**本计划只负责把这些量测准并如实
报告，不设 pass/fail 阈值，是否满意由使用者判断。**

## 文献与复现边界

### 固定来源

- 论文：[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/abs/2603.16666)，
  arXiv v2，2026-03-23。
- 官方代码：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)，
  commit `45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- 官方环境：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，
  commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`；不以同名 PyPI 包替代 checkout。
- 官方 checkpoint：[yuanty/fastwam](https://huggingface.co/yuanty/fastwam)，
  revision `139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a`。
- LIBERO checkpoint：`libero_uncond_2cam224.pt`，12,041,573,140 bytes，
  SHA-256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。
- LeRobot 集成：[huggingface/lerobot v0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0)，
  commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`；
  [FastWAM guide](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/fastwam.mdx)。
- LeRobot-format checkpoint：
  [ZibinDong/fastwam_libero_uncond_2cam224](https://huggingface.co/ZibinDong/fastwam_libero_uncond_2cam224)，
  revision `a784645f9ce367ba6953dd20a7a7f0310c85747c`；`model.safetensors` SHA-256
  `251b09499056a2cd784bbe31f12e1e74fb90d9d70960552a5131c74f3ddb791c`。

### 论文状态与可声称范围

Fast-WAM 目前的可核对出版状态是 arXiv preprint，本计划不写成“已被同行评审
会议接收”。它仍然可作为第一个文献锚点，因为它已公开：

- 完整 method/paper result；
- 训练和 LIBERO/RoboTwin evaluation 代码；
- 官方 LIBERO/RoboTwin checkpoints 与 dataset stats；
- LeRobot `policy.type=fastwam` 的 dataset/policy/checkpoint/eval 接口。

论文主张的 future target 不是 RGB pixel MSE，而是 pretrained VAE 产生的
future-video latents，并用 Flow Matching 训练：

```text
L = L_action_flow + lambda_video * L_future_video_latent_flow
```

推理时只保留 clean first-frame latent 和 action denoising，不实例化 future noisy-video
tokens，不解码 future video。

### 不能由 released checkpoint 单独证明的事

官方 Hugging Face repo 只发布了主 Fast-WAM checkpoint，没有发布
`Fast-WAM w.o. video co-train` checkpoint。因此：

- released checkpoint eval 只证明 artifact/evaluation 可复现；
- `world co-training -> better direct action` 必须由本计划的等初始化训练对照证明；
- 不允许用论文 Table 2 的数字代替本地 causal reproduction。

## 实验 Base

### 两条代码路径

| 路径 | 用途 | 是否产生机制结论 |
| --- | --- | --- |
| `UP` official FastWAM | 官方 checkpoint eval 和 `lambda_video` causal pair | 是，canonical |
| `LR` LeRobot 0.6.0 | 验证 LeRobot dataset/policy/checkpoint/eval 闭环 | 否，只是 integration cross-check |

LeRobot 文档记录的四 suite 平均值是 96.4%，而 Fast-WAM 论文与上游主路径
报告 97.6%。两者属于不同 serialization/evaluator record，必须分开报告，不要求
checkpoint logits/action 逐值等价。

### 必做 variants

| ID | 初始化 | 训练 | 作用 |
| --- | --- | --- | --- |
| `UP-REL` | 官方 released checkpoint | 不训 | artifact/eval 锚点 |
| `LR-REL` | LeRobot-format released checkpoint | 不训 | LeRobot 集成锚点 |
| `FW-V` | 同 Wan2.2-5B + ActionDiT init | 10 epoch，`lambda_video=1`，video expert 可训 | 主 Fast-WAM reproduction |
| `FW-FZ` | 与 `FW-V` 逐值相同 | 10 epoch，`lambda_video=0`，**video expert 冻结** | **主对照组**（VLA 形状 baseline） |
| `FW-NV` | 与 `FW-V` 逐值相同 | 10 epoch，`lambda_video=0`，video expert 可训 | **可选**：论文同款消融，文献交叉验证 |

`FW-NV` 只去掉 video objective，不冻结 video expert，不改 architecture、attention mask、
action inference 或训练预算。`lambda_video=0` 时可保留 video-loss forward 计算，
以避免引入 compute-path 差异。它对应论文 Table 2 的 `Fast-WAM w.o. video co-train`
（LIBERO average 93.5 vs Fast-WAM 97.6），因此同时充当**机制层面的外部锚点**：主指标对上
不代表机制对上。

### 主对照选 `FW-FZ` 的理由，以及 `FW-NV` 为何仍应尽量跑

`FW-FZ` 冻结 video expert，只训 action expert 与 `proprio_encoder`（约 1 B / 6.02 B），
`lambda_video=0`。它是三组里最接近 **VLA 形状**的一组：固定视觉主干 + 训练动作头 + 只条件在
当前观测 + 无生成式视频目标。因此「WAM 相对 VLA 买到了什么」这个问题由
`FW-V` vs `FW-FZ` 回答得最直接。

`FW-FZ` 同时也切开了论文回答不了的一个二分。`FW-V` 与 `FW-NV` 的差值能证明「保留 video
objective 有用」，但分不清两种机制：

- **A（动力学）**：预测未来这个目标让 video expert 的表征编码了对动作有用的动力学信息。
- **B（正则化）**：video loss 只是阻止 action loss 把 Wan 预训练表征拆掉。6.02 B 可训练参数
  对 1,712 条演示严重过剩，action loss 完全有能力为拟合这批轨迹而破坏预训练特征。

论文的四个变体（`Fast-WAM`/`-Joint`/`-IDM`/`w.o. co-train`）全部完整训练 video expert，
没有冻结组，所以论文数据无法区分 A 与 B。

`FW-FZ` 的三点读数：

| 观察 | 归因 |
| --- | --- |
| `FW-FZ ≈ FW-V` | 偏向 **B**：保住预训练表征就够了，不需要额外学动力学；同时说明这个任务上 WAM 相对 VLA 的增益有限 |
| `FW-FZ ≪ FW-V` | 偏向 **A**：video co-training 确实带来了 VLA 形状 baseline 拿不到的东西 |
| `FW-FZ > FW-NV`（若跑了 `FW-NV`） | 最强的 **B** 证据：只用 action loss 训 backbone 会**主动损坏**预训练表征 |

**`FW-FZ` 相对 `FW-V` 同时改变两件事**（冻结 + `lambda_video`），所以它是**架构对比**而非
单变量消融。这是有意的取舍：它回答的是「这个架构值不值」，不是「哪一项配置起作用」。

`FW-NV` 降为可选，但仍值得在算力允许时补上，理由是它是**唯一的机制级外部锚点**：论文
Table 2 的 `Fast-WAM w.o. video co-train` 定义与它一字不差（"removes only the video modeling
objective during training"），LIBERO average 93.5 vs Fast-WAM 97.6。主指标对上不代表机制对上；
少了它，`FW-V` 落在 97.6 附近只能说明主指标复现，无法验证我们复现的是同一个机制。而
`FW-V` vs `FW-FZ` 没有任何已发表参考值，效应量大小无从预期。

## 数据、模型与训练合同

### LIBERO 数据/评测单元

- dataset：官方 Fast-WAM preprocessed LIBERO，四 suites，40 tasks，每 suite
  500 demonstrations，合计 2,000 demonstrations；
- environment：LIBERO + MuJoCo `3.3.2`；
- observation：front + wrist RGB，各 `224x224`，沿 width 拼成 `224x448`；
- state：8D proprioception；action：7D EEF pose + gripper；
- action horizon：32；evaluation replan/execute steps：10；
- raw temporal window：33 steps，`action_video_freq_ratio=4`，模型看 9 video frames；
- instruction 使用官方 dataset task text/template，不改 seen/unseen 口径。

### 模型合同

- Wan2.2-TI2V-5B video expert，30 layers，hidden 3072；
- 1B ActionDiT，30 layers，hidden 1024；
- current clean first-frame latent 是 action branch 唯一可见 video token group；
- future noisy-video tokens 不得泄漏到 action query；
- action/video 均用 Flow Matching，各自独立 noise/timestep；
- `FW-V/FW-NV` 的 model/init/data/window/batch/noise-seed/optimizer 均相同，唯一
  预注册差异是 `lambda_video` 的 `1/0`。

### 训练合同

- endpoint：20,000 optimizer steps，不用 development 选 checkpoint；
- seed：42；8 GPUs；per-GPU batch 16；global batch 128；
- AdamW，lr `1e-4`，weight decay `0.01`，cosine schedule；
- bf16 mixed precision，gradient clip `1.0`；
- 10-step action denoising，CFG 1.0，action horizon 32；
- 同一 data order manifest 和每 step action/video noise seeds；
- `FW-V/FW-NV` 同时保存 step 20k endpoint，中间 checkpoint 只做健康检查。

## 预期如何达成实验目的

1. `UP-REL` 先证明官方 code/checkpoint/stats/LIBERO evaluator 可以完整跑通，
   避免把后续训练失败误判成方法失败。
2. `LR-REL` 验证该方法已进入 LeRobot 0.6.0 的标准 policy/dataset/eval 接口，
   但不替代 upstream causal pair。
3. `FW-V` vs `FW-FZ` 是主对照：同初始化、同数据、同预算、同推理路径，差别是
   video expert 是否随任务更新以及是否保留 video objective。它回答「WAM 相对 VLA 形状
   baseline 买到了什么」，代价是不是单变量消融。
   `FW-NV`（可选）只移除 video loss 而不冻结，用于对上论文的 93.5 vs 97.6，验证复现的是
   同一个机制。
4. 部署时记录 future-token allocation、video denoise/decode call count，确认提升来自
   train-time world supervision，而不是 test-time imagination。
5. 三层测量（artifact、causal、direct-deployment）齐备后由使用者判断是否继续 Light-WAM
   reproduction；本计划不自动给出解锁结论。

## 资源与环境隔离

### 运行时镜像提供环境，repo 提供 pinned 输入

当前 M2-W 环境固定 LeRobot `0.4.4` / Python 3.10，且不包含 FastWAM policy。
W0 要求运行时镜像预先提供两个独立环境：

| environment | Python | 作用 |
| --- | --- | --- |
| `fastwam-upstream` | 3.10 | 官方 repo 的 canonical eval/training |
| `fastwam-lerobot` | 3.12 | LeRobot v0.6.0 policy/checkpoint/eval cross-check |

不得在 `robot-practice` 环境里 upgrade LeRobot/Torch/MuJoCo，也不得修改 installed
LeRobot package 文件。W0 wrapper 不执行 `conda create`、`pip install`、git clone/fetch、
HuggingFace download、archive extract 或 ActionDiT preprocessing；这些必须在镜像构建或
一次性手工准备阶段完成。pinned checkout 位于 `third_party/w0_fastwam/`，权重/数据位于
`pretrained/w0_fastwam/`，wrapper 只做只读校验并强制离线运行。
镜像还应预先提供 `hf` CLI，供用户在 wrapper 之外把预训练文件直接放进仓库。

### 资源 gate

| 阶段 | 预注册资源 | 不满足时 |
| --- | --- | --- |
| source/config/unit preflight | CPU + 现有本地环境 | 继续修实现 |
| upstream single-worker eval | `1 x >=32 GiB` GPU | 不下载/不启动正式 eval |
| LeRobot 文档同口径 quantitative eval | 优先 `1 x H20 140 GiB` | 降为 API smoke，不报 quantitative parity |
| `FW-V/FW-FZ`（及可选 `FW-NV`）10-epoch causal training | `8 x >=80 GiB` GPU，同机同型号 | 只能报 artifact 层测量，因果对照未测 |

Light-WAM 论文报告的 Fast-WAM 训练峰值是 70.7 GiB/GPU，因此本计划不把
24 GiB 消费卡写成可执行正式训练资源。

## 前置条件

- [ ] 官方 paper/code/HF/LeRobot revisions 和文件 hashes 已写入 manifest。
- [ ] upstream 与 LeRobot 两个独立 environment 通过 import/version smoke。
- [ ] LIBERO 四 suite dataset 路径、episode/task counts 与 MuJoCo 3.3.2 通过审计。
- [ ] `UP-REL` checkpoint/stats 已放入 repo-local `pretrained/` 后与预注册 SHA-256 一致。
- [ ] official evaluator 的 40 tasks x 50 trials 和 seed schedule 已导出并冻结。
- [ ] `FW-V/FW-NV` 初始 state-dict、data order 和 noise manifests 的 hash 一致。
- [ ] 训练前已预留 `8 x >=80 GiB` 同型号 GPU；未满足则不启动 causal pair。

## 执行步骤

### 1. 实现 repo-local reproduction wrapper

计划新增：

```text
configs/experiments/literature/w0_fastwam_reproduction.json
configs/experiments/literature/w0_fastwam_with_video.yaml
configs/experiments/literature/w0_fastwam_no_video.yaml
src/robot_practice/experiments/literature/w0_fastwam_audit.py
src/robot_practice/experiments/literature/w0_fastwam_analysis.py
scripts/experiments/literature/run_w0_fastwam.sh
tests/test_w0_fastwam_protocol.py
tests/test_w0_fastwam_analysis.py
```

wrapper 只管 provenance、overlay config、launch manifest 和 analysis；不复制/改写
Fast-WAM 模型源码。

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_w0_fastwam_protocol.py tests/test_w0_fastwam_analysis.py

bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-config \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json
```

### 2. 冻结 source/artifact/resource preflight

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=preflight \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/preflight
```

preflight 必须在手工准备大权重前登记 GPU 型号/数量/显存、可用磁盘、Python/CUDA/
Torch 组合，并输出 `allowed_phases`。不记录机器名、用户名或 cache 绝对路径。

### 3. 准备并校验 pinned checkout 与镜像环境

代码 checkout 必须由用户/镜像准备到 `third_party/w0_fastwam/`；然后执行：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-environments \
  --vendor-root=third_party/w0_fastwam \
  --output-root=outputs/literature/w0_fastwam/environment
```

该阶段不得改动任何环境或 checkout；它只验证 detached-HEAD、Python import 和 MuJoCo
版本。环境缺失时应更新运行时镜像，而不是在实验命令中安装。

### 4. 将 artifact 直接放入 repo 并校验

checkpoint、dataset、Wan 权重、LeRobot 权重和 ActionDiT 必须先手工放入
`pretrained/w0_fastwam/`，再执行：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-artifacts \
  --cache-root=pretrained/w0_fastwam \
  --vendor-root=third_party/w0_fastwam \
  --output-root=outputs/literature/w0_fastwam/artifacts
```

该阶段只检查 repo-local 路径、目录结构、checkpoint 哈希/字节数和 ActionDiT 是否存在，
并通过 offline 环境变量阻止下游隐式下载。ignored generated artifacts 不得加入 Git。

### 5. `UP-REL` 官方 checkpoint smoke 与 2,000-trial evaluation

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-upstream-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --checkpoint=pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt \
  --trials-per-task=50 \
  --output-root=outputs/literature/w0_fastwam/upstream_released
```

smoke 先只跑 1 task x 2 trials；shape/action toggle/reset 通过后才跑
4 suites x 10 tasks x 50 trials。

### 6. `LR-REL` LeRobot integration cross-check

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-lerobot-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --policy-path=pretrained/w0_fastwam/lerobot \
  --trials-per-task=50 \
  --output-root=outputs/literature/w0_fastwam/lerobot_released
```

若只满足 API-smoke 资源，命令必须写 `quantitative_eval=false`，不生成与
paper/upstream 成功率的 parity 结论。

### 7. 训练 `FW-V/FW-NV`（gate 用）与 `FW-FZ`（归因用）

预算按 epoch 而非固定步数：`configs/task/libero_uncond_2cam224_1e-4.yaml` 是
`num_epochs=10 / max_steps=null`，trainer 由数据集推导出 **21,700 步**，同时决定 cosine
schedule 长度与 5 percent warmup。固定 20,000 步相当于 9.22 epoch = 配方的 92.2%，
因此不等价。

三组串行，每组一条命令（launcher 内建 `nohup`、重定向、`--num-gpus=8`、共享父目录，
使 `pair_contract.json` 与 sibling 预算检查都生效）：

按优先级串行，`FW-NV` 排在最后且可跳过：

```bash
bash scripts/experiments/literature/launch_w0_causal.sh with_video
W0_FREEZE_VIDEO_EXPERT=1 bash scripts/experiments/literature/launch_w0_causal.sh no_video
bash scripts/experiments/literature/launch_w0_causal.sh no_video
```

第二条的父目录由 `W0_FREEZE_VIDEO_EXPERT` 自动派生为 `causal_10ep_frozen/`，因此不会覆盖
第三条的产物，sibling 预算检查也不会把两者当成同一对。它要求镜像已应用冻结补丁
（`w0-fastwam-20260821-fwfz` 及之后）。实测步时：`FW-V` 约 2.53 s/step → 单组约 15.3 小时；
`FW-FZ` 可训练参数约 1 B，预计更快。

每组正式 launch 前跑 5-step distributed smoke，校验 init/data/noise hashes、
attention leak test、loss gradient routing、checkpoint save/load 和 peak memory。
训练期用 `scripts/experiments/literature/watch_w0_stall.sh <train.log>` 监控停滞与
宿主内存（cgroup 用量应稳定在 300 GiB 以内；若按 GiB/step 线性增长立即停机排查）。

#### `FW-FZ` 需要的代码改动

冻结开关不在配置里，需要一个**条件生效**的 vendored 补丁（未设环境变量时行为与现在完全一致，
所以同一个镜像能跑三组）。改 `FastWAM/src/fastwam/trainer.py` 两处：

```python
# _apply_dit_only_train_mode（约 287 行）末尾追加
if os.environ.get("W0_FREEZE_VIDEO_EXPERT", "").strip() not in ("", "0"):
    model.dit.mixtures["video"].eval()
    model.dit.mixtures["video"].requires_grad_(False)

# 约 85 行：按 requires_grad 过滤，否则 ZeRO-1 会为 50 亿冻结参数分配 Adam 状态
trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
```

第二处是必需的：`_apply_dit_only_train_mode` 在第 84 行、优化器构造在第 89 行，而原代码
`list(self.model.dit.parameters())` **不按 `requires_grad` 过滤**。

补丁已落成 `docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch`
（29 行，已验证可 `patch -p1` 干净应用且应用后可编译），需在 Dockerfile 加一层应用它。理由见
该目录 README：在活容器里直接改 vendor 树的改动会随容器消失，历史上曾导致三个测试只因本地
未跟踪 checkout 是脏的而通过。构建与推送可与正在运行的 `FW-V` 并行；容器重建放在
`FW-V`（及 `FW-NV`，若同容器内接着跑）结束之后。

输出父目录由同一个环境变量派生（`causal_10ep_frozen/`），manifest 记录
`freeze_video_expert`，两者都是为了让 `FW-FZ` 不可能被误当成 `FW-NV`。

`mot_checkpoint_mixed_attn` 保持 `false`：gradient checkpointing 在冻结分支上会因输入与参数
都不需要梯度而失效并告警。

### 8. 同 seed schedule 评测 causal pair

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-causal-pair \
  --with-video=outputs/literature/w0_fastwam/train/with_video/checkpoint_step_00020000.pt \
  --no-video=outputs/literature/w0_fastwam/train/no_video/checkpoint_step_00020000.pt \
  --trials-per-task=50 \
  --seed-manifest=outputs/literature/w0_fastwam/preflight/libero_eval_seeds.json \
  --output-root=outputs/literature/w0_fastwam/causal_eval
```

### 9. 汇总测量报告

`--phase=gate-w0` 保留为**汇总器**：它把三层结果、配对 bootstrap CI 与有效性检查聚到一个
JSON 里，便于一次看全。**它输出的 `passed` 字段不再作为判断依据**——阈值已从本计划移除，
是否满意由使用者读数字自行判断。

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=gate-w0 \
  --released=outputs/literature/w0_fastwam/upstream_released/summary.json \
  --causal=outputs/literature/w0_fastwam/causal_eval/comparison.json \
  --deployment=outputs/literature/w0_fastwam/causal_eval/deployment.json \
  --bootstrap-repeats=10000 \
  --bootstrap-seed=20260817 \
  --output=outputs/literature/w0_fastwam/w0_summary.json
```

要看的是里面的 SR、配对 gap 与 CI、以及有效性检查（completeness / equal budget / safety），
不是 `passed`。

## 必须报告的测量（不设通过/不通过阈值）

**本计划不再设置 pass/fail gate。** 判断由使用者依据下面的测量结果自行做出；这里只规定
「必须测什么、什么条件下这个数字才有意义」。原先的冻结阈值已移除，因此也不存在事后放宽阈值的
风险；代价是本计划不自动产出「是否解锁 W1」的结论。

### 测量有效性（不满足则数字无意义，必须修而不是叙述）

这几条不是判据，是测量成立的前提：

| 条件 | 要求 |
| --- | --- |
| completeness | 每个被评测变体完成 40 tasks x 50 trials，无缺失/重复 episode |
| equal budget | 参与对比的变体之间 init/data/noise/global-batch hash 一致；所有有意的差异（`lambda_video`、video-expert freeze）都必须写进 manifest |
| safety | non-finite action / invalid rollout count = 0 |
| paired unit | 每个 trial 按同 suite/task/initialization seed 配对，配对 bootstrap 10,000 次 |

### 必须报告的量

**artifact 层**

- `UP-REL` 与 `LR-REL` 的 overall 与逐 suite SR、逐 task 明细、失败索引
- provenance：code/HF/config/checkpoint/stats 的 revision 与 hash
- 与 paper reference（Spatial 98.2、Object 100.0、Goal 97.0、Long 95.2、average 97.6）的差，
  标注为 reference 而非本地已测

**主对照 `FW-V` vs `FW-FZ`**

- 两组 overall 与逐 suite SR，配对 gap 与 95% CI
- CI 的位置要如实叙述。三种情形都是结果，不要只报有利的一种：CI 不含 0（检出效应）、
  CI 整段落在 ±2pp 内（可称等价）、CI 宽且跨 0（**样本量不足，不得挑方向叙述**，需补 seed 或 trial）
- 该对比同时改变 `lambda_video` 与 freeze，因此是架构对比而非单变量消融，报告中必须写明

**可选 `FW-V` vs `FW-NV`**

- 跑了就报同样的量，并与 paper 的 97.6 vs 93.5（4.1pp）对照；不跑则明确写
  「未验证机制级复现」

**部署行为**

| 量 | 记录内容 |
| --- | --- |
| output contract | inference 只返回 `[B,32,7]` action chunk |
| future allocation | future noisy-video token allocation count（Fast-WAM 路径应为 0） |
| imagination calls | future video denoise steps、VAE future decode calls（应为 0） |
| action behavior | `n_action_steps=10`，gripper toggle/normalization 与官方一致 |
| latency | batch-1 median/p95 与 GPU 型号；不跨硬件与 paper 190 ms 做逐值比较 |

**LeRobot 集成**

`policy.type=fastwam` 注册、`from_pretrained`、processors、`predict_action_chunk`、
`select_action`、LIBERO gripper toggle 是否全通过；只做 smoke 时写
`api_pass/quantitative_not_run`，不得声称已复现 LeRobot benchmark number。

## 必记实验字段

| 字段 | W0 记录内容 |
| --- | --- |
| task suite | LIBERO Spatial/Object/Goal/Long，40 tasks |
| dataset size | 2,000 demonstrations；每 suite 500 |
| observation keys | front+wrist RGB 224²，8D proprio，task text |
| action space | 7D EEF pose + gripper |
| action horizon | chunk 32，execute/replan 10 |
| control frequency | 按 official LIBERO/Fast-WAM dataset/evaluator 记录，不从 StackCube 20Hz 类推 |
| train loss | action Flow、video-latent Flow、total，各自 grad norm |
| open-loop action error | denormalized action L1/L2，padding-aware |
| closed-loop success | 50 trials/task，suite 和 overall SR，paired gap/CI |
| latency | batch-1 action query median/p95、peak GPU memory、hardware/dtype |
| failure cases | suite/task/seed、timeout、gripper toggle、invalid action、load/eval failure |

## 产物路径

- `configs/experiments/literature/w0_fastwam_reproduction.json`
- `configs/experiments/literature/w0_fastwam_with_video.yaml`
- `configs/experiments/literature/w0_fastwam_no_video.yaml`
- `outputs/literature/w0_fastwam/preflight/`
- `outputs/literature/w0_fastwam/artifacts/`
- `outputs/literature/w0_fastwam/upstream_released/summary.json`
- `outputs/literature/w0_fastwam/lerobot_released/summary.json`
- `outputs/literature/w0_fastwam/train/with_video/checkpoint_step_00020000.pt`
- `outputs/literature/w0_fastwam/train/no_video/checkpoint_step_00020000.pt`
- `outputs/literature/w0_fastwam/causal_eval/comparison.json`
- `outputs/literature/w0_fastwam/w0_gate.json`

## 风险与停止条件

- 本地只有 24 GiB 级 GPU 时，停在 preflight，不用 CPU offload/降分辨率/减模型
  产生伪“官方复现”。
- 未获得 `8 x >=80 GiB` 同型号 GPU 时，不训 causal pair。
- 若 official repo/config/checkpoint revisions 变动，继续使用本计划 pinned revision；
  升级必须新建 protocol revision。
- 若 released checkpoint 的 SR 明显低于 paper reference，先审计 MuJoCo、action toggle、
  dataset stats、task text 和 seed schedule，不启动正式训练。
- 若参与对比的两组出现了 manifest 未记录的差异，两个 run 全部作废。
- 若 video target/future token 泄漏到 direct action 输入，停止并修 attention-mask test。
- 不得用 `UP-REL` 的高成功率替代因果对照的结论：artifact 能跑通与机制成立是两件事。
- 不得把 Fast-WAM 的 future VAE-latent Flow loss 写成 pixel reconstruction loss。
- 不得将 LeRobot adapter 的 96.4% 记录与 upstream/paper 97.6% 混成同一 result。

## 收尾

- [ ] 将 artifact reproduction、causal reproduction 和 LeRobot integration 分开写入执行当日报告。
- [ ] 不向 `plans/` 回写 loss/success/latency 结果。
- [ ] 三层测量齐备、有效性检查全部满足后，本文件改为 `done`。
- [ ] Light-WAM W1 是否从 `blocked` 改为 `todo`，由使用者读完测量结果决定，不由本文件的
      任何自动判据决定。
