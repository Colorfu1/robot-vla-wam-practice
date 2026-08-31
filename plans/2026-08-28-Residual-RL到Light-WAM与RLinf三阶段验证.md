# 2026-08-28 Residual RL 到 Light-WAM 与 RLinf 三阶段验证

## 状态

- 状态：`todo`
- 范围：严格按顺序完成三个问题：Policy Decorator 在 ManiSkill 的代表任务复现；自建
  Light-WAM 及其 frozen-base residual RL 对照；RLinf 在 RoboTwin 的 GRPO/PPO 复现。
  本轮 S1 只做官方 artifact/runtime audit 和 residual 接线 smoke，不自动执行 4M residual SAC；
  需要 residual 正式结果时，单独启动 pinned checkout 的官方 online 接口。
- 最后更新：2026-08-28
- 替代关系：本计划是这三个问题的唯一母执行顺序；既有 Fast-WAM/Wan 视频验证计划不是本计划
  的前置条件，其结果也不得混入本计划的 RL 因果对照。

## 目标

1. 在 ManiSkill2 `PegInsertionSide-v2` 上按 Policy Decorator 官方接口准备并审计 frozen Diffusion
   Policy；用 smoke 验证 residual 接线。本轮不做 `no RL` vs `+ residual SAC` 的性能对照。
2. 按官方 RoboTwin 配置训练并验证一个 Light-WAM，再做同一 frozen Light-WAM 的
   `no RL` vs `+ residual SAC`。
3. 在 RLinf 官方 `place_empty_cup / demo_randomized` 协议上复现同一 OpenVLA-OFT SFT 基座的
   `no RL`、`+ GRPO`、`+ PPO`；不跨基座宣称三种 RL 方法的优劣。

## 固定路线与结论边界

```text
S1  Policy Decorator / ManiSkill2
    PegInsertionSide-v2 + official Diffusion Policy
    official artifact/runtime audit ──→ residual SAC 接线 smoke（不做 formal RL）
                         |
                         v
S2  Light-WAM / RoboTwin2.0
    move_stapler_pad + locally trained Light-WAM
    frozen base ──vs── frozen base + residual SAC
                         |
                         v
S3  RLinf / RoboTwin2.0
    place_empty_cup + official OpenVLA-OFT SFT
    SFT ──vs── GRPO ──vs── PPO
```

- S1 仍只覆盖**一个任务**：smoke 阶段是官方 artifact/runtime audit + residual 接线检查；正式
  论文对齐 run 使用十个 seed，但这不等于论文四个 ManiSkill 任务的完整复现。若要生成
  residual checkpoint，只能额外启动 pinned checkout 的官方 online 脚本；S1 通过后是否补齐论文
  全任务不阻塞 S2。
- S2 的主对照只允许使用同一个本地训练 Light-WAM checkpoint；released checkpoint 只负责验证
  环境、权重和 evaluator 接通，不能代替本地构建的 base 进入主结论。
- S2 固定 `move_stapler_pad`。官方 released Light-WAM 的参考值为 clean 26%、randomized 34%
  （仅是外部锚点，不是本地结果）；该任务有 headroom，且精细放置错误比长程语义错误更适合
  测 residual correction。
- S3 固定 `place_empty_cup / demo_randomized`，因为 RLinf 在同一 OpenVLA-OFT 基座上同时发布了
  SFT 75.78%、GRPO 94.53%、PPO 92.97%。这些是 128 个固定 eval seed 的外部复现目标。
- S2 的 residual SAC 与 S3 的 full-policy GRPO/PPO 使用不同 base、观察、动作和训练预算；
  本轮 S1 不训练 residual。只能分别计算有正式 treatment 的阶段相对各自 base 的增益，禁止
  直接按最终成功率给算法排名。

## 前置条件

- [ ] 仓库根目录执行，`PYTHONNOUSERSITE=1` 保持开启；生成数据、模型、视频和日志只写入
  `datasets/`、`outputs/` 或 ignored `third_party/`，不纳入版本控制。
- [ ] S1 的 Policy Decorator checkout 为
  `92ba9ba442587ae5989c286355ace2200a0537fb`，ManiSkill2 为 `0.5.3`；现有 source/environment
  audit 和 50/50 native/strict replay smoke 全部通过。S1 不再等待旧 M1-R 的 986 条 full replay、
  `D_core` 或 state banks。
- [ ] S1 官方 Diffusion Policy checkpoint 目录同时包含 `best.pt` 与训练时 `args.json`；下载后
  先记录文件 hash，不以本地重新训练的 base 冒充官方 checkpoint。
- [ ] S1 每个官方 worker 使用单卡 16 env；十个 seed 可由外层 launcher 跨多卡并行。S2/S3
  只在远端多卡机器执行，本机单张 RTX 3090 不作为 S2/S3 的运行环境。
- [ ] S2 至少有 4 张可训练 GPU，且磁盘按下表实测值预留；空间不足时不开始下载/解包。

  | 项 | 性质 | 体积 | 备注 |
  | --- | --- | ---: | --- |
  | `lightwam-offline-cache` 的 `robotwin_3cam384_sharded` | 派生缓存 | **1,108 GB** | 6 分片（5×214.7 + 34.4）；`cat parts \| tar -xf -` 要求分片与解压结果同时在盘，**峰值约 2.2 TB**，解压后删分片可回落到 1.1 TB |
  | `robotwin2.0-fastwam` | 原始训练数据 | 79 GB 压缩 | 8 分片；解压后另占一份 |
  | `Wan2.1-T2V-1.3B` | 冻结 backbone 权重 | 17.6 GB | 训练与评测都要 |
  | `lightwam-checkpoints` | released 权重 | 全量 18.6 GB → **只取 3.72 GB** | 只需 `lightwam_robotwin/`，用 `--include` 过滤，省 14.9 GB |
  | RoboTwin `assets` | 仿真资源 | 解压 **16 GB** | 901M embodiments + 4.4G objects + 11G background_texture；解压后删 zip 再省约 15 GB |

  **cache 全下的话 S2 峰值约 2.4 TB，清理分片后稳态约 1.3 TB；不下 cache 则约 200 GB。**
  latent cache 的 1.1 TB 是产物本身大小，自行 `RUN_VIDEO=true` 生成也是同样占用，只是把下载
  换成 GPU 编码时间。它**只被 §2.4 正式训练通过 `LATENT_CACHE_DIR` 消费**，rollout、评测、
  residual RL、S1、S3 都不需要。**是否需要它由 §2.3 的「VAE 现算 vs 预编码」对照决定，该
  smoke 只需 raw data 与 Wan 权重，先做完再决定是否下载。** 若要缩小它必须缩小训练任务集，
  而官方 released Light-WAM 训练覆盖全部 50 个任务，缩集后 §2.5 的 26%/34% parity 锚点即
  失效，届时必须改研究问题并新建计划。
- [ ] S2 的 `num_envs=16` 需要 RoboTwin 侧的并行 env 支持，而 vendored RoboTwin 2.0 的
  `envs/`、`script/` 中没有 `num_envs`/`VectorEnv`/`make_vec`（S3 之所以要拉 `RLinf_support`
  分支，正是因为并行 env 需要额外机制）。开始 §2.6 前必须二选一并记录：实现并验证并行 env
  wrapper，或接受单 env 预算。单 env 实测上界（RTX 3090、`click_bell`、`replan_steps=4`）：
  `get_obs` 106.2 ms + 4×`take_action` 47.4 ms = 295.8 ms / 4 actions = **73.95 ms per
  low-level action → 1M actions ≈ 20.5 h/seed 仅环境，三 seed ≈ 61.6 h**，另加 250,000 次
  Light-WAM chunk 推理（未测）。见
  [本机 RoboTwin 渲染探针](2026-08-27-RoboTwin本机渲染与闭环吞吐探针.md)；该探针在
  `warp 1.5.1` 与驱动 580 存在 `cuDeviceGetUuid` 报错的状态下测得，curobo GPU 加速可能退化，
  且为 3090 数字，正式机器须重测。
- [ ] S2 使用 Light-WAM commit
  `b2785f66e13fd9987e94ae1ecc1c441d5059c9ae`、Wan revision
  `37ec512624d61f7aa208f7ea8140a131f93afc9a`、raw-data revision
  `aac262c35d02cc71b2f6ef670bd65fd9f2bb2547`、cache revision
  `995bd27d4848cf5b0afefb4d33657ae127205076`、released-checkpoint revision
  `7cc8593fb95423a9cfbb93f82c95c2fa7d5357bd`。
- [ ] S2 的本机 RoboTwin probe 已完成；正式机器仍须先用同一环境跑 5 episode model smoke。
- [ ] S3 至少有 8 张满足官方配置显存要求的 GPU；不足 8 张时停止，不静默修改
  `component_placement`、global batch、128 env 或训练算法。
- [ ] S3 使用 RLinf commit `7ad72d1160858971293fec4d2cca232642320574` 和 RoboTwin
  `RLinf_support` commit `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，并能使用官方
  `rlinf/rlinf:agentic-rlinf0.4-robotwin` 镜像。

## 统一实验合同

执行前先实现并冻结：

- `configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml`：所有 revision、task、seed、
  horizon、action contract、预算和阈值；S1 residual formal training 标记为禁用；
- `scripts/experiments/rl/run_three_stage_validation.sh`：`preflight`、`evaluate`、`summarize` 入口；
- `scripts/experiments/rl/run_policy_decorator_paper_multigpu.sh`：S1 论文十 seed 的单卡 worker
  并行 launcher；
- `scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh`：S1 base 评测完成后再
  启动官方 SAC 的串行 wrapper；
- `scripts/experiments/robotwin/run_lightwam_residual_rl.sh`：S2 residual 的 smoke/train/eval 入口；
- `src/robot_practice/experiments/rl/`：S2/S3 的 episode-level manifest、paired bootstrap、
  checkpoint hash、progress heartbeat 和失败分类；S1 官方结果不依赖这些本地 evaluator；
- 对应测试。任何 long-running 入口必须向 stdout 和各自 `run.log` 每 60 秒内至少写一次
  `STATUS completed=<n> total=<N>`，结束写 `DONE completed=<N> total=<N>`。

实现后先运行：

```bash
PYTHONNOUSERSITE=1 conda run -n robot-practice \
  python -m pytest -q tests/experiments/rl tests/experiments/robotwin

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=preflight \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root=outputs/rl_three_stage_20260828/preflight
```

`preflight` 必须保存 resolved config、git/Hugging Face revisions、环境包版本、GPU 信息、文件 hash、
seed 列表和 train/development/test 无交集审计；不把机器名或绝对路径写进可分享汇总。

## 执行步骤

> 可复制执行的命令以配套 guide [三阶段验证指南](../reports/guides/rl/three_stage_residual_lightwam_rlinf.md) 为准；本计划只规定顺序、冻结合同和验收标准。尤其不得继续使用本计划早期版本中把 `LATENT_CACHE_DIR` 传给 `precompute_robotwin.sh`、省略 `MAX_STEPS` 或直接依赖 `eval_robotwin.sh` 默认 `checkpoints/Wan-AI` 的旧片段。guide 已把这些路径改为显式 Hydra override、`MAX_STEPS=null` 和安全 symlink 检查。

### 1. S1：复现 Policy Decorator 的 ManiSkill 代表任务

#### 1.1 固定官方 base 与协议

官方 checkpoint 尚未就位时，按 guide §1.2 的两个固定 file id 分别下载
`best.pt` 与同目录 `args.json`；不要使用 `gdown --folder`，也不要猜测下载器生成的目录层级。
下载后在同一 Bash 会话中设置唯一的 `s1_ckpt` 变量并运行：

```bash
if [[ -n "${S1_CKPT:-}" ]]; then
  s1_ckpt="${S1_CKPT}"
else
  mapfile -t s1_checkpoints < <(
    find outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints \
      -type f -name best.pt -print
  )
  test "${#s1_checkpoints[@]}" -eq 1
  s1_ckpt="${s1_checkpoints[0]}"
fi
test -f "${s1_ckpt}"
test -f "$(dirname "${s1_ckpt}")/args.json"

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s1-audit \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --checkpoint="${s1_ckpt}" \
  --output-root=outputs/rl_three_stage_20260828/stage1/audit
```

只使用：state observation、`pd_ee_delta_pose` 7D action、20 Hz、episode limit 200、
`obs/pred/action/execute horizon = 2/16/4/4`、sparse success reward。base checkpoint 在后续全部
S1 运行中必须保持相同 SHA-256。

#### 1.2 官方 residual run（需要正式 residual 结果时才执行）

Policy Decorator 的 `online/pi_dec_diffusion_maniskill2.py` 同时负责 residual SAC 训练和周期性
闭环评测，没有独立的 eval-only CLI。`best.pt` 只是 frozen base；需要 residual 结果时，直接从
pinned checkout 运行上游命令，不经过本仓库的 runner/evaluator：

论文结果需要先做 frozen base 的十 seed 闭环，再做十个独立的官方 residual-SAC run。使用一条
wrapper 命令串行完成两阶段：base 全部成功且 checkpoint hash 不变后，才启动官方 SAC。每个
worker 只看到一张 GPU，不能改成 `torchrun`/DDP；GPU 数少于 10 时按批次运行。论文正文只公布
“十个 seed”，没有公布 canonical seed-id 列表，因此默认的 1--10 必须记录在 manifest。S2 的
三 seed 约定不受本节影响。

```bash
bash scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh \
  --gpus=0,1,2,3 \
  --base-policy-ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --output-root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac
```

launcher 固定 `PegInsertionSide-v2` 的官方任务参数：`res_scale=0.1`、`prog_explore=30,000`、
`total_timesteps=4,000,000`、`num_envs=16`、每 50,000 steps 的 50-episode 评测、每 1,000,000
steps 保存 checkpoint，其余参数保持 pinned 官方默认值。论文将每个评测点的 10 个 seed 结果求均值
并报告标准差；launcher 不自动修改或合并 TensorBoard event。正式训练未获明确批准时跳过本节；
S1 本轮只做下面的接线 smoke。

#### 1.3 residual 接线 smoke（本轮执行）

只使用 guide 中的 dummy smoke 命令。它不能读取真实 checkpoint、不能产生 residual checkpoint，
也不能作为官方成功率或论文数字。

#### 1.4 S1 结果边界

本轮不生成 residual formal 结果，不运行自定义 `policy_decorator_eval.py` 的正式闭环或 paired
汇总。若后续执行 §1.2，只报告官方脚本自身的周期评测、最终 checkpoint、`args.json`、stdout
和 TensorBoard event；本地 wrapper/evaluator 输出只能标记为诊断。

### 2. S2：构建 Light-WAM，并做 Light-WAM vs Light-WAM + residual RL

S2 必须在 S1 已得到完整正/负结论后开始；S1 未达正向阈值仍可进入 S2，但必须标记为
“迁移探索”，不得写成“已复现方法后迁移”。

#### 2.1 固定源码和官方 artifacts

```bash
practice_root="$(pwd)"
test -d third_party/lightwam/Light-WAM/.git || \
  git clone https://github.com/L1ziang/Light-WAM.git third_party/lightwam/Light-WAM
git -C third_party/lightwam/Light-WAM fetch origin
git -C third_party/lightwam/Light-WAM checkout b2785f66e13fd9987e94ae1ecc1c441d5059c9ae
```

按官方 README 建**两个**环境，用途不可互换：`lightwam` 做 precompute 与训练，
`lightwam-robotwin-eval` 只做评测。

```bash
conda create -n lightwam python=3.10 -y
conda run -n lightwam python -m pip install -U pip
conda run -n lightwam python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n lightwam python -m pip install -e third_party/lightwam/Light-WAM

conda env create -f third_party/lightwam/Light-WAM/scripts/robotwin_env.full.yml
conda run -n lightwam-robotwin-eval \
  python -m pip install -e third_party/lightwam/Light-WAM --no-deps
```

RoboTwin `assets` 解压后约 16 GB。仓库内 FastWAM vendor 已有一份，但两个 checkout 的
RoboTwin revision 不同，**只有下载清单一致时才允许复用**，否则各自下载：

```bash
practice_root="$(pwd)"
fastwam_assets="${practice_root}/third_party/w0_fastwam/FastWAM/third_party/RoboTwin/assets"
lightwam_rt="${practice_root}/third_party/lightwam/Light-WAM/third_party/RoboTwin"

if [ -f "${fastwam_assets}/_download.py" ] && \
   cmp -s "${fastwam_assets}/_download.py" "${lightwam_rt}/assets/_download.py"; then
  echo "下载清单一致，复用已解压 assets"
  for d in embodiments objects background_texture; do
    test -e "${lightwam_rt}/assets/${d}" || ln -s "${fastwam_assets}/${d}" "${lightwam_rt}/assets/${d}"
  done
  (cd "${lightwam_rt}" && conda run -n lightwam python ./script/update_embodiment_config_path.py)
else
  echo "下载清单不一致，独立下载"
  (cd "${lightwam_rt}" && bash script/_download_assets.sh)
fi
cd "${practice_root}"
```

```bash
mkdir -p outputs/rl_three_stage_20260828/artifacts/lightwam
mkdir -p datasets/raw/lightwam datasets/lerobot/lightwam

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir outputs/rl_three_stage_20260828/artifacts/lightwam/Wan-AI/Wan2.1-T2V-1.3B

hf download l1ziang/lightwam-checkpoints \
  --revision 7cc8593fb95423a9cfbb93f82c95c2fa7d5357bd \
  --include "lightwam_robotwin/*" "config.yaml" \
  --local-dir outputs/rl_three_stage_20260828/artifacts/lightwam/released

hf download yuanty/robotwin2.0-fastwam --repo-type dataset \
  --revision aac262c35d02cc71b2f6ef670bd65fd9f2bb2547 \
  --local-dir datasets/raw/lightwam/robotwin2.0-fastwam
```

`--include` 不可省：released checkpoint 只需 `lightwam_robotwin/`（3.72 GB，全量 18.6 GB 含四个
LIBERO checkpoint）。

**latent cache（1,108 GB）暂不下载。** 是否需要它由 §2.3 的 VAE 现算 vs 预编码对照决定；只有
该 smoke 判定需要 cache 时才执行下面这条，并且必须用 `--include` 只取 RoboTwin 分片（全量另含
41.8 GB LIBERO cache）：

```bash
hf download l1ziang/lightwam-offline-cache --repo-type dataset \
  --revision 995bd27d4848cf5b0afefb4d33657ae127205076 \
  --include "latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded.tar.part-*" \
  --local-dir datasets/raw/lightwam/offline-cache-release
```

按发布仓库的分片顺序解包；目标目录必须为空，并使用 `--keep-old-files`，不覆盖已存在的不匹配文件：

```bash
mkdir -p datasets/raw/lightwam/robotwin2.0
test -z "$(find datasets/raw/lightwam/robotwin2.0 -mindepth 1 -print -quit)" || exit 1
cat datasets/raw/lightwam/robotwin2.0-fastwam/robotwin2.0.tar.gz.part-* | \
  tar --keep-old-files -xzf - -C datasets/raw/lightwam/robotwin2.0

mkdir -p datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B
test -z "$(find datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B -mindepth 1 -print -quit)" || exit 1
cat datasets/raw/lightwam/offline-cache-release/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded.tar.part-* | \
  tar --keep-old-files -xf - -C datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B
```

生成 RoboTwin text cache 后做完整 artifact audit。text cache 用**训练环境** `lightwam`，具体
命令见配套 guide §3.1；必须直接调用 `precompute_text_embeds.py` 并同时覆盖 `data.train` 和
`data.val` 的 dataset/cache 路径，不能把环境变量传给 `precompute_robotwin.sh` 后假定路径已生效。
非交互 shell 统一用 `conda run`，并把 `precompute_text.log` 与 `s2-audit` 产物保留。

#### 2.2 released checkpoint 的 5-episode 接通 smoke

在 smoke 前按配套 guide §3.2 创建并校验 `third_party/lightwam/Light-WAM/checkpoints/Wan-AI`
到已下载 Wan artifact 的非覆盖符号链接；pinned `eval_robotwin.sh` 会强制把
`DIFFSYNTH_MODEL_BASE_PATH` 改为该 `checkpoints` 目录。发现已有真实目录、断链或错误目标时停止。

```bash
practice_root="$(pwd)"
cd third_party/lightwam/Light-WAM
for task_config in demo_clean demo_randomized; do
  PYTHONNOUSERSITE=1 \
  DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
  EVAL_MODE=single ROBOTWIN_TASK_NAME=move_stapler_pad TASK_CONFIG="${task_config}" \
  EVAL_NUM_EPISODES=5 REPLAN_STEPS=24 SKIP_GET_OBS_WITHIN_REPLAN=true \
  CKPT="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt" \
  OUTPUT_DIR="${practice_root}/outputs/rl_three_stage_20260828/stage2/released_smoke/${task_config}" \
  conda run --no-capture-output -n lightwam-robotwin-eval bash scripts/eval_robotwin.sh
done
cd "${practice_root}"
```

smoke 只检查环境、camera schema、14D qpos action、normalizer、checkpoint 和成功判定可运行；不把
5 episode 成功率写成正式结果。

#### 2.3 latent cache 必要性 smoke：VAE 现算 vs 预编码

这一步在**下载那 1.1 TB 之前**做，用来判断预编码 cache 是否值得。它只需要 §2.1 已下载的
raw data 与 Wan 权重，不需要 cache 本身。

**先确认 pinned 行为。** 该 checkout 的 `precompute_robotwin.sh` 将数据和 cache 路径硬编码在
`Light-WAM/data/...`，`train_robotwin.sh` 默认传 `data.train.use_latent_cache=true`；只设置
`LATENT_CACHE_DIR` 不会切换到现场 VAE。guide §3.1 已据此改为直接调用底层 precompute 脚本，
并在训练命令末尾显式传 `data.train.use_latent_cache=false`/`latent_cache_dir=null`。这一结论和
resolved config 必须写入 `s2-audit`，不得静默修改 vendored 代码。

**对照设计（单变量为 latent 来源）。** 固定一个小 probe 子集（例如 2 个任务，含
`move_stapler_pad`），A/B 两组跑**同一子集**、同 GPU 数、同 `BATCH_SIZE=16`、同 seed、同步数；
先 20 步 warmup 不计时，再计时 200 步。

A 组需要该子集的预编码 latent。**不要为此下载 1.1 TB**：pinned `precompute.sh` 不支持自定义
数据/cache 路径或 subset 参数，因此必须按配套 guide §3.1 直接调用
`scripts/precompute_text_embeds.py` / `scripts/precompute_video_latents.py`，用
`val_set_proportion=0.99`、`is_training_set=false` 生成约 1% validation-episode probe；训练 A/B
使用相同 override。guide 中的命令是本计划唯一可复制实现，不能把环境变量传给
`precompute_robotwin.sh` 后假定它生效。

顺带记录 probe 子集的预编码耗时与产物体积，据此外推全量自行生成 cache 的时间与占用，作为
「下载 1.1 TB」之外的第三个选项。

**必须记录的量：**

- 两组的 median 与 p95 `s/step`（去掉 warmup 20 步）；
- dataloader 等待占比：区分「慢在 VAE 前向」还是「慢在读 1.1 TB 随机小文件」——cached 组也
  可能被 IO 拖慢，不能默认它更快；
- **peak VRAM**：现算要在训练显存之外多放 VAE encoder，若在 `BATCH_SIZE=16` 下 OOM，就必须
  改 batch，而改 batch 会破坏官方 5 epoch 配方 → 那 cache 就是**强制项**，直接记录并下载；
- 5 epoch 的总步数（从 manifest 与 batch 算出，不靠猜），据此把 `s/step` 差换算成总墙钟差。

**决策规则（看到数字之前冻结）：**

| 情形 | 决定 |
| --- | --- |
| 现算的总训练墙钟 `<=1.3×` cached，且 `BATCH_SIZE=16` 不 OOM | 走现算，**不下载那 1.1 TB** |
| 现算 `>1.3×` cached，且磁盘满足 2.4 TB | 下载 cache，按 §2.1 的 `--include` 只取 RoboTwin 分片 |
| 现算 OOM 或需改 batch/配方 | cache 强制，下载；不接受降 batch 冒充官方配方 |
| 磁盘不足 2.4 TB 且现算可行 | 只能走现算，把额外墙钟代价明确记入 audit 与最终报告 |

该 smoke 的 loss 与成功率不作为任何结论，只用它的吞吐、显存和 IO 画像做决策。

#### 2.4 按官方 5 epoch 配置训练本地 Light-WAM

先用 `MAX_STEPS=20` 做一次训练 smoke；通过后从干净输出目录运行正式 5 epoch。训练用
`lightwam` 环境（不是评测环境）。正式 run 必须显式传 `MAX_STEPS=null`，让 Hydra 将它解析为
`None` 并按 `NUM_EPOCHS=5` 计算总步数；省略该 override 会触发 pinned wrapper 的 `500000`
默认值并覆盖 epoch budget。具体带 `VAE_MODE` 分流、显式 cache 开关和完整命令见配套 guide §3.2：

具体命令（包括 `VAE_MODE=online|cached` 分流、显式 `data.train.use_latent_cache`、安全输出目录检查和
`MAX_STEPS=null`）统一见配套 guide §3.2，不再在计划中复制一份容易漂移的 wrapper 命令。

启动 smoke 前把 `scripts/train_robotwin.sh` 的 resolved config 写入 `s2-audit`，确认
`MAX_STEPS=null` 实际解析为 `None`，并确认 `VAE_MODE` 与缓存决策一致。

禁止从 training/eval success 选模；固定使用 5 epoch 结束时最后一个完整 checkpoint。训练必须保持：
三相机、`384x320`、33 frames、32-action chunk、Wan backbone frozen + published LoRA/adapters、
StateFusionActionExpert、future-video latent loss 和官方 action loss。

#### 2.5 验证本地构建的 base，并冻结 residual 合同

对 released 与 local-final 各跑 100 episode clean/randomized、`replan_steps=24`。随后只对 local-final
补跑同一批 episode 的 `replan_steps=4`，作为 residual 主对照 base：

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-eval-base \
  --task=move_stapler_pad \
  --released-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --local-run=outputs/rl_three_stage_20260828/stage2/train_formal \
  --settings=demo_clean,demo_randomized \
  --episodes=100 \
  --official-replan-steps=24 \
  --residual-replan-steps=4 \
  --seed=42 \
  --output-root=outputs/rl_three_stage_20260828/stage2/eval_base
```

Residual 合同固定如下：

```text
base: local-final Light-WAM, frozen, hash checked before/after RL
decision/execute horizon: 4/4
actor input: frozen adapted-state pooled feature + 14D robot qpos; no privileged object pose
actor output: 4 x 12 arm-joint residual; two gripper channels remain exactly base output
combine: a_final[arm] = clip(a_base[arm] + 0.1 * action_std_arm * tanh(a_res), joint limits)
         a_final[gripper] = a_base[gripper]; action_std_arm comes from frozen physical-unit dataset stats
critic input: observation + final combined 4 x 14 physical qpos target
reward: sparse success only
algorithm: SAC; gamma=0.97, UTD=0.25, batch=1024, actor/q lr=1e-4, tau=0.01
exploration: progressive residual enable schedule over first 30,000 low-level actions
budget: 1,000,000 aggregate executed low-level actions per seed; learning_starts=8,000; seeds 1,2,3
train/eval split: train start-seed family 0; development 41; formal test 42
```

`replan_steps=24` 的官方结果与 `replan_steps=4` 的 residual 对照分开报告，禁止把 24-step base 与
4-step `+RL` 直接比较。paired evaluator 必须让 base/treatment 复用同一 episode manifest；仅有相同
seed 不构成配对。

#### 2.6 residual 接通 smoke（正式训练前必做）

§2.2 的 smoke 只验证 released ckpt 能 rollout，§2.4 的 `MAX_STEPS=20` 只验证 Light-WAM 训练能
跑。residual RL 自身的接通尚未验证，正式烧 1M×3 之前必须先做这一步，且用**已有 ckpt**
（local-final 若还没训完就先用 released，仅此 smoke 允许），不产出任何正式数字：

```bash
practice_root="$(pwd)"
mkdir -p outputs/rl_three_stage_20260828/stage2/residual_smoke

# a) rollout smoke：base 冻结、residual 恒为 0，确认组合后的动作与纯 base 逐步一致
bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-rollout \
  --task=move_stapler_pad --task-config=demo_randomized \
  --base-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --episodes=2 --replan-steps=4 --zero-residual=true \
  --output-root=outputs/rl_three_stage_20260828/stage2/residual_smoke/rollout

# b) train smoke：极短预算跑通采样→存 buffer→SAC 更新→存 ckpt 全链路
bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-train \
  --task=move_stapler_pad --task-config=demo_randomized \
  --base-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --seed=1 --total-low-level-actions=12000 --learning-starts=1024 --num-envs=2 --replan-steps=4 \
  --res-scale=0.1 --prog-explore=30000 \
  --output-root=outputs/rl_three_stage_20260828/stage2/residual_smoke/train
```

smoke 必须同时给出并记入 `s2-audit` 产物：

- `--zero-residual=true` 时组合动作与纯 base 输出逐步相等（验证 `a_final = a_base + 0.1 *
  action_std * tanh(a_res)` 的接线与 gripper 直通）；
- base checkpoint hash 在 smoke 前后不变，`requires_grad` 全为 `False`；
- 两个 gripper 通道的 residual 恒为 0；
- 实测吞吐：low-level actions/s、每 chunk 的 Light-WAM 推理耗时、SAC 更新耗时。**用它把 1M×3
  的墙钟时间算出来写进 audit**；若单 env 且超过 §前置条件 里 61.6 h 的环境下界估计，先决定
  并行 env 方案再进入 §2.7，不要直接开跑。

smoke 的成功率不作为任何结论。

#### 2.7 训练 residual 并做正式配对评测

```bash
for seed in 1 2 3; do
  bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
    --phase=train \
    --task=move_stapler_pad \
    --task-config=demo_randomized \
    --base-run=outputs/rl_three_stage_20260828/stage2/train_formal \
    --seed="${seed}" \
    --total-low-level-actions=1000000 \
    --learning-starts=8000 \
    --num-envs=16 \
    --replan-steps=4 \
    --res-scale=0.1 \
    --prog-explore=30000 \
    --output-root="outputs/rl_three_stage_20260828/stage2/residual_train/seed_${seed}"
done

for setting in demo_clean demo_randomized; do
  bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
    --phase=evaluate \
    --task=move_stapler_pad \
    --task-config="${setting}" \
    --base-run=outputs/rl_three_stage_20260828/stage2/train_formal \
    --residual-runs=outputs/rl_three_stage_20260828/stage2/residual_train \
    --seeds=1,2,3 \
    --eval-seed=42 \
    --episodes=100 \
    --replan-steps=4 \
    --output-root="outputs/rl_three_stage_20260828/stage2/eval_residual/${setting}"
done

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-summarize \
  --base=outputs/rl_three_stage_20260828/stage2/eval_base/replan4 \
  --treatment=outputs/rl_three_stage_20260828/stage2/eval_residual \
  --output-root=outputs/rl_three_stage_20260828/stage2/summary
```

### 3. S3：复现 RLinf 在 RoboTwin 的 GRPO/PPO

S3 在 S2 形成完整正/负结论后开始；S2 的 residual 是否提升不阻塞 S3。S3 使用独立 checkout、
独立模型和独立结果目录。

#### 3.1 固定 RLinf、RoboTwin 与三个官方模型

```bash
test -d third_party/rlinf/RLinf/.git || \
  git clone https://github.com/RLinf/RLinf.git third_party/rlinf/RLinf
git -C third_party/rlinf/RLinf fetch origin
git -C third_party/rlinf/RLinf checkout 7ad72d1160858971293fec4d2cca232642320574

test -d third_party/rlinf/RoboTwin/.git || \
  git clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git \
  third_party/rlinf/RoboTwin
git -C third_party/rlinf/RoboTwin fetch origin RLinf_support
git -C third_party/rlinf/RoboTwin checkout 0008ae6800df9f75fc8de7098bacb01735fd8fd2

practice_root="$(pwd)"
cd third_party/rlinf/RoboTwin
bash script/_download_assets.sh
cd "${practice_root}"

mkdir -p outputs/rl_three_stage_20260828/artifacts/rlinf
hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
  --revision 04150e05ded8b915ccf6adb7ea903ff13dc3aa27 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/sft
hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup \
  --revision 2984c5a20196b26d4e19ce3335e7caad80f1fcb6 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/grpo_released
hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-PPO-place_empty_cup \
  --revision cf9c4730d06279d6eadc11b75946f96342959796 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/ppo_released
```

安装/挂载 RoboTwin assets 后运行 `s3-audit`；必须确认 `place_empty_cup_1k` normalization key、Piper
双臂 14D action、head RGB 224、center crop、25-action chunk、demo_randomized、200-step limit、
128 个固定 eval env 与 train/eval seed 文件 hash 完全一致。

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s3-audit \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root=outputs/rl_three_stage_20260828/stage3/audit
```

#### 3.2 先复评三个 released artifacts

本步骤是环境/发布权重 parity，不是训练复现。对 sft、grpo、ppo 分别运行同一 128-seed evaluator：

```bash
mkdir -p outputs/rl_three_stage_20260828/stage3/released_eval
for arm in sft grpo_released ppo_released; do
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
      rollout.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/${arm} \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.total_num_envs=128 env.eval.rollout_epoch=1 \
      env.eval.use_fixed_reset_state_ids=true \
      env.eval.max_episode_steps=200 env.eval.max_steps_per_rollout_epoch=200 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/released_eval/${arm}" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/released_eval/${arm}.run.log"
done
```

#### 3.3 用官方配置从同一 SFT 分别重跑 GRPO 与 PPO

先各做 1 epoch smoke；正式训练不得改官方 algorithm、batch、env 或 seed，只把路径和 logger
改到本仓库输出。以下命令中的 `runner.max_epochs=1` smoke 通过后，正式改回 `1000`：

```bash
mkdir -p outputs/rl_three_stage_20260828/stage3/train_smoke
for algo in grpo ppo; do
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      export REPO_PATH=\$(pwd) EMBODIED_PATH=\$(pwd)/examples/embodiment \
      PYTHONPATH=\$(pwd):\${ROBOTWIN_PATH}:\${PYTHONPATH} HYDRA_FULL_ERROR=1 \
      python examples/embodiment/train_embodied_agent.py \
      --config-path \$(pwd)/examples/embodiment/config \
      --config-name robotwin_place_empty_cup_${algo}_openvlaoft \
      runner.max_epochs=1 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/train_smoke/${algo} \
      actor.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/sft \
      env.train.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/train_smoke/${algo}.run.log"
done
```

正式运行：

```bash
mkdir -p outputs/rl_three_stage_20260828/stage3/train_formal
for algo in grpo ppo; do
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      export REPO_PATH=\$(pwd) EMBODIED_PATH=\$(pwd)/examples/embodiment \
      PYTHONPATH=\$(pwd):\${ROBOTWIN_PATH}:\${PYTHONPATH} HYDRA_FULL_ERROR=1 \
      python examples/embodiment/train_embodied_agent.py \
      --config-path \$(pwd)/examples/embodiment/config \
      --config-name robotwin_place_empty_cup_${algo}_openvlaoft \
      runner.max_epochs=1000 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/train_formal/${algo} \
      actor.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/sft \
      env.train.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/train_formal/${algo}.run.log"
done
```

GRPO 保持 group size 8、action-level reward、token-level logprob、actor-only；PPO 保持 group size 1、
GAE、actor-critic/value head。二者都使用 actor seed 1234、128 train env、128 eval env、官方
train/eval seed files。正式判断使用所有预定 validation 点及 1000 epoch 结束点，不因看到曲线而
改超参或延长预算。

#### 3.4 汇总 artifact parity 和 training reproduction

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s3-summarize \
  --released-eval=outputs/rl_three_stage_20260828/stage3/released_eval \
  --training-runs=outputs/rl_three_stage_20260828/stage3/train_formal \
  --output-root=outputs/rl_three_stage_20260828/stage3/summary
```

分别报告：released artifact 的 128-seed success；本地训练的最终点与最高预定 validation 点；SFT
到 GRPO/PPO 的百分点变化；训练环境 steps/chunks、吞吐、loss/KL/value、P50/P95 rollout latency 和
失败类型。不得用 released RL 权重的成功率冒充本地训练结果。

### 4. 登记实验记录

三个阶段各自完成后立即追加 `reports/experiment_log.md`，并在三阶段都结束后创建日期化报告。
每个 arm 必须填写：task suite、dataset size、observation keys、action space、action horizon、control
frequency、train loss、open-loop action error、closed-loop success rate、latency、failure cases。

- 无训练 arm 的 `train loss` 写 `N/A（frozen released checkpoint）`。
- 无相同 held-out action label 的 S3 `open-loop action error` 写明 `N/A` 原因，不伪造代理指标。
- 本地训练 Light-WAM 必须报告 1% frozen validation split 的 action error；S1 base 报告 checkpoint
  自带/可重建 held-out split 的 action error。
- 原始 observations/actions/predictions、视频、权重、日志和机器 manifest 只留在 ignored outputs，
  日期化报告只写 aggregate metrics、限制和仓库相对路径。

## 判断标准

**以下标准在看到本地数字之前冻结，不得事后修改。**

| 阶段/判据 | 通过阈值 | 不满足时怎么办 |
| --- | --- | --- |
| S1 协议 | commit/version/hash/schema 全匹配；50/50 replay smoke 保持通过 | 停止，先修协议；不运行 4M RL |
| S1 base headroom | pooled success 在 `[10%, 90%]` | 停止并判为 artifact/protocol 或 headroom 不成立 |
| S1 代表复现 | 相对 base `>=+10pp`；十个 seed 增益同向；paired 95% CI 下界 `>0`（以 4M final checkpoint 计） | 记录负复现；允许进入 S2，但必须降级迁移表述 |
| S1 绝对水平（参考，非门槛） | 记录 4M final pooled success，与论文 Table 报告的 PegInsertionSide + Diffusion base 数字并列呈现 | 论文数字必须先从 [arXiv 2412.13630](https://arxiv.org/abs/2412.13630) 的表格抄录进 config 才能引用；未抄录前只报本地值，不设绝对阈值、不臆造 |
| S1 base 不变 | RL 前后 base checkpoint hash、参数 hash 完全一致 | 结果作废 |
| S2 released parity | replan24 的 clean/randomized 各自与官方 26%/34% 相差 `<=10pp` | 停止，修 evaluator/权重/环境 |
| S2 本地构建 | formal train 完成 5 epoch；无 NaN；local-final 相对本地 released 的 clean/randomized 各自不低于 `-10pp` | 不进入 residual RL；先修构建 |
| S2 latent cache 决策 | §2.3 已给出 A/B 两组的 median/p95 `s/step`、dataloader 等待占比、peak VRAM 与 5 epoch 总步数，并按冻结的决策表落定「现算 / 下载 cache / 本地生成 cache」之一 | 不进入 §2.4 正式训练；不在未测的情况下默认下载 1.1 TB，也不默认现算可行 |
| S2 磁盘与吞吐 preflight | 开跑前实测可用磁盘满足 §2.3 落定方案所需容量（下 cache 则 `>=2.4 TB` 峰值口径，现算则约 200 GB）；并行 env 方案已定并记录；用 §2.6 smoke 的实测吞吐算出 1M×3 的墙钟时间 | 不下载/不开跑，先解决容量或并行度 |
| S2 residual 接通 smoke | `--zero-residual=true` 时组合动作与纯 base 逐步相等；base hash 与 `requires_grad=False` 不变；gripper residual 恒 0；全链路（采样→buffer→SAC 更新→存 ckpt）跑通 | 修接线，不进入正式 residual 训练 |
| S2 residual headroom | local-final replan4 两设置 pooled success 在 `[10%, 90%]` | 记录任务不适合作本轮 residual probe，停止 S2 RL |
| S2 residual 提升 | clean/randomized pooled 平均 `>=+10pp`；三个 seed 平均增益同向；paired 95% CI 下界 `>0`；任一设置不回退超过 `2pp` | 记录负迁移，不做超参 sweep；继续 S3 |
| S2 安全性 | arm residual saturation `<10%`；action jerk P95 相对 base 增幅 `<=20%`；gripper residual 恒为 0；base hash 不变 | 结果作废或判为不安全 |
| S3 released parity | SFT/GRPO/PPO 分别落在 `75.78/94.53/92.97 ±5pp` | 停止训练复现，先修官方协议 |
| S3 GRPO 训练复现 | 预定 validation 点中至少一个 `>=89.53%`，且 1000-epoch final 不低于 SFT `-2pp` | 记录负复现，不追加预算 |
| S3 PPO 训练复现 | 预定 validation 点中至少一个 `>=87.97%`，且 1000-epoch final 不低于 SFT `-2pp` | 记录负复现，不追加预算 |
| 可观测性 | 所有长任务 stdout/run.log 在 60 秒内有 completed/total STATUS，结束有 DONE | 停止任务，先补 progress，不接受只有 final JSON |
| 结论完整性 | 每阶段都有 episode-level 数据、aggregate、hash manifest、预算和失败分类 | 不把阶段标为完成 |

## 产物路径

```text
configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
scripts/experiments/rl/run_three_stage_validation.sh
scripts/experiments/robotwin/run_lightwam_residual_rl.sh
src/robot_practice/experiments/rl/
tests/experiments/rl/
tests/experiments/robotwin/

outputs/rl_three_stage_20260828/preflight/
outputs/rl_three_stage_20260828/artifacts/
outputs/rl_three_stage_20260828/stage1/{audit,train,eval,summary}/
outputs/rl_three_stage_20260828/stage2/{audit,vae_cache_probe,train_smoke,train_formal,eval_base,residual_smoke,residual_train,eval_residual,summary}/
outputs/rl_three_stage_20260828/stage3/{audit,released_eval,train_smoke,train_formal,summary}/
datasets/raw/lightwam/
datasets/lerobot/lightwam/
reports/experiment_log.md
reports/2026-<完成日期>-Residual-RL到Light-WAM与RLinf三阶段验证.md
```

## 风险与停止条件

- S1 的官方 checkpoint/`args.json` 无法确认来源或 hash 时停止；不得用另训 base 替代后仍称论文
  checkpoint 复现。
- S1 只允许官方 Peg Diffusion Policy recipe。不得临时换 BeT、RGBD、ManiSkill3 或其他任务来
  “救结果”。
- S2 数据分片、latent cache、dataset stats、text cache 任一 identity/schema 不匹配时停止；不得
  现场重算一部分 cache 后与发布 cache 混用。
- S2 local-final 构建未过 parity gate 时，不允许直接拿 released Light-WAM 做主 `base + RL` 并
  声称完成“自建 Light-WAM + RL”；如确需该 fallback，另建计划并改研究问题。
- S2 residual 不得读取 object pose、success predicate 内部状态等训练时特权信息；训练时使用而
  部署时不可得的输入会使结果作废。
- 任一 online RL 出现连续 50k low-level actions 无成功、NaN/Inf、residual saturation `>=10%`、
  base hash 改变、OOM 重启三次或 simulator crash rate `>1%` 时停止该 seed 并诊断，不自动扩预算。
- S3 released parity 未通过前不烧 1000 epoch；8-GPU 官方合同不满足时停止，不用降低 batch/env
  数的非等价运行冒充复现。
- 三阶段都禁止根据 formal test 结果选任务、调超参、挑 checkpoint 或重新划 seed；任何协议变化
  必须新建版本化计划和新输出目录。

## 收尾

- [ ] S1 完成并形成正/负结论后再启动 S2
- [ ] S2 完成并形成正/负结论后再启动 S3
- [ ] 三阶段 required experiment fields 已写入 `reports/experiment_log.md`
- [ ] 创建日期化最终报告，区分 released parity、本地训练结果、限制与后续建议
- [ ] 把本文件状态改为 `done`
- [ ] 更新 `plans/README.md` 索引
