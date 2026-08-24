# W0 Fast-WAM LIBERO 复现与机制锚点：详细记录

这是主操作指南的深入记录，保留协议推导、历史结果、故障排查和统计限制。
执行入口请回到 [主指南](../01_w0_fastwam_libero.md)。

## 实验目标

本指南复现 pinned Fast-WAM 的 LIBERO artifact/evaluation 闭环，并训练同初始化的
`FW-V (lambda_video=1)` 与 `FW-NV (lambda_video=0)` 对照，判断 future-video latent
co-training 是否提升 direct action。LeRobot 版本只作为独立 integration cross-check，不能
与 upstream/paper 指标混成一个结果。

当前 W0 已有 `UP-REL` released checkpoint 的正式闭环 baseline（`1929/2000`，96.45%）和
`LR-REL` 的 2000-episode quantitative eval（`1885/2000`，94.25%）；`FW-V/FW-NV` 因果对照与
deployment evidence 仍未完成。LR-REL 的残余 2.2pp 几乎全部集中在 `libero_spatial` task 5 且
原因未解释，因此它**不构成 parity 声明**。所有正式结果分别标为 `Smoke`、`Formal` 或
`Offline`，写入 `outputs/literature/w0_fastwam/` 后再登记到实验运行记录。

重要运行约束：本指南中的 wrapper 是只读审计/启动器，不安装环境、不创建 conda 环境、不
`pip install`、不 clone/fetch 仓库，也不下载或解压预训练参数。运行时必须已经提供
`fastwam-upstream` 与 `fastwam-lerobot` 两个环境；代码 checkout 放在仓库的
`third_party/w0_fastwam/`，或由 Docker 的 `W0_VENDOR_ROOT` 指向镜像内 checkout；所有权重、
数据和 ActionDiT 放在仓库的 `pretrained/w0_fastwam/`。镜像还应预先提供 `hf` CLI，供 Node 3
的人工准备使用；正式 wrapper 阶段只做路径、版本、哈希和离线运行校验。

### 宿主机与 Docker 的统一运行时

宿主机和 W0 Docker 镜像都必须提供同名、同版本的两个环境：
`fastwam-upstream`（Python 3.10，MuJoCo 3.3.2）和 `fastwam-lerobot`（Python 3.12，
LeRobot 0.6.0）。实验命令不区分本地与容器；只要在 repo 根目录执行即可。宿主机首次使用时，
按下面的手工 bootstrap 创建环境；这不是实验 wrapper 的一部分，wrapper 仍然不会安装环境：

```bash
conda create --yes --name fastwam-upstream --clone robot-practice
conda run --no-capture-output -n fastwam-upstream python -m pip install \
  --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e third_party/w0_fastwam/FastWAM \
  -e third_party/w0_fastwam/LIBERO
conda run --no-capture-output -n fastwam-upstream python -m pip install \
  --no-cache-dir mujoco==3.3.2 easydict==1.9 opencv-python==4.6.0.66 \
  robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 future==0.18.2 \
  matplotlib==3.5.3 cloudpickle==2.1.0 gym==0.25.2

conda create --yes --name fastwam-lerobot \
  --override-channels --channel=conda-forge python=3.12 pip
conda install --yes --name fastwam-lerobot \
  --override-channels --channel=conda-forge cmake make cxx-compiler
conda run --no-capture-output -n fastwam-lerobot python -m pip install \
  --no-cache-dir --no-deps robomimic==0.2.0 docker/training/egl_probe_stub
conda run --no-capture-output -n fastwam-lerobot python -m pip install \
  --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e 'third_party/w0_fastwam/lerobot[fastwam,libero]' \
  h5py psutil tensorboardX imageio imageio-ffmpeg
```

完成后，本地与镜像都应满足：

```bash
conda run --no-capture-output -n fastwam-upstream \
  python -c "import fastwam,libero,mujoco,torch; print(torch.__version__, mujoco.__version__)"
conda run --no-capture-output -n fastwam-lerobot \
  python -c "import lerobot,torch; print(lerobot.__version__, torch.__version__)"
```

正式执行 Node 2 的 `validate-environments` 会进一步硬校验 Python 版本、关键依赖版本和
三个 pinned checkout；主机和 Docker 必须得到同一份 `environment_manifest.json` 合同，不能
只看“能 import”。

Docker 只负责提供同一套已准备好的环境。进入镜像的方式是：

```bash
W0_IMAGE='<registry>/<namespace>/practice:latest'
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$W0_IMAGE"
```

镜像内的 pinned source 默认位于 `/opt/robot-practice/w0-vendor`，由镜像中的
`W0_VENDOR_ROOT` 自动选用；宿主机则使用 repo 内的 `third_party/w0_fastwam/`。镜像和宿主机
都不包含仓库数据、checkpoint 或预训练权重，这些仍需放在挂载仓库的
`pretrained/w0_fastwam/` 下。

**读代码前先确认读的是哪一份。** `third_party/w0_fastwam/` 被 `.gitignore` 忽略，可能带有本地
实验性改动，而镜像内的 checkout 是干净的 pinned commit。凭它推断运行时行为会出错，先核对：

```bash
for d in FastWAM LIBERO lerobot; do git -C third_party/w0_fastwam/$d status --porcelain; done
```

镜像侧的权威答案：

```bash
docker run --rm --entrypoint bash <image> -lc 'for d in FastWAM LIBERO lerobot; do git -C /opt/robot-practice/w0-vendor/$d status --porcelain; done'
```

自 `w0-fastwam-20260821-fwfz` 起，预期输出恰好一行 `M src/fastwam/trainer.py`——这是
`docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch`（FW-FZ 的冻结开关，
`W0_FREEZE_VIDEO_EXPERT` 未设时与上游逐字相同）。刻意用 `git apply` 而非替换文件，好处是
**镜像通过 git 自报偏离**。除此之外应为空，HEAD 等于 `Dockerfile.w0-fastwam` 里 pin 的 commit。

所有本地适配都应放在 tracked 文件里——`sitecustomize.py` 的运行时 monkeypatch 或 Dockerfile
层，而不是编辑 vendor 文件。

## 实验思路

1. 固定 FastWAM commit、HF revision、LeRobot v0.6.0 和 checkpoint hash。
2. 先做配置、GPU/磁盘、环境与 artifact gate；gate 不满足就停止。
3. 用官方 evaluator 跑 `UP-REL`，再单独跑 `LR-REL`。
4. 用同一 Wan/ActionDiT 初始化、数据顺序、noise/update seed 和 global batch 训练两组。
5. 用同一 LIBERO initial-state/trial manifest 配对评测，做 10,000 次 paired bootstrap。
6. deployment evidence 单独检查 direct action shape 和 future-video 调用为零。

## 固定协议

| 项目 | 冻结值 |
| --- | --- |
| dataset | `yuanty/LIBERO-fastwam`；四 suite、40 tasks；训练集 1,712 episodes / 277,713 frames（见下表） |
| environment | LIBERO + MuJoCo `3.3.2` |
| observation | front+wrist RGB `224x224`，width concat 为 `224x448`；8D proprio |
| action | 7D EEF pose + gripper；chunk 32；每次执行/replan 10 |
| temporal | raw window 33；`action_video_freq_ratio=4`；9 video frames |
| training | 20,000 steps；8 GPU；per-GPU batch 16；global batch 128；bf16 |
| causal difference | 唯一预注册差异是 `model.loss.lambda_video: 1.0` vs `0.0` |

官方 pinned README/配置的入口是 `scripts/train_zero1.sh`、`run_libero_manager.py` 和
`eval_libero_single.py`；本仓库 wrapper 只负责固定参数、路径、manifest 和汇总，不改 FastWAM
模型源码。UP-REL 评测由本仓库的 Python launcher 直接启动 `eval_libero_single.py` worker，
默认每张卡只运行一个 worker，以避免多个 FastWAM/EGL 进程共享同一张卡时触发底层
`SIGABRT`；每个 worker 也会为每个 trial 新建并关闭一个 offscreen env，避免长期复用
MuJoCo/EGL context 导致 `mjr_readPixels` 直接 abort。子进程显式记录物理 GPU 的
`MUJOCO_EGL_DEVICE_ID`。可通过 `--max-tasks-per-gpu` 显式调整。不依赖 `tmux` 或终端窗口。
[FastWAM pinned README](https://github.com/yuantianyuan01/FastWAM/tree/45d8e1458921d83f8ad6cf9ce993d371208dabd0)

## 训练数据的实际构成与 split

`configs/experiments/literature/w0_fastwam_reproduction.json` 的 `protocol.demos_per_suite: 500`
和 `total_demos: 2000` 描述的是**原始 LIBERO**，不是本实验的训练集。上游发布的是
`_no_noops` 变体（剔掉纯 no-op 的 demonstration），本地实测：

| suite | episodes | frames | frames/ep |
| --- | ---: | ---: | ---: |
| `libero_spatial` | 434 | 53,229 | 122.6 |
| `libero_object` | 457 | 67,309 | 147.3 |
| `libero_goal` | 433 | 52,895 | 122.2 |
| `libero_10` | 388 | 104,280 | 268.8 |
| **合计** | **1,712** | **277,713** | 162.2 |

读法：`pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2/<suite>_no_noops_lerobot/meta/info.json`
的 `total_episodes` / `total_frames`。

这两个数与 released checkpoint 自带的
`libero_uncond_2cam224_dataset_stats.json`（`num_episodes = 1712`、`num_transition = 277713`）
**逐数字相等**，因此本地这份数据集就是上游训练所用的那一份。1,712/2,000 = 85.6%。
`libero_10` 每 episode 268.8 帧、约为其余 suite 的两倍，与它评测时用 600 步上限（其余 300）一致。

### 评测初始状态与训练 demonstration 的重叠

闭环评测的初始状态不是新采样的，来自 LIBERO 固定文件：

```
experiments/libero/eval_libero_single.py:913
    initial_states = list(task_suite.get_task_init_states(cfg.EVALUATION.task_id))
                    ↓  benchmark/__init__.py:158-165
    <init_states>/<problem_folder>/<task>.pruned_init
```

实测每 task 恰好 **50** 条（`libero_spatial (50,92)`、`libero_object (50,110)`、
`libero_goal (50,79)`、`libero_10 (50,45..123)`），而协议是每 task 50 trials——**这 50 条被
全部用掉，一 trial 一条**。LIBERO 原始每 task 也正好 50 条 demonstration，数量精确相等且文件名为
`pruned_init`，因此评测的初始配置**几乎肯定就是 demonstration 的起始配置**。

这不是标签泄漏（policy 仍要自己走完整条闭环，没有动作可抄），但它决定了措辞：评测衡量的是
**"能否从训练见过的初始配置重新执行"**，不是"能否泛化到新的物体摆放"。所有已发表的 LIBERO
数字都在这个口径下，所以 UP-REL 的 96.45% 与文献可比；**但不能写成 held-out 泛化**。

未证实的部分：没有证据表明 `pruned_init[i]` 与 demo `i` 的 `states[0]` 逐字节相等。LIBERO 不带
生成脚本，原始 HDF5 也不在 pinned artifact 里。上述是数量吻合加命名的推断。

`_no_noops` 过滤又带来一个不对称：训练集只有 1,712 条，而评测用满 2,000 条初始状态，所以
**288 个 trial 的初始配置对应的 demonstration 被过滤掉了**。按 task 统计覆盖率：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python scripts/experiments/literature/audit_w0_train_eval_overlap.py \
    --sort-by-coverage \
    --output outputs/literature/w0_fastwam/train_eval_overlap.json
```

按 task 语言字符串 join——**不能按 index**，LeRobot 转换是字母序而 LIBERO 用自己的
`libero_task_map` 顺序，`task_index != task_id`。脚本会在出现无法匹配的语言时报错。

| 量 | 实测 |
| --- | ---: |
| 训练 episodes / 评测 init states | 1,712 / 2,000 |
| 整体覆盖率 | 85.6% |
| 无对应训练 demo 的 trial | 288 |
| 单 task 覆盖率范围 | 0.640（`libero_10` task 9）– 1.000 |
| suite 覆盖率均值 | object 0.914 > spatial 0.868 ≈ goal 0.866 > libero_10 0.776 |

### 覆盖率不解释 task 5，而且在结构上不可能解释

`libero_spatial` task 5（`pick up the black bowl on the ramekin`，UP 96% vs LR 28%）的覆盖率是
**0.760（38/50），在 40 个 task 里排第 7 低，并不异常**。同为 0.760 的另外三个 task 的 LR 分别是
92%、92%、96%；覆盖率更低的（0.640/0.660/0.660/0.700/0.720/0.740）LR 落在 84–98%。

更根本的一点：**覆盖率对 UP-REL 和 LR-REL 完全相同**（同一个 released checkpoint、同一份训练集），
所以它按定义无法解释任何 UP−LR 差异。把它当作 task 5 缺口的候选解释是个结构性错误，已排除。

### 为什么覆盖率不是 100%：过滤的是失败的 demonstration，不是划分 split

OpenVLA 论文 Appendix E.1 给出了 `_no_noops` 数据集的构建规则，直接解释了缺口来源：

> we replay all demonstrations in the corresponding simulation environments and **filter out the
> demonstrations that fail to complete the task** (as determined by the environments' success
> criteria). As a result, we remove **68 of 500** LIBERO-Spatial demonstrations, **46 of 500**
> LIBERO-Object, **72 of 500** LIBERO-Goal, and **121 of 500** LIBERO-Long demonstrations.

同一节还写明评测端不做任何过滤：

> we do not change the test environments but rather use the **same initial environment
> configurations provided by the original LIBERO benchmark**

所以缺口**不是** train/test 划分，而是**数据质量过滤**：重放时没能完成任务的 demonstration 被
丢弃，而它们对应的初始配置仍然留在评测里。`_no_noops` 这个名字只描述了另一项处理（剔除近零
动作的 transition，那一项减少帧数、不减少 episode 数）。

对比我们实测的 episode 数（`yuanty/LIBERO-fastwam` 是 FastWAM 自己的 processed release，
不是 OpenVLA RLDS 的拷贝）：

| suite | 原始 | OpenVLA 剔除 | OpenVLA 保留 | 本地实测 | 差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `libero_spatial` | 500 | 68 | 432 | 434 | +2 |
| `libero_object` | 500 | 46 | 454 | 457 | +3 |
| `libero_goal` | 500 | 72 | 428 | 433 | +5 |
| `libero_10` | 500 | 121 | 379 | 388 | +9 |
| 合计 | 2,000 | 307 | 1,693 | **1,712** | **+19** |

本地一致偏高 19 条，方向一致、量级很小，符合"FastWAM 用 MuJoCo 3.3.2 独立重放并按自己的
success 判据过滤"：目录名 `libero_mujoco3.3.2`，数据配置 `raw_shape: [3,512,512]` 也与
OpenVLA 重新渲染的 256×256 不同。**这不是本地数据有问题，而是另一次独立重建。**

一个重要推论：**未覆盖的 288 个 trial 正是"原始 demonstration 重放失败"的那些初始配置**。
它们可能本身更难（当初的人类演示也没做成），所以下一节那条"LR-REL 更依赖覆盖率"的线索存在
难度混淆——覆盖率低可能只是难度高的代理变量，而非"训练没见过"在起作用。这进一步削弱该线索。

### 一条弱线索：LR-REL 的成功率比 UP-REL 更依赖覆盖率

suite 内相关（n=10，pooled 值被 suite 难度混淆，不要读）：

| suite | `r(coverage, UP%)` | `r(coverage, LR%)` |
| --- | ---: | ---: |
| `libero_spatial` | 0.008 | 0.521 |
| `libero_object` | −0.070 | n/a（全 100%） |
| `libero_goal` | 0.298 | **0.752** |
| `libero_10` | 0.107 | **0.704** |

UP-REL 基本与覆盖率无关（0.01–0.30），LR-REL 在 goal / libero_10 上明显相关。剔除 task 5 后
pooled LR 相关反而升到 0.661（UP 仍 0.297），说明这个模式不是由 task 5 撑起来的；但
`libero_spatial` 剔掉 task 5 后 LR 相关塌到 0.076，所以**4 个 suite 里只有 2 个呈现该模式，每个
n=10**。

可能的读法是：LR-REL 那个未定位的残余差异（唯一剩下的候选是两套实现的采样 RNG 消费顺序）在
policy 最不确定的地方放大，也就是训练未覆盖的初始配置上。**这只是线索，不是结论。**

能钉死它的实验目前做不了，两个缺口：LR-REL 的 `per_task` 只有 `pc_success` / `n_episodes`，没有
逐 trial 索引（UP-REL 有 `failure_indices`）；而且即使有，也需要知道**哪些** init state 索引被
过滤掉了，这又回到上面那个未证实的 `pruned_init[i] ↔ demo i` 对应关系。要推进就得先让 LR-REL
输出逐 trial 结果，并从原始 LIBERO HDF5 重建 demo→init-state 映射。



不要在报告里写 train/val/test 划分——上游配方里不存在：

| 证据 | 位置 | 含义 |
| --- | --- | --- |
| `val_set_proportion: 0.0` | `configs/data/libero_2cam.yaml` | 走 `base_lerobot_dataset.py:89-91` 的 `< 1e-6` 分支，`episodes = range(total_episodes)`，**全部 1,712 条进训练集** |
| 配置只有 `train:`，无 `val:` | 同上 | `runtime.py:335-336` 因此执行 `val_ds = train_ds`，"验证集"与训练集是同一个对象 |
| `eval_every: 200` | `configs/task/libero_uncond_2cam224_1e-4.yaml` | 触发 `trainer.py:377 evaluate()`，但它从**训练集**抽样做 teacher-forced 重建，是训练内 monitor，不是泛化指标 |

所以唯一的能力测量来源是**闭环评测**：40 tasks × 50 trials = 2,000 episodes。不存在 held-out
demonstration 集合，因此不能报告 offline validation loss 或 open-loop action error 作为泛化证据。
而且评测的初始配置与训练 demonstration 有 85.6% 重叠（见上一节），所以闭环成功率也不是 held-out
泛化指标。

## 训练配置与官方配方的逐项对照

pipeline 是官方的：本仓库 launcher 调用 vendored FastWAM 的 `scripts/train_zero1.sh` →
`scripts/train.py` → `runtime.run_training` → `Wan22Trainer`，全部为上游源码，本地无训练代码。
差异只通过 Hydra override 表达。下表把 override 与上游解析值（`configs/train.yaml` +
`configs/task/libero_uncond_2cam224_1e-4.yaml` + `configs/model/fastwam.yaml`）逐项比对：

| key | 上游解析值 | 本仓库传入 | 判定 |
| --- | --- | --- | --- |
| `batch_size`（per GPU） | 16 | 16 | 相同 |
| `num_workers` | 8 | 8 | 相同 |
| `learning_rate` | `1e-4` | `0.0001` | 相同 |
| `weight_decay` | `1e-2` | `0.01` | 相同 |
| `lr_scheduler_type` | `cosine` | `cosine` | 相同 |
| `mixed_precision` | `bf16` | `bf16` | 相同 |
| `max_grad_norm` | `1.0` | `1.0` | 相同 |
| `gradient_accumulation_steps` | 1 | 未覆盖 | 相同 |
| `seed` | 42 | 42 | 相同 |
| `save_every` | 2000 | 2000 | 相同 |
| `log_every` | 10 | 10（`steps<=10` 时降为 1） | 相同 |
| `model.mot_checkpoint_mixed_attn` | `false` | 未覆盖 | 相同（gradient checkpointing 关闭） |
| `model.model_id` | `Wan-AI/Wan2.2-TI2V-5B` | 同值显式传入 | 相同 |
| `model.tokenizer_model_id` | `Wan-AI/Wan2.1-T2V-1.3B` | 同值显式传入 | 相同 |
| `model.load_text_encoder` | `false` | `false` | 相同（上游默认已是 `false`，此 override 为空操作） |
| `model.skip_dit_load_from_pretrain` | `false` | `false` | 相同（同为空操作） |
| `model.action_dit_pretrained_path` | `checkpoints/ActionDiT_..._1024hdim.pt` | 同名文件的仓库内绝对路径 | 仅路径重定位 |
| `data.train.dataset_dirs` / `text_embedding_cache_dir` | `./data/...` | 仓库内绝对路径 | 仅路径重定位 |
| `num_epochs` / `max_steps` | `10` / `null` → **21,696 步** | `max_steps=20000` → **9.22 epochs** | **偏离 −7.8%** |
| `eval_every` | 200 | `0` | **偏离**（关掉的是训练集 monitor，不损失验证信号） |
| `model.loss.lambda_video` | 键不存在 → 代码默认 `1.0` | `+...=1.0`（FW-V）/ `0.0`（FW-FZ 与 FW-NV） | FW-V 与官方数值等价；`0.0` 是预注册干预 |

步数换算：`277,713 frames / global batch 128 = 2,169.6 步/epoch`；上游 `num_epochs: 10` 即
约 21,700 步（trainer 自己推导），而冻结配置里的 `steps: 20000` 只相当于 9.22 epochs、
配方的 92.2%。**不要把 20k 写成"与上游配方一致"。** 现在正式训练统一用 `--train-epochs=10`
（launcher 默认），因此这处取整已不适用；`--steps` 仅用于 smoke。

两处偏离的性质：

- `max_steps=20000`：真实偏离，训练量少 1,696 步。
- `eval_every=0`：因为 `val_ds is train_ds`，上游那个 eval 本来就只是在训练集上抽样重建，
  关掉它不减少任何泛化信息。是否扰动 RNG 流未验证；两组取值相同，组内一致性不受影响。

## Node 0：配置与单测（本地可执行）

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_w0_fastwam_protocol.py tests/test_w0_fastwam_analysis.py

bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-config \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json
```

结果：终端显示 `valid: true`，当前单测应为 `11 passed`（数量随测试扩展可能变化）。失败即停止，
不要进入下载或训练。

## Node 1：source/resource preflight

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=preflight \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/preflight
```

查看：

```bash
python -m json.tool outputs/literature/w0_fastwam/preflight/preflight.json
```

必须确认 `resource_gate.upstream_eval=true` 后才跑 released eval；只有
`resource_gate.causal_training=true` 才能启动 20k causal pair。正式 causal gate 是
8 张同机同型号、每张至少 80 GiB；不满足时停在这里，不能用 CPU offload、降分辨率或小模型
冒充 W0。该步骤还冻结 `preflight/libero_eval_seeds.json`，正式 pair 必须使用它。

## Node 2：准备并校验 pinned checkout 与镜像环境

宿主机先把代码 checkout 到仓库内。下面是一次性准备命令；它不是 W0 wrapper 的一部分，wrapper
只会检查 detached HEAD，不会再次 clone/fetch/checkout。Docker 中已经有
`/opt/robot-practice/w0-vendor`，直接跳过这一段：

代码块使用子 shell 包裹，避免 `set -u` 污染交互式 `zsh`/`bash`。如果之前已经执行过
旧版本并出现 `Q_LOG_LEVEL: parameter not set`，先在当前终端执行一次 `set +u`，再重新执行
下面的代码块。

```bash
(
set -euo pipefail

if [[ -z "${W0_VENDOR_ROOT:-}" ]]; then
  export W0_VENDOR_ROOT=third_party/w0_fastwam
fi

if [[ "${W0_VENDOR_ROOT}" == "third_party/w0_fastwam" ]]; then
  mkdir -p third_party/w0_fastwam
  pin_checkout() {
    local url="$1"
    local checkout="$2"
    local commit="$3"

    if [[ -e "${checkout}" && ! -d "${checkout}/.git" ]]; then
      echo "error: ${checkout} exists but is not a Git checkout" >&2
      return 1
    fi
    if [[ ! -e "${checkout}" ]]; then
      git clone "${url}" "${checkout}"
    fi
    if [[ -n "$(git -C "${checkout}" status --porcelain)" ]]; then
      echo "error: ${checkout} has local changes; refusing to change checkout" >&2
      return 1
    fi
    if ! git -C "${checkout}" rev-parse --verify "${commit}^{commit}" >/dev/null 2>&1; then
      git -C "${checkout}" fetch --depth=1 origin "${commit}"
    fi
    git -C "${checkout}" checkout --detach "${commit}"
  }

  pin_checkout \
    https://github.com/yuantianyuan01/FastWAM.git \
    third_party/w0_fastwam/FastWAM \
    45d8e1458921d83f8ad6cf9ce993d371208dabd0
  pin_checkout \
    https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    third_party/w0_fastwam/LIBERO \
    8f1084e3132a39270c3a13ebe37270a43ece2a01
  pin_checkout \
    https://github.com/huggingface/lerobot.git \
    third_party/w0_fastwam/lerobot \
    30da8e687a6dfc617fcd94afc367ac7071c376ce
else
  echo "Using pre-pinned runtime checkout: ${W0_VENDOR_ROOT}"
fi
)
```

确认运行时镜像已经提供两个环境后，执行只读校验：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-environments \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/environment
```

产物只有 `environment/environment_manifest.json`。它记录 checkout commit 和 import
smoke；`installs_performed` 必须为 `false`。若环境缺失，脚本会直接失败，并要求更新
运行时镜像；不会在当前任务里创建环境或安装依赖。

## Node 3：把 artifact 直接放入仓库并校验
FastWAM 的 released checkpoint、LeRobot checkpoint 和 LIBERO 数据使用 HuggingFace；但
Wan 的两个公共组件文件使用 ModelScope 的
`DiffSynth-Studio/Wan-Series-Converted-Safetensors` 仓库，不能用 `hf download`。需要认证时，
分别在对应环境登录；不要把 token 写进配置、日志或命令历史。下面是唯一需要联网下载的手工
准备步骤，目标目录固定为 `pretrained/w0_fastwam/`，不会写入用户级 HF cache。命令显式通过
`conda run` 调用 CLI，因此不依赖当前交互 shell 的 PATH：

```bash
conda run --no-capture-output -n fastwam-lerobot hf auth login
```

ModelScope 公共仓库通常无需登录；若远端要求认证，使用：

```bash
conda run --no-capture-output -n fastwam-upstream modelscope login
```

登录成功后，再执行下载块：

```bash
(
set -euo pipefail
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
w0_root=pretrained/w0_fastwam
mkdir -p "$w0_root"/{upstream,lerobot,dataset}
mkdir -p "$w0_root/libero_assets"
mkdir -p "$w0_root"/models/{Wan-AI/Wan2.2-TI2V-5B,Wan-AI/Wan2.1-T2V-1.3B,DiffSynth-Studio/Wan-Series-Converted-Safetensors}

conda run --no-capture-output -n fastwam-lerobot hf download yuanty/fastwam \
  libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  --revision=139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a \
  --local-dir="$w0_root/upstream"
conda run --no-capture-output -n fastwam-lerobot hf download ZibinDong/fastwam_libero_uncond_2cam224 \
  --revision=a784645f9ce367ba6953dd20a7a7f0310c85747c \
  --local-dir="$w0_root/lerobot"
conda run --no-capture-output -n fastwam-lerobot hf download yuanty/LIBERO-fastwam --repo-type=dataset --revision=main \
  --local-dir="$w0_root/dataset"
conda run --no-capture-output -n fastwam-lerobot hf download lerobot/libero-assets \
  --repo-type=dataset --revision=main \
  --local-dir="$w0_root/libero_assets"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model Wan-AI/Wan2.2-TI2V-5B \
  --local_dir="$w0_root/models/Wan-AI/Wan2.2-TI2V-5B"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir="$w0_root/models/Wan-AI/Wan2.1-T2V-1.3B"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors \
  --local_dir="$w0_root/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

mkdir -p "$w0_root/dataset/data/libero_mujoco3.3.2"
find "$w0_root/dataset" -maxdepth 1 -type f -name '*.tar.gz' -print0 \
  | while IFS= read -r -d '' archive; do
      tar -xzf "$archive" -C "$w0_root/dataset/data/libero_mujoco3.3.2"
    done
)
```

ActionDiT 不是再次下载的权重，而是用上面已经放入仓库的本地 Wan 文件做一次显式预处理：

```bash
export W0_VENDOR_ROOT="${W0_VENDOR_ROOT:-third_party/w0_fastwam}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=1 \
DIFFSYNTH_MODEL_BASE_PATH=pretrained/w0_fastwam/models \
conda run --no-capture-output -n fastwam-upstream python \
  "${W0_VENDOR_ROOT}/FastWAM/scripts/preprocess_action_dit_backbone.py" \
  --model-config "${W0_VENDOR_ROOT}/FastWAM/configs/model/fastwam.yaml" \
  --output pretrained/w0_fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

最后只做校验：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-artifacts \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --cache-root=pretrained/w0_fastwam \
  --output-root=outputs/literature/w0_fastwam/artifacts
```

`artifacts/artifact_manifest.json` 必须显示 `downloads_performed=false` 和
`installs_performed=false`，并通过 checkpoint SHA-256/字节数、四套 LIBERO 数据目录、Wan
模型目录、T5/VAE 文件、LIBERO simulator assets 及 ActionDiT 文件检查。W0 evaluator 会
通过 `W0_LIBERO_ASSETS` 直接读取 `pretrained/w0_fastwam/libero_assets/`，不使用
`~/.cache/libero/assets`。此阶段不会下载、解压或生成任何东西。

### 数据集按内容校验，不只看目录存在

`--phase=validate-artifacts` 除了检查四个 suite 目录非空，还会读每个
`<suite>_no_noops_lerobot/meta/info.json` 的 `total_episodes` / `total_frames`，与冻结协议的
`train_episodes_per_suite` / `train_episodes` / `train_frames` 逐项比对，不一致直接
`RuntimeError`，结果写入 manifest 的 `checks.dataset_fingerprint`。

这么做的理由是本项目已经两次被"看起来存在、内容却不同"的树坑到（vendored FastWAM 三份分叉、
`libero_utils.py` 少 430 字节）。数据集不对会让 17 小时的训练白跑，而且症状是"结果有点怪"
而不是报错，所以这里选择启动前硬失败。

## Node 4：UP-REL official released checkpoint

### 前置：EGL 渲染设备索引不是 CUDA 卡号

`robosuite` 用 `MUJOCO_EGL_DEVICE_ID` 直接作为 `eglQueryDevicesEXT()` 返回数组的**下标**
（`robosuite/renderers/context/egl_context.py: create_initialized_egl_device_display`）。该数组
不受 `CUDA_VISIBLE_DEVICES` 过滤、顺序不由 CUDA 决定、长度也不等于 GPU 数。8×H20 上
`egl_device_count=10`，末两项重复前面的 DRM 节点且缺 `EGL_NV_device_cuda`，不可渲染。

**所以不能假设 `egl_index == cuda_index`。** 失败是**进程级 native SIGABRT**，Python 捕获不到，
只能由 `faulthandler` 打出栈顶 `binding_utils.py read_pixels` → `mjr_readPixels`，stderr 上没有
任何 GL / MuJoCo 错误消息。

#### 实测规则（2026-08-22，8×H20，驱动 535.261.03）

用 `scripts/experiments/literature/sweep_w0_egl_pairs.py` 扫出 8×8 全矩阵，**每行恰好一格失败**：

```
cuda\egl   0   1   2   3   4   5   6   7
       0   .   .   .   .   .   .   X   .
       1   .   .   .   .   .   .   .   X
       2   .   .   .   .   X   .   .   .
       3   .   .   .   .   .   X   .   .
       4   .   .   X   .   .   .   .   .
       5   .   .   .   X   .   .   .   .
       6   X   .   .   .   .   .   .   .
       7   .   X   .   .   .   .   .   .
```

失败格 = **该 CUDA 设备自己那张物理卡**，可由 PCI 确定性算出（`nvidia-smi` 的 `pci.bus_id` ⋈
`/sys/class/drm/card*/device` ⋈ EGL 枚举的 `EGL_DRM_DEVICE_FILE_EXT`）。本机 DRM 枚举相对 CUDA
是 PCI 域倒序，得 `{0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1}`——**逐格等于实测矩阵**。

两条规则，性质不同：

| 规则 | 强度 |
| --- | --- |
| GL 不能落在**本进程**的 CUDA 卡上 | **确定性**，立刻 abort，8×8 矩阵证实 |
| GL 不能落在**其他进程**的 CUDA 卡上 | **竞态**，8 路并发下恒等映射 4/8 失败、环形映射 3/8 失败 |

第二条的证据：两次并发实验的失败目标卡集合**毫无交集**（`[0,2,5,7]` vs `[1,4,6]`），而两个映射
都是双射、每张卡负载对称。所以**「一张卡一个 EGL 索引」是必要卫生但完全不充分**。

单进程下跨卡渲染是稳定的：`(cuda 5, egl 5)` 跑 400 步、4 trial、同任务全部通过。所以探针无论
拉多长都复现不了并发故障。

#### 推论：worker 数不能占满全部 GPU

第二条规则意味着 `N` 个 worker 占满 `N` 张卡时**没有安全分配**——每张卡都被某个 worker 用作计算。
必须让渲染卡与计算卡不重叠，例如 4 计算 + 4 纯渲染：

```bash
W0_EGL_DEVICE_MAP=0:2,1:3,2:0,3:1  --num-gpus=4
```

代价是并行度减半（约 1.7 h → 3.4 h）。

#### 未解释的部分

旧节点上 UP-REL 用 `CUDA_VISIBLE_DEVICES=i` + `MUJOCO_EGL_DEVICE_ID=i`（与本机恒等映射同配置）
跑通了 2000 集 × 8 worker。同配置一台通、一台不通，**原因不明**：`compute_mode` 全为 `Default`
（排除独占模式），而旧节点的驱动版本与 GPU UUID 已无从追溯——当时 eval manifest 不记指纹，
标定文件又被下一次运行覆盖。现在两条 eval 路径的 manifest 都写 `render_fingerprint`，避免重演。

#### 公开线索（2026-08-22 检索，均无解决方案）

- [LIBERO #100](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/100)（2025-08，零回复）：
  多进程评测下报出与本节**逐字相同**的两条错误（`'EGLGLContext' object has no attribute
  '_context'`、`'MjRenderContextOffscreen' object has no attribute 'con'`）。上游未答复。
- [NVIDIA forum 18583](https://forums.developer.nvidia.com/t/failure-with-independent-devices-on-independent-processes-try-it-yourself/18583)：
  「一个进程在某设备上启动 kernel 会影响另一个进程在**不同**设备上的运行，可能阻塞也可能直接失败」，
  NVIDIA 确认为 bug，报告者注明不总可复现。签名相似，但那是驱动 256.x 且据帖已在 270.x 修复，
  **不作为本问题的根因**，只说明这类跨进程跨设备干扰在 NVIDIA 栈上有先例。
- torchrl `LiberoWrapper` 文档确认「每进程一个 sim + `render_gpu_device_id` 钉住 EGL 设备」是标准
  做法，但**没有任何上游文档要求渲染设备与计算设备不同**——4+4 分离是本机绕法，无公开先例。
- 未找到可过滤 EGL 设备枚举的环境变量。
- 补充：有 2026 年的 NVIDIA 帖子记录 compute mode 为 `Default` 时第二个进程仍可能因资源/GSP 分配
  失败而报 `CUDA_ERROR_NO_DEVICE`。因此本节用 `compute_mode=Default` 排除的只是**独占策略**这一
  机制，不排除多进程资源分配层面的问题。

**不要把上面第二条规则当成通用结论**；它只在本节点被测量。换机器先重扫矩阵。

### 解析顺序与工具

launcher 不假设映射，解析顺序：

1. **显式覆盖**：`--egl-device-map` 或 `W0_EGL_DEVICE_MAP`，缺任一目标卡即报错。该路径**不跑
   验证探针**，适合已知正确映射时绕过一切推断。
2. **PCI 推导 + 探针验证**：算出每张卡的禁用索引，选一个不同的（能用恒等就用恒等，保持可读），
   保证双射，然后逐对用探针验证；任一对失败即 `RuntimeError` 并打印三张 join 表。
3. **推导不可用**：默认**拒绝启动**。渲染探针只能证明「窗口内没炸」，不能证明配对正确——它曾
   接受过一个对八张卡全错的映射。要接受这种降级需显式设 `W0_ALLOW_EGL_PROBE_FALLBACK=1`。

映射**每次 eval 重新推导，不从缓存读**。`egl_calibration.json` 只留档并用于比对漂移，变化时打印
`map changed since ...`。缓存曾让一个错误映射跨运行存活。

渲染设备与计算设备不同时，worker 环境为 `CUDA_VISIBLE_DEVICES=<compute>,<render>`、
`MUJOCO_EGL_DEVICE_ID=<render>`；计算设备必须在前，因为 `robosuite/utils/binding_utils.py` 断言
`MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`，而 `EVALUATION.device=cuda` 解析为 `cuda:0`。

诊断命令：

```bash
conda run --no-capture-output -n robot-practice python scripts/experiments/literature/sweep_w0_egl_pairs.py --egl 0,1,2,3,4,5,6,7 --steps 120 --output outputs/literature/w0_fastwam/egl_pair_sweep.json
```

```bash
conda run --no-capture-output -n fastwam-upstream python scripts/experiments/literature/probe_egl_devices.py
```

扫描工具支持 `--cuda` / `--egl` / `--steps` / `--trials` / `--suite` / `--task-id`，最后会输出可直接
使用的 `W0_EGL_DEVICE_MAP=...`。并发故障要靠**同时起多个**扫描进程才能复现，单进程无论多少步都
不会失败。

### Smoke 与 Formal

先跑 1 task × 2 trials 的 `Smoke`：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-upstream-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --checkpoint=pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt \
  --trials-per-task=2 \
  --num-gpus=1 \
  --output-root=outputs/literature/w0_fastwam/upstream_smoke \
  --smoke
```

确认 action shape、gripper toggle、reset 和结果 JSON 正常后，再跑完整 `Formal`：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-upstream-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --checkpoint=pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt \
  --trials-per-task=50 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/upstream_released
```

`--trials-per-worker` 默认为 `0`，即**一个 task 的全部 trial 共用一个 evaluator 子进程**，
12 GiB checkpoint 每个 task 只加载一次；40 task × 50 trial 从 2000 次加载降到 40 次。
历史上曾用 `--trials-per-worker=1` 做每 trial 进程隔离，那是 EGL 设备索引错配的绕法，
不是 renderer 生命周期问题的修复；根因修好后不需要它，仅在排查新的 native 崩溃时才设为 `1`。

启动时应在日志里看到 `egl_map_source=`（`autocalibrated:` 或 `calibration:`）、完整的
`egl_device_map=`，以及每个 worker 的 `gpu_visible=<compute> egl_device_id=<render>`；
`task_status/` 里同时记录 `gpu_visible` 与 `egl_visible`。

### UP-REL baseline 结果

上面的 `Formal` run 已完成，使用上游 released checkpoint
`pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt`，固定 40 个 LIBERO tasks、每个
task 50 trials，共 2,000 episodes。汇总校验显示无缺失 task、无重复 task key、无重复 trial
index，结果如下：

| LIBERO suite | 成功 / 总数 | success rate |
| --- | ---: | ---: |
| LIBERO-10 | 467 / 500 | 93.4% |
| LIBERO-Goal | 488 / 500 | 97.6% |
| LIBERO-Object | 498 / 500 | 99.6% |
| LIBERO-Spatial | 476 / 500 | 95.2% |
| **Overall** | **1929 / 2000** | **96.45%** |

这是当前仓库记录的 **Fast-WAM W0 UP-REL baseline**：它测量的是上游 released artifact 的
闭环能力，不是本地训练结果，也不是 `lambda_video=1/0` 的机制归因结果。这两者都不能从这条
baseline 推出；`LR-REL` 已有独立测量（见 Node 5），但两者分栏记录、不相减。原始汇总位于
`outputs/literature/w0_fastwam/upstream_released/summary.json`。

结果：`upstream_released/summary.json`、`workers/` 下的 trial 分片、`task_logs/`、
`task_status/` 和 `run.log`。A gate 要求 40 tasks × 50 trials、无缺失/重复、overall 至少
94.6%，且每个 suite 不低于对应 paper reference 5 个百分点。worker 的实际 PID、GPU 映射、
trial 起始编号、退出码和结果文件都写入 `task_status/`；父进程会按全局 trial 编号合并
`workers/` 中的分片，不再依赖调度器的窗口记账。

如果要复用已经存在的 `upstream_released` 路径，必须确认旧任务已停止，然后显式加
`--overwrite`。脚本会把旧目录整体移动为同级的
`upstream_released.backup-<UTC时间>`，再创建干净的 `upstream_released`，因此不会把旧
task result 混入新汇总；不加该参数时，非空目录会直接拒绝启动。

### 运行时诊断字段必须反映实际生效值

`W0_HARD_RESET` 的历史实现只在 `libero_utils.get_libero_env` 里生效，而
`eval_libero_single.py` 的 `_w0_diag_snapshot` 是把环境变量原样回显。镜像里那份 vendored
`libero_utils.py` 没有这段转发，于是 `W0_HARD_RESET=0` **完全没有生效，日志却显示已生效**，
一度让诊断把 `hard_reset` 错误地划入「已排除」。

现在这两件事都由 `src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py`
负责，与部署的是哪份 vendored 副本无关：

- 包装 `robosuite.environments.base.MujocoEnv.__init__`，使 `W0_HARD_RESET` 真正决定
  `hard_reset`；取值非法时直接终止进程。
- 每进程打印一行 `[W0-RUNTIME] render config requested_hard_reset=… effective_hard_reset=…
  cuda_visible=… mujoco_egl_device_id=… egl_device_count=… mujoco_gl=…`，其中
  `effective_hard_reset` 读的是 env 对象属性，不是环境变量。

注意 `site.execsitecustomize()` 会吞掉 `Exception` 并让进程以 exit 0 继续，只在 stderr 留一行
`Error in sitecustomize`。所以 sitecustomize 里的配置错误必须显式终止进程（本仓库用
`os._exit(2)` 并打印原因），不能依赖抛异常；同理，`W0_LIBERO_ASSETS` 指向缺少
`scenes/libero_tabletop_base_style.xml` 的目录时也会直接退出，避免静默回落到 LIBERO 自带
assets 而改变被评测的环境。

## Node 5：LR-REL LeRobot integration

先跑 API/shape `Smoke`：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-lerobot-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --vendor-root=/opt/robot-practice/w0-vendor \
  --policy-path=pretrained/w0_fastwam/lerobot \
  --trials-per-task=2 \
  --output-root=outputs/literature/w0_fastwam/lerobot_smoke \
  --smoke --api-smoke
```

资源满足且 API smoke 通过后，才运行四 suite × 50 的 integration `Formal`：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-lerobot-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --vendor-root=/opt/robot-practice/w0-vendor \
  --policy-path=pretrained/w0_fastwam/lerobot \
  --trials-per-task=50 --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/lerobot_released
```

`lerobot-eval` 本身是单进程单卡，`--eval.n_episodes` 又是**每个 task** 的次数，所以按 suite
串行会把 2000 episodes 拖到 10 小时以上。`--num-gpus>1` 时 launcher 改为按 (suite, task) 分片
（`--env.task_ids=[N]`），40 个 shard 铺在 8 张卡上约 1.5 小时，调度日志与 UP-REL 同格式
（`START`/`DONE`/`FAIL`/`STATUS`，逐 shard 的 `task_status/` 与 `task_logs/`）。单卡串行模式用
`--num-gpus=1` 配 `--gpu-id` 保留。

结果必须单列为 LR integration；即使命令成功，也不能把它的指标替换 upstream
`UP-REL` 或论文 reference。

LR-REL 同样使用 Node 4 的 CUDA→EGL 映射。它是单进程 evaluator，因此默认取当前
`CUDA_VISIBLE_DEVICES` 的第一张卡；可用 `--gpu-id` 指定物理 CUDA 卡，或用
`--egl-device-map` 显式覆盖映射。启动日志必须出现 `LR-REL EGL mapping`，并且当渲染
设备不同于计算设备时，子进程的 `CUDA_VISIBLE_DEVICES` 会把计算卡放在第一位。
Wan 的 UMT5/VAE/tokenizer 也从 `W0_FASTWAM_MODEL_ROOT` 和 policy 目录中的本地文件
加载；正式运行保持 Hugging Face offline，不应再触发网络下载。

`--eval.n_episodes` 是**每个 task** 的次数，所以 `--trials-per-task=2` 在 10-task 的
suite 上得到 20 个 episode。指标由 `lerobot-eval` 自己写在
`<output-root>/<suite>/eval_info.json` 的 `overall.pc_success`，launcher 会读回并写进
manifest 的 `lerobot_metrics`（含按 episode 数加权的 `aggregate`）；子进程 stdout 在
`<output-root>/<suite>/run.log`。

### 两个 pin 组合后的 VISUAL 归一化冲突

checkpoint 与代码是两个独立的 pin，各自自洽不等于组合被验证过。这里就出现了一处冲突：

| pin | 内容 |
| --- | --- |
| `ZibinDong/fastwam_libero_uncond_2cam224` @ `a784645f` | `policy_preprocessor.json` 存 `norm_map.VISUAL = MEAN_STD`，烘焙的图像统计量为 `mean = std = 0.5`，等价于 `2x-1` |
| `huggingface/lerobot` @ `30da8e68` | 契约是 `VISUAL: IDENTITY`（`configuration_fastwam.py` 的 dataclass 默认值 + `processor_fastwam.py` 的 NOTE），因为 core 自己会做映射：`policies/fastwam/wan/modular.py:1086` 的 `input_image * 2.0 - 1.0` |

两者叠加后 VAE 收到 `4x-3`，而官方 evaluator 送的是 `2x-1`
（`_obs_to_model_input` 的 `x*(2/255)-1`，且官方 core 不再变换——
`src/fastwam/models/wan22/fastwam.py:261` 直接 `vae.encode`）。实测：

```
VISUAL=MEAN_STD（出厂）   core input 值域 [-3, 1]   max|diff vs 官方| = 2.000000
VISUAL=IDENTITY（修正）   core input 值域 [-1, 1]   max|diff vs 官方| = 0.000000
libero_spatial pc_success  0.0%  ->  90.0%（20 episodes）
```

症状极具误导性：模型加载、EGL、rollout、视频生成全部正常，只有成功率为 0。

因此 `--visual-normalization` 默认为 `identity`：launcher 在
`<output-root>/policy_overlay/` 建一个指向 checkpoint 的符号链接目录，**只**物化
`policy_preprocessor.json` 并把 `norm_map.VISUAL` 改成 `IDENTITY`。12 GiB 权重不复制、
pinned artifact 字节不变，改动记录在 manifest 的 `policy_overlay.visual_norm_changes`
里。`--visual-normalization=checkpoint` 保留出厂设置，用于复现 0% 的对照臂。
postprocessor 的 `norm_map` 里也有 `VISUAL`，但其 `features` 只含 `action`，是惰性的，
overlay 故意不动它。

排查工具（不需要 rollout、不需要 12 GiB DiT、不需要 EGL）：

```bash
conda run --no-capture-output -n fastwam-lerobot \
  python scripts/experiments/literature/audit_w0_lr_parity.py \
  --policy-path pretrained/w0_fastwam/lerobot \
  --visual-normalization identity
```

它在合成观测上逐级打印张量值域，并与官方表达式对比，输出
`max|LeRobot_core_input - official_VAE_input|`。同一脚本用 `--build-model` 还会报告
`_load_as_safetensor` 在 `strict=False` 下静默丢弃的 shape 不匹配张量——那类张量会停留在
随机初始化状态，症状同样是「全跑完、全失败」。

### 已核对等价、以及仍存在的差异

静态核对 + 合成观测实测确认下列环节两侧**等价**，排查时不必再怀疑：

| 环节 | 官方 | LeRobot |
| --- | --- | --- |
| 180° 旋转 | `get_libero_image` 的 `[::-1, ::-1]` | `processor/env_processor.py` 的 `torch.flip(img, dims=[2,3])`，由 `scripts/lerobot_eval.py:278` 的 `env_preprocessor` 调用（**不在** `envs/libero.py:_format_raw_obs` 里，那里只有 `render()` 为可视化做翻转） |
| 8D proprio | `eef_pos + quat2axisangle(eef_quat) + gripper_qpos` | `LiberoProcessorStep` 组装同样三段 |
| 相机拼接顺序 | `concatenate([primary, wrist], axis=1)` | `sorted(image_keys)` → `image`, `image2` → `cat(dim=-1)` |
| gripper 后处理 | `v=v*2-1; v*=-1; sign(v)` | `FastWAMActionToggleProcessorStep`: `sign(-(v*2-1))` |

仍存在且未消除的差异，引用 LR-REL 数字时必须一并声明：

- **渲染分辨率**：官方以 `LIBERO_ENV_RESOLUTION=256` 渲染再经 PIL BILINEAR 降到 224
  （`_center_crop_resize` 在 256→224 的方形情形下 crop 是空操作）；LR 默认用
  `--observation-size=224` 原生渲染。视野相同，采样/抗锯齿不同。
- **dtype**：官方 `mixed_precision=bf16`，LR 默认传 `--policy.torch_dtype=float32`。
- **采样噪声**：两套实现的 RNG 消费顺序不同，action chunk 不可逐位比较。

### LR-REL 正式结果与未解释的残余

2000 episodes（40 task × 50，`--num-gpus=8` 分片，`--visual-normalization=identity`）：

| suite | LR-REL | UP-REL | 差 |
| --- | ---: | ---: | ---: |
| libero_spatial | 444/500（88.8%） | 95.2% | −6.4pp |
| libero_object | 500/500（100.0%） | 99.6% | +0.4pp |
| libero_goal | 479/500（95.8%） | 97.6% | −1.8pp |
| libero_10 | 462/500（92.4%） | 93.4% | −1.0pp |
| **overall** | **1885/2000（94.25%）** | **96.45%** | **−2.2pp** |

`libero_spatial` 的 39 个 task 在 ±8pp 内，**残余差距几乎全部来自 task 5**
（`pick up the black bowl on the ramekin and place it on the plate`）：UP 为 48/50（96%），
LR 为 28%。它是该 suite 中唯一目标物体架在另一个小物体上的配置，抓取容差最紧；视频显示 LR 侧
场景正确、目标识别正确，**在抓取瞬间将碗碰落**，属精细定位误差而非理解错误。

已逐项排除（源码核对或实测，命令见上方 `audit_w0_lr_parity.py` /
`compare_w0_init_states.py` / `run_w0_lr_task_probe.sh`）：初始状态数组（40 task 全部
`max_abs_delta=0`）、动作执行路径（`step` 直通，无变换/裁剪/缩放）、action ensembler
（`configs/sim_libero.yaml` 两侧都是 `use_action_ensembler: false`）、settling 步数
（34.0% vs 28.0%，n=50 约 0.65σ）、渲染分辨率（40.0% vs 28.0%，约 1.28σ）、dtype
（30.0%，n=20，同口径基线 25.0%）。task 5 的五个 arm 全部落在 28–40%：

| dtype | obs | settling | n | pc_success |
| --- | ---: | ---: | ---: | ---: |
| float32 | 224 | 10 | 50 | 28.0% |
| float32 | 224 | 5 | 50 | 34.0% |
| float32 | 256 | 10 | 50 | 40.0% |
| bfloat16 | 224 | 10 | 20 | 30.0% |
| bfloat16 | 256 | 10 | 20 | 35.0% |

**该缺口未解释，不补填归因。** 剩下的唯一差异是两套实现的采样 RNG 消费顺序，无法对齐。
因此 LR-REL 不声称与 UP-REL 等价，两者分栏记录；引用 LR-REL 时必须同时声明 task 5 的缺口。
完整推导见 [2026-08-19 W0 Fast-WAM UP-REL 与 LR-REL 复现结论](../../../daily/2026-08-19-W0-FastWAM-UP-REL与LR-REL复现结论.md)。

## Node 6：FW-V/FW-NV 5-step distributed smoke

### 必须用 8 卡：单卡在结构上装不下

ZeRO-1 在 `world_size=1` 时不分片任何东西，所以单卡跑会在 ZeRO 分配 fp32 优化器状态时 OOM：

```
deepspeed/runtime/zero/stage_1_and_2.py:2128
torch.OutOfMemoryError: Tried to allocate 22.43 GiB.
GPU 0 has a total capacity of 95.22 GiB of which 6.31 GiB is free.
```

`22.43 GiB` 正是 6.02 B 可训练参数（video expert 5.00 B + action expert 1.02 B）× 4 bytes，
即**一个完整的 fp32 状态组**。每 rank 静态需求对比：

| | bf16 参数 | bf16 梯度 | fp32 master + Adam 二阶 | 冻结 VAE | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单卡（不分片） | 11.21 GiB | 11.21 GiB | **67.3 GiB** | 2.63 GiB | **92.4 GiB** |
| 8 卡 ZeRO-1 | 11.21 GiB | 11.21 GiB | **8.41 GiB** | 2.63 GiB | **33.5 GiB** |

ZeRO-1 每 rank 省 58.9 GiB。注意它**只分片优化器状态**，参数和梯度仍是全量副本——所以
纯 DDP（`accelerate_zero0.yaml`）同样需要 92.4 GiB 静态，在这个规模上也不可行。

实测 8 卡稳态为**每 rank 74.8 GiB**（rank 0 约 76.0 GiB），8 个 rank 数值几乎逐字节一致，
这正是 ZeRO-1 等分的特征。静态之外约 41 GiB 是激活 + allocator 保留 + 碎片；`nvidia-smi`
报的是保留池而非活跃张量，单卡日志里出现过 `29.81 GiB is reserved by PyTorch but
unallocated`。可用 95.22 GiB，余量约 20 GiB。

### 运行命令

```bash
for variant in with_video no_video; do
  bash scripts/experiments/literature/run_w0_fastwam.sh \
    --phase=train-causal-pair \
    --config=configs/experiments/literature/w0_fastwam_reproduction.json \
    --vendor-root=/opt/robot-practice/w0-vendor \
    --variant="${variant}" \
    --steps=5 \
    --num-gpus=8 \
    --output-root="outputs/literature/w0_fastwam/train_smoke/${variant}" \
    --overwrite
done
```

`--num-gpus=8` 配 `batch_size=16`（每卡）得到 global batch 128，与冻结协议的
`global_batch_size: 128` 一致。

### 显存余量不足时的三级手段

正式 20k 训练还会叠加周期性 checkpoint 保存，而 smoke 已观测到 rank 间瞬时相差约 38 GiB
（阶段性峰值 rank 6 为 59.1 GiB 而其余 21.2 GiB）。按「是否改变数值」排序：

| 手段 | 每 rank 节省 | 数值影响 | 其他代价 |
| --- | --- | --- | --- |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（**launcher 默认已开**） | 碎片部分 | **无**，纯 allocator 行为 | torch 2.7 仍非默认值；`nvidia-smi` 口径改变，需重建基线 |
| `per_gpu_batch_size` 8 + `gradient_accumulation_steps` 2 | 激活约一半 | **无**，global batch 仍 128 | 每步多一次 fwd/bwd 调度 |
| `model.mot_checkpoint_mixed_attn=true` | 激活的大部分 | **无**（纯重算） | **约 +30% 墙钟**；见下方两条约束 |
| ZeRO-2（`train_zero2.sh`） | 再约 9.8 GiB（梯度也分片） | 梯度归约顺序微变 | 通信增加 |

第一项由 `train_causal_variant` 默认注入，不需要手动设置。它**数值上完全无影响**（只改 CUDA
allocator 行为，不同于 gradient checkpointing 或换 ZeRO stage），而漏设的代价是 20k 步跑到
几小时后 OOM——收益成本极不对称，所以默认开启而不是交给操作者记住。作用域限于训练路径，
评测进程保持与产出 UP-REL / LR-REL 记录时一致。

想换成别的分配器策略时显式设置即可，launcher 会让位并打印 `[W0] keeping operator
PYTORCH_CUDA_ALLOC_CONF=...`：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
bash scripts/experiments/literature/run_w0_fastwam.sh --phase=train-causal-pair ...
```

gradient checkpointing 有两条容易踩的约束：

1. **不能独立开关。** `configs/model/fastwam.yaml` 里 `video_dit` 与 `action_dit` 的
   `use_gradient_checkpointing` 都写成 `${model.mot_checkpoint_mixed_attn}`，一个旋钮同时
   控制 MoT mixed-attention checkpointing 和两个 expert 的 DiT 层 checkpointing。
2. **是对 pinned 配方的偏离。** `configs/task/libero_uncond_2cam224_1e-4.yaml` 显式设为
   `false`，released checkpoint 就是在关闭状态下训练的。数值等价性成立（三处调用点
   `mot.py:90`、`mot.py:238`、`helpers/gradient.py:17` 均为
   `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，未传
   `preserve_rng_state=False`，重算时 RNG 状态保留），但开启必须记录为协议偏离。

真实步时可由 `log_every=1` 的日志读出（`speed=... step/s`、`eta=`），据此决定是否接受
+30% 再动它，不要凭估计。

### 通过标准：不能只看 status

第一次 8 卡 smoke 两组都写出了 `checkpoint_step_00000005.pt`、`run_manifest.json` 为
`status: complete`，但 253 KB 日志里 `grep -ci loss` 为 **0**——task 配置的 `log_every: 10`
大于 5 步，永远碰不到打印点。checkpoint 存在只说明文件写出，不说明优化发生。launcher 因此
在 `steps <= 10` 时自动传 `log_every=1`（该键在 task 配置中存在，故不加 `+`）。

必须同时满足三条：

```
loss = loss_video + loss_action                两组都成立
no_video 臂 loss_video ≡ 0.0000 且 loss = loss_action
两组 loss 随步下降
```

第二条是**因果对照真实存在的直接证据**。日志字段是
`self.loss_lambda_video * float(loss_video.detach().item())`（`fastwam.py:565`，已乘系数），
所以 `lambda_video=0` 时它必须恒为 0；若两组都非零，说明覆盖没到模型，FW-V/FW-NV 会是两次
相同训练，而 `run_manifest.json` 里的 `lambda_video` 只是本仓库自己的记录，不构成证据。

实测（8 卡、5 步、seed 42）：

| step | with_video: loss / action / video | no_video: loss / action / video |
| ---: | --- | --- |
| 1 | 1.8992 / 1.4161 / 0.4831 | 1.4185 / 1.4185 / **0.0000** |
| 2 | 7.2060 / 6.4309 / 0.7751 | 5.5450 / 5.5450 / **0.0000** |
| 3 | 2.6946 / 1.3738 / 1.3209 | 1.5285 / 1.5285 / **0.0000** |
| 4 | 1.7725 / 0.6963 / 1.0762 | 0.7793 / 0.7793 / **0.0000** |
| 5 | 1.5425 / 0.6652 / 0.8772 | 0.6264 / 0.6264 / **0.0000** |

加和残差 ≤ 1e-4（打印舍入）。**Node 6 通过。** 5-step smoke 不是正式成功率。

### 运行间非确定性对 gate 的约束

step 1 的 loss 在任何权重更新之前算出，理论上只取决于（初始权重、batch、噪声），且
`lambda_video=0` 只是把 video 项乘 0（video 分支照常计算，RNG 消耗顺序不变）。但两组的
step-1 `loss_action` 为 **1.4161 vs 1.4185**（相对差 0.169%）。该差值落在 bf16 于 1.4 附近的
分辨率（约 0.0055）之内，而 `global_loss` 是跨 8 rank `accelerator.gather().mean()`，归约
顺序不保证确定性——最可能是精度与归约顺序，而非第二个未受控差异。**但这未经证实。**

直接后果：`causal_pair.json` 的 `same_init_data_noise_update: true` 无法从这些日志验证到
逐位一致。因此 Node 8 的因果判据必须建立在**多 seed 方向一致**或**明确的容差**上，不能把
"两组数值不同"本身当作机制证据。要把这条钉死，可重复跑一次同参数 `with_video`，比较两次
step-1 `loss_action`：差值同为 ~0.002 量级即确认为运行间非确定性。

### 其他必查项

查看 `train_smoke/pair_contract.json` 和两个 `run_manifest.json`：`init_hash`、
`data_hash`、`noise_hash`、`update_hash`、global batch 必须一致，只有 `lambda_video`
不同。

### 启动依赖：DeepSpeed 的导入期 nvcc 探测

`import deepspeed` 会执行 `git_version_info.py`，对每个 op builder 调 `is_compatible()`；
`FpQuantizerBuilder` 进而调用 `installed_cuda_version()`，后者 shell 出
`$CUDA_HOME/bin/nvcc -V` **仅为读取版本号**。该函数只在 `CUDA_HOME is None` 时抛可捕获的
`MissingCUDAException`；当目录存在但没有编译器时，`subprocess` 的 `FileNotFoundError` 会
逃出并在训练开始前杀死 launcher：

```
deepspeed/ops/op_builder/builder.py:53
FileNotFoundError: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**这里没有任何东西在编译。** `ds_zero1_config.json` 的 `offload_optimizer.device` 和
`offload_param.device` 均为 `none` 且无 `optimizer` 块，因此既不需要 `CPUAdam` 也不需要
`FusedAdam`。所以不要安装 C/C++ 编译器或完整 CUDA toolkit——只需让探测有答案。
`DS_SKIP_CUDA_CHECK=1` **无效**，它守的是 `assert_no_cuda_mismatch`，另一个函数。

镜像的 `/usr/local/cuda` 是 runtime + headers 布局，没有 `bin/`。处理方式分两层：

- **镜像层**：`Dockerfile.w0-fastwam` 向 `fastwam-upstream` 安装 `cuda-nvcc=12.8`（必须匹配
  torch 的 cu128 主版本，否则 `FpQuantizerBuilder` 报 mismatch）。不装 CUDA dev 库和 host
  compiler，保持环境变更最小。
- **无网主机的兜底**：`docker/training/cuda_nvcc_probe_shim/bin/nvcc` 是 tracked 文件，随
  仓库同步即可用。它只应答 `-V`，任何其他调用打印说明并 exit 2，不伪装成编译器。

`w0_runtime/sitecustomize.py` 的 `_configure_cuda_home()` 按
`sys.prefix` → `/usr/local/cuda-12.8` → `/usr/local/cuda-12` → `/usr/local/cuda` → shim
顺序挑第一个含 `bin/nvcc` 的前缀，所以镜像装上真 nvcc 后 shim 自动闲置，无需改代码。该逻辑
由 `W0_CUDA_HOME_FALLBACK=1` 开启，仅训练路径设置——评测进程的 stdout 保持与产出 UP-REL /
LR-REL 记录时一致。

### 启动依赖：`+model.loss.lambda_video`

`configs/model/fastwam.yaml` 的 `loss` 段只有 `lambda_action`，**没有 `lambda_video`**，
所以 Hydra 严格结构下必须用 `+` 追加。但加 `+` 之前必须确认模型真的读它，否则会造出一个
跑得通、两组却没有区别的假因果实验。核对结果：

```
src/fastwam/runtime.py:156（及 241、328）  loss_lambda_video=float(loss.get("lambda_video", 1.0))
src/fastwam/models/wan22/fastwam.py:563    loss_total = loss_lambda_video * loss_video + loss_lambda_action * loss_action
```

默认值为 1.0，这就是配置省略它的原因——released checkpoint 即按 `lambda_video=1.0` 训练。
因此 FW-V 的 `+model.loss.lambda_video=1.0` 与上游默认数值等同，FW-NV 的 `0.0` 才是真正移除
video 项。`lambda_action` 在配置中存在，**不能**加 `+`（否则报 "An item is already at ..."）。
`tests/test_w0_fastwam_protocol.py` 读取 vendored 配置来锁定这层对应关系：上游若补上该键，
测试会提示去掉 `+`。

## Node 7：causal training（FW-V / FW-FZ / 可选 FW-NV）

只有 Node 1 的 causal 资源检查通过后执行。**预算按 epoch 而非固定步数**：
`configs/task/libero_uncond_2cam224_1e-4.yaml` 是 `num_epochs=10 / max_steps=null`，
trainer 由数据集推导出 **21,700 步**，同时决定 cosine schedule 长度与 5 percent warmup。
固定 20,000 步相当于 9.22 epoch = 配方的 92.2%，不等价。

用 launcher 而不是直接敲 runner：它内建 `nohup`、输出重定向、`--num-gpus=8`、
`--overwrite`、`unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID`，并在已有 `scripts/train.py`
在跑时拒绝启动。直接敲的等价命令约 450 字符，粘贴时被软换行截断已发生多次，会让 `&`
与重定向脱离 `nohup`、参数被静默丢弃。

三组按优先级串行，`FW-NV` 排最后且可跳过：

```bash
bash scripts/experiments/literature/launch_w0_causal.sh with_video
```

```bash
W0_FREEZE_VIDEO_EXPERT=1 bash scripts/experiments/literature/launch_w0_causal.sh no_video
```

```bash
bash scripts/experiments/literature/launch_w0_causal.sh no_video
```

| 命令 | 变体 | 落点 | 可训练 |
| --- | --- | --- | ---: |
| 第一条 | `FW-V`（`lambda_video=1`） | `causal_10ep/with_video` | 6.02 B |
| 第二条 | **`FW-FZ`**（`lambda_video=0` + 冻结 video expert，主对照） | `causal_10ep_frozen/no_video` | 约 1 B |
| 第三条 | `FW-NV`（`lambda_video=0`，论文同款消融，可选） | `causal_10ep/no_video` | 6.02 B |

`FW-FZ` 的父目录由 `W0_FREEZE_VIDEO_EXPERT` **自动派生**（`causal_10ep_frozen/`），不需要也
不应该单独传路径：它也是一个 `no_video` 运行，单独传就可能忘传，那会覆盖 `FW-NV` 的产物，
而 sibling 预算检查还会把两者当成同一对。`run_manifest.json` 记 `freeze_video_expert`
作为双重保险。启动时打印的 `freeze` 一行可直接确认。

冻结需要镜像已应用 `docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch`
（`w0-fastwam-20260821-fwfz` 及之后）。确认方式：

```bash
git -C /opt/robot-practice/w0-vendor/FastWAM status --porcelain
```

预期恰好一行 `M src/fastwam/trainer.py`。

训练期用停滞/内存监控：

```bash
L=outputs/literature/w0_fastwam/causal_10ep/with_video/train.log
```

```bash
nohup bash scripts/experiments/literature/watch_w0_stall.sh "$L" > /tmp/w0_watch.log 2>&1 &
```

结果：每组 `train.log`、`run_manifest.json`、中间 checkpoint（`save_every=2000`）与最终
`checkpoint_step_00021700.pt`。同一父目录下的两组必须来自同一 `pair_contract.json`，
否则作废。

### 实测预算与显存

修复 torchcodec 之后的稳态（见下面「宿主内存 OOM」一节）：

| 量 | 实测 |
| --- | ---: |
| 稳态步时 | **2.53 s/step**（修复前 4.0） |
| 单组 21,700 步墙钟 | **约 15.3 小时**（修复前约 24） |
| 三组串行 | 约 1.6 天（`FW-FZ` 可训练参数约 1 B，预计更快） |
| 显存峰值 = 稳态 | 64.2 GiB/rank（8 rank 近乎一致） |
| 余量 | 31.4 GiB（容量 95.6 GiB） |
| 宿主内存 | 稳定 280 GiB / 1788.9 GiB 上限（15%） |

三个坑：

- **不要用短跑估算步时，两个方向都会错。** 5 步跑报 `speed=0.08–0.10 step/s`，被 warmup
  主导，偏低 3–4 倍；100 步跑报 `0.30–0.32 step/s`，比真实稳态偏高 20–25%。只有跑过几百步
  的读数可信。
- **checkpoint 保存没有显存尖峰。** `save_checkpoint(optimizer=None)` 只存权重、不 gather
  优化器状态，实测 100 步全程（含末步保存）平在 65,749 MiB。因此
  `gradient checkpointing` / 降 batch / ZeRO-2 三级手段一个都不需要，`mot_checkpoint_mixed_attn`
  保持上游的 `false`。

### 宿主内存 OOM（已修复，2026-08-21）

四次 8 卡训练在 step 265 / 270 / 560 / 598 崩溃，同一根因。

`lerobot` 用 `importlib.util.find_spec("torchcodec")` 选解码后端，它成功（Python 包装着），但
`from torchcodec.decoders import VideoDecoder` 失败——torchcodec 只支持 FFmpeg 4–7，镜像里
只有 conda 环境内的 FFmpeg 8。`decode_video_frames` **在 except 块内**回落 pyav，于是每次解码
都留下一个异常，其 traceback 钉住失败 import 的栈帧；这些帧构成引用环，连带钉住同帧的 pyav
对象和解码出的大张量，直到全量 gc 才释放。实测每样本 2 个 traceback（每相机一次）、约
29 MiB/样本，64 个 worker 合计约 5 GiB/step，撞到 1788.9 GiB 的 cgroup 上限后 OOM killer 杀掉
一个 worker，其 rank 阻塞在 `queue.get` 再不 enqueue 梯度 allreduce，其余 7 个在 DeepSpeed 的
`allreduce_and_copy` 自旋直到 600 s NCCL watchdog 杀全场。

**修复**：`docker/training/Dockerfile.w0-fastwam-torchcodec-fix` 装 jammy 的 FFmpeg 4.4 运行库
（只增加 FFmpeg 4 时代的 soname，与 conda 里的 FFmpeg 8 不冲突，不动任何 conda 包）。同层补
`gcc` + `libc6-dev`：这是**独立的、原本就潜伏的**故障，`import deepspeed` 在 capability ≥ 8 的
设备上会导入 `TritonMLP`，triton 需现场编译 `driver.c`；老容器靠热缓存绕过，容器重建后暴露。
两者都不改训练数值（`TritonMLP` 是推理路径的类，训练从不实例化）。
镜像 `w0-fastwam-20260821-torchcodec-gcc`，digest
`sha256:ce864b5ed496ac0c63c7ec8d5b946ab3a7d6c6f9c6e22aa9b78429e69d95a058`。

修复前后同配置实测：

| 量 | 修复前 | 修复后 |
| --- | ---: | ---: |
| 宿主内存斜率 | 5–6 GiB/step | 0.017 GiB/step（step 560→620） |
| cgroup 用量 | 撞 1788.9 GiB 上限 | 280 GiB（15%） |
| 稳态步时 | 4.0 s/step | 2.53 s/step |
| 单组 21,700 步 | 约 24 小时 | 约 15.3 小时 |

提速来自 torchcodec 的精确 seek：pyav 路径因 `keyframes_only=True` 每次要从前一个关键帧解到
目标时间戳。**副作用需记录**：训练解码器由 pyav 变为 torchcodec，AV1→RGB 可能有舍入差异，
取帧也变准；FW-V/FW-NV 两组同解码器故因果对照不受影响，且更接近上游。

被测量排除的假设，避免重复走：FUSE 存储阻塞（缺席 rank 是 `S` 不是 `D`）、
`expandable_segments`（那是 GPU allocator）、降 `num_workers`（8→2 斜率不变，泄漏按样本数累积）、
glibc arena 碎片（`MALLOC_ARENA_MAX` 等三项斜率不变，配置已移除）、`__getitem__` 异常重试
（`_get` 抛出率 0/120）、`torch.compile`（本路径不调用）。

**磁盘不是约束**：挂载点是 1.5 PiB 的 vePFS，可用约 421 TB；单组单次 save 约 102 GB，11 次约
1.1 TB。注意 `df` 不带 `-h` 输出的是 1K 块而非字节。

### 排查工具

- `watch_w0_stall.sh <train.log>`：轮询 step，停滞即对 agent+rank 发 `SIGUSR1` dump 栈（两次间隔
  60 s 以区分卡死与慢），并记录内核态 `state`/`wchan`、cgroup 内存、匿名内存按 rank/worker 归属。
  两个坑：`SIGUSR1` 默认动作是 Term，`sitecustomize` 必须以 `chain=False` 注册；
  `pgrep -f scripts/train.py` 会同时匹配 agent/rank/worker（命令行相同），必须按 `/proc` 进程树
  深度筛。
- `probe_w0_dataset_memory.py`：单进程迭代数据集，不加载模型。`--cycles` 用 `gc.DEBUG_SAVEALL`
  点名引用环成员，`--stage {full,lerobot,text}` 逐层归因，`--video-backend` 强制解码后端。
  **census 前必须 `gc.collect()`**，否则「尚未回收」会被读成「泄漏」。

### 单 seed 一对能支持什么结论

实测同配置重跑在**第一次反传之前**就不一致：`with_video` 两次的 step-1 `loss_action` 为
`1.4161` vs `1.4316`（相对 1.09%），而跨臂只差 `1.4161` vs `1.4185`（0.169%）。
`num_workers=0` 不能消除（单 suite 两次为 `2.0042` vs `1.8533`），因此
`pair_contract.json` 的 `same_init_data_noise_update` 只表示"两组被给了相同的 seed/数据/配方"，
**不表示逐位可复现**。

但这些数字只刻画**单 batch 采样噪声**，不能外推成收敛后成功率的方差——20k 步会看到约 256 万个
样本，"第一个 batch 是哪些"基本被冲掉。所以本 Node 默认只跑**两次**（FW-V + FW-NV，同 seed 42），
按结果决定是否需要更多：

| 观察到的闭环差距 | 可写的结论 |
| --- | --- |
| 方向清楚且量级远超 `causal_min_gap`（2pp） | 机制证据成立，注明单 seed |
| 差距 < 2pp 或方向不明 | **不得归因**，必须每臂补第二个 seed 后再判 |

补 seed 用 `--train-seed=43`，并放到**另一个父目录**（例如
`outputs/literature/w0_fastwam/causal_seed43/`）：`pair_contract.json` 的
`init_hash`/`data_hash`/`noise_hash`/`update_hash` 都随 seed 变，同一父目录下混放会在启动前
报错并点名两边的 seed。单次 20k 约 18 小时，双 seed 四次共约 3 天。

## Node 8：同 seed schedule 的 causal evaluation

先跑 8 GPU `Smoke`：`libero_spatial` 的 10 个 task × 2 trials；两组 checkpoint 都做真实推理和
MuJoCo/EGL 渲染。不要把 50-trial seed manifest 传给 smoke。这样才能在正式评测前发现多进程
推理或渲染问题：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-causal-pair \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --with-video=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoint_step_00021700.pt \\
  --no-video=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoint_step_00021700.pt \\
  --pair-contract=outputs/literature/w0_fastwam/causal_10ep/pair_contract.json \\
  --trials-per-task=2 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/causal_smoke \
  --smoke
```

正式 `Formal` 使用同一 frozen seed manifest：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-causal-pair \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --with-video=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoint_step_00021700.pt \\
  --no-video=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoint_step_00021700.pt \\
  --pair-contract=outputs/literature/w0_fastwam/causal_10ep/pair_contract.json \\
  --seed-manifest=outputs/literature/w0_fastwam/preflight/libero_eval_seeds.json \
  --trials-per-task=50 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/causal_eval
```

`--no-video` 这个参数名是历史遗留的**槽位名**，指「对照组的 checkpoint」，不代表该组必须是
`FW-NV`。主对照传 `causal_10ep_frozen/no_video/` 下的 `FW-FZ`；若也跑了可选的 `FW-NV`，
再单独跑一次评测传 `causal_10ep/no_video/`，两次用不同 `--output-root`。

结果：`causal_eval/comparison.json`、两组 per-task results 和 `deployment.json`。

**没有 pass/fail 阈值**（计划已移除）。要读的是：两组 overall 与逐 suite SR、配对 gap 与
95% CI，以及三项有效性检查——completeness（各 40×50 无缺失/重复）、equal budget
（contract hash / global batch 一致，差异只有 `lambda_video` 与 freeze 且都在 manifest 里）、
safety（invalid rollout 为 0）。CI 的位置要如实叙述：不含 0 是检出效应；整段落在 ±2pp 内可
称等价；**又宽又跨 0 说明样本量不足，不得挑一个方向叙述**。

## Node 9：direct-deployment evidence 与测量汇总

`deployment.json` 不能由模板自动判通过。先将 one-task runtime probe 的实测计数整理成
JSON，字段必须为：

```json
{
  "status": "pass",
  "output_shape": ["B", 32, 7],
  "future_noisy_video_token_allocations": 0,
  "future_video_denoise_steps": 0,
  "future_vae_decode_calls": 0,
  "n_action_steps": 10,
  "gripper_contract": "toggle_last_dimension"
}
```

导入并检查证据：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=audit-deployment \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --deployment-evidence=outputs/literature/w0_fastwam/causal_eval/deployment_evidence.json \
  --output=outputs/literature/w0_fastwam/causal_eval/deployment.json
```

然后生成总 gate：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=gate-w0 \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --released=outputs/literature/w0_fastwam/upstream_released/summary.json \
  --causal=outputs/literature/w0_fastwam/causal_eval/comparison.json \
  --deployment=outputs/literature/w0_fastwam/causal_eval/deployment.json \
  --lerobot=outputs/literature/w0_fastwam/lerobot_released/summary.json \
  --bootstrap-repeats=10000 \
  --bootstrap-seed=20260817 \
  --output=outputs/literature/w0_fastwam/w0_gate.json
```

只检查 gate 命令是否可调用、而不读取任何正式结果时，可加 `--dry-run`；它只会输出
`status: dry_run`，不会伪造通过：

```bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=gate-w0 \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output=outputs/literature/w0_fastwam/w0_gate_smoke.json \
  --dry-run
```

`--phase=gate-w0` 现在只是**汇总器**：把三层结果、配对 bootstrap CI 与有效性检查聚到一个
JSON 里。计划已移除 pass/fail 阈值，所以**输出里的 `passed` 字段不作为判断依据**，要看的是
SR、配对 gap 与 CI、以及 completeness / equal budget / safety 三项有效性检查。是否继续 W1
由使用者读数字决定。

## 结果落点与必记字段

- preflight：`outputs/literature/w0_fastwam/preflight/`
- artifacts manifest：`outputs/literature/w0_fastwam/artifacts/`
- repo-local inputs：`third_party/w0_fastwam/` 与 `pretrained/w0_fastwam/`
- released eval：`upstream_released/summary.json`、`lerobot_released/summary.json`
- causal training：`causal_10ep/with_video/`（FW-V）、`causal_10ep_frozen/no_video/`（FW-FZ）、
  `causal_10ep/no_video/`（可选 FW-NV）；10 epoch 推导为 21,700 步
- causal eval：`causal_eval/comparison.json`、`deployment.json`
- final gate：`w0_gate.json`

报告至少记录 suite/task、dataset/demo 数、front+wrist RGB、8D state、7D action、horizon 32、
execute 10、action/video/total loss、open-loop action error、success_once、paired gap/CI、
batch-1 latency/peak memory、timeout/gripper/invalid action/load failure。

## 停止条件

- pinned revision/hash、环境 import/version、dataset 结构或 checkpoint hash 不一致：停止。
- GPU memory gate 不满足：停止，不降级为伪复现。
- released artifact 未过 A：先审计 MuJoCo、stats、action toggle、task text 和 seed schedule，
  不启动 causal pair。
- causal pair 不是唯一 `lambda_video` 差异，或 future noisy-video tokens 泄漏到 action：两组作废。
- 未获得实测 deployment evidence：gate 必须保持失败，不能用预期/静态推断替代。
