# 2026-08-19 W0 Fast-WAM UP-REL 与 LR-REL 复现结论

## 重要进展

- **UP-REL 正式闭环完成**：40 tasks × 50 trials = 2000 episodes，`1929/2000`（96.45%）。
- **修复两个阻断性缺陷**：`MUJOCO_EGL_DEVICE_ID` 被当作物理卡号（导致 `mjr_readPixels` 处
  native SIGABRT），以及 LeRobot checkpoint 的 `norm_map.VISUAL=MEAN_STD` 与 pinned core 的
  `*2-1` 重复映射（导致 LR-REL 全程 0 成功）。
- **LR-REL 从 0/20 提升到 `1885/2000`（94.25%）**，`libero_object` 达到 100.0%。
- **残余 2.2pp 几乎全部来自 `libero_spatial` task 5**（LR 28–40% vs UP 96%）；已逐项排除七类
  差异，该 task 的失败原因**未解释**，不编造归因。
- 每 trial 重载 12 GiB checkpoint 的绕法已删除；LR-REL 也改为按 (suite, task) 分片，2000
  episodes 从 10+ 小时降到约 1.5 小时。

## 评测与训练数据的重叠（2026-08-20 补充）

闭环评测的初始状态来自 LIBERO 的 `<task>.pruned_init`，每 task 恰好 50 条，而协议是每 task
50 trials，因此全部用尽。LIBERO 原始每 task 也是 50 条 demonstration，数量精确相等，所以
**评测的初始配置几乎肯定就是 demonstration 的起始配置**——评测衡量的是"能否从训练见过的初始
配置重新执行"，不是对新物体摆放的泛化。这是 LIBERO 的标准口径，与文献可比，但不能写成 held-out
泛化。未证实 `pruned_init[i]` 与 demo `i` 的 `states[0]` 逐字节相等（LIBERO 不带生成脚本）。

`_no_noops` 过滤使训练集为 1,712 条而评测用满 2,000 条，因此 **288 个 trial 的初始配置对应的
demonstration 被过滤掉了**，整体覆盖率 85.6%，单 task 范围 0.640–1.000。落点
`outputs/literature/w0_fastwam/train_eval_overlap.json`，脚本
`scripts/experiments/literature/audit_w0_train_eval_overlap.py`（按 task 语言 join，因为
LeRobot 转换是字母序、`task_index != task_id`）。

对 `libero_spatial` task 5 缺口的检验结果是**否**：它的覆盖率 0.760 在 40 个 task 中排第 7 低、
并不异常，同覆盖率的另三个 task 的 LR 为 92%/92%/96%。更根本的是覆盖率对两个评测器完全相同
（同 checkpoint、同训练集），按定义无法解释 UP−LR 差异。该候选解释已排除。

附带观察到一条弱线索：suite 内 `r(coverage, LR%)` 在 `libero_goal` 为 0.752、`libero_10` 为
0.704，而 `r(coverage, UP%)` 全部在 0.01–0.30。剔除 task 5 后 pooled LR 相关升到 0.661。
但 4 个 suite 只有 2 个呈现该模式且每个 n=10，`libero_spatial` 剔除 task 5 后塌到 0.076，
**因此只作线索记录，不作结论**。要推进需要 LR-REL 输出逐 trial 结果，并从原始 LIBERO HDF5
重建 demo→init-state 映射。

## 一、UP-REL 正式结果

固定 40 tasks × 50 trials、`physx` 无关（LIBERO/MuJoCo 3.3.2）、front + wrist RGB 224×224、
8D proprio、7D 相对末端动作、`replan_steps=10`、`num_steps_wait=5`、`use_action_ensembler=false`。

| suite | successes / episodes | 成功率 | paper reference |
| --- | ---: | ---: | ---: |
| libero_spatial | 476/500 | 95.2% | 98.2% |
| libero_object | 498/500 | 99.6% | 100.0% |
| libero_goal | 488/500 | 97.6% | 97.0% |
| libero_10 | 467/500 | 93.4% | 95.2% |
| **overall** | **1929/2000** | **96.45%** | 97.6% |

无缺失或重复 trial。结果落在 `outputs/literature/w0_fastwam/upstream_released/summary.json`。
这是**上游 released checkpoint 的本地闭环 baseline**，不是本地重新训练，也不等同于尚未完成的
`FW-V/FW-NV` 因果对照。

## 二、缺陷一：EGL 渲染设备索引不是 CUDA 卡号

`robosuite` 用 `MUJOCO_EGL_DEVICE_ID` 直接索引 `eglQueryDevicesEXT()` 返回的数组
（`renderers/context/egl_context.py`）。该数组不受 `CUDA_VISIBLE_DEVICES` 过滤，顺序不由 CUDA
决定，长度也不等于 GPU 数——8×H20 节点上是 10 项，索引 8/9 不可渲染。

实测标定（每格 405 步 ×2，另经 705 步复核）：

| 配对 | 结果 |
| --- | --- |
| `(cuda i, egl i)`，i = 0…6 | 通过 |
| **`(cuda 7, egl 7)`** | **确定性失败**，两次都在第 50–75 步 `mjr_readPixels` 处 SIGABRT |
| `(cuda 7, egl 0…6)` / `(cuda 0…6, egl 7)` / `(cuda i, egl i+1)` | 全部通过 |

64 个 GPU 组合里只有 1 个失败，**这在数学上不可能由「索引映射整体错位」产生**，因此该现象
只被测量、未被解释：本仓库没有 EGL↔物理 GPU 的驱动级映射证据，不声称「EGL 索引 7 就是物理卡
7」或「同卡 CUDA+EGL 共存不安全」。

失败特征需记住：进程级 native SIGABRT，Python 捕获不到，stderr 上没有任何 glibc / GL / MuJoCo
消息；同一坏配对下 CPU 观测可跑满 400 步，即**同进程有 CUDA 活动是必要条件**；与 `hard_reset`、
context churn、trial 边界、累积步数、显存均无关（`rss` 与 `cuda_free` 全程恒定）。

**修复**：launcher 不再假设映射，而是自动标定并按机器指纹（GPU UUID + 驱动版本 + EGL 设备数）
缓存；先试 identity，只有失败的卡才扫其他索引；探针不加载 checkpoint，每格约 13 秒。换机器时
指纹不匹配会自动作废重测；某张卡无可用索引时在启动前拒绝。渲染设备与计算设备不同时，worker
环境为 `CUDA_VISIBLE_DEVICES=<compute>,<render>`（计算设备在前，满足 robosuite 的包含断言）。

## 三、缺陷二：VISUAL 归一化被应用两次

两个独立 pin 各自自洽，组合后冲突：

| pin | 立场 |
| --- | --- |
| `ZibinDong/fastwam_libero_uncond_2cam224` @ `a784645f` | `policy_preprocessor.json` 存 `norm_map.VISUAL=MEAN_STD`，图像统计量为 `mean=std=0.5`（等价 `2u-1`，是人为常量而非数据集统计） |
| `huggingface/lerobot` @ `30da8e68` | 契约为 `VISUAL: IDENTITY`，映射由 core 负责：`policies/fastwam/wan/modular.py` 的 `input_image * 2.0 - 1.0` |

官方链路是 `p*(2/255)-1 = 2u-1`，且官方 core 不再变换（直接 `vae.encode`）。LeRobot 实际链路
是 `2u-1` 之后再 `*2-1`，即 **`4u-3`**：

```
误差 = x_actual − x_correct = 2(u − 1)
白 u=1 误差 0；中灰 u=0.5 误差 −1.0；黑 u=0 误差 −2.0
```

实测 `max|LeRobot_core_input − official_VAE_input| = 2.000000`，均值关系
`2×(−0.0604) − 1 = −1.1208` 与实测 `−1.1207` 一致。危害不是「变暗」，而是**大部分像素越出
Wan VAE 的训练域**（最低 −3），DiT 被条件化在从未见过的 latent 区域。

`make_pre_post_processors(pretrained_path=...)` 加载 checkpoint 里**保存好的** pipeline，代码
里 `VISUAL: IDENTITY` 的默认值没有参与决策的机会，所以这个保护被静默绕过。

**修复**：`--visual-normalization` 默认 `identity`，launcher 在 `<output-root>/policy_overlay/`
建符号链接目录，**只**物化 `policy_preprocessor.json` 并把 `norm_map.VISUAL` 改成 `IDENTITY`。
12 GiB 权重不复制、pinned artifact 字节不变，改动记录在 manifest 的
`policy_overlay.visual_norm_changes`。修正后 `max|diff| = 0.000000`。

## 四、LR-REL 正式结果

2000 episodes（40 task × 50，8 卡按 (suite, task) 分片），`--visual-normalization=identity`。

| suite | LR-REL | UP-REL | 差 |
| --- | ---: | ---: | ---: |
| libero_spatial | 444/500（88.8%） | 95.2% | −6.4pp |
| libero_object | 500/500（100.0%） | 99.6% | +0.4pp |
| libero_goal | 479/500（95.8%） | 97.6% | −1.8pp |
| libero_10 | 462/500（92.4%） | 93.4% | −1.0pp |
| **overall** | **1885/2000（94.25%）** | **96.45%** | **−2.2pp** |

修复前同一配置为 **0/20**。

## 五、未解释的残余：libero_spatial task 5

`libero_spatial` 逐 task 对照下，39 个 task 在 ±8pp 内（LR 常常更高），只有 task 5 掉了 68pp：

| task | UP-REL | LR-REL |
| ---: | ---: | ---: |
| 0–4, 6–9 | 84%–100% | 88%–100% |
| **5** | **48/50（96%）** | **28%** |

task 5 = `pick up the black bowl on the ramekin and place it on the plate`，是该 suite 中**唯一
目标物体架在另一个小物体上**的配置，抓取容差最紧。视频显示 LR 侧**场景正确、目标识别正确，
在抓取瞬间将碗碰落**——是精细定位误差，不是理解错误。

已排除的差异（均经源码核对或实测）：

| 项 | 结论 |
| --- | --- |
| 180° 旋转 | 等价。LeRobot 在 `processor/env_processor.py` 的 `LiberoProcessorStep` 里做（由 `scripts/lerobot_eval.py` 的 `env_preprocessor` 调用），**不在** `envs/libero.py:_format_raw_obs`；后者只在 `render()` 里为可视化翻转 |
| 8D proprio 组装与 MIN_MAX 模式 | 等价（官方 `configs/data/libero_2cam.yaml` 为 `min/max`，无 per-key 例外） |
| dataset stats | checkpoint 自带 JSON 与官方 JSON md5 相同 |
| prompt 模板 / `text_cfg_scale` / `num_inference_steps` / `negative_prompt` | 逐字与逐值相同（checkpoint 未覆盖，走代码默认值） |
| gripper 后处理 | 数学等价：官方 `sign(−(2v−1))`，LeRobot `FastWAMActionToggleProcessorStep` 同式 |
| 相机拼接顺序 | 等价（`sorted(image_keys)` → primary, wrist） |
| 初始状态数组 | 40 个 task 全部 `max_abs_delta = 0` |
| 动作执行路径 | LeRobot `step` 直通 `self._env.step(action)`，无变换/裁剪/缩放 |
| action ensembler | 两侧都关（`configs/sim_libero.yaml` 为 `use_action_ensembler: false`） |
| settling 步数（10 vs 5） | 无效：34.0% vs 28.0%（n=50，约 0.65σ） |
| 渲染分辨率（原生 224 vs 256→224） | 无效：40.0% vs 28.0%（n=50，约 1.28σ） |
| dtype（bfloat16 vs float32） | 无效：30.0%（n=20）vs 同口径基线 25.0% |

task 5 的全部 arm：

| dtype | obs | settling | n | pc_success |
| --- | ---: | ---: | ---: | ---: |
| float32 | 224 | 10 | 50 | 28.0% |
| float32 | 224 | 5 | 50 | 34.0% |
| float32 | 256 | 10 | 50 | 40.0% |
| bfloat16 | 224 | 10 | 20 | 30.0% |
| bfloat16 | 256 | 10 | 20 | 35.0% |

五个 arm 全部落在 28–40%，彼此在噪声内，无任何单变量能接近 96%。

**结论**：仍存在的差异只剩两套实现内部 RNG 的消费顺序，无法对齐，也不值得继续投入。
因此 **LR-REL 不声称与 UP-REL 等价，两者分栏记录**；引用 LR-REL 数字时必须同时声明
task 5 的未解释缺口。

## 六、工程改动

- launcher 自动标定 CUDA→EGL 映射并按机器指纹缓存；无可用索引时启动前拒绝。
- `--trials-per-worker` 默认改为 `0`（一个 task 一个进程）：40 task × 50 trial 的 checkpoint
  加载次数从 2000 次降到 40 次。原先的每 trial 独立进程只是 EGL 缺陷的绕法，不是修复。
- LR-REL 改为按 (suite, task) 分片（`--env.task_ids=[N]`），`--num-gpus=8` 时 2000 episodes
  从 10+ 小时降到约 1.5 小时；指标按 episode 数加权写入 manifest 的 `lerobot_metrics`。
- `sitecustomize.py`（tracked，随同步生效，不依赖镜像里哪份 vendored 副本）新增：
  `W0_HARD_RESET` 真正生效并打印**实际生效值**、`W0_GL_CONTEXT_GUARD`、
  `W0_LIBERO_NUM_STEPS_WAIT`。注意 `site.execsitecustomize()` 会吞掉异常并让进程以 exit 0
  继续，所以配置错误一律 `os._exit(2)` 并打印原因。
- 新增诊断脚本：`probe_egl_devices.py`、`probe_w0_render_lifecycle.py`、
  `calibrate_w0_egl_devices.py`、`audit_w0_lr_parity.py`、`compare_w0_init_states.py`、
  `run_w0_lr_task_probe.sh`。

## 七、限制与下一步

- UP-REL 是 released checkpoint 的本地闭环 baseline；`FW-V/FW-NV` 因果对照、LR-REL
  quantitative parity 声明和 deployment gate 均未完成，W0 gate 仍在进行中。
- LR-REL 的 94.25% 不能替代 UP-REL，也不能与论文 reference 直接相减。
- `libero_spatial` task 5 的 68pp 缺口未解释；如需继续，唯一剩下的方向是逐步对齐两套实现的
  采样 RNG，成本高、收益不确定。
- 镜像里 `lerobot/policies/fastwam/wan/components.py` 有 +175 行本地加载适配（离线 Wan
  组件），只存在于两个 gitignore 的树中、无任何记录。建议落成 tracked patch 文件，否则重建
  镜像时会静默丢失。
