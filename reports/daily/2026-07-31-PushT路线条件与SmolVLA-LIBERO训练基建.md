# 2026-07-31 PushT 路线条件与 SmolVLA LIBERO 训练基建

## 重要进展

- 完成 PushT E5 Prototype-ID 条件实验的正式收口：Mode-Diffusion 的 oracle
  closed-loop success 为 12/21（57.1%），高于 Mode-BCChunk 的 7/21（33.3%），但两者
  requested→achieved 混淆矩阵都没有形成对角占优，不能证明 P0/P1/P2 可作为稳定控制命令。
- 将下一阶段条件表示从轨迹聚类 ID 改为 **RF7 condition**：每条轨迹编码为固定 BEV
  坐标系中的 `[4,7,7]` occupancy、progress 和二维 direction field，全部 165 条 train
  episodes 以及其中全部合法 action chunks 均可参与训练。
- 完成 RF7 数据 sidecar、RouteField-BCChunk、RouteField-Diffusion、单卡 Smoke 和
  Accelerate 多卡训练入口；当前状态是“可直接启动训练”，尚未产生 RF7 正式训练或闭环结果。
- 打通 SmolVLA B0 的完全离线加载：本地 base policy 同时提供外层权重、SmolVLM 结构配置、
  tokenizer 和 processor，模型与 LeRobot tokenizer processor 都不再依赖 Hugging Face 网络。
- 完成 B0 单卡 Smoke、单机 8 卡 BF16 正式训练和严格 checkpoint resume 契约；B0 使用
  37 个背景任务的 1,412 条 train episodes / 236,470 frames，目标三任务不会进入 B0 训练。
- 完善 LIBERO SmolVLA 闭环评测、离线资产检查、任务可辨识的 Rerun/录像命名，以及训练
  镜像中的 `watch` 动态库隔离；这些属于接口与环境基建，不视为正式模型效果。

## 已完成工作

### PushT E5 正式结论收口

E5 使用 165 条 train demonstrations 构造 K=3 trajectory prototypes，并完成
Mode-BCChunk 与 Mode-Diffusion 的 development checkpoint selection 和 21 条 audit
episodes 的 held-out closed-loop 评测。最终选中 BC step 175000 和 DP step 125000。

在 oracle prototype 条件下，DP 的成功率高于 BC，但 adherence 仍然偏低：DP 的
P0/P1/P2 adherence 为 14.3%/28.6%/19.0%，BC 为 19.0%/23.8%/19.0%。固定 DP 的
initial noise 和完整 reverse noise stream 后，切换 condition 会改变 action chunk，说明
condition 输入链路确实有效；负结果主要来自 prototype 的可选择性不足，以及训练集没有
“同一初始状态、多条成功路线”的配对监督。

因此 E5 支持的结论是：prototype embedding 会影响策略，但目前没有建立可靠的语义模态
控制。P0/P1/P2 是完整 GT 轨迹的事后聚类标签，不能直接解释成任意初始状态下都可切换的
规划指令。

### PushT E6 RF7 condition

RF7 使用 T-block 中心的稳定网格路线生成固定空间条件。每个 episode 的 condition 为：

```text
route_field.shape = [4, 7, 7]
channels = occupancy, progress, direction_x, direction_y
```

7×7 相比 5×5 保留了更多路线结构。使用 3 帧稳定判定时，165 条 train episodes 的审计
结果为：

- 路线长度中位数为 4；只有 3 条 episode 至多确认一个稳定格；
- 44 条原始路线存在重复格；
- 支持数不低于 3 的局部有向转移覆盖约 89.8%；
- 回退和闭环只在 condition 表示中执行可审计的 backtrack/loop erasure，不删除 episode
  或 action chunk。

首个 A/B 诊断路线固定为 `16→17→24` 与 `16→23→24`，分别表达先向右再向下、先向下再
向右。现有 demonstrations 没有同初始状态下的严格 A/B 配对，因此正式评测必须固定
observation 和 DP noise，再检查切换 RF7 后的实际 T-block 路径，而不能只比较动作数值。

已准备 sidecar 构建、可视化、BCChunk 和 Diffusion 的单卡/多卡启动脚本。该阶段只完成
实现与离线数据审计，尚未启动正式训练。

### SmolVLA B0 完全离线训练

B0 从本地 materialized `smolvla_base` 开始，排除三个 held-out 目标任务，只训练
LIBERO-Goal 的 37 个背景任务。正式配置冻结为 8 卡、每卡 batch 8、BF16、30,000
updates，每 5,000 steps 保存一次。

离线加载修复同时覆盖两条此前会访问 Hub 的路径：

1. SmolVLA 内部使用 `vlm_assets/` 构造 SmolVLM 结构和 image/text processor；
2. LeRobot 的独立 tokenizer processor 也把 `tokenizer_name` 重定向到同一本地目录。

SmolVLM 参数不重复保存到 `vlm_assets/`，而是由外层 SmolVLA checkpoint 一次性加载。
训练入口固定开启 Hugging Face、Transformers 和 Datasets offline mode，并在创建 dataset
前执行严格本地预检；缺文件时立即失败，不会联网重试。

B0 的 checkpoint 同时保存 model、optimizer、scheduler、RNG 和 global step。续训只能从
同一 run 的最新完整 checkpoint 开始，并恢复原 world size、BF16 和训练配置，不能在
resume 时更改 batch、学习率、GPU 数或输出目录。

### LIBERO 评测、Rerun 与镜像运行环境

统一 SmolVLA LIBERO evaluator 支持 Hub checkpoint、本地普通 checkpoint 和本地 PEFT
adapter，并输出 metrics、Parquet rollout、MP4 和 Rerun Recording。录像与 Rerun 文件名
加入 suite、task、init state 等上下文，避免多任务评测产物互相覆盖或难以辨认。

离线评测增加 LIBERO XML/scene 资产预检，避免 rollout 中途才尝试联网。训练镜像同步补充
环境锁定和验证脚本，并为 `watch` 命令增加动态库隔离，防止宿主或 Conda 的库搜索路径
污染工具自身运行环境。

## 验证与结果边界

| 项目 | 类别 | 截至 7 月 31 日的状态 |
| --- | --- | --- |
| E5 checkpoint selection | Closed-loop / Development | BC 175k，DP 125k |
| E5 held-out audit | Closed-loop / Formal | Oracle BC 33.3%，DP 57.1%；未证明 prototype adherence |
| RF7 数据构建与入口 | Offline / Smoke-ready | 165/165 train episodes 可用；正式训练未启动 |
| SmolVLA B0 launcher | Offline / Smoke-ready | 单卡 Smoke、8 卡 BF16 与 resume 入口就绪 |
| SmolVLA B0 模型效果 | Formal | 尚未在仓库登记正式 validation/test 指标 |
| LIBERO evaluator | Closed-loop infrastructure | 接口和产物链路就绪，不等同于正式成功率 |

E5 的实验字段如下：Task suite 为 PushT；dataset 为 206 episodes（165 train / 20
development / 21 audit）；observation 为 96×96 RGB 与二维 agent position、历史长度 2；
action 为二维绝对 target position；内部 action horizon 16、闭环执行 8 步；control
frequency 为 10 Hz。选中 checkpoint 的 held-out normalized MSE 为 BC 0.01333、DP
0.01315；replan p95 为 BC 10.22 ms、DP 608.93 ms。主要失败包括任务超时、轨迹不可分类
和 requested/achieved prototype 不一致。

B0 的冻结字段为：Task suite 为 LIBERO-Goal 37 个背景任务；dataset 为 1,412 train
episodes / 236,470 frames；observation 为两路 256×256 RGB 与 8 维 state；action 为 7 维
相对末端动作；prediction/execution horizon 为 50/10；control frequency 为 10 Hz。正式
train loss、open-loop error、closed-loop success、latency 和 failure cases 均尚未登记，
不能从 launcher 或 Smoke 状态推断模型效果。

## 限制与风险

- E5 的 P0/P1/P2 来自事后聚类，不是已经验证可任意切换的控制命令。
- RF7 的 loop erasure 无法表达同一格被多次访问的完整时间顺序；7×7 也会增加精确路线
  稀疏性。
- 当前 PushT 数据没有同一初始状态下的多路线配对 demonstrations，observation 可能继续
  压过 route condition。
- SmolVLA 的完全离线加载和训练入口只证明工程链路可用；B0 validation、held-out test
  以及目标任务迁移仍需要正式闭环数据。
- Docker/Volc 训练环境和本地挂载模型必须保持相同的仓库相对结构；本阶段没有把本地
  artifact 或模型权重加入 Git。

## 后续工作

- 启动 RF7 RouteField-BCChunk 与 RouteField-Diffusion 正式训练，并按固定
  observation/noise 的 A/B 协议评测路线 adherence。
- 完成 B0 checkpoint selection 和 held-out background test，冻结唯一 selected B0。
- 从同一 selected B0 开始训练 C1 Default Non-PEFT、C2 Canonical LoRA 和 C3 LoRA +
  Paraphrase，保持数据、预算和 selection protocol 一致。
- 正式结果继续登记到 `reports/experiment_log.md`，避免把 Smoke 或接口验证写成模型结论。

完整操作说明见 `reports/guides/pusht_diffusion/e5_mode_conditioned.md`、
`reports/guides/pusht_diffusion/e6_route_field_7x7.md` 和
`reports/guides/smolvla_libero/README.md`。
