# E6 7×7 Route Field 条件控制

## 统一名称

后续统一把这个模型输入称为 **RF7 condition**（RouteField-7×7 路线条件），简称
**RF7**。代码中的 batch key 是 `route_field`。`route_manifest.json`、
`episode_assignments.parquet` 和 `route_fields.npz` 是 RF7 sidecar；真正传给模型的是
每个训练样本携带的 `[4,7,7]` 浮点 tensor。

## 实验定位

RF7 使用固定 BEV 坐标系中的空间路线场替代 E5 的离散 prototype ID。它要验证的是：
固定 observation 和 Diffusion noise 时，切换一条带有明确局部方向的粗粒度路线，
能否稳定改变动作和完整 rollout 的 T-block 中心路径。

7×7 相比 5×5 能保留更多轨迹结构。使用 3 帧稳定判定时，165 条 train episode 的
原始路线统计为：

- 5×5：路线长度中位数 3，7 条 episode 至多确认一个稳定格；
- 7×7：路线长度中位数 4，3 条 episode 至多确认一个稳定格；
- 5×5 有 52 条原始路线含重复格，7×7 有 44 条；
- 7×7 的局部有向转移中，训练支持数不低于 3 的转移覆盖约 89.8%。

首版固定 7×7，不加 NULL、condition dropout、classifier-free guidance、waypoint cursor
或外部阶段机。

## 数据和 condition 契约

每帧从已有 `trajectories.parquet` 读取 `block_position`，按 512×512 PushT 世界坐标映射到
7×7 row-major 网格。进入新格后连续保持 3 帧才确认切换，并压缩连续重复。

RF7 不再按路线长度、重复格或精确路线支持度过滤训练 episode：

- 全部 165 条 train episode 参与训练；
- LeRobot 原有的全部合法随机 index/action chunk 都可采样；
- 只有一个稳定格的 episode 也合法，编码为单格 occupancy/progress，方向为 0；
- 精确路线支持度只用于数据统计和选择 A/B 诊断路线，不决定训练资格。

固定空间场不能直接表达同一格被多次访问的先后顺序，因此 sidecar 同时保存：

- `raw_route_cells`：去抖后的原始稳定路线；
- `route_cells`：模型实际使用的无重复 macro route；
- `backtrack_removed_cells`：迭代压缩 `A→B→A` 删除的格数；
- `loop_removed_cells`：对残余重复格做按时间顺序 loop erasure 删除的格数。

这项简化只改变 condition 表达，不删除 episode 或训练 chunk。后续分析可按审计字段单独
检查受路线简化影响的样本。

每个 macro route 编码为 `[4,7,7]`：

1. `occupancy`：路线格为 1；
2. `progress`：按路线顺序从 0 线性增加到 1，单格路线取 1；
3. `direction_x`：当前路线格指向下一格的归一化 x 分量；
4. `direction_y`：当前路线格指向下一格的归一化 y 分量。

路线外和终点方向都为 0。同一 episode 中所有 action chunk 使用同一份 route field；
模型从当前 observation 自行判断路线进度。

首个 A/B 诊断对使用相同起点和终点、不同转向顺序的路线：

```text
Route A: 16 → 17 → 24   support=3   先向右、再向下
Route B: 16 → 23 → 24   support=2   先向下、再向右
```

现有 demonstration 没有严格的同初始状态 A/B 配对监督。因此“condition 被模型读取”
不等于“路线可任意切换”；正式结论必须来自固定 observation/noise 的切换诊断和相同环境
初态的闭环 A/B 对照。

## 生成 sidecar 与图示

先确认 E0 已生成完整轨迹表，然后执行：

```bash
bash scripts/experiments/pusht/build_route_fields.sh
```

默认读取：

```text
outputs/pusht_diffusion/data_inspection/trajectories.parquet
```

默认输出：

```text
outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/
├── route_manifest.json
├── episode_assignments.parquet
├── route_fields.npz
├── metrics.json
└── metrics.md
```

重新生成时显式加 `--overwrite`。训练入口会校验 split hash、artifact hash 和
网格/通道契约，并在创建 run 前拒绝损坏或不匹配的输入。

生成实际 A/B condition 图：

```bash
conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.route_field_visualization
```

图像默认保存到
`outputs/pusht_diffusion/pilot/route_field_7x7/visualizations/rf7_route_a_b.png`。

## Smoke 训练

```bash
bash scripts/experiments/pusht/train_route_field_bc_chunk.sh \
  --mode=smoke \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=1000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda
```

```bash
bash scripts/experiments/pusht/train_route_field_diffusion.sh \
  --mode=smoke \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=1000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda
```

## 正式多卡训练

推荐 8 卡、每卡 batch 8，使 global batch size 为 64：

```bash
bash scripts/experiments/pusht/train_route_field_bc_chunk_multi_gpu.sh \
  --num-gpus=8 \
  --mode=pilot \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=200000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_bc_chunk
```

```bash
bash scripts/experiments/pusht/train_route_field_diffusion_multi_gpu.sh \
  --num-gpus=8 \
  --mode=pilot \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=200000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_diffusion
```

`--batch-size` 是每卡 batch size。运行前可追加 `--print-command` 检查最终 Accelerate
命令、resolved episodes 和模型配置，此时不会创建输出目录。

## 训练后必须做的评测

- 固定同一 observation，只切换 Route A/B；DP 同时固定 initial noise 和 reverse noise
  stream；
- 相同环境初始状态分别请求 A/B，记录实际 T-cell 序列；
- requested route 与 achieved route 的有序 waypoint adherence 和编辑距离；
- Route A/B 各自 success、final reward、final rotation error；
- 同一初始状态下 A/B 都成功的比例；
- agent/T 完整轨迹和 Rerun Recording。

第一阶段成功标准不是“condition 让动作数值变化”，而是切换 A/B 后实际 T 中心路径按预期
先右/先下地改变，且 adherence 明显高于错误路线或随机基线。

## 已知限制

- 精确 RF7 路线族更稀疏；全量训练依赖共享的空间卷积编码和局部转移复用，而非每个
  exact route 都有大量样本；
- loop erasure 会丢失回退和闭环的时间顺序，必须结合 sidecar 审计字段报告其影响；
- 现有数据不是同一状态下的 A/B 配对数据，observation 可能压过 condition；
- 7×7 hard assignment 更靠近边界，3 帧 debounce 只能缓解抖动；
- route field 只约束 T 中心平移，不直接表达最终旋转。
