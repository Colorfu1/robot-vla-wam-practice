# ACT Action Chunking 启动脚本可读性重构

## 目标

让使用者打开一个公开脚本，就能直接看到最终执行的 `conda run ... lerobot-train`
命令，同时继续复用 A0/A1 的公共数据、训练参数和参数校验。

## 方案

新增公开入口：

```text
scripts/experiments/act/action_chunking.sh
```

调用方式：

```bash
scripts/experiments/act/action_chunking.sh --variant=a0 --steps=20000
scripts/experiments/act/action_chunking.sh --variant=a1 --steps=20000
```

脚本在同一文件内完成以下工作：

1. 解析 `--variant`、`--steps`、`--policy-path` 和 `--print-command`；
2. 用一个清晰的 `case` 将 A0 映射为 `chunk_size=1, n_action_steps=1`，将 A1
   映射为 `chunk_size=20, n_action_steps=5`；
3. 固定相同的 20 条 episodes、batch size、seed 和其他训练参数；
4. 对 `steps=0` 调用现有 step-0 initializer；
5. 对正训练步数，在文件底部直接写出完整的 `conda run ... lerobot-train`
   命令，不再经由内部命令数组和另一层执行脚本。

保留两个便捷别名：

```text
a0_single_step.sh -> action_chunking.sh --variant=a0
a1_chunking.sh    -> action_chunking.sh --variant=a1
```

删除内部实现：

```text
run_action_chunking_variant.sh
```

实验指南以统一入口作为主启动方式，同时保留两个别名示例。

## 错误处理

- 统一入口必须恰好提供一次 `--variant=a0|a1`，缺失、重复或其他值均报错；
- 别名固定注入一次 variant；调用别名时再传 `--variant` 会按重复参数拒绝；
- `steps=0` 禁止 `--policy-path`，只执行对应 variant 的 initializer；
- `steps>0` 未提供 `--policy-path` 时从随机初始化开始训练；
- `steps>0` 提供 path 时先验证它与同 variant 兼容，再通过 `--policy.path` 加载；
- 保留现有 steps、checkpoint、数据目录、输出目录和并发启动检查；
- 使用文件描述符获取 `flock`，锁成功后直接 `exec conda ...`，让最终训练命令保持可见；
- `--print-command` 完成参数、数据和 checkpoint 验证，但不创建输出目录、不获取锁、
  不启动 initializer 或训练；`steps=0` 打印 initializer 命令，正步数打印完整的
  `conda run ... lerobot-train` 命令。

## 验证

- A0/A1 别名与统一入口生成相同参数；
- A0/A1 使用完全相同的 20 条 episodes；
- `steps=0` 仍调用对应 variant 的 initializer；
- 重复、未知或无效参数仍被拒绝；
- 指南以 `action_chunking.sh --variant=...` 为主命令，并说明两个别名；
- 运行 Shell 语法检查、launcher 测试和全仓库测试。
