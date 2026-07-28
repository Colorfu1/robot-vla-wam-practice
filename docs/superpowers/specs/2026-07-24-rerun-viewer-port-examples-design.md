# Rerun Viewer 端口示例统一设计

## 目标

避免用户同时打开 A0/A1 或其他两份 RRD 时，因为两个 Rerun 进程默认监听 `9876`
而把第二份数据流发送到第一个 Viewer。

## 范围

更新当前使用的用户指南和 evaluator 脚本头部：

- `reports/guides/act/act_action_chunking.md`
- `reports/guides/act/act_a_sanity.md`
- `reports/guides/act/act_temporal_ensemble.md`
- `scripts/experiments/act/eval_offline.sh`
- `scripts/experiments/act/eval_libero.sh`

不修改 evaluator、Rerun recording、application ID 或自动启动逻辑，不新增辅助脚本。

## 示例约定

单窗口示例显式使用：

```bash
rerun --persist-state --detach-process --port=9876 <result.rrd>
```

并行双窗口示例使用不同端口：

```bash
rerun --persist-state --detach-process --port=9876 <first.rrd>
rerun --persist-state --detach-process --port=9877 <second.rrd>
```

每处说明以下语义：

- `--detach-process` 只分离进程，不自动更换端口；
- 每个并行 Viewer 必须使用不同 `--port`；
- 把多个 RRD 路径传给同一条 `rerun` 命令只会在同一个 Viewer 中加载；
- `--persist-state` 才会保存用户调整后的布局；
- 同一 application ID 的两个并行 Viewer 共享持久化布局，最后关闭者可能覆盖布局状态。

指南中的命令继续通过项目 Conda 环境执行：

```bash
conda run --no-capture-output -n robot-practice rerun ...
```

## 验证

- 所有现用 `rerun` 命令都包含 `--persist-state`、`--detach-process` 和显式端口；
- A0/A1 双窗口示例分别使用 `9876/9877`；
- 两个 evaluator Shell 脚本通过 `bash -n`；
- 文档不引入本机绝对路径或私人信息。
