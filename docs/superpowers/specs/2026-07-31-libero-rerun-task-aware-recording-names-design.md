# LIBERO Rerun 任务可辨识 Recording 名设计

## 问题

多个 LIBERO 闭环 `.rrd` 同时载入 Rerun 时，当前名称只有 episode、seed、初始状态
和结果，例如：

```text
Episode 000 | seed=2000 | init=0 | success
```

名称缺少 suite、task ID 和任务语义，用户无法从 Recording 列表判断对应任务。

## 决策

所有共享 LIBERO 闭环 Recording 使用以下名称：

```text
<suite>/task_<id>/<task_name> | Episode <episode> | seed=<seed> | init=<id> | <outcome>
```

例如：

```text
libero_goal/task_0/open_the_middle_drawer_of_the_cabinet | Episode 000 | seed=2000 | init=0 | success
```

无逐 episode 可视化、只生成 summary Recording 时使用：

```text
<suite>/task_<id>/<task_name> | Summary
```

`task_name` 使用 LIBERO 的稳定 snake_case 标识，不使用可能包含标点或改写内容的
language prompt。

## 实现边界

- 扩展共享 `libero_episode_recording_name`，要求调用方传入 suite、task ID 和
  task name。
- 增加同格式的 summary 名称生成函数。
- `evaluate_libero` 已持有解析后的 `LiberoTaskIdentity`，在发送最终 Recording 名时
  传入该身份。
- ACT 和 SmolVLA 都复用 `evaluate_libero`，因此无需各自实现命名逻辑。
- 不改变模型输入、推理随机数、环境状态、动作、成功判定、指标或 RRD 实体路径。
- 已生成的 RRD 不原地修改；当前可视化样本在代码修复后重新运行，以生成可信的新
  Recording 元数据。

## 验证

- 单元测试固定成功、失败和 summary 名的精确格式。
- 更新现有闭环集成测试中的期望 Recording 名。
- 运行相关评测测试，确认合并 RRD 的 Recording 数量和名称校验仍通过。
- 用一个训练内任务和一个 target 任务生成 RRD，同时载入 Rerun，确认列表无需打开
  内容即可辨认任务。

