# SmolVLA 本地 Base Policy 设计

## 目标

将 B0 初始化权重从隐式 Hugging Face cache 移到工作区内的固定、可同步位置：

```text
policies/smolvla_or_pi0/smolvla_base/
```

远端无网络时，B0 默认从该目录读取原始 `lerobot/smolvla_base`，不需要访问 Hugging
Face。模型内容是本地 ML artifact，始终 Git 忽略。

## 方案

采用实体副本，而不是软链接：Hugging Face snapshot 通常由指向 blob cache 的软链接组成；
将这些链接复制到工作区会使远端同步后失效。materialize 操作必须解开链接，保证目标目录
自包含 `config.json`、模型权重和 processor 文件。

不把两步 B0 Smoke checkpoint 作为源；它已经更新过参数。源只能是原始
`lerobot/smolvla_base` snapshot。

## 行为契约

- B0 `--base-policy` 默认值改为仓库相对的本地目录。
- 默认目录不存在或缺少必要文件时，在启动 Accelerate/创建输出目录前报出明确错误。
- 用户显式给出 Hugging Face repo ID 时，仍按该 repo ID 处理；它是有网络场景下的显式
  覆盖，不是远端正式训练的默认路径。
- 新增一个本地 materialize 入口：只从本机已经完整的 cache 复制，默认不下载；若 cache
  不存在则失败并说明需要先在有网机器下载。
- 复制和远端 rsync 均不使用 `--delete`，不删除已有用户模型文件。

## 文件与同步

新增：

```text
scripts/experiments/smolvla_libero/materialize_base_policy.sh
policies/smolvla_or_pi0/smolvla_base/   # ignored generated artifact
```

更新 `.gitignore`，忽略该目录的全部内容但保留目录说明文件（不含模型 hash、缓存路径或
机器信息）。B0 guide 给出 materialize、单次 rsync 和远端正式训练的完整命令。

远端需要先同步本地目录到相同的仓库相对路径；之后正式训练不传 `--base-policy` 即使用
本地副本。也可以显式传同一路径以便日志更直观。

## 验证

- 单元测试覆盖默认本地路径、显式 Hub 覆盖和缺失本地模型的 fail-closed 行为；
- materialize 后确认目标不是软链接、包含 config/weights/processor 文件；
- B0 `--print-command` 默认显示本地 `--base-policy`；
- 在无网络环境以本地目录运行训练的模型解析阶段不产生 Hub 请求。

本变更不重新训练 B0，也不修改数据 split、优化器、BF16 或 8 卡训练契约。
