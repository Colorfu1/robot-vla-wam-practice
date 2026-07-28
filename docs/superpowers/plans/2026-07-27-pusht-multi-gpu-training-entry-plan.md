# PushT 多卡训练入口实施计划

## 兼容性边界

- 保留现有 `train_bc_chunk.sh`、`train_diffusion.sh` 及其参数和默认值。
- 不修改模型 factory、数据 split、初始化 checkpoint、训练 loss 或推理/评测代码。
- 新增的多卡入口最终仍调用现有 `train_bc_chunk.py` / `train_diffusion.py`。
- 共享 runtime 在没有分布式环境变量时固定解析为 rank 0、world size 1。

## 实施步骤

1. 在 `training_common.py` 增加分布式进程元数据、规范化 worker 参数和单次输出目录解析，
   并让 resolved configuration 同时记录每卡与全局 batch。
2. 在 `training_runtime.py` 让持久化日志可由 rank 0 独占；保持默认启用，保证单卡行为
   不变。
3. 重构两个现有模型训练入口，暴露只负责构造 resolved config 的函数；`main()` 仍直接
   调用 `lerobot_train.train()`。
4. 新增共享 `distributed_training.py`，显式构造不受用户 Accelerate 默认配置影响的
   单机 DDP 命令。
5. 新增 BCChunk/Diffusion 两个 Python 多卡 launcher 和两个薄 Bash 入口。
6. 增加参数、命令、rank/log、batch 元数据和单卡回归测试。
7. 更新 PushT README、Smoke 和 Benchmark Guide。
8. 运行 Ruff、相关 Pytest、两个单卡 `--print-command` 和多卡命令构造验证；本机只有
   一张 CUDA GPU，因此明确记录未执行真实双 GPU update。

## 验收

- 现有单卡测试全部通过。
- 现有单卡 dry-run 仍显示原模型 module，且不创建输出。
- 新多卡 dry-run 的内部命令固定包含 `--multi_gpu`、`--num_processes`、
  `--gpu_ids=all`、`--mixed_precision=no`、`--dynamo_backend=no` 和随机空闲端口。
- rank 0 是唯一共享日志写入者。
- `--batch-size=64 --num-gpus=2` 明确显示全局 batch 128。
