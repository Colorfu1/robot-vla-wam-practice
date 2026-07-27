# PushT 多卡训练入口设计

## 目标

为 PushT 的 BCChunk 和 Diffusion 正式训练增加清晰、可复现的单机多 GPU
入口，同时保持现有模型配置和 LeRobot 训练循环不变。

完成后提供两个独立命令：

```text
scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh
scripts/experiments/pusht/train_diffusion_multi_gpu.sh
```

多卡实现使用 LeRobot 已集成的 Hugging Face Accelerate/DDP。项目不实现自己的
梯度同步、优化器或 checkpoint 逻辑。

## 非目标

- 不支持多机训练。
- 不引入 DeepSpeed、FSDP、梯度累积或混合精度新变量。
- 不改变 BCChunk 或 Diffusion 的模型结构、loss、数据 split 和初始化权重。
- 不替换现有单卡入口。
- 不自动把单卡 batch size 除以 GPU 数量。

## 用户接口

两个 Bash 文件只进入 `robot-practice` conda 环境并调用各自明确的 Python
多卡 launcher：

```text
train_bc_chunk_multi_gpu.sh
  -> robot_practice.experiments.pusht_diffusion.train_bc_chunk_multi_gpu

train_diffusion_multi_gpu.sh
  -> robot_practice.experiments.pusht_diffusion.train_diffusion_multi_gpu
```

两个 Python launcher 最终分别通过 `accelerate launch` 启动现有训练 module：

```text
train_bc_chunk_multi_gpu.py
  -> accelerate launch
  -> robot_practice.experiments.pusht_diffusion.train_bc_chunk
  -> lerobot_train.train()

train_diffusion_multi_gpu.py
  -> accelerate launch
  -> robot_practice.experiments.pusht_diffusion.train_diffusion
  -> lerobot_train.train()
```

模型配置仍只由以下 factory 创建：

```text
model_configs/bc_chunk.py
model_configs/diffusion.py
```

多卡 launcher 不包含 dataset 参数、模型超参数或 optimizer 配置。

### 多卡专用参数

```text
--num-gpus=N
```

- 必填。
- 必须大于等于 2。
- 使用当前 `CUDA_VISIBLE_DEVICES` 中的前 N 张可见 GPU。
- N 大于 `torch.cuda.device_count()` 时，在创建 worker 前失败。
- `--num-gpus=1` 不回退到单卡，而是提示使用原有单卡脚本。

其余参数与现有训练入口完全相同：

```text
--mode=smoke|pilot
--steps=N
--train-seed=N
--split-file=PATH
--output-dir=PATH
--batch-size=N
--num-workers=N
--device=auto|cuda
--print-command
```

多卡入口拒绝 `--device=cpu` 和 `--device=mps`。

### Batch size 语义

`--batch-size` 表示每个 DDP worker、也就是每张 GPU 的 batch size，保持
LeRobot/Accelerate 原生语义：

```text
global batch size = per-GPU batch size * world size
```

例如：

```text
--num-gpus=2 --batch-size=64
```

表示每卡 64、全局 128。入口不会为了保持单卡实验的全局 batch 而自动缩放。

启动时必须打印：

```text
per-GPU batch size
world size
global batch size
```

这些字段也写入 resolved configuration。

## 父进程启动流程

共享模块 `distributed_training.py` 负责进程编排，模型专用 launcher 只提供：

```text
policy_type
worker module
description
```

父进程按以下顺序执行：

1. 从参数中取出并校验 `--num-gpus`。
2. 使用现有共享训练参数解析器解析剩余参数，拒绝重复和未知选项。
3. 检查 CUDA 可用性和可见 GPU 数量。
4. 只在父进程中解析一次最终输出目录。
5. 将所有训练参数规范化，并显式加入同一个 `--output-dir`。
6. 构造并执行：

   ```text
   python -m accelerate.commands.launch
     --multi_gpu
     --num_processes=N
     --num_machines=1
     --gpu_ids=all
     --mixed_precision=no
     --dynamo_backend=no
     --main_process_port=0
     -m <worker-module>
     <canonical-training-arguments>
   ```

7. 原样返回 Accelerate 的退出码。

默认时间戳只能由父进程生成一次，禁止各 worker 独立生成时间戳。

上述 Accelerate 参数必须显式传入，不能依赖用户执行过的 `accelerate config`：

- `--gpu_ids=all` 保留当前进程的 `CUDA_VISIBLE_DEVICES`，worker 使用其中前 N
  个逻辑设备；
- `--mixed_precision=no` 保持现有训练精度；
- `--dynamo_backend=no` 不启用额外编译；
- `--num_machines=1` 固定为本设计唯一支持的单机模式；
- `--main_process_port=0` 让每次启动选择空闲 rendezvous 端口。

因此，用户本地 Accelerate 配置中的 GPU ID、mixed precision 或 Dynamo 设置不能改变
本实验语义。

### `--print-command`

`--print-command` 不执行输出目录/日志的存在性检查，不创建输出目录，不加载数据，也
不创建 worker；但仍执行参数、CUDA 数量和 Smoke/Pilot 路径隔离等纯语义检查。它输出：

- 完整 Accelerate 命令；
- 明确的 worker module；
- 当前可见 GPU 数和请求 GPU 数；
- per-GPU、world size 和 global batch；
- 原有模型训练入口生成的 resolved configuration。

## Worker 与 rank 行为

Accelerate 为每个 worker 设置标准分布式环境变量。训练入口通过一个小型
`DistributedProcessContext` 读取并校验：

```text
RANK
LOCAL_RANK
WORLD_SIZE
```

单卡直接启动时默认：

```text
rank=0
local_rank=0
world_size=1
```

训练仍由 `lerobot_train.train(cfg)` 创建官方 `Accelerator`，不在本项目重复实现
DDP 初始化。LeRobot 继续负责：

- `accelerator.prepare()`；
- DataLoader 分片；
- 梯度同步；
- 主进程 checkpoint 保存；
- worker barrier；
- 训练结束清理。

### 输入校验

校验分为两类：

1. 所有 rank 都执行只读 artifact 校验，包括 dataset view、split、初始化文件、
   manifest、fingerprint 和哈希。
2. 只有 rank 0 校验 output directory 和临时日志不存在。

非主 rank 不检查也不创建共享日志路径，避免把 rank 0 正常创建的临时日志误判为
冲突。

### 日志

- 只有 rank 0 使用现有 `persistent_training_log()`。
- 只有 rank 0 打印 resolved configuration。
- `train.log` 保存 rank 0 的训练主日志。
- 非主 rank 的异常仍输出到启动终端，由 Accelerate 汇报 worker rank。
- 成功后 rank 0 将临时日志移动到 `<output-dir>/train.log`。
- 失败时保留 `<output-dir-name>.train.log`，不覆盖已有结果。

## 错误处理

以下情况必须在启动 worker 前给出清晰错误：

- 缺少 `--num-gpus`；
- `--num-gpus < 2`；
- CUDA 不可用；
- 可见 GPU 数不足；
- 多卡入口使用 CPU/MPS；
- 重复或未知参数；
- Smoke/Pilot 输出目录混用。

数据、初始化或 fingerprint 错误由 worker 的既有校验报告。任意 worker 返回非零时，
父 launcher 必须返回非零，不能打印成功提示。

## 测试

### 命令与参数测试

- 两个 Bash 文件分别转发到正确的模型专用多卡 module。
- Bash 文件不包含 dataset 或模型配置。
- `--num-gpus` 缺失、等于 1、超过可见卡、重复时失败。
- 未知训练参数继续由共享 parser 拒绝。
- CPU/MPS 被多卡入口拒绝。
- 默认输出目录只解析一次，并显式传给 worker。

### Dry-run 测试

对 BCChunk 和 Diffusion 分别执行 `--print-command`，确认：

- 命令包含 `accelerate.commands.launch --multi_gpu`；
- `--num_processes` 与请求一致；
- `-m` 后是正确的现有训练 module；
- per-GPU、world size 和 global batch 正确；
- 不创建输出目录或日志。

### Rank 与日志测试

- 单进程默认 context 为 rank 0/world size 1。
- 模拟 rank 0 时创建并搬移日志。
- 模拟非主 rank 时不接触共享日志。
- resolved configuration 记录分布式 batch 信息。

### 集成验证

若当前机器至少有两张 CUDA GPU，使用 Smoke 数据执行：

```text
2 GPU
1 optimizer update
batch size 1 per GPU
num_workers 0
```

验证输出目录、checkpoint/配置和单一 `train.log`。

如果当前机器不足两张 CUDA GPU，则不为生产 launcher 增加 CPU 例外，也不把
`--num-gpus` 校验改成可注入的测试后门。此时只运行命令构造测试、模拟
rank/environment 的日志测试和现有单卡回归测试，并明确记录真实双 GPU 集成验证尚未
在本机执行。

## 文档

更新以下指南：

```text
reports/guides/pusht_diffusion/README.md
reports/guides/pusht_diffusion/smoke.md
reports/guides/pusht_diffusion/benchmark.md
```

指南必须：

- 同时保留单卡和多卡命令；
- 给出 BCChunk 和 Diffusion 的完整多卡示例；
- 明确 `--batch-size` 是每卡 batch；
- 展示 global batch 计算；
- 说明多卡使用当前可见设备中的前 N 张卡；
- 说明 `--num-gpus=1` 应改用单卡脚本。

## 完成标准

1. 两个模型拥有名称明确、互不分发的多卡入口。
2. 多卡入口最终进入原有模型训练 module 和 LeRobot 官方训练循环。
3. 输出目录只由父进程确定一次。
4. rank 0 独占共享日志和配置打印。
5. 每卡和全局 batch 语义在终端、配置和指南中一致。
6. 参数错误在创建 worker 前失败。
7. 单卡入口和既有训练配置保持兼容。
