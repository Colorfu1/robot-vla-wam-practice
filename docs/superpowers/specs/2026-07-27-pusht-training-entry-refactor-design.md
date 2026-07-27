# PushT 训练入口与模型配置重构设计

日期：2026-07-27

## 1. 问题

当前 `train_bc_chunk.sh` 和 `train_diffusion.sh` 同时负责参数解析、数据路径推导、
episode 选择、初始化权重定位、训练命令拼装、日志转存和错误检查。相同逻辑在两个
脚本中重复，VSCode Debug 又维护了第三份训练参数。

模型结构配置虽然存在于 JSON 模板和生成的 step-zero checkpoint 中，但训练入口只
通过 `--policy.path` 间接加载它。阅读启动脚本时无法直接确认模型类型、配置来源和
最终进入的 Python 代码。

## 2. 目标

- 使用明确的 Python 模块分别构造 `BCChunkConfig` 和 `DiffusionConfig`。
- 使用 LeRobot 官方的 `DatasetConfig` 和 `TrainPipelineConfig`。
- 继续使用原始 `lerobot.scripts.lerobot_train.train()` 完成全部训练工作。
- Bash 脚本仅激活环境并转发参数，不包含实验业务逻辑。
- Bash 与 VSCode Debug 使用完全相同的 Python 入口。
- 保留 BCChunk 和 Diffusion 的 paired step-zero 初始化。
- 保留当前针对 LeRobot episode 子集 sampler 的本地兼容修复。
- 删除重复的 JSON model config，避免多个配置源产生漂移。

## 3. 非目标

- 不修改安装目录中的 LeRobot 源码。
- 不复制或重写 LeRobot 的 dataset、optimizer、scheduler、训练循环和 checkpoint
  保存逻辑。
- 不改变当前 PushT 数据划分、模型结构、损失函数或训练超参数。
- 不在本次重构中增加新的实验变量。

## 4. 目录结构

```text
src/robot_practice/experiments/pusht_diffusion/
├── model_configs/
│   ├── __init__.py
│   ├── bc_chunk.py
│   └── diffusion.py
├── train_config.py
├── train_bc_chunk.py
├── train_diffusion.py
└── training_runtime.py
```

职责如下：

- `model_configs/bc_chunk.py`
  - 显式 import `BCChunkConfig`。
  - 提供 `make_bc_chunk_config()`。
  - 在函数中完整列出本实验关心的 policy 配置。
- `model_configs/diffusion.py`
  - 显式 import LeRobot `DiffusionConfig`。
  - 提供 `make_diffusion_config()`。
  - 在函数中完整列出本实验关心的 policy 配置。
- `train_config.py`
  - 定义统一的训练命令行参数。
  - 读取 `split.json` 并解析 smoke/pilot episode。
  - 构造官方 `DatasetConfig` 和 `TrainPipelineConfig`。
  - 设置 batch size、steps、seed、checkpoint 频率和输出目录。
  - 将 paired step-zero 目录赋给 `policy.pretrained_path`。
- `train_bc_chunk.py`
  - import `make_bc_chunk_config()`。
  - 构造 BCChunk 的 `TrainPipelineConfig`。
  - 安装本地 runtime 兼容层。
  - 调用 `lerobot_train.train(cfg)`。
- `train_diffusion.py`
  - import `make_diffusion_config()`。
  - 构造 Diffusion 的 `TrainPipelineConfig`。
  - 安装本地 runtime 兼容层。
  - 调用 `lerobot_train.train(cfg)`。
- `training_runtime.py`
  - 只保存 `SubsetEpisodeAwareSampler` 及其安装函数。
  - 提供训练 stdout/stderr 的 tee context，使日志同时显示在终端并持久化。
  - 不承担 CLI 解析或配置构造。

现有 `initialization.py` 也必须改为直接 import 两个 model config factory。paired
step-zero 生成与正式训练由此共享同一个模型配置源；删除 JSON 后不允许保留任何按文件名
加载 model config 的路径。

## 5. 模型配置

两个 model config 使用真实的 LeRobot dataclass，不再使用本地 JSON 作为配置源。

BCChunk 示例：

```python
from robot_practice.policies.bc_chunk import BCChunkConfig


def make_bc_chunk_config(*, device: str) -> BCChunkConfig:
    return BCChunkConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        crop_shape=(84, 84),
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        fixed_timestep=0,
        do_mask_loss_for_padding=False,
        drop_n_last_frames=7,
        optimizer_lr=1e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
        device=device,
        push_to_hub=False,
    )
```

Diffusion 配置保持与 BCChunk 相同的 observation/action horizon、视觉结构和优化器配置，
仅保留其生成方法所需的 scheduler、timestep 和 prediction 配置。

## 6. 训练数据流

```text
train_bc_chunk.sh / train_diffusion.sh
                    │
                    ▼
train_bc_chunk.py / train_diffusion.py
                    │
                    ├── import 对应 model config
                    ├── 解析 mode/steps/seed/output
                    ├── 读取 split.json
                    ├── 设置 paired initialization
                    └── 构造 TrainPipelineConfig
                                      │
                                      ▼
                      lerobot_train.train(cfg)
                                      │
                         cfg.validate / make_dataset
                                      │
                            make_policy / optimizer
                                      │
                         LeRobot 原始 training loop
                                      │
                         LeRobot 原始 checkpoint
```

`lerobot_train.main()` 是 CLI 适配层；本设计绕过其字符串解析，但不绕过 LeRobot 的正式
训练 pipeline。`lerobot_train.train(cfg)`、`cfg.validate()` 以及其后的全部代码保持
原样。

## 7. Bash 与 Debug

两个 Bash 脚本只保留环境转发：

```bash
#!/usr/bin/env bash
set -euo pipefail

exec conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.train_bc_chunk "$@"
```

Diffusion 脚本只替换 Python module 名称。

VSCode Debug 分别使用相同 module：

```text
robot_practice.experiments.pusht_diffusion.train_bc_chunk
robot_practice.experiments.pusht_diffusion.train_diffusion
```

Debug 可以覆盖 `--mode`、`--steps`、`--train-seed`、`--output-dir` 和调试专用
`--batch-size`，但不能维护另一份模型结构参数。

## 8. 参数与输出

公共 Python CLI 保留：

```text
--mode=smoke|pilot
--steps=N
--train-seed=N
--split-file=PATH
--output-dir=PATH
--batch-size=N
--num-workers=N
--device=auto|cpu|cuda|mps
--print-command
```

默认值仍遵循当前实验：

- smoke：默认 1,000 steps、batch size 8；
- pilot：默认 200,000 steps、batch size 64；
- train seed：默认 1000；
- device：默认 `auto`，按 CUDA、MPS、CPU 的顺序选择当前可用设备，并把解析后的明确
  device 字符串写入 policy config；
- save frequency：不超过 25,000 steps；
- log frequency：不超过 100 steps。

`--print-command` 是无副作用路径：打印薄 Bash 最终转发的 Python module invocation
以及 resolved configuration，然后在创建目录、加载数据或训练前退出。它保留现有脚本
用于审阅命令的合同。

启动前打印 resolved configuration，至少包括：

```text
policy type
model config module
mode
dataset root/revision
episode indices
observation/action horizon
batch size
steps
seed
paired initialization path
output directory
```

输出目录维持 new-only 语义，避免静默覆盖已有实验。

Python runtime 保留当前日志合同：训练期间写入与目标目录相邻的临时
`<output-dir>.train.log`，同时原样输出到终端；训练成功后移动为
`<output-dir>/train.log`，训练失败则保留临时日志并传播原始非零退出状态。日志持久化
不放回 Bash。

## 9. 错误处理

Python 入口必须在进入 LeRobot 前检查：

- mode 合法；
- steps、batch size 和 worker 数合法；
- split 文件存在且包含所需 split；
- dataset view 已准备；
- paired initialization 的 config、processor 和权重完整；
- initialization manifest 的 mode、seed、split fingerprint 与本次 resolved config
  一致；
- smoke/pilot 输出目录未串用；
- 输出目录不存在。

检查失败时给出单一、明确且可执行的错误信息。训练开始后由 LeRobot 负责训练异常和
checkpoint 行为，不在外层捕获并改写异常。

## 10. 验证

- 配置单元测试：
  - 两个 factory 返回正确的官方 config 类型；
  - 共享结构字段一致；
  - Diffusion 专用字段和 BCChunk `fixed_timestep` 正确。
- 训练配置测试：
  - smoke/pilot 数据、batch size、steps 和 paired initialization 解析正确；
  - 非法路径和已存在输出目录被拒绝。
- paired initialization 测试：
  - `initialization.py` 直接使用两个 Python factory；
  - manifest 记录并校验 mode、seed、split fingerprint 和 model config fingerprint；
  - model config fingerprint 在 factory 返回后、dataset feature 推导和 checkpoint 路径注入
    前计算；明确排除 `device`、`pretrained_path` 及 Hub 目标等 runtime-only 字段；
  - BCChunk 与 Diffusion 的共同 tensor hash 仍一致；
  - 从两个 paired 路径实际加载 policy 和 processor，加载后的 state-dict hash、policy
    类型及 resolved config 与 manifest/factory 一致。
- 启动器测试：
  - Bash 仅包含环境转发；
  - VSCode 与 Bash 指向同一 Python module；
  - 删除的 JSON 不再被引用。
  - `--print-command` 不创建输出目录、不加载 dataset、不启动训练。
- 回归测试：
  - subset sampler 仍只产生相对索引；
  - BCChunk 与 Diffusion 各完成一个真实 optimizer step；
  - checkpoint 仍包含 LeRobot 标准 `train_config.json`、policy config、processor 和权重。

## 11. 迁移

1. 创建 Python model config 和训练入口。
2. 将 `initialization.py` 切换为直接使用两个 factory，并扩充 manifest。
3. 把 sampler 与日志 tee 兼容代码移入 `training_runtime.py`。
4. 将两个 Bash 脚本替换为薄转发层。
5. 将 VSCode Debug 切换到模型专属 Python module。
6. 更新 PushT guide 中的入口说明。
7. 删除 `configs/experiments/pusht_diffusion/bc_chunk.json` 和
   `configs/experiments/pusht_diffusion/diffusion.json`。
8. 运行单元测试、静态检查和两次单步训练。
