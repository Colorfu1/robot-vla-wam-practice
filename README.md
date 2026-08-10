# Robot VLA WAM Practice

一个以 LeRobot 为统一工作流的机器人操作与 VLA 实验工作区，包含 ACT、Diffusion
Policy、SmolVLA、ManiSkill 以及后续的 Mini-WAM / Residual RL 实验。

根 README 只负责导航、环境准备和仓库约定。所有实验与项目的可复现说明统一从
[`reports/guides/`](reports/guides/README.md) 进入；不要在根 README 里维护第二份实验手册。

## 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 查看所有实验与项目 | [`reports/guides/`](reports/guides/README.md) |
| 查看某次实际运行及结果 | [`reports/experiment_log.md`](reports/experiment_log.md) |
| 查看指标定义与判断边界 | [`reports/metrics.md`](reports/metrics.md) |
| 查看尚未执行的工作 | [`plans/README.md`](plans/README.md) |
| 查看报告、进展和文档职责 | [`reports/README.md`](reports/README.md) |

## Guide：实验与项目总览

每个项目族都在 `reports/guides/<project>/` 下维护一个 README；README 负责说明实验
顺序和入口，具体实验文件负责给出完整命令、产物路径和判断标准。

| 项目 | Guide | 覆盖范围 |
| --- | --- | --- |
| ACT | [`guides/act/`](reports/guides/act/README.md) | A-Sanity、单步预测、Action Chunking、动作差分损失、Temporal Ensemble、observation 等权与 VAE |
| PushT Diffusion | [`guides/pusht_diffusion/`](reports/guides/pusht_diffusion/README.md) | 数据检查、BCChunk / Diffusion baseline、采样多样性、闭环 rollout、采样延迟、Prototype-ID 与 Route Field 条件控制 |
| SmolVLA LIBERO | [`guides/smolvla_libero/`](reports/guides/smolvla_libero/README.md) | B0 domain adaptation、C1/C2/C3 目标适配、LoRA、paraphrase 和 Prompt Swap 评测 |
| ManiSkill | [`guides/maniskill/`](reports/guides/maniskill/README.md) | PickCube-v1 replay、LeRobotDataset、Diffusion 闭环探针和论文参照的 4D 基线 |

### 一次实验的文档流转

```text
待执行目标       → plans/
可复现实验说明   → reports/guides/
实际运行记录     → reports/experiment_log.md
阶段性结论       → reports/daily/ 或 reports/
```

如果要运行实验，先读对应的 Guide README，再进入具体实验页；执行完成后把结果补到
实验运行记录。计划描述“要做什么”，Guide 描述“怎么做”，报告描述“跑出了什么”，三者不要混写。

## 环境准备

项目目标环境是 Python 3.10 的 `robot-practice` conda 环境：

```bash
conda env create -f environment.yml
conda activate robot-practice
pip install -r requirements.txt
pip install -e .
```

依赖发生变化后重新执行 `pip install -r requirements.txt`。环境设计保留
`PYTHONNOUSERSITE=1`，避免用户目录中的 Python 包泄漏进实验环境。

## 环境检查

```bash
conda activate robot-practice
./scripts/doctor.sh
./scripts/quickstart_info.sh
```

`doctor.sh` 检查 Python、ffmpeg、PyTorch、LeRobot 和 CLI；`quickstart_info.sh` 展示
LeRobot CLI 帮助以及当前安装位置。实验指南中的命令默认在这个环境中执行。

## 仓库结构

```text
reports/guides/       所有实验与项目的可复现 Guide（统一入口）
reports/              实验运行记录、指标说明、阶段性报告
plans/                尚未执行工作的计划与完成闸门
src/robot_practice/   可复用的本地 Python 代码
scripts/experiments/  训练、评测和数据准备启动脚本
policies/             按策略族整理的本地策略代码
eval/                 rollout、指标和扰动工具
configs/              实验配置
envs/                 ManiSkill / LIBERO 环境说明
datasets/             原始数据与 LeRobotDataset 产物
outputs/              训练、评测、视频和 checkpoint 等生成物
```

LeRobot 是主要的数据、训练、评测和策略工作流层；ManiSkill 用于仿真训练与快速闭环
评测；LIBERO 用于语言条件操作和 VLA benchmark。项目不是 LeRobot 的源码 checkout，
不要直接修改已安装的 LeRobot 包。

## 运行结果与产物约定

每条正式实验记录至少包含：

- task suite
- dataset size
- observation keys
- action space
- action horizon
- control frequency
- train loss
- open-loop action error
- closed-loop success rate
- latency
- failure cases

数据、checkpoint、模型权重、预测结果、视频、Rerun recording、日志和其他生成物放在
`datasets/` 或 `outputs/`，不提交到 Git。实际结果应区分 Smoke、Offline、Closed-loop
和 Formal，不能把计划、冒烟检查或未测量结果写成正式结论。

## 许可证与数据

项目代码和文档使用 [Apache License 2.0](LICENSE)。第三方软件、ALOHA 几何/标定资料和
数据集说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。示例数据集（例如
[`lerobot/aloha_mobile_cabinet`](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)
和 [`lerobot/libero`](https://huggingface.co/datasets/lerobot/libero)）不包含在仓库中，
使用时请遵守各自的数据集卡和许可证。
