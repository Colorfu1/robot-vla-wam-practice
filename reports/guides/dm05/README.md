# DM0.5 本地 LIBERO 推理与闭环验证

- [Square 零样本离线推理与单场景闭环](02_square_zero_shot.md)

- [模型结构与完整张量维度](01_dm05_model_architecture.md)：训练/推理图、逐层 K/V、
  动作 mask、参数拆分，以及真实权重形状探针。

## 实验目标

验证官方 DM05-libero 在单张 RTX 3090 24GB 上能否加载、返回有限动作并驱动
LIBERO 环境。合成输入测试与单任务闭环均为 `Smoke`，不构成正式 benchmark。

## 实验思路

- 独立 Python 3.10 环境运行 OpenDM；现有 robot-practice 负责 LIBERO 仿真。
- 用本地 HTTP 连接两套环境，避免 Transformers 和仿真依赖互相覆盖。
- 默认 PyTorch 后端、BF16、视觉 SDPA、关闭 Liger；不安装 FlashAttention/TensorRT。
- 本地加载子类在模型构造前覆盖 checkpoint 中的注意力配置，避免先触发 FA2 缺包；
  上游运行参数在构造后应用，单独指定视觉 SDPA 不足以解决初始化问题。
- 合成输入跑 5 次，前 2 次作为预热，检查动作形状、有限值、显存与延迟。
- 闭环使用双相机、8 维状态、7 维动作，连续夹爪，每 10 步重新规划。
- 图像双轴翻转与状态顺序跟随官方 dexbotic-benchmark。

## 固定版本与依赖

- OpenDM：`fbab441b63c789e5c37f7293e61fea9ed356c6c5`。
- checkpoint：`Dexmal/DM05-libero`，revision `25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088`。
- 评测协议参考：dexbotic-benchmark `789b87f50d9fadc7663d2e8bac057941221aab81`。
- 本地入口从 OpenDM 源码导入；依赖文件仅覆盖推理，不代表上游完整训练环境。
- 权重由用户下载；下面的执行入口强制离线，不自动下载任何模型。

## 完整命令

所有命令从本仓库根目录执行。环境和源码已存在时跳过创建步骤。

```bash
conda create -y -p ./outputs/dm05/env python=3.10 pip
git clone https://github.com/dexmal/opendm.git outputs/dm05/opendm
git -C outputs/dm05/opendm checkout fbab441b63c789e5c37f7293e61fea9ed356c6c5
PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python -m pip install \
  -r envs/dm05/requirements-inference.txt
```

权重目录须包含完整 safetensors、配置、tokenizer、processor 和 norm_stats.json。

```bash
mkdir -p outputs/dm05/smoke
set -o pipefail
PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python \
  src/robot_practice/experiments/dm05_local.py \
  --opendm-dir outputs/dm05/opendm \
  --checkpoint outputs/dm05/checkpoints/DM05-libero \
  --output outputs/dm05/smoke --requests 5 --seed 7 \
  2>&1 | tee outputs/dm05/smoke/run.log
```

闭环需要两个终端。终端一启动服务（仅监听回环地址）：

```bash
mkdir -p outputs/dm05/server
set -o pipefail
PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python \
  src/robot_practice/experiments/dm05_local.py \
  --opendm-dir outputs/dm05/opendm \
  --checkpoint outputs/dm05/checkpoints/DM05-libero \
  --output outputs/dm05/server --serve --port 7891 --seed 7 \
  2>&1 | tee outputs/dm05/server/run.log
```

待服务完成初始化后，终端二运行单任务闭环：

```bash
mkdir -p outputs/dm05/libero_smoke
set -o pipefail
PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  conda run --no-capture-output -n robot-practice python \
  src/robot_practice/experiments/dm05_libero_smoke.py \
  --url http://127.0.0.1:7891/v1/infer \
  --libero-assets-dir datasets/raw/libero_assets \
  --task-id 0 --episodes 1 --seed 7 --output outputs/dm05/libero_smoke \
  2>&1 | tee outputs/dm05/libero_smoke/run.log
```

服务用完后按 Ctrl-C 停止，释放模型显存。

## 结果落点

- `outputs/dm05/smoke/metrics.json`：合成输入推理形状、预热与稳定阶段延迟、显存。
- `outputs/dm05/libero_smoke/metrics.json`：逐 episode 成败、步数、HTTP 延迟。
- 各输出目录的 `progress.log`：周期性 STATUS/DONE；`run.log`：完整运行输出。
- 这些路径均为忽略的运行产物，不纳入发布。

## 判断标准

- 推理通过：5/5 请求完成，动作均为有限值且形状为 `(10, 7)`，无 OOM。
- 闭环连通：完成官方 10 步静置和最多 220 步 rollout，动作由真实模型返回。
- 任务成功只按环境完成信号记录；未成功也必须写结果，不能隐藏失败。
- 20Hz 是仿真控制频率；HTTP 耗时和动作块复用决定实际吞吐，不等同于实时 20Hz。
- 单 episode 成功不代表 LIBERO suite 成功率；与官方数字对比需后续完整协议评测。
- 若出现 OOM、非有限动作或 HTTP 失败，停止，不继续统计为正常 episode。

## 来源

- [OpenDM 源码](https://github.com/dexmal/opendm/tree/fbab441b63c789e5c37f7293e61fea9ed356c6c5)
- [官方推理指南](https://github.com/dexmal/opendm/blob/fbab441b63c789e5c37f7293e61fea9ed356c6c5/docs/en/dm05_inference.md)
- [官方 LIBERO 评测协议](https://github.com/dexmal/dexbotic-benchmark/blob/789b87f50d9fadc7663d2e8bac057941221aab81/evaluation/evaluator/libero_evaluator.py)
