# FastWAM Square原生双视角训练

## 实验目标

从原始Wan视频预训练与ActionDiT初始化重新训练，不加载Square step_020262。比较高清双视角与长预测窗口组合配置；不能将收益单独归因于腕相机或分辨率。

## 实验思路

- 保留原100条Square训练轨迹的动作及9D自身状态，按完整动作序列匹配HDF5示教。
- 每帧恢复示教XML和仿真状态，原生渲染agentview与robot0_eye_in_hand各224×224，左外部右腕部拼接448×224。
- 控制频率20Hz；动作预测32步，闭环只执行前4步；视频采样t,t+4,…,t+32，共9帧，覆盖1.6秒。
- 状态窗口与视频同索引，策略只条件于当前自身状态；视频和动作损失权重均为1。
- 数据采用只读mmap，避免8卡与spawn数据进程复制整份图像；丢弃不足32个未来步的窗口，不跨轨迹。
- 首先独立测8卡吞吐，再从原始初始化启动12小时计时训练，不沿用benchmark更新后的权重。

## 实验命令

在同步后的远端仓库根目录执行。全部依赖与预训练资产必须已本地部署，入口不联网下载。

```bash
export DUAL_DATA=datasets/derived/square_dual224_v1
bash scripts/experiments/rl/run_fastwam_square_dual.sh export
```

目录必须不存在；若上次导出中断，保留证据并换一个新目录，不将未完成数据用于训练。完整图像数组约8.2GB（十进制），还需预留checkpoint空间。

```bash
export DUAL_DATA=datasets/derived/square_dual224_v1
export GPUS=0,1,2,3,4,5,6,7
export MICROBATCH=2 ACCUMULATION=16
export OUT="outputs/fastwam_square_dual_bench_$(date +%Y%m%d_%H%M%S)"
bash scripts/experiments/rl/run_fastwam_square_dual.sh benchmark
```

上述batch为保守起点，启用梯度检查点；远端显存与吞吐尚未实测。只有benchmark正常完成且显存安全后再训练。

```bash
export DUAL_DATA=datasets/derived/square_dual224_v1
export GPUS=0,1,2,3,4,5,6,7
export MICROBATCH=2 ACCUMULATION=16 TRAIN_HOURS=12
export OUT="outputs/fastwam_square_dual_train12h_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs
nohup bash scripts/experiments/rl/run_fastwam_square_dual.sh train > "${OUT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUT"
tail -F "${OUT}.launch.log"
```

新AdamW与按训练计时的cosine计划：1%时间warmup到1e-4，再衰减到1e-6。每4小时保存权重，并在训练结束时保存最终权重；最终保存可能使总墙钟时间略超12小时，初始加载不计入训练预算。

训练完成后用新入口评测，不能用旧单视角4步预测入口。以下为单GPU串行80例命令；将CHECKPOINT指向实际新训练权重。

```bash
export CHECKPOINT="$(find "$OUT/checkpoints" -maxdepth 1 -name 'step_*.pt' -type f | sort | tail -n 1)"
test -n "$CHECKPOINT" && test -s "$CHECKPOINT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_dual_eval \
  --checkpoint "$CHECKPOINT" --episodes 80 --seed 10001 --max-steps 600 \
  --output "outputs/fastwam_dual_eval_$(date +%Y%m%d_%H%M%S)"
```

## 结果落点

### EGL 数据导出可行性（Smoke，远端待测）

生产导出默认仍为 OSMesa。独立目录测试同样的前两条示教：OSMesa 一次、
NVIDIA-only EGL 三次，每次独立进程。按物理 GPU 索引查询唯一 EGL ID，
不把 CUDA ordinal 当作 EGL ID。此测试不执行策略，也不修改生产数据。

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2
export PROBE_OUT="outputs/square_egl_export_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs
nohup conda run --no-capture-output -n dppo python -u \
  scripts/experiments/rl/test_square_egl_export.py \
  --output "$PROBE_OUT" --gpu 0 --rounds 3 --trajectories 2 \
  > "${PROBE_OUT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$PROBE_OUT"
```

查看 `metrics.json` 的退出码、实际 renderer、恢复/渲染/写入吞吐、每路相机的
跨后端 MAE；`mapping.log` 和各次 `.log` 保留失败证据。
要求各进程退出码为零、manifest 完成、状态和动作数组一致，再人工检查原生预览。
小样本通过不保证长跑或多进程 rollout 稳定；并行生产时吞吐受资源竞争影响。
已有导出入口也支持显式 `--render-backend egl`，但生产切换需另行决定。

本地改动后 OSMesa 导出 127 帧完成；远端 EGL 测试尚未执行。

后续远端 Smoke：OSMesa 250 帧完成，EGL 三次均 SIGSEGV；用户提供的栈均位于
新增 `glGetString` 信息查询处，该调用在首帧双视角渲染返回之后。
因此不能把这次失败直接归因于 `mjr_readPixels` 或设备错配。
可在上述测试命令追加 `--skip-renderer-query` 并使用新的输出目录，
只跳过两种后端的 renderer 信息查询，保留全部渲染、状态核验与图像对比。
导出模块也接受同名开关；manifest 标记 `renderer_query=skipped`。
此对照尚待远端执行，即使通过，实际 GL renderer 身份仍未由查询验证，
也不代表已确定查询崩溃的底层原因。

- 数据：`manifest.json`（相机和时序协议、源及数组哈希、逐轨迹恢复检查）、mmap数组、原生双视角预览PNG。
- 训练：`metrics.json`、`run.log`、各rank日志、`checkpoints/step_*.pt`及独立恢复状态。
- 评测：`metrics.json`、按seed实际GIF和32步预测动作记录；每次只执行前4步，记录400/600步结果。

## 判断标准与验证边界

- 导出要求完整动作匹配唯一、状态误差不超过1e-8、自身状态误差不超过1e-3；相机缺失即失败。
- 训练要求100条轨迹、完整数据manifest及数组哈希通过；恢复训练不能混用新旧协议。
- Smoke已完成：本地127帧双视角导出；同seed随机重置一致，模拟器返回3×224×448图像并执行4步。相关18项测试通过。
- 尚未验证：8卡显存、吞吐、完整模型前反向及正式闭环效果。此次Smoke不证明训练收敛或成功率改善。
- 评测是种子随机重置，不是恢复示教状态；跨模型正式比较需另核验实际初始状态和成功统计口径一致，不只依据相同seed。
