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

### 多卡 EGL 闭环测试（Smoke，待运行）

这次使用最终双相机权重，仍为32步动作预测、只执行前4步、20Hz控制，
每个策略步从仿真器读取原生agentview和腕相机各224×224图像。
8张物理GPU各运行一个策略进程和一个DPPO仿真进程；启动器先按PCI身份查询
物理GPU到CUDA ordinal及EGL ID的映射，不能把这些编号直接视为相同。
仿真子进程使用NVIDIA-only EGL且取消CUDA_VISIBLE_DEVICES；策略进程仍只看自己的GPU。
用渲染context的Python类型验证EGL，不调用先前崩溃位置的`glGetString`。

在远端仓库根目录、8卡无其他高显存作业时先跑每卡一例，共8例的稳定性测试：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CHECKPOINT=outputs/fastwam_dual_b16_acc2_12h_20260922_164223/checkpoints/step_006954.pt
export OUT="outputs/fastwam_dual_egl8_smoke_$(date +%Y%m%d_%H%M%S)"
test -s "$CHECKPOINT"
mkdir -p outputs
nohup conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_dual_eval_multigpu \
  --checkpoint "$CHECKPOINT" --output "$OUT" \
  --gpus 0,1,2,3,4,5,6,7 --episodes 8 --seed 10001 --max-steps 600 \
  --render-backend egl --cpu-threads 2 \
  --hdf5 datasets/raw/robomimic/square/mh/demo_v141.hdf5 \
  --normalization datasets/raw/dppo/robomimic/square-img/normalization.npz \
  > "${OUT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUT"
tail -F "${OUT}.launch.log"
```

`Ctrl-C`只停止tail，不停止nohup任务。先验收`metrics.json`中`status=complete`、
`evaluated=8`、每个shard退出码0及`rendering.context_class=EGLGLContext`，
并检查`mapping.log`、各`gpu_*.launch.log`和`gpu_*/env_worker.log`无EGL/readPixels崩溃。
若这8例通过，再用新输出目录、把`--episodes 8`改为`--episodes 80`运行同一命令；
80例使用连续seed 10001–10080，汇总分别给出400/600步成功率。
这两个命令的输出不能覆盖已有OSMesa评测或训练输出。

### 官方 DPPO reset/seed 协议对照

上述10001–10080是我们自定的配对初始状态集合，不是官方DPPO默认80局。
官方Square evaluator使用`seed=42`、20个异步环境（初始seed42–61）、每个环境
连续执行400个策略调用；每次执行4个控制动作，100次调用后由官方`MultiStep`
包装器自动reset，合计20×4=80个完整400控制步episode。
下面的FastWAM入口复用官方向量环境和成功判据，将8卡各分配连续的2–3个环境。
双视角输入是FastWAM所需的改动：官方图像包装器先输出6×224×224，策略进程再
拼成训练时的3×224×448（agentview左、腕相机右）。因此这对齐了环境的seed、
reset、动作执行和统计协议，但不是把FastWAM变成官方DP模型；两者网络和动作采样仍不同。

先在远端仓库根目录执行三环境、100次调用的非正式smoke，验证最满的单卡分片：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CHECKPOINT=outputs/fastwam_dual_b16_acc2_12h_20260922_164223/checkpoints/step_006954.pt
export OUT="outputs/fastwam_dual_official_smoke_$(date +%Y%m%d_%H%M%S)"
test -s "$CHECKPOINT"
nohup conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_dual_official_multigpu \
  --checkpoint "$CHECKPOINT" --output "$OUT" --gpus 0 --smoke \
  > "${OUT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUT"
tail -F "${OUT}.launch.log"
```

只有`status=complete`、`evaluated=3`、退出码0，且`gpu_0/first_observation.png`
左右相机顺序正确时，才启动完整80局：

```bash
export OUT="outputs/fastwam_dual_official80_$(date +%Y%m%d_%H%M%S)"
nohup conda run --no-capture-output -n fastwam-upstream python -u \
  -m robot_practice.experiments.rl.fastwam_dual_official_multigpu \
  --checkpoint "$CHECKPOINT" --output "$OUT" --gpus 0,1,2,3,4,5,6,7 \
  > "${OUT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUT"
tail -F "${OUT}.launch.log"
```

顶层`metrics.json`必须有`status=complete`、`evaluated=80`、20个环境各4局、
每个shard退出码0；子目录`gpu_*/env_worker.log`不可有native/EGL错误。
这两个运行使用全新输出目录，不覆盖旧的10001–10080评测。

## 结果落点

### EGL 数据导出可行性（Smoke，已测）

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

本地OSMesa导出127帧完成。第一次远端Smoke中OSMesa 250帧完成，EGL三次均
SIGSEGV；栈在新增`glGetString`查询处，不能直接归因于`mjr_readPixels`。
跳过该查询后的独立远端Smoke：OSMesa 250帧6.26帧/秒；EGL三轮均完成、
57.8–59.0帧/秒；EGL对OSMesa的agentview平均MAE约0.89/255、腕相机约0.54/255。
这些是静态状态渲染结果，不证明闭环rollout稳定，也不确定底层查询崩溃原因。
官方示教图像转换另有两条轨迹的EGL退出清理Smoke，见
[EGL退出清理记录](../../daily/2026-09-23-Square图像转换EGL退出清理修复.md)。

- 数据：`manifest.json`（相机和时序协议、源及数组哈希、逐轨迹恢复检查）、mmap数组、原生双视角预览PNG。
- 训练：`metrics.json`、`run.log`、各rank日志、`checkpoints/step_*.pt`及独立恢复状态。
- 评测：`metrics.json`、按seed实际GIF和32步预测动作记录；每次只执行前4步，记录400/600步结果。

## 判断标准与验证边界

- 导出要求完整动作匹配唯一、状态误差不超过1e-8、自身状态误差不超过1e-3；相机缺失即失败。
- 训练要求100条轨迹、完整数据manifest及数组哈希通过；恢复训练不能混用新旧协议。
- Smoke已完成：本地127帧双视角导出；同seed随机重置一致，模拟器返回3×224×448图像并执行4步。相关18项测试通过。
- 多卡EGL闭环尚未远端验证；静态EGL导出通过不代表高频闭环读取和退出清理通过。
- 评测是种子随机重置，不是恢复示教状态；跨模型正式比较需另核验实际初始状态和成功统计口径一致，不只依据相同seed。
