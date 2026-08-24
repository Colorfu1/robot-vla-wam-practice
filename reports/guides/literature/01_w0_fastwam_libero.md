# W0 Fast-WAM LIBERO 复现与机制锚点

这是 W0 的操作入口。主文档只保留当前结论、运行顺序、命令和失败处理；协议推导、历史结果、
排查证据和统计限制见 [details/](details/README.md)。

## 当前重点

- UP-REL：1,929 / 2,000，96.45%，released checkpoint baseline。
- LR-REL：1,885 / 2,000，94.25%，仅作 integration cross-check，不声明 parity。
- FW-V：1,921 / 2,000，96.05%，10 epoch / 21,700 步，已完成。
- FW-FZ：主对照，尚未训练；需要 lambda_video=0 并冻结 video expert。
- causal evaluation、deployment evidence 和 W0 汇总 gate：待完成。

详细结果与限制：

- [协议与数据说明](details/01_w0_fastwam_libero_full.md#固定协议)
- [UP-REL / LR-REL 评测排查](details/01_w0_fastwam_libero_full.md#node-4up-rel-official-released-checkpoint)
- [训练、显存和 causal 诊断](details/01_w0_fastwam_libero_full.md#node-6fw-vfw-nv-5-step-distributed-smoke)

## 运行前提

- 需要 fastwam-upstream（Python 3.10、MuJoCo 3.3.2）和 fastwam-lerobot（Python 3.12、
  LeRobot 0.6.0）。
- 代码使用 pinned checkout：宿主机是 third_party/w0_fastwam/，Docker 是
  /opt/robot-practice/w0-vendor/。
- 权重、数据和 ActionDiT 放在 pretrained/w0_fastwam/。
- wrapper 只校验和启动，不安装环境、不下载 artifact、不修改 vendor checkout。
- causal training 需要同机 8 张、每张至少 80 GiB 的 GPU。

### 宿主机一次性环境准备

Docker 镜像已包含环境时跳过。

~~~bash
conda create --yes --name fastwam-upstream --clone robot-practice
conda run --no-capture-output -n fastwam-upstream python -m pip install \
  --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e third_party/w0_fastwam/FastWAM \
  -e third_party/w0_fastwam/LIBERO
conda run --no-capture-output -n fastwam-upstream python -m pip install \
  --no-cache-dir mujoco==3.3.2 easydict==1.9 opencv-python==4.6.0.66 \
  robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 future==0.18.2 \
  matplotlib==3.5.3 cloudpickle==2.1.0 gym==0.25.2

conda create --yes --name fastwam-lerobot \
  --override-channels --channel=conda-forge python=3.12 pip
conda install --yes --name fastwam-lerobot \
  --override-channels --channel=conda-forge cmake make cxx-compiler
conda run --no-capture-output -n fastwam-lerobot python -m pip install \
  --no-cache-dir --no-deps robomimic==0.2.0 docker/training/egl_probe_stub
conda run --no-capture-output -n fastwam-lerobot python -m pip install \
  --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e 'third_party/w0_fastwam/lerobot[fastwam,libero]' \
  h5py psutil tensorboardX imageio imageio-ffmpeg
~~~

验证：

~~~bash
conda run --no-capture-output -n fastwam-upstream \
  python -c "import fastwam,libero,mujoco,torch; print(torch.__version__, mujoco.__version__)"
conda run --no-capture-output -n fastwam-lerobot \
  python -c "import lerobot,torch; print(lerobot.__version__, torch.__version__)"
~~~

### Docker

~~~bash
W0_IMAGE='<registry>/<namespace>/practice:latest'
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  "$W0_IMAGE"
~~~

### 确认实际使用的 vendor

宿主机：

~~~bash
for d in FastWAM LIBERO lerobot; do git -C third_party/w0_fastwam/$d status --porcelain; done
~~~

Docker：

~~~bash
docker run --rm --entrypoint bash <image> -lc \
  'for d in FastWAM LIBERO lerobot; do git -C /opt/robot-practice/w0-vendor/$d status --porcelain; done'
~~~

FW-FZ 必须使用已应用冻结补丁的镜像（w0-fastwam-20260821-fwfz 或更新版本）。预期
FastWAM 只显示 M src/fastwam/trainer.py；原因和补丁内容见
[vendor patch 说明](../../../docker/training/vendor-patches/README.md)。

## 固定协议

| 项目 | 值 |
| --- | --- |
| 数据 | yuanty/LIBERO-fastwam，4 suites，训练 1,712 episodes / 277,713 frames |
| 观测 | front + wrist RGB，224×224，拼接 224×448；8D proprio |
| 动作 | 7D EEF pose + gripper；chunk 32；execute/replan 10 |
| 时间 | raw window 33；action_video_freq_ratio=4；9 video frames |
| 训练 | 10 epochs；8 GPU；per-GPU batch 16；global batch 128；bf16 |
| 评测 | 40 tasks × 50 trials = 2,000 episodes |

评测初始状态与训练 demonstration 有重叠，因此 success rate 不是 held-out 泛化结果。详见
[协议与数据说明](details/01_w0_fastwam_libero_full.md#评测初始状态与训练-demonstration-的重叠)。

## Node 0：配置与单测

失败就停止，不进入下载或训练。

~~~bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_w0_fastwam_protocol.py tests/test_w0_fastwam_analysis.py

bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-config \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json
~~~

## Node 1：资源与 seed preflight

只有 resource_gate.upstream_eval=true 才能跑 released eval；只有
resource_gate.causal_training=true 才能启动 causal training。

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=preflight \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/preflight
~~~

~~~bash
python -m json.tool outputs/literature/w0_fastwam/preflight/preflight.json
~~~

## Node 2：pinned checkout 与环境

Docker 内已有 /opt/robot-practice/w0-vendor 时跳过 checkout 准备。

~~~bash
(
set -euo pipefail

if [[ -z "${W0_VENDOR_ROOT:-}" ]]; then
  export W0_VENDOR_ROOT=third_party/w0_fastwam
fi

if [[ "${W0_VENDOR_ROOT}" == "third_party/w0_fastwam" ]]; then
  mkdir -p third_party/w0_fastwam
  pin_checkout() {
    local url="$1"
    local checkout="$2"
    local commit="$3"

    if [[ -e "${checkout}" && ! -d "${checkout}/.git" ]]; then
      echo "error: ${checkout} exists but is not a Git checkout" >&2
      return 1
    fi
    if [[ ! -e "${checkout}" ]]; then
      git clone "${url}" "${checkout}"
    fi
    if [[ -n "$(git -C "${checkout}" status --porcelain)" ]]; then
      echo "error: ${checkout} has local changes; refusing to change checkout" >&2
      return 1
    fi
    if ! git -C "${checkout}" rev-parse --verify "${commit}^{commit}" >/dev/null 2>&1; then
      git -C "${checkout}" fetch --depth=1 origin "${commit}"
    fi
    git -C "${checkout}" checkout --detach "${commit}"
  }

  pin_checkout \
    https://github.com/yuantianyuan01/FastWAM.git \
    third_party/w0_fastwam/FastWAM \
    45d8e1458921d83f8ad6cf9ce993d371208dabd0
  pin_checkout \
    https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    third_party/w0_fastwam/LIBERO \
    8f1084e3132a39270c3a13ebe37270a43ece2a01
  pin_checkout \
    https://github.com/huggingface/lerobot.git \
    third_party/w0_fastwam/lerobot \
    30da8e687a6dfc617fcd94afc367ac7071c376ce
else
  echo "Using pre-pinned runtime checkout: ${W0_VENDOR_ROOT}"
fi
)
~~~

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-environments \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output-root=outputs/literature/w0_fastwam/environment
~~~

## Node 3：artifact 准备与校验

不要把 token 写入配置、日志或命令历史。以下是唯一需要联网下载的步骤。

~~~bash
conda run --no-capture-output -n fastwam-lerobot hf auth login
conda run --no-capture-output -n fastwam-upstream modelscope login
~~~

~~~bash
(
set -euo pipefail
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
w0_root=pretrained/w0_fastwam
mkdir -p "$w0_root"/{upstream,lerobot,dataset}
mkdir -p "$w0_root/libero_assets"
mkdir -p "$w0_root"/models/{Wan-AI/Wan2.2-TI2V-5B,Wan-AI/Wan2.1-T2V-1.3B,DiffSynth-Studio/Wan-Series-Converted-Safetensors}

conda run --no-capture-output -n fastwam-lerobot hf download yuanty/fastwam \
  libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  --revision=139eebb6d90cdd9bdbbe465f72c6edc9ad5a518a \
  --local-dir="$w0_root/upstream"
conda run --no-capture-output -n fastwam-lerobot hf download ZibinDong/fastwam_libero_uncond_2cam224 \
  --revision=a784645f9ce367ba6953dd20a7a7f0310c85747c \
  --local-dir="$w0_root/lerobot"
conda run --no-capture-output -n fastwam-lerobot hf download yuanty/LIBERO-fastwam --repo-type=dataset --revision=main \
  --local-dir="$w0_root/dataset"
conda run --no-capture-output -n fastwam-lerobot hf download lerobot/libero-assets \
  --repo-type=dataset --revision=main \
  --local-dir="$w0_root/libero_assets"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model Wan-AI/Wan2.2-TI2V-5B \
  --local_dir="$w0_root/models/Wan-AI/Wan2.2-TI2V-5B"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir="$w0_root/models/Wan-AI/Wan2.1-T2V-1.3B"
conda run --no-capture-output -n fastwam-upstream modelscope download \
  --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  models_t5_umt5-xxl-enc-bf16.safetensors Wan2.2_VAE.safetensors \
  --local_dir="$w0_root/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

mkdir -p "$w0_root/dataset/data/libero_mujoco3.3.2"
find "$w0_root/dataset" -maxdepth 1 -type f -name '*.tar.gz' -print0 \
  | while IFS= read -r -d '' archive; do
      tar -xzf "$archive" -C "$w0_root/dataset/data/libero_mujoco3.3.2"
    done
)
~~~

~~~bash
export W0_VENDOR_ROOT="${W0_VENDOR_ROOT:-third_party/w0_fastwam}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=1 \
DIFFSYNTH_MODEL_BASE_PATH=pretrained/w0_fastwam/models \
conda run --no-capture-output -n fastwam-upstream python \
  "${W0_VENDOR_ROOT}/FastWAM/scripts/preprocess_action_dit_backbone.py" \
  --model-config "${W0_VENDOR_ROOT}/FastWAM/configs/model/fastwam.yaml" \
  --output pretrained/w0_fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
~~~

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=validate-artifacts \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --cache-root=pretrained/w0_fastwam \
  --output-root=outputs/literature/w0_fastwam/artifacts
~~~

数据校验按 info.json 的 episode/frame 数和文件 hash 执行；内容不匹配就停止。

可选的 train/eval 初始状态覆盖审计：

~~~bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python scripts/experiments/literature/audit_w0_train_eval_overlap.py \
    --sort-by-coverage \
    --output outputs/literature/w0_fastwam/train_eval_overlap.json
~~~

## Node 4：UP-REL released checkpoint

### 失败处理

EGL index 不是 CUDA index；多进程渲染时计算卡与 EGL 卡必须分离。遇到 native SIGABRT 或
mjr_readPixels，先重新扫描，不要直接增加 worker。

~~~bash
conda run --no-capture-output -n robot-practice \
  python scripts/experiments/literature/sweep_w0_egl_pairs.py \
  --egl 0,1,2,3,4,5,6,7 --steps 120 \
  --output outputs/literature/w0_fastwam/egl_pair_sweep.json

conda run --no-capture-output -n fastwam-upstream \
  python scripts/experiments/literature/probe_egl_devices.py
~~~

### Smoke

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-upstream-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --checkpoint=pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt \
  --trials-per-task=2 \
  --num-gpus=1 \
  --output-root=outputs/literature/w0_fastwam/upstream_smoke \
  --smoke
~~~

### Formal

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-upstream-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --checkpoint=pretrained/w0_fastwam/upstream/libero_uncond_2cam224.pt \
  --trials-per-task=50 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/upstream_released
~~~

结果检查：40 tasks、每 task 50 trials、无缺失/重复；结果写入
outputs/literature/w0_fastwam/upstream_released/summary.json。

## Node 5：LR-REL LeRobot integration

LR-REL 单独记录，不替换 UP-REL。默认使用 visual-normalization=identity，否则两个 pin
叠加会导致输入归一化错误、成功率可能为 0。详细差异见
[LR-REL 详情](details/01_w0_fastwam_libero_full.md#node-5lr-rel-lerobot-integration)。

### Smoke

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-lerobot-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --vendor-root=/opt/robot-practice/w0-vendor \
  --policy-path=pretrained/w0_fastwam/lerobot \
  --trials-per-task=2 \
  --output-root=outputs/literature/w0_fastwam/lerobot_smoke \
  --smoke --api-smoke
~~~

### Formal

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-lerobot-released \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --vendor-root=/opt/robot-practice/w0-vendor \
  --policy-path=pretrained/w0_fastwam/lerobot \
  --trials-per-task=50 --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/lerobot_released
~~~

归一化审计：

~~~bash
conda run --no-capture-output -n fastwam-lerobot \
  python scripts/experiments/literature/audit_w0_lr_parity.py \
  --policy-path pretrained/w0_fastwam/lerobot \
  --visual-normalization identity
~~~

## Node 6：causal 5-step smoke

必须用 8 GPU；单卡会在 ZeRO-1 优化器状态分配阶段 OOM。smoke 必须同时检查：

- with_video：loss = video + action；
- no_video：video loss 恒为 0，loss = action；
- 两组 init/data/noise/update hash 和 global batch 一致。

FW-V / FW-NV：

~~~bash
for variant in with_video no_video; do
  bash scripts/experiments/literature/run_w0_fastwam.sh \
    --phase=train-causal-pair \
    --config=configs/experiments/literature/w0_fastwam_reproduction.json \
    --vendor-root=/opt/robot-practice/w0-vendor \
    --variant="${variant}" \
    --steps=5 \
    --num-gpus=8 \
    --output-root="outputs/literature/w0_fastwam/train_smoke/${variant}" \
    --overwrite
done
~~~

FW-FZ：

~~~bash
W0_FREEZE_VIDEO_EXPERT=1 \
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=train-causal-pair \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --vendor-root=/opt/robot-practice/w0-vendor \
  --variant=no_video \
  --steps=5 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/train_smoke_frozen/no_video \
  --overwrite
~~~

如果出现 /usr/local/cuda/bin/nvcc 不存在，使用带 cuda-nvcc=12.8 的镜像；不要用
DS_SKIP_CUDA_CHECK=1 代替。

## Node 7：causal training

正式预算使用 10 epoch，约 21,700 steps。FW-FZ 不是从 FW-V resume，而是相同初始化下
重新训练 lambda_video=0 + frozen video expert。

FW-V：

~~~bash
bash scripts/experiments/literature/launch_w0_causal.sh with_video
~~~

FW-FZ：

~~~bash
W0_FREEZE_VIDEO_EXPERT=1 \
bash scripts/experiments/literature/launch_w0_causal.sh no_video
~~~

可选 FW-NV：

~~~bash
bash scripts/experiments/literature/launch_w0_causal.sh no_video
~~~

输出目录：

- FW-V：outputs/literature/w0_fastwam/causal_10ep/with_video/
- FW-FZ：outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/
- FW-NV：outputs/literature/w0_fastwam/causal_10ep/no_video/

FW-FZ 启动前确认：

~~~bash
git -C /opt/robot-practice/w0-vendor/FastWAM status --porcelain
~~~

应显示 M src/fastwam/trainer.py。训练监控：

~~~bash
L=outputs/literature/w0_fastwam/causal_10ep/with_video/train.log
nohup bash scripts/experiments/literature/watch_w0_stall.sh "$L" \
  > /tmp/w0_watch.log 2>&1 &
~~~

训练日志、manifest、checkpoint 和 pair_contract.json 都必须保留。详细 OOM 根因、显存
预算和非确定性说明见 [训练诊断](details/01_w0_fastwam_libero_full.md#node-7causal-trainingfw-vfw-fzfw-nv)。

## Node 8：causal evaluation

先 Smoke，再 Formal。--no-video 是对照 checkpoint 槽位名；主对照应传 FW-FZ checkpoint。

Smoke：默认 8 GPU，跑 `libero_spatial` 的 10 个 task、每 task 2 trials；两组 checkpoint 都执行真实
推理和 MuJoCo/EGL 渲染。它不是正式结果，但能尽早暴露多进程推理、渲染初始化和 worker 崩溃。
不传 50-trial seed manifest。

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-causal-pair \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --with-video=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --no-video=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --pair-contract=outputs/literature/w0_fastwam/causal_10ep/pair_contract.json \
  --trials-per-task=2 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/causal_smoke \
  --smoke
~~~

Formal：

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=eval-causal-pair \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --with-video=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --no-video=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --pair-contract=outputs/literature/w0_fastwam/causal_10ep/pair_contract.json \
  --seed-manifest=outputs/literature/w0_fastwam/preflight/libero_eval_seeds.json \
  --trials-per-task=50 \
  --num-gpus=8 \
  --output-root=outputs/literature/w0_fastwam/causal_eval
~~~

判断只看：完整性、equal budget、invalid rollout、两组 SR、paired gap 和 95% CI。没有固定
pass/fail 阈值；CI 跨 0 且很宽时，结论是样本不足。

## Node 9：deployment evidence 与总 gate

先把 one-task runtime probe 的实测结果写入：

~~~json
{
  "status": "pass",
  "output_shape": ["B", 32, 7],
  "future_noisy_video_token_allocations": 0,
  "future_video_denoise_steps": 0,
  "future_vae_decode_calls": 0,
  "n_action_steps": 10,
  "gripper_contract": "toggle_last_dimension"
}
~~~

导入 evidence：

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=audit-deployment \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --deployment-evidence=outputs/literature/w0_fastwam/causal_eval/deployment_evidence.json \
  --output=outputs/literature/w0_fastwam/causal_eval/deployment.json
~~~

生成 gate：

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=gate-w0 \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --released=outputs/literature/w0_fastwam/upstream_released/summary.json \
  --causal=outputs/literature/w0_fastwam/causal_eval/comparison.json \
  --deployment=outputs/literature/w0_fastwam/causal_eval/deployment.json \
  --lerobot=outputs/literature/w0_fastwam/lerobot_released/summary.json \
  --bootstrap-repeats=10000 \
  --bootstrap-seed=20260817 \
  --output=outputs/literature/w0_fastwam/w0_gate.json
~~~

只检查命令：

~~~bash
bash scripts/experiments/literature/run_w0_fastwam.sh \
  --phase=gate-w0 \
  --config=configs/experiments/literature/w0_fastwam_reproduction.json \
  --output=outputs/literature/w0_fastwam/w0_gate_smoke.json \
  --dry-run
~~~

## 结果落点

- preflight：outputs/literature/w0_fastwam/preflight/
- artifact manifest：outputs/literature/w0_fastwam/artifacts/
- released eval：upstream_released/summary.json、lerobot_released/summary.json
- causal training：causal_10ep/with_video/、causal_10ep_frozen/no_video/
- causal eval：causal_eval/comparison.json、deployment.json
- final gate：w0_gate.json

## 停止条件

- revision、环境版本、数据 fingerprint 或 checkpoint hash 不一致；
- resource gate 不通过；
- smoke 未证明 loss 路由或冻结状态；
- causal pair 不是预注册差异；
- deployment evidence 缺失；
- 正式结果存在缺失、重复、invalid rollout 或无法解释的运行状态。
