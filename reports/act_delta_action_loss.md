# ACT 实验二：动作差分损失

## 1. 实验要回答什么

在 A1 的 `chunk_size=20, n_action_steps=5` 基础上增加动作差分损失：

\[
L=L_{ACT}+\beta L_{\Delta a},\qquad \beta=0.1
\]

实验问题：

> 要求预测动作的变化与 demonstration 一致，是否能在不明显降低成功率的前提下减少闭环动作
> 抖动？

A1 与 A2 使用相同数据、网络结构、step-0 权重、训练步数和评测场景。唯一训练变量是
`delta_action_loss`。

> 当前状态：本文件是实施指南。`act_delta` policy、checkpoint 转换器和 A2 启动脚本需要按
> 本指南实现后才能使用。

## 2. 固定配置

沿用 [实验一](act_action_chunking.md) 的 20 条 demonstrations：

```text
823, 826, 834, 840, 841,
845, 847, 869, 875, 878,
910, 918, 920, 929, 944,
948, 949, 953, 966, 972
```

| 项目 | A1 | A2 |
|---|---:|---:|
| `chunk_size` | 20 | 20 |
| `n_action_steps` | 5 | 5 |
| `use_vae` | false | false |
| `delta_loss_weight` | 0 | 0.1 |
| Batch size | 8 | 8 |
| Updates | 20,000 | 20,000 |
| Seed | 1000 | 1000 |

## 3. 需要实现的文件

```text
src/robot_practice/policies/__init__.py
src/robot_practice/policies/act_delta/__init__.py
src/robot_practice/policies/act_delta/configuration_act_delta.py
src/robot_practice/policies/act_delta/modeling_act_delta.py
src/robot_practice/experiments/act_delta_checkpoint.py
scripts/experiments/act/a2_delta_loss.sh
tests/test_act_delta_policy.py
tests/test_act_delta_checkpoint.py
```

还需要让现有 evaluator 接受本地 ACT 子类：

```text
src/robot_practice/evaluation/common.py
tests/test_evaluation_common.py
```

不要修改 Conda 环境中的 `site-packages/lerobot`。

## 4. 实现 `ACTDeltaConfig`

在 `configuration_act_delta.py` 中继承 `ACTConfig`：

```python
from dataclasses import dataclass
import math

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_delta")
@dataclass
class ACTDeltaConfig(ACTConfig):
    delta_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not math.isfinite(self.delta_loss_weight)
            or self.delta_loss_weight < 0
        ):
            raise ValueError("delta_loss_weight must be finite and non-negative")
```

默认值使用 `0.0`，这样可以直接验证本地 policy 在关闭附加损失时与原始 ACT 一致。正式 A2
启动脚本固定传入 `0.1`。

在包的 `__init__.py` 中导入配置类，使 LeRobot plugin discovery 能完成注册。

## 5. 实现 `ACTDeltaPolicy`

类名和模块名必须遵守 LeRobot 0.4.4 的第三方 policy 约定：

```text
configuration_act_delta.ACTDeltaConfig
modeling_act_delta.ACTDeltaPolicy
```

基本定义：

```python
class ACTDeltaPolicy(ACTPolicy):
    config_class = ACTDeltaConfig
    name = "act_delta"
```

### 5.1 为什么不能直接调用 `super().forward()`

当前 `ACTPolicy.forward()` 只返回 loss，不返回 `actions_hat`。如果先调用
`super().forward()`，再调用一次 `self.model(batch)` 计算 delta loss，会造成每个训练 batch
做两次网络前向，并让 dropout 使用两套随机结果。

正确做法是在本地子类中保留一次 `self.model(batch)`，复制 LeRobot 0.4.4 中很短的 loss
wrapper，原样计算原始 L1/KL，然后追加 delta loss。不要复制 ACT 网络结构。

在方法注释中记录：

```text
Compatible with LeRobot 0.4.4
Upstream reference: lerobot/policies/act/modeling_act.py::ACTPolicy.forward
```

### 5.2 关键 forward 逻辑

```python
def forward(self, batch):
    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [
            batch[key] for key in self.config.image_features
        ]

    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

    l1_loss = (
        F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        * ~batch["action_is_pad"].unsqueeze(-1)
    ).mean()

    loss_dict = {"l1_loss": l1_loss.item()}
    if self.config.use_vae:
        mean_kld = (
            -0.5
            * (
                1
                + log_sigma_x2_hat
                - mu_hat.pow(2)
                - log_sigma_x2_hat.exp()
            )
        ).sum(-1).mean()
        act_loss = l1_loss + mean_kld * self.config.kl_weight
        loss_dict["kld_loss"] = mean_kld.item()
    else:
        act_loss = l1_loss

    valid = ~batch["action_is_pad"]
    delta_pred = actions_hat[:, 1:] - actions_hat[:, :-1]
    delta_target = batch[ACTION][:, 1:] - batch[ACTION][:, :-1]
    delta_valid = valid[:, 1:] & valid[:, :-1]
    delta_error = (delta_pred - delta_target).abs()

    delta_loss = (
        delta_error * delta_valid.unsqueeze(-1)
    ).sum() / delta_valid.sum().clamp_min(1)

    loss = act_loss + self.config.delta_loss_weight * delta_loss
    loss_dict.update(
        {
            "delta_action_loss": delta_loss.item(),
            "delta_loss_weight": self.config.delta_loss_weight,
            "loss": loss.item(),
        }
    )
    return loss, loss_dict
```

这个 loss 在 normalization processor 处理后的 action 空间计算。它是每个有效相邻动作对的
7D L1 和。

Offline evaluator 的 `delta_action_mae` 则是在反归一化 action 空间中计算逐元素 MAE。这两个
指标单位和尺度不同，不应直接比较数值大小；只比较 A1 与 A2 各自同名指标。

## 6. 从 A1 step-0 生成 A2 step-0

A1 和 A2 的模型张量结构相同。不要重新随机初始化 A2，而应从 A1 step-0 生成一份新的完整
checkpoint。

实现：

```text
python -m robot_practice.experiments.act_delta_checkpoint create
```

建议接口：

```text
--source-policy=PATH
--output-dir=PATH
--delta-loss-weight=0.1
```

转换器必须：

1. 拒绝覆盖已有目录；
2. 验证 source 是完整的标准 ACT checkpoint；
3. 复制模型、processor config 和 processor state；
4. 将 `config.json.type` 改为 `act_delta`；
5. 写入 `delta_loss_weight=0.1`；
6. 同步更新 `train_config.json.policy` 中的有效类型和字段；
7. 写入转换 provenance 和源模型 SHA256；
8. 加载 A1/A2 state dict 并逐张量确认完全一致；
9. 原子地完成输出目录。

本实验新建的 A1 initializer 必须保存 `train_config.json`。转换器遇到缺少 train config 的
legacy checkpoint 时应明确报错，不要生成来源语义不完整的 A2 初始化。

使用：

```bash
a1_b0="outputs/act/a1_chunking/<timestamp>/checkpoints/000000/pretrained_model"

python -m robot_practice.experiments.act_delta_checkpoint create \
  --source-policy="${a1_b0}" \
  --output-dir=outputs/act/a2_delta_loss/initialization \
  --delta-loss-weight=0.1

a2_b0="outputs/act/a2_delta_loss/initialization/checkpoints/000000/pretrained_model"
```

## 7. 让 evaluator 认识 `act_delta`

当前 `common.py` 中有几处假定 policy type 只能是 `"act"`。需要统一成：

```text
policy family: act
concrete type: act 或 act_delta
```

至少处理：

- `validate_complete_checkpoint()`；
- `load_act_checkpoint()` 在反序列化前导入本地插件；
- `build_run_config()`；
- `_valid_run_config_sentinel()` 的安全覆盖检查。

推荐在 `run_config.json` 中记录：

```json
{
  "policy": {
    "family": "act",
    "type": "act_delta"
  }
}
```

`ACTDeltaConfig` 是 `ACTConfig` 子类，所以仍可复用现有 ACT processor、LIBERO contract、offline
指标和 Rerun 可视化。

## 8. 最小测试

```bash
pytest -q \
  tests/test_act_delta_policy.py \
  tests/test_act_delta_checkpoint.py \
  tests/test_evaluation_common.py
```

必须覆盖：

- `delta_loss_weight=0` 时，本地总 loss 与原始 ACT 一致；
- 一次 policy forward 只调用一次底层 model；
- GT 与 Pred 差分相同时 `delta_action_loss=0`；
- padding 后的动作不参与 loss；
- `chunk_size=1` 或没有有效相邻对时返回有限的零；
- backward 后梯度有限；
- A1/A2 step-0 state dict 逐张量相同；
- `act_delta` checkpoint 可保存并被两个 evaluator 加载；
- `--overwrite` 仍只允许覆盖合法 evaluator 输出。

## 9. 训练 A2

`a2_delta_loss.sh` 应固定：

```text
policy.type=act_delta
delta_loss_weight=0.1
chunk_size=20
n_action_steps=5
use_vae=false
20 episodes
batch_size=8
steps=20000
seed=1000
```

加载本地插件：

```text
--policy.discover_packages_path=robot_practice.policies.act_delta
```

从零创建 policy 时传 `--policy.type=act_delta`。使用 `--policy-path` 加载转换后的 A2
checkpoint 时不能同时再传 `--policy.type`；脚本应改为验证 checkpoint 中：

```text
type=act_delta
delta_loss_weight=0.1
chunk_size=20
n_action_steps=5
```

启动：

```bash
scripts/experiments/act/a2_delta_loss.sh \
  --steps=20000 \
  --policy-path="${a2_b0}"
```

保存：

```bash
a2_policy="outputs/act/a2_delta_loss/<timestamp>/checkpoints/020000/pretrained_model"
```

训练日志中应同时出现：

```text
l1_loss
delta_action_loss
loss
```

## 10. Offline 评测

```bash
episodes=(
  823 826 834 840 841 845 847 869 875 878
  910 918 920 929 944 948 949 953 966 972
)
```

```bash
scripts/experiments/act/eval_offline.sh \
  --policy-path="${a2_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --agentview-image-convention=lerobot_180 \
  --episodes "${episodes[@]}" \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir=outputs/act/a2_delta_loss/offline_20
```

## 11. Closed-loop 评测

Clean：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${a2_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a2_delta_loss/closed_clean
```

Noisy：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${a2_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=proprio_gaussian \
  --proprio-noise-std=0.01 \
  --seed=1000 \
  --output-dir=outputs/act/a2_delta_loss/closed_noisy
```

## 12. 结果判定

| 配置 | Clean SR | Noisy SR | First MAE | Delta MAE | Action TV | Action Jerk | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | | | | | | | |
| A2 | | | | | | | |

支持动作差分损失假设的 development 门槛：

- A2 clean success rate 相比 A1 下降不超过 5 个百分点；
- A2 closed-loop action jerk 至少降低 15%；
- A2 offline `delta_action_mae` 低于 A1。

如果只看到训练 `delta_action_loss` 下降，但闭环 jerk 没有改善，不能声称实验成功。记录失败
的 init state、抓取阶段、放置阶段、夹爪切换和明显抖动，并更新
`reports/experiment_log.md`。
