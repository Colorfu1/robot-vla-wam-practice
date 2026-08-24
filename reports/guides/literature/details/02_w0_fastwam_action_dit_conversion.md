# W0 Fast-WAM ActionDiT 权重转换与 MoT 维度说明

相关文档：

- 主指南：[reports/guides/literature/01_w0_fastwam_libero.md](../01_w0_fastwam_libero.md)
- 模型整体前向结构：[01_w0_fastwam_model_architecture.md](../01_w0_fastwam_model_architecture.md)

代码位置：

- third_party/w0_fastwam/FastWAM/scripts/preprocess_action_dit_backbone.py
- third_party/w0_fastwam/FastWAM/src/fastwam/models/wan22/action_dit.py
- third_party/w0_fastwam/FastWAM/src/fastwam/models/wan22/mot.py
- third_party/w0_fastwam/FastWAM/configs/model/fastwam.yaml

## 结论

ActionDiT 不是从 Wan2.1 的 action 模型加载而来，而是用 Wan2.2 video DiT 的 Transformer 主干权重，经过逐轴 1-D 线性插值和 alpha scaling，转换成一个更窄的 ActionDiT。

~~~text
Wan2.2 video DiT                         ActionDiT
hidden=3072                              hidden=1024
ffn=14336                                ffn=4096
attention=24×128=3072       ───────▶    attention=24×128=3072
30 layers                                30 layers

                                           + action_encoder: 7 → 1024，随机初始化
                                           + head: 1024 → 7，随机初始化
~~~

Wan2.1 只出现在 tokenizer_model_id 中，是文本编码器/Tokenizer 的来源；不是 action backbone 的权重来源。

## 1. 配置中的源模型和目标模型

| 配置项 | Video DiT | ActionDiT |
| --- | ---: | ---: |
| hidden dim | 3072 | 1024 |
| FFN dim | 14336 | 4096 |
| layers | 30 | 30 |
| attention heads | 24 | 24 |
| attention head dim | 128 | 128 |
| attention width | 24 × 128 = 3072 | 24 × 128 = 3072 |
| text dim | 4096 | 4096 |
| time/frequency dim | 256 | 256 |
| action dim | — | 7 |

这里的关键是：两个 expert 的 hidden dim 可以不同，但 attention width、head 数和层数必须相同，才能进入 MoT。

## 2. 转换脚本逐段做什么

### 2.1 读取并解析 YAML

脚本首先读取 video_dit_config 和 action_dit_config，解析配置引用，并把以下字段转换成整数：

~~~text
hidden_dim, action_dim, ffn_dim, num_layers,
num_heads, attn_head_dim, text_dim, freq_dim
~~~

eps 单独转换成浮点数。

--dtype 只控制模型参数的加载 dtype，例如 bfloat16；实际插值时会临时转成 float32。

### 2.2 加载源 Video DiT

脚本调用：

~~~python
components = load_wan22_ti2v_5b_components(...)
video_expert = components.dit
~~~

真正参与权重转换的是 components.dit，也就是 Wan2.2 video DiT。

当前 loader 还会加载 VAE，以及默认的 Wan2.1 text encoder/tokenizer。转换本身不需要这些组件，因此这是当前 GPU OOM 的主要来源之一。

### 2.3 创建目标 ActionDiT

ActionDiT 的结构是：

~~~text
action [B, S_action, 7]
    ↓ action_encoder: Linear(7, 1024)
[B, S_action, 1024]
    ↓ 30 个 DiTBlock
[B, S_action, 1024]
    ↓ head: Linear(1024, 7)
[B, S_action, 7]
~~~

action_encoder 和 head 是 action 专用层，转换脚本不从 video DiT 复制它们。

### 2.4 检查 MoT 的结构兼容性

脚本检查：

~~~text
video.num_heads == action.num_heads
video.attn_head_dim == action.attn_head_dim
video.num_layers == action.num_layers
~~~

当前值都是兼容的：

~~~text
24 heads
128 dim per head
30 layers
~~~

### 2.5 选择需要转换的 key

ActionDiT 使用：

~~~python
ACTION_BACKBONE_SKIP_PREFIXES = (
    "action_encoder.",
    "head.",
)
~~~

因此目标 ActionDiT 中这些 key 被跳过：

~~~text
action_encoder.weight: [1024, 7]
action_encoder.bias:   [1024]
head.weight:           [7, 1024]
head.bias:             [7]
~~~

它们在创建 ActionDiT 时随机初始化，后续由训练学习。

反过来，Video DiT 独有的 patch_embedding 和 video output head 也不会被复制，因为 ActionDiT 中没有对应目标 key。

## 3. 全局 embedding 参数的维度转换

以下 shape 按 PyTorch Linear.weight 的存储形式表示，即 [out_features, in_features]。

| 参数 | Video DiT 源 shape | ActionDiT 目标 shape | 转换 |
| --- | --- | --- | --- |
| text_embedding.0.weight | [3072, 4096] | [1024, 4096] | 第 0 维插值 |
| text_embedding.0.bias | [3072] | [1024] | 1-D 插值 |
| text_embedding.2.weight | [3072, 3072] | [1024, 1024] | 两个维度插值 |
| text_embedding.2.bias | [3072] | [1024] | 1-D 插值 |
| time_embedding.0.weight | [3072, 256] | [1024, 256] | 第 0 维插值 |
| time_embedding.0.bias | [3072] | [1024] | 1-D 插值 |
| time_embedding.2.weight | [3072, 3072] | [1024, 1024] | 两个维度插值 |
| time_embedding.2.bias | [3072] | [1024] | 1-D 插值 |
| time_projection.1.weight | [18432, 3072] | [6144, 1024] | 两个维度插值 |
| time_projection.1.bias | [18432] | [6144] | 1-D 插值 |

其中：

~~~text
18432 = 6 × 3072
6144  = 6 × 1024
~~~

这 6 组值分别用于：

~~~text
shift_msa, scale_msa, gate_msa,
shift_mlp, scale_mlp, gate_mlp
~~~

## 4. 每个 DiTBlock 的参数转换

下面的转换对 blocks.0 到 blocks.29 重复 30 次。

### 4.1 Q/K/V 投影

Video DiT：

~~~text
q/k/v.weight: [3072, 3072]
q/k/v.bias:   [3072]
~~~

ActionDiT：

~~~text
q/k/v.weight: [3072, 1024]
q/k/v.bias:   [3072]
~~~

原因是：

~~~text
Video:  Linear(3072, 24×128) = Linear(3072, 3072)
Action: Linear(1024, 24×128) = Linear(1024, 3072)
~~~

所以输入维度从 3072 缩到 1024，但 attention 输出维度仍保持 3072。Q/K/V 的 bias shape 不变，直接复制。

这套转换同时适用于：

~~~text
blocks.N.self_attn.q/k/v
blocks.N.cross_attn.q/k/v
~~~

### 4.2 Attention 输出投影

~~~text
Video:  o.weight [3072, 3072], o.bias [3072]
Action: o.weight [1024, 3072], o.bias [1024]
~~~

方向是：

~~~text
attention width 3072 → expert hidden width
Video: 3072 → 3072
Action: 3072 → 1024
~~~

### 4.3 Attention norm

~~~text
norm_q.weight: [3072] → [3072]
norm_k.weight: [3072] → [3072]
~~~

因为 attention width 相同，这些参数直接复制。

### 4.4 Block modulation 和 norm3

~~~text
modulation: [1, 6, 3072] → [1, 6, 1024]
norm3:      [3072]       → [1024]
~~~

norm1、norm2 设置了 elementwise_affine=False，没有可训练参数；gate 也没有参数，因此不参与转换。

### 4.5 FFN

Video DiT 的 FFN：

~~~text
3072 → 14336 → 3072
~~~

ActionDiT 的 FFN：

~~~text
1024 → 4096 → 1024
~~~

具体 shape：

| 参数 | Video DiT | ActionDiT |
| --- | --- | --- |
| ffn.0.weight | [14336, 3072] | [4096, 1024] |
| ffn.0.bias | [14336] | [4096] |
| ffn.2.weight | [3072, 14336] | [1024, 4096] |
| ffn.2.bias | [3072] | [1024] |

## 5. 线性插值的具体算法

_resize_tensor_to_shape(src, target_shape) 不理解当前参数是 Linear、bias 还是 modulation，只按 tensor 的每个轴做通用处理。

例如：

~~~text
[14336, 3072] → [4096, 1024]
~~~

处理顺序是：

~~~text
第 0 维：14336 → 4096
第 1 维：3072  → 1024
~~~

每个维度都执行：

1. 将目标维度移动到最后一维。
2. 将其他维度展平。
3. 用 F.interpolate(mode="linear", align_corners=True) 做 1-D 插值。
4. 恢复原来的维度顺序。

因此这是逐轴的 separable 1-D interpolation，不是矩阵截断，也不是随机投影。

插值计算临时使用 float32：

~~~python
out = src.to(torch.float32)
~~~

完成后再转回目标 dtype，通常是 bfloat16，并放到 CPU 保存。

shape 完全相同的参数不会插值，直接复制。

## 6. alpha scaling

如果参数至少是二维，并且最后一维发生变化，脚本会额外计算：

~~~python
alpha = sqrt(src.shape[-1] / target.shape[-1])
~~~

### 3072 → 1024

~~~text
alpha = sqrt(3072 / 1024)
      = sqrt(3)
      ≈ 1.732
~~~

主要影响：

~~~text
text_embedding.2.weight
time_embedding.2.weight
time_projection.1.weight
self_attn.q/k/v.weight
cross_attn.q/k/v.weight
ffn.0.weight
blocks.N.modulation
~~~

### 14336 → 4096

~~~text
alpha = sqrt(14336 / 4096)
      = sqrt(3.5)
      ≈ 1.871
~~~

主要影响：

~~~text
ffn.2.weight
~~~

如果只是第 0 维变化、最后一维没有变化，例如：

~~~text
[3072, 4096] → [1024, 4096]
~~~

则不乘 alpha。

## 7. 输出文件和训练时加载

转换脚本保存：

~~~python
{
    "policy": {...},
    "backbone_state_dict": {...},
    "meta": {...},
}
~~~

含义：

- backbone_state_dict：已经转换成 ActionDiT shape 的主干参数。
- policy：记录 skip prefixes、alpha scaling 和插值方法。
- meta：记录 hidden、FFN、层数、head 数等结构信息，用于加载时校验。

ActionDiT.from_pretrained() 的流程是：

1. 创建完整的 ActionDiT。
2. 随机创建 action_encoder 和 head。
3. 读取 payload 中的 backbone。
4. 将 backbone 合并到当前 ActionDiT。
5. 严格检查所有目标 shape。

因此预处理文件只包含 backbone，不包含 action encoder/head 的训练结果。

## 8. 运行时的 ActionDiT 和 MoT 维度

### ActionDiT 前处理

~~~text
action: [B, S_action, 7]
    ↓ action_encoder 7 → 1024
tokens: [B, S_action, 1024]
~~~

文本 context：

~~~text
context: [B, L, 4096]
    ↓ action text_embedding 4096 → 1024
action context: [B, L, 1024]
~~~

时间条件：

~~~text
timestep [B]
    ↓ sinusoidal embedding
[B, 256]
    ↓ time embedding
[B, 1024]
    ↓ time projection
[B, 6144]
    ↓ unflatten(6, 1024)
[B, 6, 1024]
~~~

### MoT 混合注意力

Video 和 action 分支分别用自己的 block 权重计算 Q/K/V：

~~~text
video q/k/v:  [B, S_video,  3072]
action q/k/v: [B, S_action, 3072]
~~~

沿 sequence 维拼接：

~~~text
q_cat/k_cat/v_cat: [B, S_video + S_action, 3072]
~~~

attention 内部拆成：

~~~text
[B, 24, S_video + S_action, 128]
~~~

混合注意力结束后再按原 sequence 长度切回：

~~~text
video:  [B, S_video,  3072]
action: [B, S_action, 3072]
~~~

然后各自经过 output projection 和 FFN：

~~~text
video output: 3072 → 3072
action output: 3072 → 1024

video FFN:  3072 → 14336 → 3072
action FFN: 1024 → 4096  → 1024
~~~

最后 action head：

~~~text
[B, S_action, 1024]
    ↓ Linear(1024, 7)
[B, S_action, 7]
~~~

MoT 的核心就是：hidden width 可以不同，但两个 expert 的 Q/K/V 输出宽度必须相同。

## 9. Debug 断点

### 调查 OOM

当前转换脚本在加载完整组件后才进入参数循环，因此先在：

~~~text
preprocess_action_dit_backbone.py:171
~~~

观察 loader 是否已经加载 video DiT、T5 和 VAE。

### 观察实际 shape 转换

在：

~~~text
preprocess_action_dit_backbone.py:201
~~~

查看：

~~~python
key
tuple(src.shape)
tuple(target.shape)
src.dtype
target.dtype
~~~

建议依次观察：

~~~text
text_embedding.0.weight
blocks.0.self_attn.q.weight
blocks.0.self_attn.o.weight
blocks.0.ffn.0.weight
blocks.0.ffn.2.weight
~~~

对应关系分别是：

~~~text
[3072, 4096] → [1024, 4096]
[3072, 3072] → [3072, 1024]
[3072, 3072] → [1024, 3072]
[14336, 3072] → [4096, 1024]
[3072, 14336] → [1024, 4096]
~~~

如果要观察插值结果，需要在 :205 查看 value；如果要观察 alpha scaling，需要在 :206 查看 alpha。

### Debug 结论

当前 GPU 容易 OOM 的原因不是参数转换本身，而是转换前加载了不必要的 VAE 和 text encoder，并且 video DiT 与 ActionDiT 同时驻留 GPU。正确的轻量调试路径应只加载 video DiT 的权重，再在 CPU 或分块方式上执行 shape 转换。

