# 2026-08-17 StackCube W0 Action-Conditioning 机制验证

## 状态

- 状态：`dropped`
- 范围：保留旧 W0 被替代的路由记录；本文件不再可执行。
- 放弃原因：WAM 主线已改为先复现 Fast-WAM/Light-WAM 官方 LIBERO 机制，
  再回到 StackCube 做文献对齐小模型对照。
- 最后更新：2026-08-17

## 目标

阻止后续人员误执行已经失效的 D1/N0、action replace/noise/mask 和 `Δ=4` latent
decoder 协议。当前 W0 唯一入口是
[Fast-WAM LIBERO 复现与机制锚点](2026-08-17-W0-Fast-WAM-LIBERO复现与机制锚点.md)。

## 前置条件

- [x] 用户已明确将 W0 改为 Fast-WAM literature reproduction。
- [x] 新 W0 计划已接管 base、模型、训练、评测和 gate。

## 执行步骤

本计划无执行命令。不得生成旧 `w0/action_conditioned`、`w0/no_action` 或 latent
intervention 产物。

## 判断标准

| 判据 | 状态 | 处理 |
| --- | --- | --- |
| 旧 W0 是否仍可执行 | 否 | 一律转到新 W0 |
| latent action-conditioning 是否被证明 | 未证明 | 不得将旧计划写成已完成结果 |

## 产物路径

- 无新产物。
- 新 W0 产物以 `outputs/literature/w0_fastwam/` 为根目录。

## 风险与停止条件

- 任何仍引用本文件的下游计划均属过期依赖，在执行前必须修订。
- 旧 action-conditioning 设计不得与 Fast-WAM/Light-WAM 复现并行或作为绕行路线。

## 收尾

- [x] 本文件标记为 `dropped`。
- [x] `plans/README.md`、WAM 总入口和母计划全部转向新 W0。
