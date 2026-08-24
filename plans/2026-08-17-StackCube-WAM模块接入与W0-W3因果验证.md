# 2026-08-17 StackCube WAM 模块接入与 W0-W3 因果验证

## 状态

- 状态：`dropped`
- 范围：保留旧“直接在 StackCube 上自设计 WAM”路线的替代记录。
- 放弃原因：新路线先复现 Fast-WAM/Light-WAM 官方 LIBERO 机制，再回到
  StackCube 做小模型对齐对照。
- 最后更新：2026-08-17

## 目标

防止继续执行旧 Mini-WAM/latent W0-W3。当前唯一总入口是
[WAM 文献锚点复现与 StackCube 迁移路线](2026-08-17-WAM文献锚点复现与StackCube迁移路线.md)。

## 前置条件

- [x] 用户已决定 W0 不做 Mini-WAM，先复现 Fast-WAM。
- [x] 新总路线和 Fast-WAM W0 计划已创建。

## 执行步骤

本文件无执行命令。不得再生成 `w0_mini_wam_aligned_native4d` 或旧
latent horizon/shuffle/formal 产物。

## 判断标准

| 判据 | 状态 |
| --- | --- |
| 旧 StackCube-first WAM 路线是否可执行 | 否 |
| 新 W0 是否从 M2-W checkpoint warm-start | 否；W0 是官方 LIBERO Fast-WAM reproduction |

## 产物路径

- 无新产物。
- 新 W0 产物根目录：`outputs/literature/w0_fastwam/`。

## 风险与停止条件

- 任何仍引用本文件作为 active WAM contract 的计划都必须先修订。
- 不得在 Fast-WAM/Light-WAM 锚点完成前恢复 StackCube 自设计 world head。

## 收尾

- [x] 本文件标记为 `dropped`。
- [x] 新总入口与 W0 详细计划已建立。
