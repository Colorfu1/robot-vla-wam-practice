# Rerun Dock 图标实现计划

**Design:** `docs/superpowers/specs/2026-07-24-rerun-dock-icons-design.md`

## 1. Recording identity 与 CLI 分类

- 新建独立 launcher 模块及测试；
- 定义 Offline、LIBERO、Unknown/Mixed identity；
- 实现有效 `run_config.json` 快速路径；
- 实现 `rerun.dataframe.load_archive()` fallback；
- 保守解析 Rerun 0.26.2 顶层参数，始终原样保留 argv。

## 2. 官方 CLI 透传

- 在 `pyproject.toml` 注册 `robot-rerun`；
- 定位并启动官方 `rerun`；
- 非原生 Viewer 调用不运行 X11 逻辑；
- 保留标准流、信号和退出码。

## 3. 图标与 X11 窗口标记

- 添加共享基础图形的 L/O/R raster icon；
- 将资源作为 package data 安装；
- 启动前快照 `_NET_CLIENT_LIST`；
- 唯一匹配新 Rerun 窗口后写 `_NET_WM_ICON` 和 `WM_CLASS`；
- X11 不可用、候选歧义或超时时仅警告。

## 4. evaluator 与文档接入

- `launch_rerun_viewer()` 优先使用 `robot-rerun`，缺失时降级到 `rerun`；
- 当前 evaluator 脚本和报告中的手动打开示例使用 `robot-rerun`；
- 不修改历史 spec/plan 或 site-packages。

## 5. 验证

- 定向运行 launcher、evaluation common/interface/rerun 测试；
- 运行完整 pytest、Ruff、doctor 和 quickstart smoke checks；
- 在 Ubuntu GNOME/X11 使用不同端口同时打开 Offline/LIBERO RRD；
- 检查 Dock 的 O/L 图标、官方 Viewer 行为和残留进程。
