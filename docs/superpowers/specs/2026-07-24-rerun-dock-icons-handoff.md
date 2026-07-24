# Rerun 图标与可视化切换交接说明

## 1) 安装流程（给其他人/agent）

仓库已经有可复用安装流程，别人拿到库后只要：

1. 安装依赖并激活 `robot-practice` 环境（`rerun` 需可用）
2. 在仓库根目录执行：

```bash
cd /path/to/robot-practice
./scripts/install_rerun_desktop_package.sh 0.1.1
```

脚本会做三件事：
- 运行 `build_rerun_desktop_deb.sh` 生成 `.deb`
- `dpkg -i` 安装
- 尝试更新桌面和图标缓存

这条流程要求 `sudo` 权限；如果系统不允许自动输入密码，需手动输入一次。
安装结果是系统级资源，不需要再手动创建 `.desktop` 或链接 `WM_CLASS`。

## 2) 图标机制（图标怎么绑定到窗口）

- `rr_of.desktop` -> `rr_of` 二进制 -> `StartupWMClass=rr_of` -> `Icon=rr_of`
- `rr_li.desktop` -> `rr_li` 二进制 -> `StartupWMClass=rr_li` -> `Icon=rr_li`
- `rerun.desktop` -> `/usr/bin/rerun` -> `StartupWMClass=rerun` -> `Icon=rerun`

安装内容对应路径：
- `/opt/robot-practice-rerun/bin/rr_of`
- `/opt/robot-practice-rerun/bin/rr_li`
- `/usr/share/applications/{rr_of,rr_li,rerun}.desktop`
- `/usr/share/icons/hicolor/256x256/apps/{rr_of,rr_li,rerun}.png`

绑定核心来自两个来源：
1. 修改后的 rerun 二进制里内置 app-id（由 `robot_practice.rerun_launcher` 里 `native_app_id` 控制）
2. `.desktop` 里的 `StartupWMClass` 与 `Icon` 组合（GNOME Dock 基于这条链条匹配）

## 3) 如何切换可视化（offline/libero/unknown）

当前是自动判断，不需要手工带参数：
- `*.rrd` 旁边有 `run_config.json` 且 `mode` 为：
  - `offline` -> 显示 Offline 图标
  - `libero` -> 显示 LIBERO 图标
- 其他情况：
  - fallback 到 archive 的 `application_id` 集合；
  - 识别失败 -> `rerun` 默认图标（`R`）

对应入口：
- 手动：直接运行
  - `rr_of <某 prediction.rrd> --port ...`
  - `rr_li <某 rollout.rrd> --port ...`
- 评测脚本：按文件路径和 sidecar 自动落到对应 launcher

## 4) 如果要新增/替换图标（给其他 agent 的改动点）

集中在 `src/robot_practice/rerun_launcher.py`：

- `VIEWER_IDENTITIES`：
  - `desktop_id`（desktop 文件名）
  - `native_app_id`（窗口类）
  - `icon_filename`（资源文件名）
- `install_desktop_integration()` 会用上面三项批量写入

新增类型时建议的最小修改：
1. 在 `ViewerIdentity` 增一条
2. 在 `LEGACY_DESKTOP_IDS` 增旧名清理
3. 补对应 PNG 到 `src/robot_practice/assets/`
4. 如需打包，重建 `.deb` 并分发即可
