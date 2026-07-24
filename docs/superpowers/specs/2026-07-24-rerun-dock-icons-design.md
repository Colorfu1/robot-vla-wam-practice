# Rerun Dock 图标与手动启动器设计

## 目标

为本项目生成的 Rerun Viewer 窗口提供可辨认的 Ubuntu Dock 图标，并让用户手动
打开 RRD 时也能自动区分：

- Offline prediction：`O` 图标；
- LIBERO closed-loop rollout：`L` 图标；
- 未知或混合 recording：中性的 `R` 图标。

用户通过新的 `robot-rerun` 命令获得这一行为。官方 `rerun` 命令保持不变。
evaluator 自动打开 Viewer 时复用同一套 launcher。

## 非目标

- 不修改 LeRobot 或 Rerun 的 site-packages；
- 不覆盖 Conda 环境中的官方 `rerun` 命令；
- 不安装用户级 `.desktop` 文件或修改 GNOME 全局配置；
- 不改变 Rerun 的端口选择、Viewer 复用、持久化布局或数据加载语义；
- 不为 Web Viewer、headless 模式或 `rerun rrd` 子命令设置原生窗口图标；
- 不保证 Wayland 上的自定义 Dock 图标；第一版面向当前 Ubuntu GNOME/X11 环境。

## 用户接口

项目在 `pyproject.toml` 中注册 console command：

```bash
robot-rerun [原始 rerun 参数...]
```

示例：

```bash
robot-rerun outputs/.../prediction.rrd
robot-rerun --persist-state --detach-process --port=9877 outputs/.../rollout.rrd
robot-rerun rrd verify outputs/.../rollout.rrd
robot-rerun --help
```

除原生 Viewer 的图标处理外，参数、标准输入输出、信号和退出码都保持官方
`rerun` 的语义。launcher 不自动增加 `--new`，也不修改用户指定的端口。

## Recording 类型

类型映射集中维护，避免 recording 生成和图标选择各自定义字符串：

| 类型 | RRD application ID | 角标 | WM class |
|---|---|---|---|
| Offline | `robot_practice_act_offline` | `O` | `robot-practice-rerun-offline` |
| LIBERO | `robot_practice_act_libero` | `L` | `robot-practice-rerun-libero` |
| Unknown/Mixed | 其他或多种已知类型 | `R` | `robot-practice-rerun` |

类型检测按以下顺序执行：

1. 对 RRD 优先读取同目录 `run_config.json` 的 `mode`。只有同时满足以下条件时，
   sidecar 才对应当前输入并具有权威性：
   - `schema_version` 为 `1`；
   - `mode` 是 `offline` 或 `libero`；
   - 输入文件名符合该 mode 的 evaluator 输出契约：
     `offline/prediction.rrd` 或 `libero/rollout.rrd`。
   有效 sidecar 是 evaluator 正常产物中的快速路径；检测器信任它，不再为冲突检查
   加载同一 RRD。
2. sidecar 缺失、无效、文件被重命名或不符合上述对应规则时，使用官方
   `rerun.dataframe.load_archive()` 读取 RRD 中所有 data recording 的
   `application_id`。
3. 一个 RRD 中的多个 recording 都属于同一已知类型时，返回该类型。当前合并后的
   Offline RRD 即属于这种情况。
4. 一条命令传入的所有 RRD 都属于同一已知类型时，使用该类型图标；包含 Offline
   和 LIBERO、包含冲突信息或无法识别时，使用 `R` 图标。

fallback 加载完成后应释放 archive 引用，再启动 Viewer，避免类型检测和 Viewer
长期同时持有同一大型 RRD。文件有效性最终仍由官方 Rerun 判断。

## CLI 调用分类

`robot-rerun` 对所有参数始终执行字节级顺序不变的透传；这里的分类只决定是否运行
额外的类型检测和 X11 图标 worker。分类器针对已安装的 Rerun 0.26.2 顶层 CLI，
采用保守策略：

1. 识别顶层 subcommand：
   `analytics`、`auth`、`man`、`mcap`、`reset`、`rrd`、`server`、`help`。
   发现 subcommand 时直接透传，不运行图标逻辑。
2. 明确跳过已知 option 的 value，包括独立 token 形式和 `--option=value` 形式。
   至少覆盖 `--bind`、`--memory-limit`、`--server-memory-limit`、`--port`、
   `--save`、`--screenshot-to`、`-j/--threads`、`--web-viewer-port`、
   `--window-size`、`--renderer` 和 `--video-decoder`，确保 `--save output.rrd`
   不会被误认为 Viewer 输入。
3. `--` 后的 token 全部按 positional input 处理。
4. 只有被分类为 positional、扩展名为 `.rrd` 且存在的本地普通文件才参与
   recording identity 检测。URL、stdin、`.rbl`、图像、mesh 和不存在的路径不被
   当作可检测 RRD。
5. `--web-viewer`、`--serve-web`、`--serve-grpc`、`--headless`、`--connect`、
   help/version/test 模式明确禁用原生窗口图标逻辑。
6. 遇到当前版本表中未知的 option、可疑的 option/value 组合，或无法确定 token
   是 option value 还是 positional input 时，分类结果为“不处理图标”，但仍把
   完整 argv 原样交给官方 Rerun。这样未来 Rerun 新增参数时不会被 wrapper
   错误解释。
7. 多个 positional 本地 RRD 全部参与类型合并；混合类型按 Unknown/Mixed 处理。
   positional RRD 与其他合法 Viewer input 同时出现时，图标只表达已识别 RRD 的
   类型，不声称其他输入属于该类型。

分类器不是 Rerun CLI 的第二份完整 parser。它只对白名单内、能无歧义识别的原生
Viewer 调用增加图标，其余调用一律安全透传。

## 组件与数据流

实现划分为三个边界清晰的单元：

1. **Recording identity**
   - 从 CLI 参数中识别作为输入的本地 `.rrd` 路径；
   - 读取 sidecar 或 RRD application ID；
   - 返回 Offline、LIBERO 或 Unknown/Mixed；
   - 不负责启动进程或访问 X11。

2. **Official CLI delegation**
   - 定位官方 `rerun` 可执行文件；
   - 原样启动并传递 CLI 参数；
   - 处理信号和退出码；
   - 只在确认将打开原生 Viewer 且含 RRD 输入时启动图标 worker。

3. **X11 window identity**
   - 在启动官方 Rerun 前记录 `_NET_CLIENT_LIST`；
   - 启动后在有界时间内轮询新出现的顶层窗口；
   - 综合新窗口 ID、`_NET_WM_PID`、进程关系、现有 class 和窗口标题识别本次
     Rerun Viewer；
   - 只有候选唯一时才写入 `_NET_WM_ICON` 和对应 `WM_CLASS`；
   - 不负责解析 RRD 或决定图标类型。

整体数据流：

```text
robot-rerun args
    -> classify CLI invocation
    -> detect recording identity
    -> snapshot X11 windows
    -> launch official rerun with unchanged args
    -> match the newly created Viewer window
    -> apply L/O/R icon and WM_CLASS
    -> propagate official rerun exit status
```

`--detach-process` 可能让原始子进程在 Viewer 窗口创建前退出。launcher 在这种情况
下保留一个短暂、有上限的等待窗口，以新窗口集合和 Rerun 特征完成匹配；成功或
超时后退出，不留下常驻 helper。

## 端口复用

如果目标端口已有 Viewer，官方 Rerun 可能把 RRD 发送给已有窗口，而不创建新窗口。
launcher 不重新标记已有窗口，因为已有 Viewer 可能已经包含其他类型的数据。
此时输出简短提示，建议显式创建独立 Viewer：

```bash
robot-rerun --new file.rrd
```

或为并行窗口使用不同端口：

```bash
robot-rerun --port=9876 first.rrd
robot-rerun --port=9877 second.rrd
```

## 图标

图标资源放在 `src/robot_practice/assets/` 并随 Python 包安装。三张图使用同一个深色
圆角基础图形，在小尺寸下依靠颜色和右下角大角标区分：

- LIBERO：青蓝色强调和 `L`；
- Offline：橙色强调和 `O`；
- Unknown/Mixed：中性色和 `R`。

提供适合 `_NET_WM_ICON` 的多个 raster 尺寸，至少包含 32、48、64 和 128 像素。
角标在 48×48 的 Ubuntu Dock 尺寸下必须能直接辨认。图标不依赖用户系统主题中的
外部文件。

## evaluator 与文档接入

`launch_rerun_viewer()` 优先使用已安装的 `robot-rerun`，缺失时降级到官方
`rerun` 并记录一次提示。这样 Offline evaluator、LIBERO evaluator 和用户手动命令
使用相同逻辑。

当前用户指南、实验报告和 evaluator shell 脚本中的手动 RRD 打开示例改为
`robot-rerun`。历史 spec 和 plan 不回写，以保留决策记录。

## 错误处理

以下问题只影响图标，不阻断 Viewer：

- `DISPLAY` 不存在；
- 当前 session 不是 X11；
- `python-xlib` 不可用；
- window manager 不支持所需属性；
- 找不到新 Viewer 窗口；
- 出现多个无法消歧的候选窗口；
- 图标资源缺失或属性写入失败；
- RRD 类型无法识别。

launcher 对这些情况输出短警告或提示，并继续官方 Rerun 流程。无效 RRD、非法 CLI
参数、端口冲突等官方错误保留官方退出码和诊断。launcher 不吞掉 `Ctrl-C`，也不因
图标 worker 超时而让命令无限等待。launcher 自己的警告统一写到 stderr；这是在
官方 CLI 输出之外允许增加的诊断，而不是承诺 stdout/stderr 逐字节完全相同。
轮询时限和间隔是有界的内部常量并由测试覆盖，不增加用户可调参数。

## 测试

自动测试覆盖：

- 从合法 `run_config.json` 识别 Offline 和 LIBERO；
- sidecar 只有在 schema、mode 和 evaluator 输出文件名都匹配时才是权威快速路径；
- sidecar 缺失时从真实小型 RRD 的 application ID 识别；
- 多 recording Offline archive 仍识别为 Offline；
- 无效 sidecar、重命名 RRD、混合输入、未知 ID 和无效文件安全降级；
- CLI 参数不重排、不删改，官方子进程退出码原样返回；
- `--save output.rrd` 不会把输出路径误判为输入；`--` 后的 RRD 能被识别；
- 未知 option、URL、stdin 和顶层 subcommand 保守透传且不运行图标逻辑；
- `--help`、`rrd verify`、Web Viewer 和 headless 模式不运行 X11 图标逻辑；
- X11 新窗口集合、PID/class/title 消歧、属性写入、超时和异常使用 mock 测试；
- evaluator 找到 `robot-rerun` 时使用它，找不到时安全降级到 `rerun`；
- 图标资源包含要求的尺寸，并能编码为合法的 `_NET_WM_ICON` CARDINAL 数据。

在依赖或入口变化后运行项目 smoke checks：

```bash
./scripts/doctor.sh
./scripts/quickstart_info.sh
```

人工验收在 Ubuntu GNOME/X11 下执行：

```bash
robot-rerun --port=9876 path/to/offline/prediction.rrd
robot-rerun --port=9877 path/to/libero/rollout.rrd
```

验收标准：

- Dock 同时显示可辨认的 `O` 和 `L`；
- 两个窗口不错误合并为同一类；
- Viewer 数据、blueprint 和持久化布局行为不变；
- 使用 `--detach-process` 时仍能设置图标；
- 关闭窗口后没有遗留 launcher/helper 进程；
- 图标失败模拟不影响 Viewer 正常打开和退出码。
