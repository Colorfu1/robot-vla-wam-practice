# 文献模型复现实验指南

- [DM0.5 模型结构与张量维度](../dm05/01_dm05_model_architecture.md)：参照 FastWAM 架构说明，
  展开本地 DM05-libero 的视觉语言前缀、动作专家、Flow Matching 与形状实测。

当前 WAM 路线先做 LIBERO 文献锚点，再迁移到 StackCube；W0/W1 的文献复现结果与 W2/W3 的
本地 StackCube 结果不合并。总依赖、gate 和停止条件见
[WAM 文献锚点复现与 StackCube 迁移路线](../../../plans/2026-08-17-WAM文献锚点复现与StackCube迁移路线.md)。

- [W0 Fast-WAM LIBERO 复现与机制锚点](01_w0_fastwam_libero.md)：**UP-REL baseline 与 LR-REL quantitative eval 均已完成，W0 gate 仍进行中**。UP-REL released checkpoint 为 `1929/2000`（96.45%）；LR-REL 为 `1885/2000`（94.25%），残余 2.2pp 几乎全部集中在 `libero_spatial` task 5（UP 96% vs LR 28%）且**原因未解释**。FW-V/FW-NV 因果对照与 deployment evidence 尚未完成
- [W0 Wan 视频生成能力保持性快速验证](02_w0_wan_video_generation_preservation.md)：VBench-I2V-mini-21 与 W0-Video-40 的配对离线验证命令；当前仅完成实现与单测，尚无正式指标结果
- W1 Light-WAM：**blocked**，等待 W0 的 artifact、mechanism 和 deployment gate
- W2/W3 StackCube tiny：**blocked**，等待 W0/W1 机制冻结；W3 formal bank 当前保持关闭

复现过程中修掉两个阻断性缺陷，两者的失败症状都极具误导性（管路全绿、成功率异常），细节和
排除过程见 [2026-08-19 W0 Fast-WAM UP-REL 与 LR-REL 复现结论](../../daily/2026-08-19-W0-FastWAM-UP-REL与LR-REL复现结论.md)：

- `MUJOCO_EGL_DEVICE_ID` 被当作物理卡号，而 robosuite 用它索引 `eglQueryDevicesEXT()`；
  `(cuda 7, egl 7)` 这一个配对在 `mjr_readPixels` 处 native SIGABRT，stderr 无任何消息。
  launcher 现在自动标定并按机器指纹缓存。
- LeRobot released checkpoint 的 `norm_map.VISUAL=MEAN_STD(0.5,0.5)` 与 pinned core 的
  `*2-1` 重复映射，VAE 收到 `4x-3` 而非 `2x-1`，LR-REL 全程 0 成功。launcher 现在默认用
  符号链接 overlay 把 VISUAL 改回 `IDENTITY`，pinned artifact 字节不变。
