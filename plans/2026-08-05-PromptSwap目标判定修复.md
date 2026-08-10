# 2026-08-05 Prompt Swap 目标判定修复

## 状态

- 状态：`done`
- 范围：修复 Prompt Swap 协议中「参考场景原生目标被 reset 掉导致该目标恒判失败」的 bug，
  重跑 Z0/C1/C2 的 30 条 prompt_swap，并更正四份已发表文档中的 prompt compliance。
  canonical 与 held-out paraphrase 两个协议不受影响，不重跑。
- 最后更新：2026-08-06

## 目标

已确认的矛盾（修复前）：同一个模型、同一句指令、**同样那 10 个初始状态**——

| 协议 | 判定方式 | C1 | C2 |
|---|---|---:|---:|
| `canonical` task 1 | 仿真器原生 `is_success` | 9/10 | 10/10 |
| `prompt_swap` stove | 自定义 `goal_classifier` | **0/10** | **0/10** |

stove 的 prompt 就是 task 1 的 canonical（`put the bowl on the stove`），参考状态
`[1,2,3,4,5,7,9,10,11,12]` 都取自 task 1 的 test 集。步数是决定性证据：

- cabinet / plate 那 20 条全部跑满 **300 步**，分类正确（10/10、8/10）
- stove 那 9 条在 **76–88 步**就终止，正落在 canonical 成功步数区间（74–91）内

也就是说那 9 条其实达成了目标，随后被环境 reset，`classify_bowl_goal` 在 reset 之后才评估，
看到碗回到桌上，判 `no_goal_reached`。

**这个 bug 只打击"恰好等于参考场景原生目标"的那一个目标。** 参考场景是 task 1（stove），
所以只有 stove 被打击；换成 task 4 作参考场景，坏掉的就会是 cabinet。它与目标难易无关。

后果：已发表的 prompt compliance **18/30 = 60% 是判定错误造成的旧读数**，不能继续作为当前结论。
C1 与 C2 的混淆矩阵逐格相同也由此解释——那不是模型行为，是判定机制的固定失效。修复后的正式
结果只登记在 `reports/`，本计划不重复记录指标。

## 前置条件

- [ ] 复现矛盾：

  ```bash
  conda run --no-capture-output -n robot-practice python -c "
  import pandas as pd
  df=pd.read_parquet('outputs/smolvla_libero/eval/c1_default_non_peft/episodes.parquet')
  sw=df[df.protocol=='prompt_swap']
  print(sw.groupby('expected_goal').agg(n=('success','size'), steps=('num_steps','unique')))"
  ```

      预期 stove 的步数含 76–88，cabinet/plate 只有 300。

- [ ] 已知的代码事实（供第 1 步定位用，不要当作结论）：
  - LeRobot wrapper `lerobot/envs/libero.py` 第 328–348 行：
    `terminated = done or is_success`，`is_success = self._env.check_success()`，
    `if terminated: self.reset()`——**reset 发生在 step 返回之前**
  - `robosuite` 的基础 `done` 与 horizon 有关，但当前 LIBERO 的
    `BDDLBaseDomain.step()` 在 `super().step()` 后执行
    `done = self._check_success()`，所以参考目标成功会从内部 `done` 通道终止
  - `libero_goal_classifier.disable_reference_goal_termination` 当时只覆盖了外层
    `bridge.check_success()`，没有覆盖内部 `raw._check_success()`；因此外层 patch
    看似生效，内部 `done` 仍会触发 reset
  - `libero_act.py` 第 1134–1135 行只在 episode 循环**之前**调用一次该函数；循环内每个
    episode 都会 `env.reset()`（第 1209 行）
  - 现有测试 `tests/test_smolvla_libero_target_protocol_evaluation.py:193` 用的是
    `SimpleNamespace` 假 env，只证明 patch 能作用在玩具对象上，**无法覆盖真实 env 链**——
    这就是 bug 漏过测试的原因

## 执行步骤

### 1. 用真实 env 定位失效原因

**不要先写修复。** 先跑一个最小探针，把「为什么 patch 没生效」钉死。探针要在真实 LIBERO env
上回答：

- `env.envs[0]` 是 LeRobot env 本身，还是被 Gymnasium wrapper 包了一层？
- `getattr(env.envs[0], "_env")` 与 wrapper 内部 `self._env` 是否同一对象（比 `id()`）？
- patch 之后 `env.envs[0]._env.check_success()` 是否返回 False？
- 手动步进到成功时刻，`terminated` 是否仍为 True？若是，它来自 `done` 还是 `is_success`？
- 每次 `env.reset()` 之后 patch 是否还在（`ControlEnv` 是否在 reset 时重建内部 env）？

候选原因（探针需要能区分它们，不要预设）：

1. `env.envs[0]` 经 Gymnasium wrapper 委派，`_env` 取到的不是 wrapper `step` 用的那个对象；
2. `ControlEnv.reset()` 重建了内部 robosuite env，patch 被丢弃；
3. `check_success` 在别处被提前绑定为 bound method，实例属性覆盖不到；
4. 终止其实来自 `done`，说明本仓库的 LIBERO/robosuite 版本与上面读到的实现不同。

把探针结论写进计划执行记录，再定修法。

### 2. 修复：让 prompt_swap 的 episode 不因参考目标而终止

按第 1 步的结论修。无论采用哪条路径，必须满足：

- prompt_swap 的 30 条 episode **全部跑满 300 步**（与 cabinet/plate 现状一致）；
- 修复只作用于 `goal_classifier is not None` 的路径，**canonical 与 held-out paraphrase
  的原生 `is_success` 判定不得受影响**；
- 每个 episode 的 `env.reset()` 之后修复仍然有效（若原因是 2，需在每个 episode 重新施加）。

### 3. 同时记录「首次达成」而非只看终态

现在 `classify_bowl_goal` 只在循环结束后调用一次（`libero_act.py` 第 1397 行），语义是
「第 300 步时目标是否成立」。而 canonical 协议的语义是「是否曾经达成」
（`success_seen = success_seen or success_this_step`）。两者不同源，不可比。

改为在循环内每步分类，记录：

- `first_goal_reached` 与 `first_goal_step`：首次有谓词成立的目标与步数；
- `terminal_goal`：第 300 步的分类结果（即现有语义）；
- 两者都写进 `episodes.parquet` 与 `metrics.json`。

报告时以 `first_goal_reached` 为主指标（与 canonical 同源可比），`terminal_goal` 作为「达成后
是否保持」的补充读数。这一条能顺带回答一个现在无法回答的问题：策略达成目标后继续动作 220 步，
会不会把物体推离目标区。

每步分类的成本是 300 × 30 次谓词求值，可忽略。

### 4. 回归测试

`tests/test_smolvla_libero_target_protocol_evaluation.py`：

- **保留**现有假 env 测试，但新增一条注释说明它不覆盖真实 env 链；
- 新增：模拟「达成参考目标后环境自动 reset」的假 env，断言修复后 `first_goal_reached` 仍能
  捕获到该目标（这是本 bug 的最小复现）；
- 新增：断言修复只在 `goal_classifier is not None` 时生效；
- 新增：`first_goal_reached` 与 `terminal_goal` 在「达成后又被推离」的构造场景下取值不同；
- 新增：prompt_swap 的 30 条 episode 步数全部为 300（用假 env 断言 step 循环不提前 break）。

```bash
conda run --no-capture-output -n robot-practice python -m pytest \
  tests/test_smolvla_libero_target_protocol_evaluation.py \
  tests/test_smolvla_libero_evaluation.py -q
conda run --no-capture-output -n robot-practice ruff check \
  src/robot_practice/evaluation tests/test_smolvla_libero_target_protocol_evaluation.py
conda run --no-capture-output -n robot-practice ruff format --check \
  src/robot_practice/evaluation tests/test_smolvla_libero_target_protocol_evaluation.py
```

### 5. 只重跑 prompt_swap

`eval_target_protocols.sh` 支持 `--protocols`，因此只需重跑 30 条，不动另外 180 条。
对三个已评测的策略各跑一次，**输出到新目录**，不覆盖原产物：

```bash
for pair in \
  "z0_b0_zero_shot outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model" \
  "c1_default_non_peft outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected/pretrained_model" \
  "c2_canonical_lora outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/selected/pretrained_model"; do
  set -- $pair
  bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
    --policy-path=$2 \
    --model-label=$1 \
    --protocols=prompt_swap \
    --num-gpus=8 \
    --libero-assets-dir=datasets/raw/libero_assets \
    --output-dir=outputs/smolvla_libero/eval/prompt_swap_fixed/$1
done
```

C3 尚未评测，**修复完成后再跑 C3**，它将直接使用修好的判定，不需要重跑。

### 6. 更正已发表文档

prompt compliance 与相关解读出现在四处，全部需要更正，并保留一段说明「原值因判定 bug 偏低」：

- `reports/2026-08-04-Z0基线与C1目标适配正式评测结论.md`
- `reports/2026-08-05-C2 Canonical LoRA 正式评测结论.md`
- `reports/guides/smolvla_libero/03_c1_non_peft.md`
- `reports/guides/smolvla_libero/04_c2_canonical_lora.md`

其中**必须撤回**的错误解读（原文大意）：「stove 指令 10/10 全失败且无一跑向其他目标——是做
不到，不是听错」。修复后的实际数值与终态保持诊断统一登记在 `reports/`；`experiment_log.md` 的
两条相关条目同样更正。

## 判断标准

**执行前写死。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| 根因已定位 | 第 1 步的探针能明确指出终止来自 `done` 还是 `is_success`，以及 patch 为何失效 | 不得在未定位前提交修复 |
| 步数 | 修复后 prompt_swap 全部 30 条为 300 步 | 修复未生效，继续查 |
| stove 可测 | C1 的 stove 首次达成数 ≥ 8/10（canonical 在同 10 个状态上是 9/10，允许差 1 条） | 停止，说明还有第二个失效源 |
| 隔离性 | canonical 与 held-out paraphrase 的成绩与修复前逐值相同（不重跑，用现有产物中未受影响的协议核对代码路径） | 修实现，不得波及原生判定 |
| 双读数 | `episodes.parquet` 同时含 `first_goal_reached`/`first_goal_step` 与 `terminal_goal` | 修实现 |
| 协议一致性 | 重跑的三份 manifest 除策略字段外与原产物逐项相同（split/paraphrases/协议 sha256、seed、horizon） | 停止，查清后重跑 |
| 产物隔离 | 新结果写入 `eval/prompt_swap_fixed/`，原产物不被覆盖 | 停止 |
| 测试与 ruff | 全绿，且新增用例能在**未修复**的代码上失败 | 补测试，否则无法防回归 |
| 文档更正 | 四份文档加 `experiment_log.md` 的 prompt compliance 全部更新，并显式撤回错误解读 | 补文档 |

**不预设修复后的数值。** stove 可测之后 compliance 可能升到接近 30/30，也可能暴露新的失败
模式（例如达成后被推离）。照实记录。

## 产物路径

- 代码：`src/robot_practice/evaluation/libero_goal_classifier.py`、
  `src/robot_practice/evaluation/libero_act.py`
- 测试：`tests/test_smolvla_libero_target_protocol_evaluation.py`
- 新结果：`outputs/smolvla_libero/eval/prompt_swap_fixed/{z0_b0_zero_shot,c1_default_non_peft,c2_canonical_lora}/`
- 文档：上列四份 + `reports/experiment_log.md`

## 不在本计划范围

- canonical 与 held-out paraphrase 的重跑。这两个协议用各自任务的原生 `is_success`，与跨目标
  classifier 无关，不受本 bug 影响；96.7% / 33.3% / 21.1% 这些数字保持有效。
- C3 的评测。修完之后再跑，直接得到正确判定。
- 背景回退评测。它也用原生 `is_success`，不受影响。
- Prompt Swap 协议本身的设计变更（换参考场景、增加目标、改判定谓词）。本计划只修实现与语义
  记录，不动冻结的协议配置。

## 风险与停止条件

- **不要在未定位根因前提交修复。** 当前根因已由真实 wrapper 链和安装版本源码核对：内部
  `BDDLBaseDomain.step()` 通过 `_check_success()` 产生成功 `done`，而原 patch 只改了外层
  `check_success()`。后续仍需由用户在正式环境完成 rollout 验证，不能把单元测试当作数值结果。
- **修复必须只作用于 classifier 路径。** canonical 与 held-out paraphrase 依赖原生
  `is_success` 提前终止来记成功；若把终止全局关掉，那两个协议的语义也会被改变，180 条已发表
  成绩就会失效。
- **假 env 测试不足以防回归。** 本 bug 恰恰是被假 env 测试放过的。新增用例必须模拟「达成后
  自动 reset」这一真实行为，且要验证它在未修复的代码上会失败。
- **原产物不得覆盖。** 修复后的数值与原值不可比（判定语义变了），必须并存，报告里同时给出
  并说明差异来源。
- **可能暴露新问题。** 若修复后 stove 的 `first_goal_reached` 高但 `terminal_goal` 低，说明
  策略达成后会把物体推离——那是一个新的、真实的发现，不是修复失败，应单独记录。

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
- [ ] 文档更正与结果登记按 `reports/AGENTS.md` 处理
