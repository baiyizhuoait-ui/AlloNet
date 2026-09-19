# Phase 6 / G3 —— 跨架构复现：预注册与执行前审计

**版本 v1　2026-09-15**
**上游**：`docs/PHASE6_G123_EXECUTION_PLAN.md` §3（缺口 G3）、`docs/PHASE6_PUBLICATION_STRATEGY.md`
**本文性质**：**预注册 + 只读审计报告**。所有判据、阈值、红线在**任何 G3 数据存在之前**写定。
执行件已就绪（见 §7）；本文不放宽、不追认任何事后阈值。

---

## 0. 执行摘要

按 `跨架构验证预审` 技能的清单做只读预审时，**在 arch-1 台账里查出两处制度性混淆**。
它们不影响"参数/FLOPs 几何"（那是设计事实），但**足以伪造我们要复现的那个结论**：

| # | 混淆 | 证据 | 后果 |
|---|---|---|---|
| **C1** | **监督制度**：锚框默认值在 2026-09-08 23:16 被换成 IoU-k-means；此前所有**未显式写 `anchors:` 的配置**都在用被翻转的旧默认（代码注释自述令 ~48.5% GT 零正样本） | 用 `config.yaml` 的 mtime 推出每次训练启动时间：phase2d 09-04/05、phase3a 09-05、phase4a 09-07 —— **全在换锚之前**；而 FINAL-100 在 09-13 之后。**同一模型同 20ep 的配对实测：旧锚 mAP50 = 0.3452/0.3521，k-means 锚 = 0.4984/0.4938，Δ = +0.148** | 台账内部**锚框自洽**（不是自相矛盾），但**整体活在项目已废弃的制度里**；而论文主表用 k-means。Δ=0.148 是台账编码器臂所声称优势（+0.036）的 **4 倍** |
| **C2** | **检测拓扑**：台账各单元（phase2b/2d/3a）只写 `detection: {nc: 1}` ⇒ 检测读 encoder 的 F2/F3/F4、**绕过 Z**（"R0"头）。当前模型用 `DetFromZ`，三任务**都流经 Z**（"R2"头） | 参数实测：phase2d z16 = 0.189M vs phase4a r2 z16 = 0.2014M，同一 z 不同头 | "把参数花在 encoder 还是 Z 上？"这个问题**只在三任务真的争夺 Z 时才成立**。R0 下检测不吃 Z，z=16"足够"是设计产物，不是发现 |

**结论**：G3 **不能**把新的 arch-2 数字去比历史 arch-1 台账。两个架构必须在**同一套配置**
（R2 检测读 Z + lean 分割 + 显式 k-means 锚框 + 20ep + seed 0）下重测，
使**编码器拓扑成为唯一变量**。

**这是 G3 设计的唯一改动，且对两个架构对称施加。**

---

## 1. 为什么需要 G3

`docs/PHASE6_G123_EXECUTION_PLAN.md` §3.2 已判定：`light_encoder.py` 的 `stages/blocks`
只能做**宽度与深度的等比缩放**，而等比缩放**恰恰就是基线做法（Model U = uniform encoder ×1.40）**。
⇒ 用缩放后的 encoder 当"第二架构"等于**把 Model U 再算一遍**，得到的是循环论证，不是泛化证据。

因此 arch-2 必须是**不同拓扑**。已实现 `models/encoder/ir_encoder.py`：

| | arch-1 `light_encoder.py` | arch-2 `ir_encoder.py` |
|---|---|---|
| 血统 | ESPNet / 深度可分离 | MobileNetV2/V3 逆残差 |
| 块结构 | DW+PW 再 DW+PW | 1×1 扩张 → k×k DW → 1×1 投影（**投影后无激活**，线性瓶颈） |
| 残差位置 | **宽张量**上（add 后接 ReLU） | **窄瓶颈张量**上，stride≠1 时丢弃短路 |
| 下采样 | 块内首个 DW 上 stride 2，逐级 1/2→1/4 | **主干两步到 1/4**；阶段 1 不再下采样 |
| 卷积核 | 全 3×3 | 阶段混用 3×3 / 5×5 |

**实测结构差异**（encdoer 级，640×640，`g3_arch_probe.json`）：

| | 卷积数 | 其中 1×1 | 其中 DW | ReLU | encoder FLOPs |
|---|---:|---:|---:|---:|---:|
| arch-1 dws | 33 | 18 | 14 | 36 | 0.6513 G |
| arch-2 ir | 23 | 14 | 7 | 16 | 1.0135 G |

**它是有效 independent testbed 的理由**（技能强制必须回答的一句）：

> 迁移性问题**可测** —— 结构差异足够大（**不同下采样与残差拓扑 + 块内运算顺序不同**，不是缩放）；
> 干预原理可搬（同一条"边际价格 → 分配"规则）；评测侧已接好（harness 直接吃 encoder cfg，
> 评测脚本零改动）。**不是**"换了模型所以新"—— 后半句是 novelty 违规。

---

## 2. 设计：六个单元，一条配置

**共同固定项**（对 6 个单元逐字相同）：`from_z: true` / `z_proj: true` / `det_ch: 32` /
`segmentation.hidden: 32`（lean，无 lane 修复）/ **锚框显式写入 k-means 三尺度** /
`stage A` / `tri_train` / bs16 / lr 1e-3 / cosine / wd 5e-4 / 20ep / seed 0。

**唯一变量：encoder 拓扑 + 该架构自己的容量档位。**

| 单元 | 架构 | 编码器 | z | 总参数 | FLOPs |
|---|---|---|---:|---:|---:|
| `a1_base` | dws | stem16 `[32,64,96,128]` ×2 | 16 | 0.2014 M | 1.0796 G |
| `a1_zspend` | dws | 同上 | **128** | 0.3019 M | 2.0889 G |
| `a1_encspend` | dws | stem24 `[40,80,128,168]` ×2 | 16 | 0.3020 M | 1.4790 G |
| `a2_base` | ir | stem16 `[16,32,56,72]` expand4 | 16 | 0.2018 M | 1.1424 G |
| `a2_zspend` | ir | 同上 | **128** | 0.2881 M | 2.0865 G |
| `a2_encspend` | ir | stem16 `[24,48,72,88]` expand4 | 16 | 0.2921 M | 1.6154 G |

几何由 `scripts/phase6_g3_tier_solve.py` **求解**而非手填：先解 base 档使总参数匹配，
再**实测**同编码器 z=128 的总参数，最后解编码器档匹配到该总参数 ⇒ 配对由构造保证近等预算。

**配对（Section D 的判定对象）**：

| 架构 | z-spend | enc-spend | Δparams | ΔFLOPs | 在 ±5% 内 |
|---|---|---|---:|---:|---|
| a1 | 0.3019 M / 2.0889 G | 0.3020 M / 1.4790 G | **+0.0%** | **−29.2%** | ✅ |
| a2 | 0.2881 M / 2.0865 G | 0.2921 M / 1.6154 G | **+1.4%** | **−22.6%** | ✅ |

> 两架构的 base 档总参数几乎相同（0.2014 / 0.2018 M），但 **FLOPs 不同**
> （1.0796 / 1.1424 G，+5.8%）—— 这是逆残差把算力花在块内的固有属性，
> 如实报告，不用来挑选结果。

---

## 3. 冻结判据（R0–R4，写于任何 G3 数据之前）

> 实现在 `scripts/phase6_g3_ledger.py`，并由 `phase6_g3_ledger_selftest.py`
> 用**合成数据**在结果未知时先跑通 5 个用例。**判定由脚本给出，不手抄数字。**

| 规则 | 内容 |
|---|---|
| **R0** 预算闸门 | 每个架构：`\|params(encspend) − params(zspend)\| / params(zspend) ≤ 0.05`。**不过闸 = 该架构 `INSTRUMENT_INCONCLUSIVE`**，不计胜不计负 |
| **R1** 局部判定 | enc-spend 参数 ≤ 且 FLOPs ≤ z-spend，且 **≥4/6 指标**不劣 → `DOMINATES`；2–3 个 → `PARTIAL`；≤1 个 → `Z_WINS` |
| **R2** 噪声标尺 | 指标差需 `\|Δ\| > 2·sd`，sd 由**该架构自己**的 base 单元 3 个种子测得。**禁止跨架构沿用标尺**（技能陷阱 6）。plus-seeds 阶段未跑 → 标尺 `ABSENT`，**不得做显著性声称**，只报序 |
| **R3** 迁移判定 | 两架构均过 R0，且 a2 的类别**至少不弱于** a1，且"enc−Z 差"的**符号在 ≥4/6 指标上一致** → `TRANSFERRED`；弱一档或 3/6 → `PARTIAL`；否则 `NOT_TRANSFERRED`；数据缺或闸门失败 → `INCONCLUSIVE` |
| **R4** 报告纪律 | 报告里每个数字必须出自脚本 stdout 或 JSON，禁止手抄 |

**指标集**（与 arch-1 台账逐字相同，6 个）：`mAP50, mAP50_95, da_mIoU, da_fg, lane_mIoU, lane_fg`。

### 与 arch-1 台账的两处**有意不同**（对称施加）

1. **Section B 的起点**：arch-1 用 `E-small → E-large`（起点与 Section A 不同，台账脚本自己
   在 docstring 里标注了这个 caveat）。G3 改为 **`base → enc-spend`**，使 A、B 两臂**从同一单元
   出发**，边际收益可比。该改动对两个架构**同时**施加，不造成不对称。
2. 增加 **R0 显式预算闸门**（arch-1 台账只在打印里写"almost equal"，没有阈值化）。

---

## 4. 成本与排期（实测口径）

单单元实测：20ep ≈ 2.40 h（seed0 100ep = 719 min ⇒ 20ep ≈ 144 min）。

| 阶段 | 内容 | GPU | 累计 |
|---|---|---|---|
| **A** 管道探针 | 2 个单元 × 4ep（测通 + 量速 + 早信号） | 1.0 h | 1.0 h |
| **B** 主实验 | **6 单元 × 20ep**（§2 表） | 14.4 h | 15.4 h |
| **C** 分支：重标尺 | 4 单元 × 20ep（两架构 base 各补 seed1/2） | 9.6 h | 25.0 h |

- **只跑 A+B（15.4 h）**：得到完整台账 + 迁移判定，但 **R2 标尺缺失** ⇒ 只能报"序"，
  不能报"显著"。**这是最省的可用配置。**
- **加 C（+9.6 h）**：两架构各有自己的 3 种子噪声标尺，`TRANSFERRED/PARTIAL` 才带显著性。
- 单卡约束下与 G1（200ep fresh，24 h）**串行**：G3 结束 → G1 开始。

---

## 5. 红线（写死，不得事后放宽）

1. 新 encoder **不得**是 `stages/blocks` 缩放的结果（已换拓扑，见 §1 表）。
2. 干预必须是**同一条分配规则**：6 个单元共用同一锚框集合 + 同一 TAL 分配，**不另发明一条**。
3. 第二架构必须**重新标定噪声标尺**；未标定就不得声称显著。
4. 某维度**无 headroom** 时必须**如实报告为不可测**，而不是"先劣化再恢复"。
5. **不得**把 G3 数字与历史 arch-1 台账（旧锚 + R0 头）直接比较。
6. **不得**因为结果不合预期而改动 `phase6_g3_ledger.py` 的阈值；任何修改必须进本文附录并说明理由。

---

## 6. 工程缺口（本轮审计发现，未修）

| # | 项 | 风险 | 处置 |
|---|---|---|---|
| E1 | `baselines/{TriLiteNet,TwinLiteNet,TwinLiteNetPlus}` 被 `.gitignore` 第 92 行整目录忽略，`git ls-files` = **0**，且仓库内**无 vendored 清单**记录上游 commit | 对标表**无法从仓库自证**（"引用静默失效"家族）。三个嵌套仓库的 HEAD：TriLiteNet `4ac4947`(2025-07-15)、TwinLiteNet `5241418`(2025-03-10)、TwinLiteNetPlus `90f1b86`(2026-05-05)；工作树脏改动**只有 `__pycache__` 噪声**，源码未被改动 | 待办：加 `docs/PHASE6_VENDORED_BASELINES.md` 记录 `owner/repo@commit` |
| E2 | arch-1 台账 CSV 里记录的 `git_commit`（`51b68394`、`d0ce3e55`、`81a09cb9`）在当前仓库中 **`git log` 查不到** | 台账溯源断链（历史被重写/压缩所致） | 待办：在文档中标注为"不可达 commit"，或以现有结果表为准 |
| E3 | `evaluate_baseline.py` 用 `strict=False` + 形状过滤载入权重 | 架构不匹配时**静默**只载入部分键，指标只是"难看"而不报错 | **已修**：G3 runner 增加 fail-closed 断言，要求 `missing=0` |

---

## 7. 已就绪的执行件

| 文件 | 作用 | 状态 |
|---|---|---|
| `models/encoder/ir_encoder.py` | arch-2 逆残差 encoder（新拓扑） | 已写、契约验证通过 |
| `models/encoder/factory.py` | `build_encoder` 按 `arch` 分派，**缺省仍返回 `LightEncoder`** | 已写 |
| `models/static_model.py` | **唯一一处训练路径改动**：`LightEncoder(enc_cfg)` → `build_encoder(enc_cfg)` | 已改 |
| `scripts/phase6_g3_arch_probe.py` | 等价性（逐位）/ 契约 / 档位链 / 算子清单 | ✅ 全绿 |
| `scripts/phase6_g3_tier_solve.py` | 解两架构的档位链 + 生成 6 个配置 + 读回校验 | ✅ 全绿 |
| `scripts/phase6_g3_anchor_forensics.py` | 从 checkpoint 追锚框（结论：锚框不入 state_dict，改用 `config.yaml` mtime 取证） | 已写 |
| `scripts/phase6_g3_ledger.py` | 台账 + R0–R4 判定 | 已写 |
| `scripts/phase6_g3_ledger_selftest.py` | 合成数据自测判据 | ✅ 5/5 通过 |
| `scripts/phase6_g3_run.sh` | 单单元 train+eval+写表（含 VRAM 闸门、`missing=0` 断言、readback） | 已写 |
| `scripts/phase6_g3_chain.sh` | 链式驱动（`core` / `plus-seeds`），**拒绝与 seed 链同时占用 GPU** | 已写 |
| `configs/phase6_g3_*.yaml` | 6 个单元配置（生成 + 读回校验） | ✅ 已生成 |

**等价性证明（默认路径未被改动）**：`encoder_forward_maxdiff = 0.0`、
`model_forward_maxdiff = 0.0`、`state_dict` 键与参数量完全一致（201366 = 201366）。
即在**无 `arch` 键**的配置上，补丁后的行为与补丁前**逐位相同**。

---

## 8. 什么会推翻结论

- 若 a2 的 **R0 闸门失败** → 迁移研究对这个架构**不可识别**，如实写"不可测"，不调档位去凑。
- 若 a2 出现 **`Z_WINS`** → 强有力的负结论：**"编码器优先"不是架构无关的分配律**，
  而是 arch-1 家族的特性。**这正是 G3 存在的意义**，照实报告。
- 若 a2 与 a1 **符号系统不一致**（<3/6 同号） → `NOT_TRANSFERRED`。
- 若两边都 `DOMINATES` 但 **C 阶段显示差异落在噪声内** → 只能声称"序一致"，
  **不得**声称"显著性跨架构成立"。

---

## 附录 A —— 本轮回合实测依据

- 锚框默认值变更 commit：`c8aca35`（2026-09-08 23:16:11 +0800），前值内联文档于
  `models/representation/det_from_z.py:40-56`。
- 运行启动时间（`config.yaml` mtime）：`expD_z16_e20` 09-04 23:34、`expD_z128_e20` 09-05 06:42、
  `exp3A_esmall_z16` 09-05 11:12、`exp3A_elarge_z16` 09-05 13:10、`exp4A_r2_z16/z32/z128`
  09-07 15:35 / 10:48 / 13:07、`B100` 09-13 18:25、`B100_s1` 09-14 13:32。
- 锚框制度的配对实测：`experiments/phase6/phase6_round4_results.csv`
  `R4-A0lean_old` seed1/2 mAP50 = 0.3452 / 0.3521 与 `R4-A0lean_km` seed1/2 = 0.4984 / 0.4938
  （两者 params 0.2014 M、FLOPs 1.0796 G、20ep 完全相同）。
- 档位链与配置校验：`experiments/phase6/g3/g3_tier_solve.json`、`g3_arch_probe.json`。
