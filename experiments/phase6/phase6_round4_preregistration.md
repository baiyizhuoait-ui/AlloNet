# Phase 6 · Round 4 预注册 — Final Bottleneck-Aware Architecture

**冻结时刻：** 2026-09-11 17:06 (CST)　**基线提交：** `13b670b`
**上游判决：** `experiments/phase6/phase6_round3_decision.md`（Case C → REVISE）
**状态：** 判据在任何 Round 4 训练指标存在**之前**写死；本文档 §12 起为 append-only 修订区。

---

## 0. 本轮的定位变更（必须先读）

用户 brief 给出的 provisional principle 是：

> Detection → supervision/assignment bound；Lane → spatial bound；DA → no clear bottleneck

并附了一张「Detection=supervision-aware / Lane=spatial-aware / DA=baseline」的示意图，同时明确写道
**「这不是要求你照着这个结构机械实现。必须根据 Round 2–3 的实际结果决定最终 topology。」**

而 Round 3 的实际判决是：

| block | Round 3 判决 | 含义 |
|---|---|---|
| H1.instrument | **SUPPORTED**（47.6× floor，零训练精确） | 锚框/分配是检测的瓶颈 |
| H1.accuracy | **SUPPORTED**（20.8× floor，双架构） | 同上，跨架构 |
| **H2.lane（spatial > channel）** | **WEAK SUPPORT（0.20× floor，非单调）** | **未成立，也**未**证伪** |
| DA | 无架构瓶颈（3A/4B/5 一致） | — |

因此本轮**不能**把「lane 是 spatial bound」当作已建立的 premise 去照着搭 1/4 支路。Case C 的处置是
**REVISE（重新推导架构假设）**，且 novelty matrix 已把
`task-conditional channel-vs-spatial allocation` 标为 **DOWNGRADED**。

**本轮的重新推导（本文档的核心，替代照抄示意图）：**

Round 3 之外还有一条更硬、此前未被组织起来的证据 —— `experiments/phase6c/phase6c_e9_decision.md`（EXP-09C，20ep）：

```
低容量 Δ_anchor = +0.1428 (r2_z16 : old 0.3543 -> k-means 0.4971)
高容量 Δ_anchor = -0.0007 (A-uniform : old 0.5339 -> k-means 0.5332)
interaction     = -0.1435   (2σ_20 = 0.0096)
```

即：**监督质量与表示容量在争同一块 headroom，互为替代品。**
把它与上述判决合起来读，得到的不是「哪些任务需要哪种模块」，而是**每个维度上「一份增益要花多少算力」的价目表**：

| 维度 | 现状读法 | 边际价格（实测） |
|---|---|---|
| **supervision / assignment（检测）** | H1 双架构 SUPPORTED | **0 FLOPs，+0.1365 mAP50**（lean 容量档，seed0） |
| **capacity（检测）** | EXP-09C 替代律 | **+0.5854 GFLOPs / +65.8% params 买 +0.0424 mAP50** |
| **operating resolution（lane）** | 本轮 EXP-10 要测的 | 待测；预估极廉（见 §4 成本表） |
| **channel width（lane）** | lch64 20ep lane_fg +0.0004；TLP 58× params → +0.031 lane_mIoU | 近 0 收益 |
| **远处语义（DA）** | H-18 SUPPORTED，非容量 | 不可用架构修 |

**因此本轮的科学问题不是「哪个任务需要哪种模块」，而是：**
> 在极端压缩的多任务模型里，**每一条可用的资源维度都有各自的边际价格；按边际价格而不是按统一扩宽去分配算力，是否能在等预算下取得更优的任务组合？**

架构只是这个问题的**载体**，不是主张本身。若 EXP-10 显示 lane 的边际价格也接近 0（"thin 1/4 保留全部增益"），
则最终架构的正确形态是**比 uniform baseline 更便宜**，而不是更高的分数 —— 这是 §7 F6 要单独检验的分支。

---

## 1. 文献碰撞检查（本轮实际执行，2026-09-11）

三条新增检索，只针对本轮真正要主张的两条：*"supervision 与 capacity 是替代品"* 与 *"task-conditioned 资源分配"*。

| 方向 | 碰撞源 | 对本轮主张的影响 |
|---|---|---|
| 非对称 / task-aware 投影 | **MT-TPPNet**（Computers 2025）提出 "Asymmetric Projection with Expanded-value (APEX)" + "Selective Channel–Spatial Coupling (SC²)" | **命名与机制均撞车**。"asymmetric" 与 "channel–spatial" 这两个词本身**不可作为新颖点**；APEX/SC² 是 attention-bias 组合，与我们"由实测瓶颈价目表推导"的推导链不同 |
| 逐任务维度路由 | **MDANet**（IEEE TITS 2026）"Triple-Dynamic Sampling" 显式 decouple spatial/channel/semantic 三个维度做上采样 | **最接近的碰撞**。它已把"按维度分配"当设计语言。差异必须落在：**预算配平对照 + 由测量推导 + 替代律**，而不是"我们也有多维度" |
| 分配质量对小目标 | **YOLO26 STAL**（2026，Ultralytics）Small-Target-Aware Label Assignment + one-to-many/one-to-one 双头；DALA/RLA/IHANet/eRFOTA 一族 | "分配质量对小目标重要"是**已知且已工业化**的。我们的主张**不得**是"我们修好了分配" |
| 监督↔表征强度 | NMS-aware alignment（Neurocomputing 2022）："one-to-one provides less foreground information, making the network difficult to learn strong representations" | **方向性已被隐含陈述（定性）**。我们的差异只能落在：**定量、固定 params/FLOPs、双架构、并转成分配规则**这一层 |
| 边缘算力目标 | MaixCAM2 = AX630C，3.2 TOPS INT8 / 12.8 TOPS INT4；YOLO11n 640² 达 113–140 FPS | 本模型 0.19–0.33M / 1.0–1.7G ≈ YOLO11n 的 ~1/4 算力、~1/10 参数 → **边缘可行性充裕**，且"在 3.2 TOPS 上买不起容量、买得起监督"正是本工作最现实的应用动机 |

**由碰撞推出的表述红线（写死）：**
1. 不主张 "we propose an asymmetric module"。
2. 不主张 "we fix label assignment"。
3. 不得使用 APEX/SC²/MDANet 式 attention/动态模块；本轮**零新增模型代码**（见 §3 规则 R0）。
4. 可主张的只有：**边际价格表 + 替代律 + 预算配平的分配规则**，且必须在等预算下可复现。

---

## 2. 证据基座（可用 / 不可用）

**可用（20ep，磁盘实测，全部 bs16 + tri_train 69 863 图）**

| cell | 构造 | params | FLOPs | mAP50 | lane_mIoU | seeds |
|---|---|---|---|---|---|---|
| C1 `A0lean_old` | lean + old anchors | 0.2014M | 1.0796G | 0.3543 | 0.5847 | **0 only** |
| C2 `A0lean_km` | lean + k-means | 0.2014M | 1.0796G | 0.4908 / 0.4982 / 0.4960 | 0.5844 / 0.5824 / 0.5813 | **3** |
| S_old `l14f1` | lean + full 1/4 + old | 0.2019M | 1.6391G | 0.3612 / 0.3553 / 0.3600 | 0.5988 / 0.5954 / 0.5999 | **3** |
| S_km `combo` | lean + full 1/4 + km | 0.2019M | 1.6391G | 0.5047 / 0.4931 / 0.4964 | 0.5962 / 0.5938 / 0.5952 | **3** |
| C3 `Ufloor_old` | uniform + old | 0.3339M | 1.6650G | 0.5339 | 0.5879 | 0 only |
| C4 `Ufloor_km` | uniform + km | 0.3339M | 1.6650G | 0.5332 | 0.5874 | **0 only** |

**不可用（必须记录，不得当作证据）**
- H2 在第二架构（TwinLiteNetPlus-small）上的读数：**WEAK**，且该 lane 头已饱和（58× params 只 +0.031）→ **本轮不得把 TLP 当作 lane 轴的第二测试床**，也不能引用它作支持或反驳。
- 任何 4ep 读数：本项目在案两次 4ep→20ep 反转。4ep **只做 falsification / 排雷**，不作主张。
- `round3_invalid_evidence/` 全部内容。

---

## 3. 架构设计规则（本轮自我约束，可审计）

- **R0 — 零新增模型代码。** 所有档位只调 `lane_res` / `lane_use_f1` / `lane_hidden` / 编码器宽度。理由：brief 的 Rule 1 要求每个计算单元对应一条瓶颈证据；目前**没有任何一条证据要求新增层**（lch64 证明宽度无效；R4_uponly 将检验 lateral 是否必要）。若 R4_uponly 显示 lateral 不必要，则该单元应被**移除**而不是保留。
- **R1 — 每个候选必须有实测成本。** 见 §4，成本在训练前测量并冻结。
- **R2 — 单一变量。** 每一对要比较的 cell 之间的模型树差异必须被 `scripts/phase6_round4_isolation_check.py` 证明，不允许口头声明。已通过（EXP-11 对：params 逐位相同、FLOPs 四位相同、模型树 delta = `{detection.anchors}`；EXP-10 档：delta ⊆ `segmentation.*`）。
- **R3 — 尺子固定。** 全体 Round 4 cell 共用 byte-identical 的 k-means 锚框集；分配规则保持 `j[ai] = (ratio < 2.0).all(-1) & (ratio > 0.5).all(-1)`（`losses/yolo_loss.py:106`，sha256 前 16 位 `0835b8a4b5e8ecf8`）不动。
- **R4 — 不引入任何后处理、attention、SE/CBAM、transformer、动态路由、额外融合。**
- **R5 — 不以 FPS 作判据**（项目约定）；FPS/latency 只作辅助报告。

---

## 4. 实测成本模型（零训练，`scripts/phase6_round4_cost_model.py`）

`experiments/phase6/phase6_round4_cost.csv`，640×640，CPU 前向 + `profiling/flops_real.count_flops`：

| tag | 构造 | params | FLOPs | ΔFLOPs vs 1/8 | ×基线 |
|---|---|---|---|---|---|
| `R0_no_repair_1_8` | lean, lane 1/8 h32 | 0.2014M | **1.0796G** | — | 1.000 |
| `R3_min_1_4_h8` | lean, 1/4 h8 | 0.1896M | **1.0174G** | **−0.0623G** | 0.942 |
| `R2_thin_1_4_h16` | lean, 1/4 h16 | 0.1926M | **1.1656G** | +0.0860G | 1.080 |
| `R4_uponly_1_4_h32` | lean, 1/4 h32, 无 lateral | 0.2014M | 1.6129G | +0.5333G | 1.494 |
| `R1_full_1_4_h32` | lean, 1/4 h32 (= l14f1) | 0.2019M | 1.6391G | +0.5595G | 1.518 |
| `U_floor_km` | uniform, 1/8 h32 | 0.3339M | 1.6650G | +0.5854G | 1.542 |
| `U_min_km` | uniform, 1/4 h8 | **0.3224M** | **1.6158G** | +0.5362G | 1.497 |

**成本模型本身就是一个结果：**
> 把 lane 头从 1/8 h32 换成 1/4 **h8**，FLOPs 反而**下降 5.8%**，同时把运行分辨率提高 4 倍。
> 也就是"破 1/8 天花板"的最小算力代价可能是**负数** —— 前提是宽度那一半算力本来就是浪费的（lch64 已证）。

**pre-registered 预测（在任何 Round 4 训练前写死）**

- **P1**：R2_thin（+0.086G）保留 R1_full（+0.560G）的 lane 增益的绝大部分 → 「分辨率是有效变量、宽度是无效变量」。
- **P2**：R3_min（−0.062G）仍显著优于 R0（1/8）。若成立，"最小算力破天花板 = 0 或负"。
- **P3**：R4_uponly ≈ R1_full（20ep）→ **lateral 这个计算单元不必要**（Architecture Design Rule 1 的移除分支）。4ep 时两者无法区分（0.5836 vs 0.5839），20ep 才可判。
- **P4**：thin 1/4 对检测**无影响**（lane 支路不参与 det 路径）→ det 数字应与 R0 在噪声内一致。

---

## 5. 最终竞争模型定义（EXP-12）

三个模型 + 一个"近乎免费修复"变体，全部按 §4 实测成本列出：

| 名称 | 定义 | params | FLOPs | 备注 |
|---|---|---|---|---|
| **Model U** | uniform 编码器（×1.40）+ 1/8 头 + k-means | 0.3339M | 1.6650G | 容量路线；seed0 已有，本轮补 s1/s2 |
| **Model S** | lean + **full** 1/4（h32 + lateral）+ k-means | 0.2019M | 1.6391G | 朴素 spatial-heavy（= Round 2 的 combo）；3 seeds 已有 |
| **Model B** | lean + **thin** 1/4（h16 + lateral）+ k-means | 0.1926M | **1.1656G** | 本轮主张的 bottleneck-aware 形态；**待训** |
| **Model U⁻** | uniform 编码器 + 1/4 **h8** + k-means | **0.3224M** | **1.6158G** | 在两条轴上都不劣于 U，且带 1/4 lane；**待训** |

> 与 brief 的表 `Model U / S / B` 对应关系：U↔U，S↔S，B↔B；U⁻ 是证据推导出的**额外**竞争点，
> 若它成立，则"非对称分配"这一侧的主张可以完全不依赖"牺牲检测精度"。

**等预算的现实（必须如实陈述）**：params 与 FLOPs 不可能同时配平 ——
uniform 把算力花在**通道宽度**（参数重、空间轻），non-uniform 把算力花在**1/4 的像素**（参数轻、空间重）。
`U` 与 `B` 的 FLOPs 差 30%、params 差 42%。因此本轮**不追求单一配平标量**，而是
(a) 报告 (params, FLOPs, 任务得分) 的 **Pareto 前沿**，(b) 增加 U⁻ 这个**同编码器内的单变量对照**（U vs U⁻ 只差 lane 头），
使"分配形状"这一维至少在一个干净的对内可判。

**禁止**：用一个任意加权的标量分数下结论（见 §7 的判据设计）。

---

## 6. 实验格与臂（全部 20ep @bs16，tri_train）

### EXP-10 — Efficient spatial bottleneck repair（lane 算力阶梯）

| 格 | 构造 | 状态 | FLOPs |
|---|---|---|---|
| R0 | lean 1/8 h32 + km | 已有（C2，3 seeds） | 1.0796G |
| **A = R1_full** | 1/4 h32 + lateral | 已有（combo，3 seeds） | 1.6391G |
| **B = R2_thin** | 1/4 h16 + lateral | **新训** | 1.1656G |
| **C = R3_min** | 1/4 h8 + lateral | **新训** | 1.0174G |
| **R4_uponly** | 1/4 h32，**无** lateral | **新训**（机制隔离） | 1.6129G |

报告：lane_mIoU、lane foreground IoU、params、FLOPs、latency、VRAM，及 **Δlane_mIoU / ΔFLOPs**。
目的：找**破 1/8 天花板的最小算力**，而不是找最高 lane 分。

### EXP-11 — Detection supervision integration（实现隔离）

对照：`{old anchors} × {k-means}`，在 **lean** 与 **uniform** 两个容量档上（C1/C2 与 C3/C4）。
隔离已由 §3 R2 证明（params 逐位相同、FLOPs 四位相同、模型树仅 `detection.anchors`）。
报告：mAP50、mAP50-95、per-size AP50/AP95/recall50、n_pred。
**不重新发明锚框算法**；k-means 本身在 novelty matrix 中已标 BANKED / not a claim。

### EXP-12 — Final equal-budget architecture

U / S / B / U⁻ 四行 × 3 seeds（U⁻ 若 U 的 3 seeds 完成后仍有预算再加），
按 §4 Pareto 报告，附 efficiency-adjusted utility：
`mAP50/FLOPs`、`mAP50/params`、`lane_mIoU/FLOPs`、`lane_mIoU/params`（逐任务，不合成标量）。

### EXP-13 — Ablation（2×2，已完成 3/4）

| 臂 | 构造 | 3 seeds 状态 |
|---|---|---|
| Baseline | lean + old anchors（C1） | seed0 有；**本轮补 s1/s2** |
| +Detection component | lean + k-means（C2） | 3 seeds 已有 |
| +Lane component | lean + full 1/4 + old（l14f1） | 3 seeds 已有 |
| +Both | lean + full 1/4 + k-means（combo） | 3 seeds 已有 |
| （+ full model） | = Model B（thin 1/4 + km） | 来自 EXP-10 |

本轮**只补 2 个臂**（baseline 的 s1/s2），不做其它无意义 ablation。

### 本轮新训臂清单（8 臂 ≈ 19–20 GPU·h）

| # | tag | config | seed | cell |
|---|---|---|---|---|
| 1 | `R4R1up` | `phase6_r4_R4_uponly14_z16` | 0 | R4-R1up14 |
| 2 | `R4R2thin` | `phase6_r4_R2_thin14_z16` | 0 | R4-R2thin14 ← Model B |
| 3 | `R4R3min` | `phase6_r4_R3_min14_z16` | 0 | R4-R3min14 |
| 4 | `R4A0s1` | `phase4a_r2_z16` | 1 | R4-A0lean_old |
| 5 | `R4A0s2` | `phase4a_r2_z16` | 2 | R4-A0lean_old |
| 6 | `R4Umin` | `phase6_r4_U_min_km` | 0 | R4-Umin_km ← Model U⁻ |
| 7 | `R4Us1` | `phase6c_e9_aunif_km` | 1 | R4-Ufloor_km |
| 8 | `R4Us2` | `phase6c_e9_aunif_km` | 2 | R4-Ufloor_km |

阶段闸门：**A（臂 1–3）先跑**，读完才决定 B/C/D 是否需要（`touch experiments/phase6/round4/STOP_CHAIN` 可任意臂间停机）。

---

## 7. 冻结判据（在任何 Round 4 指标存在前写死）

**噪声标尺**（由已有 20ep 多 seed 数据 pooled within-arm sd 计算，不使用 Round 4 数据）：

| 量 | σ | 2σ |
|---|---|---|
| mAP50 | 0.0048（EXP-08 §3.1） | **0.0096** |
| lane_mIoU | 0.0019（l14f1 3 seeds + combo 3 seeds pooled） | **0.0038** |
| lane_fg_iou | 0.0031（同上） | **0.0062** |
| da_mIoU | 0.0011（同上） | **0.0023** |

**参考水平（已有实测，冻结为常数）**

- `lean+km` lane_mIoU 3-seed mean = **0.5827**；`lean+full14+km`（combo）lane_mIoU 3-seed mean = **0.5951**
- `combo` mAP50 3-seed mean = **0.4981**；`lean+km` mAP50 3-seed mean = **0.4950**
- `lean+old` mAP50 seed0 = **0.3543**

| ID | 判据（seed0，除非注明） | 通过条件 |
|---|---|---|
| **F1** | thin 1/4 是否保留 full 1/4 的 lane 增益？ | `lane_mIoU(R2_thin) ≥ 0.5951 − 0.0038 = **0.5913**` |
| **F2** | 修复本身是否真实（相对 1/8）？ | `lane_mIoU(R2_thin) ≥ 0.5827 + 0.0038 = **0.5865**` |
| **F3** | 监督集成的幅度（lean 档，3-seed 后） | `mAP50(lean+km) − mAP50(lean+old) ≥ **0.0096**` |
| **F4** | 最小算力档是否也成立？ | `lane_mIoU(R3_min) ≥ **0.5865**`（同 F2） |
| **F5** | lateral 是否必要？ | 若 `lane_mIoU(R4_uponly) ≥ 0.5951 − 0.0038` → lateral **不必要**，最终形态应移除它 |
| **F6** | Model B 是否与 Model S 等效用但更便宜？ | `lane_mIoU(B) ≥ 0.5951 − 0.0038` **且** `mAP50(B) ≥ 0.4981 − 0.0096 = 0.4885` **且** `|da_mIoU(B) − da_mIoU(S)| ≤ 0.0023` → 判定 "B ≡ S at −28.9% FLOPs" |
| **F7** | 检测是否被 lane 支路影响（P4 的检验） | `|mAP50(B) − mAP50(lean+km)| ≤ 0.0096` |
| **F8** | U⁻ 是否严格支配 Model U？ | `FLOPs(U⁻) ≤ FLOPs(U)` **且** `lane_mIoU(U⁻) ≥ lane_mIoU(U) + 0.0038` **且** `mAP50(U⁻) ≥ mAP50(U) − 0.0096` |

**判级（机械，`scripts/phase6_round4_report.py` 编码）**

- 全部 F 不通过 → Level 1（只有模块，无原则）。
- F3 通过、F1/F2 不通过 → Level 2（发现 task-specific bottleneck，但不足以成为模型主张）。
- **F1 ∧ F2 ∧ F4 通过 ∧ EXP-11 隔离成立 ∧ EXP-13 四格齐备（3 seeds）→ Level 3。**
- **Level 3 ∧ Round 3 的双架构 H1 成立 ∧（F6 通过 或 F8 通过）→ Level 4。**
- F5 通过时，最终架构**必须**移除 lateral；若报告仍保留它，需要显式说明理由（否则算 Level 降级）。

**停机**：一旦 Level 3 或 Level 4 成立，**停止搜索新模块**，转入写最终文档，不再启动新训练。

---

## 8. Seed 协议与预算纪律

- baseline（C1）与最终架构（Model B / U / U⁻）**至少 3 seeds**；已有 3 seeds 的 cell 不重跑。
- 4ep 只允许用于排雷，**不作主张**。
- 所有最终 performance claim 必须基于 20ep（本轮冻结预算：69 863 图 × 20ep @bs16，lr 1e-3，与 C1–C4、l14f1、combo 完全一致）。
- 不允许在结果出来后改 lr / 改预算 / 改噪声标尺。任何修订进 §12（append-only）。
- FPS 不作判据。

---

## 9. 边缘部署评估（零额外训练）

目标平台：**MaixCAM2 / AX630C（3.2 TOPS INT8, 12.8 TOPS INT4）**，参考点 YOLO11n 640² 在该平台 113–140 FPS。

报告项（不虚构实测 FPS）：
1. **Params**（实测）与 fp32/INT8 模型体积；
2. **MACs / FLOPs**（实测，`flops_real`）；
3. **activation memory**：逐层峰值由 forward hook 实测（脚本内实现，不估算）；
4. **INT8 结构兼容性**：算子清单审计 —— 必须全部落在 `conv2d / batchnorm / relu / interpolate / sigmoid / concat` 内；**无 LayerNorm/Softmax/attention/dynamic-shape 算子**；
5. **对照声明**：`我们的 FLOPs / YOLO11n FLOPs` 与 `我们的 params / YOLO11n params`，**只报相对比，不报我们自己的 FPS**。

---

## 10. 最终产物（生成方式先写死，数据后填）

| 文件 | 生成者 | 内容 |
|---|---|---|
| `experiments/phase6/phase6_final_models.csv` | `scripts/phase6_round4_report.py` | U / S / B / U⁻ 四行 × seed，全部指标 + 成本 |
| `experiments/phase6/phase6_ablation.csv` | 同上 | EXP-13 五臂 × 3 seeds 的 Δ 表 |
| `experiments/phase6/phase6_efficiency.csv` | 同上 | per-task utility（mAP50/GFLOPs、lane_mIoU/GFLOPs、…/params）+ Pareto 标记 |
| `experiments/phase6/phase6_final_statistics.csv` | 同上 | F1–F8 的判定值、参考常数、比值、通过与否 |
| `docs/PHASE6_FINAL_ARCHITECTURE.md` | 手写（数据后） | finding / bottleneck map / principle / final architecture / equal-budget / cross-arch / ablation / efficiency / limitations / novelty risk / recommendation |
| `docs/PHASE6_FINAL_DECISION.md` | 手写（数据后） | Level 1–4 判定 + Q1–Q5 论文定位测试 + 停机决定 |
| `phase6_experiment_registry.csv` | 追加 | EXP-10~13 |
| `phase6_architecture_hypotheses.csv` | 追加 | H-36（替代律）→ 状态、H-32/H-33 重新推导后的状态 |
| `phase6_novelty_matrix.csv` | 更新 | 依据 §1 红线重写措辞；删除 "asymmetric module" 类主张 |

**报告生成脚本必须在数据存在前写好并自检**（沿用 Round 3 的纪律：判决逻辑先于指标）。
门禁：任一 cell 缺失 → 脚本**拒绝**输出 Level 判定（fail-closed，不写"部分结论"）。

---

## 11. 诚实性限制（必须随结论一并陈述）

1. **lane 轴的第二架构失效**：TLP-small 的 lane 头在其家族内已饱和，Round 3 的 H2 = WEAK。本轮的 lane 结论**只有单架构（lean R2 家族）支撑**；Level 3/4 若依赖 lane 轴，必须显式标注"单架构"。
2. **容量轴不干净**（沿袭 EXP-09C §5.1）：U 与 B 的差异同时含"容量多少"与"容量加在哪一维"，**不可表述为参数量的作用**。
3. **U 与 B 不可能同时配平 params 与 FLOPs**（§5），故 Pareto 前沿是唯一诚实的比较形式。
4. **锚框集由训练集拟合**：k-means 在 tri_train 上拟合，存在轻微的训练集信息优势；该优势对四个模型一致，不影响模型间比较，但影响与"不调锚框"基线比时的外推。
5. **20ep 老锚框臂在退化**（C1：0.3879@4ep → 0.3543@20ep），Δ_anchor 含"收敛/退化"成分。
6. **n=1 的风险**：EXP-10 的 R1/R2/R3/R4_uponly 均为 seed0 单次；只有 F1/F2/F4 通过后才值得补 seed。**不得**用单 seed 做 3σ 级陈述。
7. **`peak_gpu_mem_mib` 的历史列为 NA**（旧 run.sh 抓 `peak_mem=` 字段，R2 训练器不打印该格式）；本轮 runner 已改为解析 `mem a/bMiB` 的**分子**，Round 3 之前的 NA 不回填。

---

## 12. 时间线与预算

| 阶段 | 内容 | GPU 时间 |
|---|---|---|
| R4-A（先跑） | EXP-10 阶梯：R1up / R2thin / R3min | ≈ 7.0 h |
| R4-B | EXP-13 baseline seeds s1/s2 | ≈ 4.5 h |
| R4-C | Model U⁻ | ≈ 2.7 h |
| R4-D | Model U seeds s1/s2 | ≈ 5.4 h |
| 合计 | 8 臂 | **≈ 19.6 h** |
| 报告 | 零 GPU（机械判决 + 文档） | 0 |

墙钟 ≈ 2 天（本机 8151 MiB 下 0.19–0.33M 模型峰值 ≈ 2.4–2.9 GiB，远离 Round 3 的 96% 悬崖阈值，无租卡需求）。
断点续训已就绪（每 epoch 落盘 + `--resume`），掉电最坏损失 1 个 epoch（≈7 min）。

---

## 13. 修订区（append-only，任何事后修改必须在此登记并说明理由）

### 13.1 规划期发现的第 6 个登记类缺陷：H-34 行字段位移（2026-09-11，零训练）

在把 Round 4 写入 `phase6_architecture_hypotheses.csv` 时，逐行校验列数发现 **H-34 行有 13 格而不是 11 格**：
它的 `status` 值内含三个**未加引号**的逗号，于是 `status` 被拆成第 9/10/11 三格，真正的 `next_action` 落到第 12 格。

- **与 Round 3 的 D5 完全同类**（CSV 未引号 → 静默列位移），只是这次发生在**手写编辑**的登记表里，而不是脚本写出的结果表里。
- **危害**：任何按列位读取该行的消费者会拿到错位的字段；且它已经存在了一段时间，没有被任何自动检查捕获。
- **被谁抓到**：Round 4 的登记脚本在写回后做了 `all(len(row) == 11)` 断言 —— 不是肉眼。
- **修复**：`experiments/phase6/round4_register.py` 把 9..11 重新拼回 `status`、把 12 移入 `next_action`，并**整表经 `csv.writer` 重写**（此后自动加引号）；附带把 H-34 的状态更新为「4ep REJECTED → 20ep 被 EXP-09C 直接反证（interaction −0.1435，轴是替代关系）→ 由 H-36 取代」。
- **教训（工具层）**：**手写 CSV 的字段位移不会被阅读发现**。任何对活登记表的写入都必须配 `csv.writer` + 读回列数断言；本轮把这一步固化进登记脚本。

### 13.2 本轮新增登记

- `phase6_experiment_registry.csv`：新增 **EXP-09**（此前 Round 3 的结果从未登记，属遗漏，本轮补上）、EXP-10、EXP-11、EXP-12、EXP-13。
- `phase6_architecture_hypotheses.csv`：新增 **H-36**（监督-容量替代律，EXP-09C 建立）与 **H-37**（边际价格分配 = 本轮主假设）。
- `phase6_novelty_matrix.csv`：更新 4 行状态（H-32/H-33 模块优先读法 **DEAD** → 重述为 H-37；channel-vs-spatial **DOWNGRADED**；H-34 **SUPERSEDED**；跨架构监督瓶颈 **ESTABLISHED**，并显式记入其弱点），新增 2 行（H-36、H-37）。
- 措辞红线（§1）已随更新一并生效：novelty matrix 中不再存在 "asymmetric module" 类主张。

### 13.3 执行后发现的第 7 个缺陷：D6 显存守卫结构性哑火（2026-09-12，零训练）

- **现象**：Round 4 全部 8 臂的守卫行都打印 `peak mem 0MiB / 0MiB = 0%`。
- **复现（非猜测）**：`grep -aoE "mem +[0-9]+/[0-9]+MiB"` 保留了字面量 `mem`，经 `tr '/' ' '` 后 awk 看到 `$1="mem"`，于是 `$1+0 = 0 > m` 永假，峰值**永不更新** → 恒为 `0 0 0`。已在一次调用中同时跑出新旧两种解析作单元测试：新式 `peak 3338/8151 = 40%`，旧式 `0 0 0`。
- **结论**：该 fail-closed 守卫**从设计上不可能触发**。
- **后果（实测，非估计）**：8 臂真实峰值最高 **41.0%**（R4Umin 3338/8151 MiB），最低 31.6%；每臂墙钟 132–167 min，与正常速率一致，**无静默换页迹象**。故 Round 4 结果本身**未被污染**，但安全网是假的。
- **修复**：`scripts/phase6_round4_run.sh` 改为只匹配数字（`[0-9]+/[0-9]+ ?MiB`），并新增**零样本即 ABORT**——「未知」不得被记作「安全」。
- **附带修复**：`peak_gpu_mem_mib` 列自此写入真实值（此前因 D6 一律写 0，Round 4 表内该列不可用）。

### 13.4 D7：锚框来源错标（2026-09-12，零训练；只改标签，不改任何指标）

- **事实链**：`configs/phase4a_r2_z16.yaml` **无 `anchors` 块** → 落到代码默认；commit `c8aca35`(2026-09-08) 已把默认改为 IoU-k-means；而 `scripts/phase6_round4_run.sh` 的 `supervision` 参数**只作为 CSV 标签**，**不存在**任何据其选择锚框的代码路径。
- **因此**：标着 `old` 的两个 EXP-13 基线臂，实际训练的是 **k-means**。
- **佐证（指标层）**：两行 mAP50 = 0.4984 / 0.4938，落在 lean+km 带（0.4950）内，远离历史 old 单元（0.3543）。
- **处置**：标签 `R4-A0lean_old` → `R4-A0lean_km`（它们**恰是该 cell 缺失的 seeds 1/2**，缺陷因此转化为零成本补种）；`supervision` 同步由 `old` 改 `km`。**未改动任何指标**；经 `csv.writer` 重写 + 读回自证（行数不变、旧标签零残留、mAP50 未漂移）。
- **fail-loud**：`scripts/phase6_round4b_fix_d7.py` 在指标不匹配 k-means 签名时**拒绝改标签**——若本结论有误会被当场抓住，而不是被平滑掉。
- **真正的 old 单元**由新臂 `R4-A0lean_old` 产出，其配置 `configs/phase6_r4_lean_old.yaml` 把**历史锚框显式写进文件**（BOTH `model.detection.anchors` 与 `train.anchors`）；隔离性由 `scripts/phase6_round4b_isolation_check.py` 以解析字典 delta 证明 = `{model.detection.anchors, train.anchors}`。
- **对 F3 的影响**：Round 4 报告中的 F3 obs=0.0462 是**用错标基线算出的**；正确值为 0.1407（低容量 Δ_anchor）。报告器 cell 标签与 F3 须据此重算（零 GPU，下一步执行）。**重算完成前，F3 数值不得对外引用。**

### 13.5 Round 4B 新增臂 + 「是否租卡」的冻结判据（在任何 Round 4B 指标存在之前写死）

**背景**：需在周一前回答「是否租高性能卡跑 100 epoch」。

新增臂（每臂单旋钮，隔离性已证）：

| 臂 | 配置 | 目的 | 预计 |
|---|---|---|---|
| `R5-R2thin40` | R2_thin，**epochs 40**，seed 0 | **决定性**：20→40 的边际增益 = 租卡有无希望 | ≈4.7 h |
| `R6-R2thinL4` | R2_thin，**lr 1e-4**，20ep，seed 0 | 20ep 数字是「欠训练」还是「欠调参」 | ≈2.4 h |
| `R4-R2thin14` s1/s2 | R2_thin，20ep | Model B 补至 3 seeds（Level-3 要求） | ≈4.7 h |
| `R4-A0lean_old` s1/s2 | lean + **历史锚框** | EXP-13 补齐 2×2×3（D7 修复） | ≈4.6 h |

合计 ≈16.4 h（墙钟，本机顺序执行）。

**为什么探针是 fresh 40ep 而非 `--resume` 续训**：`training/train.py` 用 `CosineAnnealingLR(opt, T_max=epochs*len(loader))`，resume 时会 `load_state_dict(sched_state)` 把 `T_max` 与 `last_epoch` 一并恢复；20→40 续训将在 **LR 地板**上再跑 20 个 epoch，产出平线，从而**伪造**「没有余量」。fresh 40ep 的余弦在 40 视界内正确展开，20 与 40 两个端点都是各自收敛点——这才是该问题所需的对照。

**噪声标尺**（Round 3 冻结，不重估）：2σ = mAP50 0.0096 / lane_mIoU 0.0038 / lane_fg_iou 0.0062 / da_mIoU 0.0022。轴集合 A = {mAP50, da_mIoU, lane_mIoU, lane_fg_iou}。

**F9（租卡决策）**，比较 `R5-R2thin40` 与既有 R2thin(20ep, seed0)：

- **RENT（有余量）** ⟺ `∃ a ∈ A: Δ_a(40−20) ≥ 2σ_a` **且** `∀ a ∈ A: Δ_a(40−20) > −2σ_a`（无退化）。
- **STOP（已收敛）** ⟺ `∀ a ∈ A: |Δ_a(40−20)| < 1σ_a`。
- **AMBIGUOUS**：其余情形 → **不足以支持租卡**，须先看 F10。

**F10（预算归属）**：若 `R6-R2thinL4` 相对 R2thin(20ep, 1e-3) 取得任一轴 ≥2σ 增益，则该增益**首先**归因于**超参数**而非 epoch；此时即便 F9=RENT，**正确动作也是本地调参，而不是租卡**——在超参未定位前租卡，是在放大一个尚未定位的误差。

**声明边界（必须随结论一起引用）**：F9 只回答「我们的模型在更长训练下**有无余量**」。它**不回答**「能否支配已发表方法」——后者是结构性的：同为 ≈1.7-1.8G FLOPs 档，TriLiteNet tiny 的 DA mIoU 0.8796 vs 我们 0.8629；TLP nano 以 1/10 参数在 DA/lane 上与我们打平。**该差距不随 epoch 数改变。** 故 **F9=RENT ≠ Level 3/4**；本项目对外 claim 仍限定为「机制 / 成本规则」。

### 13.6 Round 4B 执行顺序

按**决策价值**排序（非按便利）：Stage 1（F9 决定性探针）→ Stage 2（F10 超参）→ Stage 3（Model B seeds）→ Stage 4（EXP-13 历史锚框）。任意臂边界可 `touch experiments/phase6/round4/STOP_CHAIN` 优雅停机。

---

### 13.7 结果记录（Round 4B 第一二臂，append-only，只记录不改判据）

**记录时刻：** 2026-09-13 00:20 (CST)

#### F9（40ep 相对 20ep）→ **RENT**

Model B（`R4R2thin`，1/4 h16，lr 1e-3，batch 16）在**完全相同**的 config 下，只把 epoch 从 20 提到 40：

| 轴 | 20ep | 40ep | Δ | σ | z |
|---|---:|---:|---:|---:|---:|
| mAP50 | 0.4954 | 0.5198 | +0.0244 | 0.0048 | **+5.08** |
| DA mIoU | 0.8528 | 0.8594 | +0.0066 | 0.0011 | **+6.00** |
| lane_mIoU | 0.5946 | 0.5992 | +0.0046 | 0.0019 | **+2.42** |
| lane_fg_iou | 0.2121 | 0.2203 | +0.0082 | 0.0031 | **+2.65** |

四轴全涨、零轴下跌 → 按 §13.5 冻结规则判 **RENT**。
40ep 训练 277 min、峰值 `self` 由 D9 修复后的守卫记录（见 13.8）。

#### F10（lr 1e-4 相对 lr 1e-3，同为 20ep）→ **NO GAIN（强负）**

| 轴 | lr 1e-3 @20ep | lr 1e-4 @20ep | Δ |
|---|---:|---:|---:|
| mAP50 | 0.4954 | 0.3648 | **−0.1306** |
| mAP50-95 | 0.2146 | 0.1358 | −0.0788 |
| DA mIoU | 0.8528 | 0.8301 | −0.0227 |
| lane_mIoU | 0.5946 | 0.5839 | −0.0107 |
| lane_fg_iou | 0.2121 | 0.1902 | −0.0219 |

**裁定**：F10 不但没有给出 ≥2σ 增益，反而在全部轴上大幅劣化（最小劣化 5.6σ）。
→ **瓶颈不包含"学习率尚未调好"这一项**；当前的 lr 1e-3 是正确的。
→ 与 F9=RENT 合并读：**余量来自训练预算（epoch），不是超参。**

该臂由此获得一个额外用途：它是**超参轴的阴性对照**——证明"我们只是没调参"这一替代解释被排除。

---

### 13.8 新缺陷 **D9**：显存悬崖守卫是"全卡口径"，会被同卡其他进程污染

**发现时刻：** 2026-09-13 00:16 (CST)，在核查 `R4R2thinL4` 守卫行 `peak mem 7692MiB / 8151MiB = 94%` 时发现。

**事实链（全部来自时间戳，非推断）：**

| 证据 | 值 |
|---|---|
| `/tmp/r4_gpu_probe.py` 写入时刻 | 23:36:48 |
| `/tmp/r4_bs_probe.py` 写入时刻 | 23:37:13 |
| 该日志中 17 条 >3000MiB 读数的分布 | 23:36:57 – 23:38:32（约 95 秒） |
| 同一日志紧邻前一条读数 | 23:36:53 `mem 2468/8151MiB` |
| 同一日志紧邻后一条读数 | 23:37:03 `mem 2468/8151MiB` |
| 全臂 4380 条读数中越界条数 | 17（0.39%），全部落在上述窗口 |

即：本会话在 **该臂训练进行中**（21:34–23:57）于**同一张卡**上跑了两个诊断探针（含大 batch 分配），而 `gpu_usage()` 的 `mem` 字段取自 `nvidia-smi --query-gpu=memory.used` —— **设备级口径**，因此把探针的占用记到了训练臂头上。

**严重性定级：无后果，但有近失（near-miss）。**
- 该臂真实进程峰值 ≈ **2468 MiB = 30%**，训练全程 96–104 ms/step 稳定，无换页迹象 → **结果有效**。
- 但 94% 距 96% 阈值仅 **132 MiB**。若探针多申请一点，守卫会在臂**已顺利完成之后**写下 `ABORTED_94pct` 并**停掉整条链**——损失一个 2.4h 的有效臂。**fail-closed 守卫变成了 fail-noisy 的杀伤源。**
- 污染窗口内步时从 98 → 126 ms 短暂上升（约 95 秒），随后回落；步数不变，对学习无实质影响（占 142 min 的 1.1%）。

**修复（已实施并单测）：**
1. `training/train.py::gpu_usage()` 新增**进程级**字段 `self <peak_alloc>/<peak_reserved>MiB`（`torch.cuda.max_memory_allocated/max_memory_reserved`）；**保留** `mem <used>/<total>MiB` 原样，以免打断 2026-09-13 之前所有解析该 token 的脚本（phase2b/2d/3a/3b/4a/4b/4c 等）。
2. `scripts/phase6_round4_run.sh` 守卫改为**只以 `self` 的 reserved 峰值 / 显存总量**做悬崖判定；若日志无 `self` 字段（旧格式）则回退到设备级并在日志中显式标注 `FALLBACK`。
3. 单测（`/tmp/d9_guard_test.sh`，合成日志）：
   - 含 `self` 的污染日志 → `peak SELF 2500/8151 = 30%`（正确忽略 7692 的污染读数）
   - 剥离 `self` 的同一日志 → `peak DEVICE-WIDE 7692/8151 = 94% (FALLBACK)`（回退正确）
   - **在跑的 `R4R2s1` 真实日志（旧格式，463 条样本）→ 2461/8151 = 30% → SAFE**（回退不会误杀）

**教训（与 D1/D5/D6 同类，第 4 次）：** 又一次"字段口径被静默接受"。前三次是**解析错位**，这一次是**测量口径错**（进程级 vs 设备级）。共同点：**守卫的读数从来没有被独立验证过**。已固化的规则：任何 fail-closed 守卫都必须附带一个**对照组单测**（正确输入→通过、污染输入→仍通过、超限输入→拦截），否则不予采信。

**操作约束（写给未来的自己）：** 训练链运行期间**不得在同一张卡上启诊断探针**。若必须测，用 `CUDA_VISIBLE_DEVICES=""` 走 CPU，或等臂间边界。

### 13.9 D9 的数据面修复：被同卡探针污染的显存样本（2026-09-13 02:5x，零训练，append-only）

13.8 修的是**代码**（守卫改用进程级 `self alloc/reserved`）；本节修的是修复前**已落表的数据**。脚本 `scripts/phase6_round4b_fix_d9_scope.py`（备份 `experiments/phase6/phase6_round4_results.csv.pred9.bak`，`csv.writer` 重写 + 读回断言）。

| 臂 | 字段 | 原值 | 修正为 | 依据 |
|---|---|---|---|---|
| R6-R2thinL4 | `peak_gpu_mem_mib` | 7692 | **2468** | 该臂共 4380 个显存样本，其中仅 **19** 个 ≥2500MiB，且**全部**落在 23:36:55–23:39:17 —— 正是本会话两个 GPU 探针的窗口（探针 mtime 23:36:48 / 23:37:13）。其余 **4361** 个样本 ≤2468MiB（其中 3814 个恰为 2468、544 个为 2461）。本臂自身真实峰值 = **2468MiB**，与同族 R4R2thin40（2468，8760/8760 样本）及 R4R2s1（2461）一致。 |
| R4-R2thin14 seed1 | `git_commit` | 131c763 | **ba7b23d** | R2s1 训练于 00:02→02:23，早于 131c763 的创建时间（00:18）；其日志含 **0** 条 `self` 样本，证明跑的是 D9 之前的 `train.py`。该行系事后补登记（见 13.10），自动写入的 HEAD 不是训练提交。 |

- **F10 是否被该污染推翻？没有**（接受结论前已核查，非事后辩解）：
  - `diff configs/phase6_r4_R2_thin14_z16.yaml configs/phase6_r4_R2thin_lr1e4_z16.yaml` 的实质差异**只有** `lr: 1e-3 → 1e-4` 与注释；batch / accum / schedule 全同 → F10 的**单旋钮性成立**。
  - 污染窗口 = 19/4380 样本（约 0.4% 步数、142 min 中的约 2.5 min）。LR 与批序均按 step 索引，卡变慢**不改变** MSE / mAP 数值，只污染墙钟（142 min 含约 2 min 争用）。
- **列读取规则**：`peak_gpu_mem_mib` 在最初 8 臂上为 **0**，那是 D6 的「未测量占位符」，**不是测量值**；只有 R5-R2thin40、R6-R2thinL4、R4-R2thin14/seed1 三臂（D6 修复后运行）有真值。

### 13.10 D10：一个 2.3h 臂不是死在显存，而是死在「改脚本」（2026-09-13 02:23，零训练）

- **现象**：4B 链在 R2s1 之后自报 `VRAM guard aborted R4R2s1 - stopping chain`，表面看像撞了显存悬崖。
- **真因**：`scripts/phase6_round4_run.sh` 在该链**运行期间**被修改（00:18 的 D9 修复）。bash 按**字节偏移**增量读取脚本文件，文件长度一变，偏移即错位，于是 02:23 读到半截字符串：

  ```
  scripts/phase6_round4_run.sh: line 93: syntax error near unexpected token `)'
  ... line 93: `s=${MEMN}; FALLBACK - no per-process field in this log)" | tee -a ...'
  ```

- **放大路径**：非零退出被链映射为 `rc=2`（= 显存守卫），于是打印出**假悬崖判决**；其后的 3 个臂因此从未启动。R2s1 的真实峰值是 2461MiB/8151MiB = **30%**。
- **损伤面**：R2s1 的训练（140 min）与评估（02:28）**均已完成且有效**，只丢了两张表的登记行。已按 13.9 用既有 `metrics.json` 事后补登记（墙钟 140 min、峰值 2461MiB 取自其自身日志），并在 `round4/round4_chain.log` 留 NOTE；**未重训任何臂**。
- **补登记结果（新增数据）**：R4-R2thin14 seed1 = mAP50 **0.4967** / mAP50_95 0.2168 / DA mIoU 0.8587 / lane mIoU 0.5936 / lane_fg 0.2105；对照 seed0 的同臂 20ep（0.4954 / 0.8528 / 0.5946），**mAP50 种子间差 0.0013，远小于 2σ=0.0096 噪声底**——这是 20ep 结论稳健性的第二个独立证据。
- **修复**：`scripts/phase6_round4b_chain.sh` 启动时把 `phase6_round4_run.sh` 复制为 `round4/run4_frozen.sh` 并**只执行该快照**。独立纪律：**永不修改正在被链执行的脚本**；要改就先改主体、再从快照重启。

### 13.11 D11：链无单实例锁 → 两个链并行踩同一目录（2026-09-13 02:34，零训练）

- **现象**：R2s2 训练约 2 分钟后**无 traceback** 地消失（`no metrics.json`）；同时出现两个 `train.py` 进程。
- **真因**：第一次启动的命令被工具 SIGTERM 掉，但 `setsid` 已把链放进**独立会话并存活**；第二次启动未察觉，于是两个链同时驱动同一 outdir 与同一 `train_<tag>.log`（`tee` 打开即**截断** → 双方日志互相覆盖，崩溃现场被抹掉）。
- **损伤面**：R2s2 / A0o1 的残留目录与日志已移入 `experiments/phase6/round4/dup_quarantine/`；窄表中 2 条 `failed-no-metrics` 垃圾行已删除（备份 `.predup.bak`）。**垃圾行必须删**——否则 `run.sh` 的幂等检查（按 cell+seed 判重）会把 R2s2 **永久 SKIP**。
- **修复**：链头加 `flock -n` 单实例锁（fail-closed，被占用则 `exit 3`）；实测第二次启动返回 **3**。附注：`setsid cmd` 会**等待**子进程，故包装器会一直挂着——这正是链的保活机制，不是卡死。

### 13.12 重启后的剩余链（2026-09-13 02:35:26 起）

- 已正确 SKIP：R5-R2thin40 seed0、R6-R2thinL4 seed0、R4-R2thin14 seed1。
- 正在跑：R4R2s2（20ep seed2）→ R4A0o1 → R4A0o2；单臂约 2.3h + 评估约 6 min，预计 **09:45–10:15** 收尾（周一截止线之前）。
- 收尾后仍为**零 GPU** 步骤：机械判级、四张表、两份 FINAL 文档。

### 13.13 Round 4B 链闭环 + D7 结案 + F4 裁定 + 判级（2026-09-13 09:41，零训练）

**链闭环。** 单实例链 02:35:26 起，09:40:39 打印 `ROUND 4B DONE`。断点续训 SKIP 掉已完成的 3 臂，新跑 3 臂，全程 **0 ABORT**，守卫按进程级 `self` 口径工作（D9 修复在线）：

| 臂 | cell | seed | 墙钟 | 峰值 SELF | mAP50 | DA mIoU | lane mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|
| `R4R2s2` | R4-R2thin14 | 2 | 140 min | 2328/8151 = 28% | 0.5012 | 0.8570 | 0.5994 | 0.2199 |
| `R4A0o1` | R4-A0lean_old | 1 | 134 min | 2236/8151 = 27% | 0.3452 | 0.8552 | 0.5808 | 0.1884 |
| `R4A0o2` | R4-A0lean_old | 2 | 134 min | 2236/8151 = 27% | 0.3521 | 0.8530 | 0.5862 | 0.1968 |

#### 13.13.1 D7 结案 —— 由推理链升级为实测确认

D7（§13.4）原本是**纯推理**断案：`configs/phase4a_r2_z16.yaml` 无 `anchors` 块 + 代码默认已被 `c8aca35` 改成 k-means ⇒ 两个被标 `old` 的臂其实训的是 k-means。Round 4B 产出了**真正的** old-anchor 臂，于是该推理第一次有了实测对照：

| 量 | mAP50 | 与历史 seed0 (0.3543) 之差 | 判定 |
|---|---:|---:|---|
| EXP-13 Baseline seed0（历史，未改动） | 0.3543 | — | 参考 |
| `R4-A0lean_old` seed1（本轮新增） | 0.3452 | −0.0091 | 2σ = 0.0096 内 ⇒ 同分布 |
| `R4-A0lean_old` seed2（本轮新增） | 0.3521 | −0.0022 | 同分布 |
| 被改标的两个臂（现记 `R4-A0lean_km`） | 0.4984 / 0.4938 | +0.1441 / +0.1395 | **15σ 之外** ⇒ 确非 old |

⇒ **D7 结论被独立实测确认**：历史 Baseline 格是真实的 old-anchor 单元；被改标的两臂确实是 k-means（它们同时充当了该 cell 缺失的 seeds 1/2，缺陷已转化为零成本补种）。**EXP-13 的 2×2 现在四格齐备、每格 3 seeds。**

#### 13.13.2 判据重算与新增补充行

`scripts/phase6_round4_report.py` 的 cell 映射按 §13.4 裁定改正（`base_s1/s2 → r4_R4A0o1/o2`），并新增 **D7 fail-loud 守卫**：任一 `base_*` 的 mAP50 > 0.45 即**拒绝出判决**（`return 2`）。复现该错误会当场被抓，而不是被平滑掉。

**F3：0.0462（污染值）→ 0.1445（真值）**，门槛 0.0096 ⇒ PASS（≈15σ）。

另新增 4 行**补充行**（id 带 `b*`，在 `basis` 字段显式标注 supplementary，**不参与**冻结阶梯）：F1b/F2b/F6b/F7b 用 Model B 的 3-seed 均值重算。买 seed1/2 是 §13.5 在数字存在之前预注册的动作（理由写明是「Level-3 的 3-seed 要求」），故这不是事后放宽；**噪声标尺未重估**。

| id | 量 | 观测 | 门槛 | 结果 |
|---|---|---:|---:|---|
| F1b* | lane_mIoU(B)，3-seed 均值 | 0.5959 | 0.5913 | PASS |
| F2b* | lane_mIoU(B)，3-seed 均值 | 0.5959 | 0.5865 | PASS |
| F6b* | B vs S（3-seed 均值）：lane / mAP50 / \|ΔDA\| | 0.5959 / 0.4978 / 0.0012 | 0.5913 / 0.4885 / 0.0023 | PASS |
| F7b* | \|mAP50(B) − mAP50(lean+km)\|，3-seed 均值 | 0.0028 | 0.0096 | PASS |

**F6 的刀锋（记录在案）。** 冻结的 F6 是 seed0 口径：`|da(B_s0) − da(S_s0)| = |0.8528 − 0.8551| = 0.0023`，**恰好等于** 2σ = 0.0023，按 `≤` 判 PASS。这不是浮点侥幸，是判据在边界上的巧合；3-seed 口径（F6b = 0.0012）远离边界。**两种读法都是 PASS**，且 F6 只在 Level 3 已达成时才影响 Level 4，故对本轮判级无影响。

#### 13.13.3 F4 裁定 —— 唯一需要「裁定」而非「读数」的判据

**字面结果：F4 FAIL。** `lane_mIoU(R3_min) = 0.4959 < 0.5865`。

**但 F4 失败的原因是**一个新机制**，不是「算力不够」：**

- R3_min（lean + 1/4 **h8**）与 U⁻（uniform + 1/4 **h8**）是**两个不同编码器、不同锚框**的臂，却给出**完全相同到 4 位**的 lane 读数：`lane_mIoU = 0.4959`、`lane_fg_iou = 0.0`。
- 自洽性检验：两臂的 `lane_pixel_acc = 0.9917`，而 `mIoU = (0.9917 + 0)/2 = 0.49585 ≈ 0.4959` ⇒ 这是「**全部预测背景**」的确定性签名。评估器加载 257 keys、`missing=0 unexpected=0` ⇒ 权重完整，**不是评估缺陷**。
- 两臂唯一的共同变量是 **`lane_hidden = 8`**（h16 与 h32 都正常）⇒ 分辨率不是原因，**头宽 h 才是**。

⇒ **P2 被证伪**（预注册预测 P2：「最小算力破天花板 = 0 或负」）。lane 头在 1/4 分辨率下存在**正的下界 h ≥ 16**；h16 的价格是 **+0.0860G = +8.0% FLOPs**（相对 1/8 h32），而不是 §4 成本模型期待的 −5.8%。

**F4 的问句被回答了，只是答案不是「是」。** 故本条的处置是：**字面判据保持 FAIL，不修订、不重述、不把 h16 塞进 F4 的位置**。

**关于「方案 B（修订 F4）」的登记。** 用户选定的方案 B，其最小必需集（`PHASE6_EI_GO_NOGO_ASSESSMENT.md` §4.1）为 6 臂，实际执行中 2 臂被 F9/F10 探针替换（用户改优先级），2 臂未跑：

| 方案 B 臂 | 状态 |
|---|---|
| EXP-13 补 baseline（显式 pin 老锚框）×2 | ✅ 已跑（`R4A0o1/o2`）→ D7 结案 |
| Model B 补 seed1/2 | ✅ 已跑（`R4R2s1` + `R4R2s2`）→ B 达 3 seeds |
| F9 40ep 探针（替 1 臂，用户改优先级） | ✅ 已跑（`R5-R2thin40`）§13.7 |
| F10 lr 探针（替 1 臂，用户改优先级） | ✅ 已跑（`R6-R2thinL4`）§13.7 |
| **1/4 h16 无 lateral（真最小点）** | ❌ **未跑** |
| **U⁻ 改 h16（重建「免费非对称」候选）** | ❌ **未跑** |

未跑两臂合计 ≈5.0 GPU·h。**关键事实：它们都不能把 F4 转为 PASS** —— F4 定义在 h8 上，而 h8 是已证的确定性死格。若要让 F4 出 PASS，只能**改判据所指向的 cell**（h8 → h16），那是看到数据之后改判据；本项目纪律禁止把它用作 claim 依据，**只能作为登记在案的偏离，并须在论文中披露**。未跑两臂的真实价值是：完成「最小算力夹逼」，并给 §7 的强制项（「F5 通过 → 最终架构**必须**移除 lateral」）一个实测对象。

#### 13.13.4 判级：UNRESOLVED（严格按冻结阶梯，不挑 Level）

`phase6_final_statistics.csv` 记分（F1–F8 为冻结口径，F*b 为补充口径）：

| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|
| PASS | PASS | **PASS (0.1445)** | **FAIL** | PASS | PASS | PASS | **FAIL** |

逐条套 §7 阶梯：

- 「全部 F 不通过 ⇒ Level 1」：不适用。
- 「F3 通过、F1/F2 不通过 ⇒ Level 2」：不适用（F1/F2 都通过）。
- 「F1 ∧ F2 ∧ F4 通过 ∧ EXP-11 隔离成立 ∧ EXP-13 四格 3-seed 齐备 ⇒ Level 3」：**F4 不通过 ⇒ 不成立**。
- Level 4 以 Level 3 为前提 ⇒ 不成立。

⇒ 阶梯**没有覆盖**「F1∧F2∧F3 通过但 F4 失败」这一格。按 fail-closed 纪律**不自行挑 Level**：

> **判级 = UNRESOLVED，且带严格界：Level 2 ＜ 本轮状态 ＜ Level 3。**

需要强调两点，以免它被读成「什么都没达成」：

1. Level 3 的**另外两个条件现在都已满足** —— **EXP-13 四格齐备 × 3 seeds**（本轮补齐）、**EXP-11 隔离成立**（早已证明）。**唯一的阻塞项就是 F4。**
2. 在本轮的 8 个判据里，**有 6 个 PASS**，其中 F3 以 15σ 通过；F5/F6/F7 也都成立。

**停机裁定：** §7 的停机触发条件是「Level 3 或 Level 4 成立」——当前不成立，故**「停止搜索模块」的触发条件未被触发**；但同时也**没有任何证据要求新增计算单元**（§3 Rule 1），且补跑**无法**产生 F4 = PASS。因此本轮的结论是：**停在当前状态，按 UNRESOLVED 写收尾文档**。Level 3 的正式达成需要一次**判据修订**，而该修订必须作为**事后偏离**披露，不能当既定成果。

#### 13.13.5 一个必须写进 limitations 的观察（不重估标尺）

Model B 自身 3 seeds 的 lane_mIoU = 0.5946 / 0.5936 / 0.5994 ⇒ **sd = 0.0031**，**高于** §7 冻结的 σ(lane_mIoU) = 0.0019（该值由 l14f1 与 combo 池化得到）。§8 禁止在结果出来后重估噪声标尺，故**标尺不变**、F1/F2/F1b/F2b 的 PASS 判定不变。但读者应知道：F1b 的裕度（0.5959 − 0.5913 = 0.0046）约等于 **1.5× B 自身的种子间 sd**。按 B 自身的 n=3 做 t 区间（df=2，t=4.303），lane_mIoU 的 95% CI ≈ [0.5882, 0.6036]：**下界高于 F2 门槛（0.5865），但低于 F1 门槛（0.5913）**。⇒ 「修复相对 1/8 为真」（F2）稳健；「thin 完全保留 full 的增益」（F1）**依赖冻结标尺**，在 B 自身方差下是边缘的。此观察不改判定，只降低 F1 的强度表述。

---

### 13.14 未跑两臂的零训练裁定：一条被成本模型证伪、一条仍需训练（2026-09-13 09:5x，零训练，append-only）

§13.13.3 登记了「方案 B 最小必需集」中未跑的 2 臂。本节用**零训练成本探针**（`scripts/phase6_round4b_cost_probe.py`，CPU，`CUDA_VISIBLE_DEVICES=""`）把它们从「未知」推进到「已知」，不烧任何 GPU 时间。

**实测（单旋钮枚举，锚框与匹配尺与全轮一致）：**

| 候选 | params | FLOPs | vs `FLOPs(U)` = 1.6650G |
|---|---:|---:|---:|
| `U` uniform + 1/8 h32（Model U） | 333862 | 1.6650G | — |
| `U⁻` uniform + 1/4 h16 **+ lateral** | 325318 | **1.7641G** | **+0.0991G** |
| `U⁻` uniform + 1/4 h16 **无 lateral** | 324550 | **1.7248G** | **+0.0598G** |
| `U⁻` uniform + 1/4 **h8**（原候选，已证死格） | 322390 | 1.6158G | −0.0492G |
| `B` lean + 1/4 h16 **+ lateral**（Model B） | 192566 | 1.1656G | — |
| **`B′` lean + 1/4 h16 无 lateral（未跑臂）** | **192054** | **1.1394G** | **+0.0598G vs 1/8** |
| `R4_uponly` lean + 1/4 h32 无 lateral | 201366 | 1.6129G | — |
| `R0` lean + 1/8 h32（参考） | 201366 | 1.0796G | — |

#### 13.14.1 「U⁻ 改 h16」—— **已证伪，无需训练**

F8 的第 1 条要求 `FLOPs(U⁻) ≤ FLOPs(U)`。h8 死格已排除该分辨率，而 uniform 主干上**任何非退化的 1/4 lane 修复都比 U 更贵**：h16 + lateral = 1.7641G（+0.0991G）、h16 无 lateral = 1.7248G（+0.0598G），**两者都 > 1.6650G**。

⇒ 「对称主干可以**免费**获得空间修复」这一候选（Pareto 图上那个「U⁻ 在两条轴上都不劣于 U」的点）**在成本模型层面就不可能成立**。这是一条**零训练**的证伪，使原计划的 1 臂（≈2.7 GPU·h）从「未知」变为「已排除」。**F8 的失败因此是结构性的，不是预算性的。**

#### 13.14.2 「lean 1/4 h16 无 lateral」—— 仍**需要**训练

`B′ = 1.1394G`，即 lateral 在 h16 上的价格是 **0.0262G（B 的 2.25%）**；去掉它在「最小算力」夹逼上再省 5.54% → 2.25%（相对 1/8 参考：B 是 +8.0%，B′ 是 +5.54%）。

**成本探针不能替代训练**：`B′` 是否保留 lane 增益（h16 上 lateral 是否必要）**是行为问题，不是成本问题**。该臂**未跑**，故：

- §7 的强制项「F5 通过 ⇒ 最终架构必须移除 lateral」**在 h16 上没有对照**（F5 只在 h32 上测过）；
- 最终文档（`docs/PHASE6_FINAL_ARCHITECTURE.md` §4）因此**如实保留 lateral 并显式声明该偏离**，而不是移除一个未测过的单元；
- `B′` 是**唯一**仍有正面价值的补跑项（≈2.3 GPU·h），但**它不能把 F4 转为 PASS、不能提到 Level 3**。

#### 13.14.3 附带产出：Model B 的 INT8 算子审计（预注册 §9 报告项）

在 Model B 上逐模块枚举（168 个模块实例），**叶子算子只有** `Conv2d`(49) / `BatchNorm2d`(41) / `ReLU`(28) / `Identity`；其余名字（`ConvBNAct` / `DepthwiseSeparableConv` / `DWSBlock` / `DetFromZ` / `LightEncoder` / `DynamicSegHead`）经逐层展开确认**全部是 Conv+BN+ReLU 的组合包装**。**无 LayerNorm、无 Softmax、无注意力、无动态 shape 算子。**

**但必须附一条部署路径约束**：`DynamicConv2d` / `DynamicBatchNorm2d` 是「嵌套前缀宽度切片」包装器（默认 width = 1.0 时退化为普通 Conv2d）；仓库中 `models/router/`（含 `torch.softmax` / `F.gumbel_softmax`）与 `models/adaptive_model.py` **不在 Model B 的调用图上**，导出时必须**钉死静态 width = 1.0 路径**，否则会把 softmax / 动态宽度选择带进图。

**未完成**：逐层 activation memory、与 YOLO11n 的相对比 —— 两项均**未产出**，登记为缺口（见 `PHASE6_FINAL_DECISION.md` §6），不估、不填。

---
