# PercepFlex / trac — Phase 6 投稿策略与审稿风险评估

**版本 v1　2026-09-14**
**数据来源**：`docs/PHASE6_FORMAL_COMPARISON_REPORT.md`（认证版对比）、
`experiments/phase6/consistency/stageB_in384/consistency.json`（18/18 认证）。
**用途**：在动笔之前先钉死"能说什么、不能说什么、审稿人会打哪里"。

---

## 0. 一句话结论

同一批数据，**声称 SOTA 必拒，声称"预算约束下的容量分配研究"则可能中稿**。
差距不在数据，在主张。本文按此把风险、资产与缺口逐条列清。

---

## 1. 资产清算（能写进论文的）

### 1.1 性能轴（4 项，同台同协议）

| 项 | 数值 | 强度 | 使用条件 |
|---|---|---|---|
| 检测 mAP50 | 53.82（vs TL-tiny 49.6，+4.2） | **强** | 唯一被独立认证的可比轴（Δ≤0.07）；对手须限定为"同量级（0.15M）唯一发表模型" |
| 车道像素精度 | 84.02（9/9 第一） | **中强** | 必须同时说明：这是**同台内**结论，非对文献声明；该指标是 YOLOPv3 推荐的 IoU 替代 |
| 参数 / FLOPs | 0.193M / 1.166G | **强** | 参数口径安全（1.00–1.01×）；**FLOPs 禁止与文献并排** |
| 评测可信度 | 18/18 复现 ±1.0 @n=10000 | **强（方法论）** | 这是论文的信任基础，应写在方法章而非结论章 |

### 1.2 分析性结论（论文的真正贡献轴）

| 结论 | 状态 | 备注 |
|---|---|---|
| Encoder 容量 E-s→E-b→E-l 在 **6/6 指标严格单调** | 已确证 | 干净的剂量-响应关系 |
| **detection 是最敏感指标** | 已确证 | DA/Lane 在 E-large 已饱和，detection 仍随容量上升 |
| **DA/Lane 在 E-large 饱和** | 已确证 | 容量投入的边际收益递减点 |
| 等参数下 E-large+z16 比 baseline+z128 **FLOPs 低 29.4%** | 已确证 | 可操作 takeaway |
| 40ep→100ep 把 DA 差距 **−0.0202 → −0.0123**（Δ>2σ） | 已确证 | **DA 赤字的预算归因证据** |

### 1.3 诚实性资产（审稿人加分项）

- 22 篇本地文献逐条对账，每条都可回溯到本地 PDF（`docs/PHASE6_ALGORITHM_LEDGER.md`）
- 双 GT 逐像素审计：DA 同源（Jaccard 0.9885，可比）／Lane 异源（0.3804，不可比）
- 车道列不对称已量化并归因（我方在官方 GT 下为**悲观值**，被罚 2.32）
- 主动撤回不严谨的对比项（跨文献 lane 排名、accuracy/GFLOP、FLOPs 并排）

---

## 2. 缺口清单（审稿人会打的地方）

| # | 缺口 | 严重度 | 现状 | 补法 |
|---|---|---|---|---|
| G1 | **DA 赤字未完全归因** | 高 | 已知"至少一半是预算"，但只到 40ep→100ep 一条证据 | 把预算轴做完整：加更长预算或加中间点，把"预算 vs 架构上限"分离干净 |
| G2 | **单 seed，无方差** | 高 | seed0 完成（FINAL-100）；seed1 在跑（ep20），seed2 顺位 | 等 seed1/2 落盘即给 ±σ；**补完前不得使用 parity/持平措辞** |
| G3 | **insight 的泛化性** | 高 | 只在单一架构上验证容量分配结论 | 跨架构验证（第二个 backbone / 第二个瓶颈设计），是提升论文档次的最高杠杆 |
| G4 | 无新方法/新架构 | 中 | 架构为既有设计的组合 | 不掩盖：把论文定位为**经验研究**，贡献声明写 conclusions 而非 method |
| G5 | 车道 IoU 列不可对文献 | 中 | 已加引用禁令三条 | 论文中凡出现 lane 数字必须同时给口径 + 悲观值说明 + 封顶事实，缺一不引 |
| G6 | 延迟/FPS 无可信读数 | 中 | D8 探针已证旧仪器不对称（reps 3 vs 100） | 如有需要，用 FIXED 协议（warmup200/reps300/CUDA events）补齐 9 基线 |

---

## 3. 审稿人意见预演（two-scenario rehearsal）

### 3.1 场景 A：写"性能可比/SOTA"

**预期意见（单句即拒）**

> "The proposed model underperforms the smallest published baseline (TriLiteNet-tiny, 0.15M)
> on drivable-area segmentation by 2.0 mIoU while carrying 28% more parameters.
> The competitive-performance claim is not supported."

**根因**：主张与数据不符，且该不符在 30 秒内可查。
**处置**：**禁用**。这是初版报告被撤回的同一错误。

### 3.2 场景 B：写"预算约束下的容量分配"

**预期意见（四问）**

1. *"The protocol certification is unusually thorough; it materially increases my confidence in the remaining numbers."*
   → **正面资产，主动前置**到方法章。
2. *"The capacity ablation is clean and the 6/6 monotonicity is convincing."*
   → **核心贡献，展开写。**
3. *"Why is DA still 6 points behind? The budget explanation is plausible but only partially demonstrated."*
   → **必须在讨论章正面回答**，用 40ep→100ep 的收窄证据 + 承认残差未归因（fail-open 比含糊强）。
4. *"Single seed. No variance reported."*
   → 补完 seed1/2 后此条消失；补完前不投。

**额外会出现的追问**：*"How general is the capacity-allocation insight beyond this one architecture?"*
→ **G3，唯一需要新实验的洞。**

---

## 4. 目标会场分级（诚实判断）

| 档位 | 会场 | 判断 | 前置条件 |
|---|---|---|---|
| 顶会主会 | CVPR / ICCV / NeurIPS | **当前很难** | 需 G1+G2+G3 全补，且 novelty 仍偏弱 |
| 顶会 workshop | CVPR/ICCV 自动驾驶、高效视觉 workshop | **较有戏** | G2 必补；G1 建议补 |
| 领域会议 | ITSC / IV / ICRA / JSAE | **有戏** | G2 必补；主线走"效率 + 可信评测" |
| 期刊 | T-ITS / T-IV / Sensors / IEEE Access | **现实可中** | G2 必补；G1 强烈建议 |

**共同前提**：**G2（seed 方差）不补完，任何档位都不投。** 单 seed 在现代评审里是无需讨论的硬伤。

---

## 5. 写作纪律（钉死，防止旧错复发）

1. **禁止**任一形式的 parity / 持平 / "追平"措辞（D-2 触发中，解除条件 = seeds 1–2 完成）。
2. **禁止**将 FLOPs、FPS 与文献数字并排（口径 3.33× 与硬件差异）。
3. **禁止**对文献做车道 IoU 比较（异源 Jaccard 0.3804 + 指标封顶）。
4. 车道数字出现时必须三件套：口径 + 悲观值说明 + 封顶事实。
5. 检测轴对手必须限定"同量级（0.15M）唯一发表模型"，不得泛化为"优于同类"。
6. 检测的 384 画布读数**有意不跑**（会改变 anchor 网格，得不可比而非更有信息之数）——若审稿人要求，说明这是设计选择，不补。
7. 贡献声明写在 conclusions 层面（可复用结论），不写 method 层面（无新方法）。

---

## 6. 缺口关闭优先级（按性价比）

| 序 | 动作 | 关闭缺口 | 成本 | 杠杆 |
|---|---|---|---|---|
| 1 | 等 seed1 / seed2 落盘 → 给 ±σ | G2 | 已在跑 | **最高**（决定能否投稿） |
| 2 | 预算轴做完整（更长预算 / 中间点） | G1 | 中（GPU） | 高（把最大弱点转成证据） |
| 3 | 跨架构复现容量分配结论 | G3 | 高 | 高（决定档位上限） |
| 4 | 9 基线延迟 FIXED 补测 | G6 | 中 | 低（可选章节） |

---

## 7. 结论

**本工作的可发表价值不来自"更准"，来自三件事：**

1. 一批**别人没有的**、可复用的容量分配结论（单调性、敏感度排序、饱和点、等价预算下的 FLOPs 节省）。
2. 一个**别人没有的**方法论资产（18/18 协议认证 + 22 篇文献对账 + 双 GT 审计），
   它把审稿人对数据的怀疑成本降到很低。
3. 一种**诚实的**弱点处理：DA 末位被量化、被归因、被给出预算解释，而非隐藏或含糊。

**因此：这是一篇"严谨的经验研究"，不是一篇失败的 SOTA 论文。**
前者的审稿标准是可信与可复用，我们达标；后者的标准是更准，我们不达标——且不应去争。
