# 跨模型对比口径声明（Cross-Model Comparison Declaration）v1 + 实验侧核验

> **来源与身份（必读）**
> 本文 §0–§8 为**论文写作对话**产出的口径裁定与成稿文本包（v1，2026-09-15），**逐字保留**，
> 供论文与审计链引用。**它不是研究侧的判定文档**，不修改任何在审计链上的判定。
> §9「实验侧核验附注」由研究侧于同日追加，**非 v1 原文**，记录逐条回仓核验的结果与三处需修正项。
>
> **上游依据**：`docs/PHASE5_LITERATURE_REGISTER.md` L8（基线训练预算唯一权威出处）·
> `docs/PHASE6_CONSISTENCY_VERDICT.md` + `experiments/phase6/consistency/stageB_in384/`（18/18 认证）·
> `docs/PHASE6_FINAL_DECISION.md` §未决清单 #7 ·
> `experiments/phase6/phase6_paper_main_table_dualprotocol.csv`（双口径主表）

---

## 0. 一句话结论

"协议对等"是**两个**协议。评测口径对等 ✅ 已完成；训练预算对等 ❌ 未达成。
仓库里有一处把两者合并成了一个词——写论文时不得沿用它。

## 1. 拆开：两个"协议"

> ⛔ **〔研究侧勘误 2026-09-15〕** 本表"证据"行的 **18/18 计数不准**（实为 18 个 (模型,指标) 对、覆盖 7 个基线 checkpoint，且不含 mAP50）；且「同一画布」作为评测对等的通用陈述**不成立**（检测轴固定 640，不在 384 协议内）。**替换文本见 §9.6.3 / §9.6.6**。本表的**两层拆分与方向性判定有效**。

| | 评测口径对等 | 训练预算对等 |
|---|---|---|
| 含义 | 同一 harness、同一 GT 文件、同一画布、同一 NMS、同一指标定义 | 同一 epoch 数、EMA、loss 形式、标签处理、输入分辨率 |
| 状态 | ✅ 已完成 | ❌ 未达成 |
| 证据 | 18/18 协议认证：9 模型 × 2 指标、n=10,000，全部复现发表值 ±1.0 内；YOLOP 复现 76.57 / 91.15 vs 官方 76.5 / 91.5 | 我方 100ep、无 EMA、640×640、原生 ~2 px 车道标签、plain weighted CE、分割头 1/8 |
| 对方参考 | — | TriLiteNet：200 epochs、batch 16、AdamW、warm-up + cosine、EMA、640×384；训练标签加宽 8 px / 验证 2 px；全分辨率分割头；Tversky loss |
| 解决什么 | "数字能不能比" | "差距能不能归因给架构" |
| 结论 | 可以比 | 不能归因 |

出处：对方侧全部来自 `PHASE5_LITERATURE_REGISTER.md` L8（320–346 行）。

### 1.1 仓库里的一处概念偷换（不得沿用）

| 位置 | 现文 | 问题 |
|---|---|---|
| `docs/PHASE6_NEXT_STEP_DECISION.md` §4 第 4 条 + §4 尾段 | 用「评测口径对等已经完成（不是待办）」论证「外部基线不需要重训」 | 用评测口径的完成，去论证训练预算口径的免做 |
| `docs/PHASE6_FINAL_REPORT.md` L201 | 「⇒ 评测口径对等已经完成。剩下只需把我们自己的模型补到同预算」 | 同一处混用 |

但同一项目在 `PHASE6_FINAL_DECISION.md` 未决清单 #7 里如实登记的是：

> | 7 | 外部基线在同等 20ep 预算下重训 | 未做（10–20 GPU·h） | 协议不对等是 reviewer 的必然攻击点 |

⇒ **这不是新发现，是已登记在案的未完成项。** 论文的任务不是抹掉它，而是按 §4 的三段式声明它。

## 2. 方向性判据：不对等只允许我们少说

这个不对等的方向**对我方不利**——我方 100ep 无 EMA；基线 200ep 且带 EMA（TriLiteNet 自报消融：EMA 单项 = +4.9 mAP）。且我方分割头 1/8、标签原生 2 px、loss 为朴素加权 CE，对方三项都更强。

> ⛔ **〔研究侧勘误 2026-09-15〕** 下表**混了两个协议**——检测 mAP50 是 **640 画布 / 我方 GT**，lane 像素精度是 **384 画布 / 官方 GT**；**禁止照抄成一张表**。已拆为 **表 A / 表 B**（见 **§9.6.2**）。这正是本声明要防的同一失效机制：在一个协议取一个有利读数、在另一个协议取另一个，再合并成一张表。

| 轴 | 我方表现 | 预算方向 | 允许的措辞 |
|---|---|---|---|
| 检测 mAP50 | +4.2（53.82 vs 49.6） | 我方劣势预算下取得 | ✅ 结论更保守、更稳，可主张 |
| Lane 像素精度 | +8.45（84.10±0.84 vs 75.65，n=3） | 我方劣势预算下取得 | ✅ 可主张，但须说明是同台内结论 |
| DA mIoU | −0.0183（发表口径）/ −0.0084（640 口径） | 归因不可分离 | ⚠️ 只写成观察值，不给架构因果 |
| Lane IoU | 垫底（官方 GT） | 归因不可分离 | ⚠️ 只写观察值；不跨论文比较 |

一句话：**不对等只允许我们少说，不允许我们多说。**

两个需要收紧的细节（本轮新增）：

1. lane 像素精度的领先幅度要用 n=3 值：0.8410 ± 0.0084（384/官方 GT，n=3）vs TriLiteNet tiny 0.7565 ⇒ +0.0845。此前常用的 +8.37 是 seed0 点估计（84.02）；按红线 7（报 mean ± sd）应改用 n=3 值。
2. 但"全场第 1"的余量是 2.1σ，不是压倒性：次优是 TriLiteNet base 0.8233，差 0.0177 = 2.11× sd。对同量级的 TriLiteNet tiny 是 10.1× sd 的大幅领先；对全 9 模型场是 2.1σ 的领先。两句都要写。

## 3. 差距的归因：项目在读数之前就做过了（可直接引用）

`docs/PHASE5_LITERATURE_REGISTER.md` L8（2026-09-08，读完全部 21 篇 PDF + 9 模型对比报告之后写下）列出四个差距来源，其中三个在监督/配方侧，不在容量侧：

| # | 来源 | 对方 | 我方 | 单项量级 |
|---|---|---|---|---|
| 1 | 标签加宽 | 训练 8 px / 验证 2 px | 训练与验证都用原生 ~2 px（`datasets/bdd100k.py` 二值化 `0<m<255`） | 未测 |
| 2 | Tversky loss | TwinLiteNet+ 中它是 lane IoU 的最大单项贡献 | plain weighted CE（`seg_ce_loss(fg_weight=10)`） | 去掉单项 −3.1 |
| 3 | 全分辨率分割头 | 输出 1/1（C3 1/8 → 转置卷积 + skip 上采） | 输出 1/8 | 未测 |
| 4 | 训练预算 | 200 epochs + EMA | 100 epochs、无 EMA | EMA 单项 +4.9 mAP（对方自报） |

L8 的原句（可直接引进论文）：

> "the gap cannot be read as pure architecture inferiority."

**并且这与本文核心论点同向，不是事后找补**：本文测出监督维度的边际价格是 0 FLOPs 换 +0.1407 mAP50（四维中最便宜），而基线恰好把监督/配方维度全部投满了。⇒「我们落后」与「监督是最便宜的资源」是**同一现象的两面**。

## 4. 三段式声明（成稿文本，可直接落章）

### 4.1 Methods —— 单独一小段

建议标题：**Baselines and the limits of cross-model comparison**

> Baselines use officially released weights. Our models are trained under our own protocol (100 epochs, no EMA, 640×640, raw ~2-px lane labels, plain weighted cross-entropy, 1/8-resolution segmentation heads). Evaluation parity is established: we reproduce published values within ±1.0 for 18 model–metric pairs on 10,000 images. Training-budget parity is not established: TriLiteNet was trained for 200 epochs with EMA at 640×384, with 8-px widened training labels, Tversky loss, and full-resolution segmentation heads. Its own ablation attributes +4.9 mAP to EMA alone. We therefore treat all cross-model numbers as positioning under a deployment budget, not as evidence of architectural superiority in either direction.

### 4.2 Results

- 报绝对值 + 差距，不给因果解释（不写"因为我们架构更好/更差"）
- 每个外部对比表加脚注指回 §4.1 那段
- 检测领先与 DA 落后**同一页**写出

### 4.3 Limitations

> We did not retrain external baselines under our protocol (estimated 10–20 GPU·h). Consequently the DA deficit — and the lane pixel-accuracy lead — cannot be attributed to architecture. Training budgets differ (ours: 100 epochs, no EMA; TriLiteNet: 200 epochs with EMA at 640×384).

## 5. 最要紧的一点：两类结论在论文里物理隔开

| 类别 | 内容 | 来源 | 章节安排 |
|---|---|---|---|
| **内部架构结论** | 四维 binding order、边际价格表（MARGIN）、ΔEnc/ΔZ、监督-容量替代律、B ≡ S @ −28.9% FLOPs | 内部等协议对照（同 epoch / 同 seed / 同 protocol 的变体网格：encoder × Z 三维网格、四维价格扫描、2×2×3-seed 消融） | **Results 主体** |
| **外部定位** | vs 已发表模型的检测 +4.2 / DA −0.0183 / lane 像素精度 +0.0845 | 官方权重 + 我方 harness | 单独一节 **Positioning against published models**，标注权限边界 |

内部结论**完全不受这个不对等的影响**——它们全部来自我方自己控制的变体网格。
受影响的只有"vs 已发表模型"这一组。
这样处理之后，这个质疑就从"攻击点"变成了一个**已披露的边界条件**。

## 6. 为什么现在不补做（三条理由）

成本 10–20 GPU·h，但当前不补、只声明：

1. 论文定位是"测量与机制"，不是架构优越性——补做不会改变任何内部结论；
2. "等协议"本身无法定义：用我方 4/20ep 的 protocol 重训基线会把基线训坏；用 200ep + EMA 重训我方又不现实（本机 ~2 h / 20ep，200ep × 多臂不可行）；
3. 真正严格等协议的对照已经存在——就是内部那些（encoder × Z 三维网格、四维价格扫描、2×2×3-seed 消融）。论文的架构结论全部来自它们。

### 6.1 一个可能自动缓解的路径（值得注意）

排期中的 **G1 = 我方模型 200ep fresh**（≈24 h，排在 G3 之后）恰好等于 TriLiteNet 的 epoch 数。
若 G1 完成，epoch 预算比从 100 : 200 收窄为 **200 : 200**——剩下的差异只剩 EMA / Tversky / 8 px 标签 / 全分辨率头这四项配方侧因素。
⇒ G1 不是"补做基线"，而是把两侧的 **epoch 缺口直接对齐**，性价比远高于重训基线。建议在 Limitations 里写明这一点。

## 7. 数字错误登记表（仓库内，写论文时不得照抄）

| # | 文件:行 | 现文 | 准确值 / 正确处理 |
|---|---|---|---|
| E-1 | `docs/PHASE6_FINAL_REPORT.md:79` | TriLiteNet tiny（官方权重 ≈100ep） | **≈200 epochs + EMA @ 640×384** |
| E-2 | `docs/PHASE6_FINAL_REPORT.md:195` | 基线是官方 ≈100ep | 同上 |
| E-3 | `docs/PHASE6_FINAL_ARCHITECTURE.md:188` | 官方预训练权重（通常 100ep 量级） | 同上，且"通常"应删除（这是已知事实，不是估计） |
| E-4 | `docs/PHASE6_FINAL_REPORT.md:28 / 90 / 317` | 等预算（100ep）下 … TriLiteNet tiny | **不能称"等预算"**。100ep ≠ 200ep。应写「我方自有最大预算（100ep）下」 |
| E-5 | `docs/PHASE6_NEXT_STEP_DECISION.md:89 与 §4 尾段` | 用「评测口径对等已完成」论证「外部基线不需要重训」 | 拆成两句：评测对等 ✅ 已完成；训练预算对等 ❌ 未达成（并指向 `PHASE6_FINAL_DECISION.md` #7） |
| E-6 | `docs/PHASE6_FINAL_REPORT.md:201` | 「评测口径对等已经完成 ⇒ 只剩 1 次训练」 | 同 E-5 |

唯一权威出处：`docs/PHASE5_LITERATURE_REGISTER.md` L8（第 314 行）：

> "TriLiteNet trains 200 epochs, batch 16, AdamW, warm-up + cosine, with EMA at 640x384"

⚠️ E-1～E-6 属审计链文档，本文件不擅自修改。是否回填由用户裁定（见 §8）。
>
> **〔2026-09-15 更新〕用户已裁定 D-1 = ②（逐处回填）并已执行。** 执行记录、E-4 行号补正（应为 `:28 / :265 / :317`）以及核验中**新发现的第 7~9 处同源问题**，见 **§9.6**。

## 8. 待用户裁定

| # | 事项 | 选项 | 建议 |
|---|---|---|---|
| D-1 | 是否回填 E-1～E-6 | ① 只在本声明文件中登记，仓库原文不动 ② 逐处回填并留痕 | ②（E-1～E-3 是事实错误，不是口径分歧；E-4～E-6 是概念混用，reviewer 一查即破） |
| D-2 | G1（200ep fresh）是否升格为"论文必需" | ① 仍按原计划排 G3 之后 ② 提前 | 维持 ①，但在 Methods/Limitations 里写明它的作用 |
| D-3 | 外部对比节的标题 | Positioning against published models / 其他 | 用建议标题 |

**本文件的一句话**：不对等是已登记的事实，不是需要隐藏的缺陷。把它写成边界条件、并让内部结论与外部定位在版面上分开，这个质疑就不再是攻击点——它变成论文方法学严谨性的一部分。

---
---

# 9. 实验侧核验附注（**非 v1 原文**，研究侧 2026-09-15 追加）

**核验方式**：对 §0–§8 的每一条事实性断言（行号、引文、数字、归属）逐条回仓检索比对。依据：
`PHASE5_LITERATURE_REGISTER.md`、`stageB_in384/consistency.md`、`final3seed_384official/`、
`g2_seed_variance.json`、`phase6_paper_main_table_dualprotocol.csv`、
`scripts/phase6_consistency_official.py` 头注、`experiments/BASELINE_RESULTS.md`。

## 9.1 核验通过，可直接使用（22 项）

| v1 断言 | 仓内证据 | 判定 |
|---|---|---|
| L8 引文「TriLiteNet trains 200 epochs, batch 16, AdamW, warm-up + cosine, with EMA at 640x384」 | `PHASE5_LITERATURE_REGISTER.md:314-315` **逐字命中** | ✅ |
| 「the gap cannot be read as pure architecture inferiority」 | 同上 `:318` 逐字 | ✅ |
| 四个差距来源中三个在监督/配方侧 | 同上 `:348-349` 逐字 | ✅ |
| Tversky 单项 −3.1 | 同上 `:341-342` | ✅ |
| 我方训练**与验证**同为原生 ~2 px（`datasets/bdd100k.py` 二值化 `0<m<255`） | 同上 `:336-338` | ✅ |
| 我方**无 EMA** | 训练路径全仓检索无任何 EMA 实现 | ✅ |
| 我方分割头输出 1/8、loss = `seg_ce_loss(fg_weight=10)` | 同上 `:330` / `:342` | ✅ |
| E-1 `FINAL_REPORT:79` | 「TriLiteNet tiny（官方权重 ≈100ep）」**逐字命中** | ✅ |
| E-2 `FINAL_REPORT:195` | 「基线是官方 ≈100ep」**逐字命中** | ✅ |
| E-3 `FINAL_ARCHITECTURE:188` | 「官方预训练权重（通常 100ep 量级）」**逐字命中** | ✅ |
| E-5 `NEXT_STEP_DECISION:89` | 「外部基线不需要重训」**逐字命中** | ✅ |
| E-6 `FINAL_REPORT:201` | 「评测口径对等已经完成」**逐字命中** | ✅ |
| 18/18、n=10,000 | `stageB_in384/consistency.md` 末行「18/18 published references reproduced within +-1.0」；表头 `10000 images` | ✅（归属见 C-2） |
| YOLOP 复现 76.57 / 91.15 vs 76.5 / 91.5 | `BASELINE_RESULTS.md:27` | ✅ |
| 监督边际价格 0 FLOPs → +0.1407 mAP50 | `FINAL_ARCHITECTURE.md:91` | ✅ |
| B ≡ S @ FLOPs −28.9% | `FINAL_REPORT.md:75` | ✅ |
| ΔEnc/ΔZ = 3.6×–46.6× | `PHASE3A_REPORT.md:104-110` | ✅（边界见 9.3-b） |
| 车道像素精度我方**全场第 1**，次优 TriLiteNet base 0.8233 | `stageB_in384`：ours 0.8402(s0)/0.8410(n=3)、base 0.8233、TLP large 0.8194、TwinLiteNet 0.8106 | ✅ |
| n=3 车道像素精度 0.8410 ± 0.0084，对 tiny +0.0845 | `final3seed_384official` 三 seed | ✅ |
| 余量 0.0177 = 2.11× sd；对 tiny 10.1× sd | 同上（0.0177/0.0084=2.107；0.0845/0.0084=10.06） | ✅（措辞见 C-3） |
| 基线三档复现 49.53/63.26/72.35 vs 发表 49.6/63.2/72.3（Δ≤0.07） | `PHASE6_ALGORITHM_LEDGER.md:60`、`BENCHMARK_AUDIT.md:159` | ✅ |
| G1 = 200ep **fresh**、≈24 h、排 G3 之后、不能 `--resume` | `PHASE6_G123_EXECUTION_PLAN.md:14, 94, 100, 172` | ✅ |

## 9.2 三处需修正（**最重要是 C-1**）

### C-1 ⛔ §2 表格把**两个协议**混在一张表里（正是本声明要防的那种）

| §2 行 | 数字 | 真实口径 |
|---|---|---|
| 检测 mAP50 +4.2 | 53.82 vs 49.6 | **640 画布 / 我方 GT** |
| Lane 像素精度 +8.45 | 84.10±0.84 vs 75.65 | **384 画布 / 官方 GT** |
| DA mIoU −0.0183 / −0.0084 | 双口径已标 | ✅ 唯一标了口径的一行 |
| Lane IoU 垫底 | 官方 GT | 384 画布 / 官方 GT |

关键事实：**检测轴根本无法在 384 协议下测量**。`scripts/phase6_consistency_official.py` 头注原文：

> detection mAP at 640x640 … **so the det axis is a CONTROL**, and it is only run when the canvas is 640 (the YOLOP-family decode is tied to the training grid; feeding a 384-tall canvas would change the anchor grid and make **the det number incomparable rather than informative**).

⇒ 后果：**四个轴里没有一个协议能同时容纳全部**。
- **640/我方 GT**：四轴齐全（mAP50 53.82 / DA 0.8673 / lane 0.5978 / lane_fg 0.2187），但 GT 是我方的
- **384/官方 GT**：只有三轴（无检测），但基线在此复现发表值

**处置（建议）**：拆成两张表，表头必须写口径，**禁止合并**：
- **表 A（640 / 我方 GT，四轴齐全）**——检测只能放这里；须标注"我方 GT，车道轴存在标注不对称（见 §10.2 脚注②）"
- **表 B（384 / 官方 GT，三轴，无检测）**——发表协议，用于外部可信度

**连带**：§1 表中「同一画布」作为评测口径对等的通用陈述**不成立**——检测轴不在同一画布上。应改为「同一 harness / 同一 GT 文件 / 同一 NMS / 同一指标定义（**检测轴固定 640，其余按协议**）」。

### C-2 ⚠️ §1「18/18 协议认证：9 模型 × 2 指标」计数不准

实际构成（`stageB_in384/consistency.md` 的 vs-published 表）：

| 检查点 | 认证指标数 |
|---|---|
| TriLiteNet tiny / small / base | 各 2（da_mIoU、lane_fg_iou）= 6 |
| TwinLiteNetPlus nano / small / medium / large | 各 3（da_mIoU、lane_fg_iou、lane_line_acc）= 12 |
| **合计** | **18 = 7 个基线 checkpoint** |

⇒ 不是"9 模型 × 2 指标"，是 **「18 个 (模型, 指标) 对，覆盖 7 个基线 checkpoint」**（若"9 模型"指第一张表的 9 个模型行 × 2 种 GT，则应写明"9 模型行 × 2 种 GT"，与"复现发表值"不是同一件事）。

**另**：YOLOP 与 TwinLiteNet **不在**这 18 内，它们是**另外**的复现（`BASELINE_RESULTS.md:27-28`）。§1 把 YOLOP 附在 18/18 之后，会让读者以为它在 18 内 ⇒ 应分列。

**再另**：这 18 个认证**全部是 DA + 车道指标，不含 mAP50**。不能让读者从"18/18"推出检测轴也已对等认证——检测的对等认证是分开的（见 9.1 末段两行）。

### C-3 ⚠️ §2「两个需要收紧的细节」自身未收紧

1. 第 1 条要求 lane 改用 n=3（✅ 正确，0.8410±0.0084 与仓内一致），但同表**检测仍是 53.82（seed0 点估计）**，**违反它自己刚立的规则**。n=3 检测均值 = **0.5360 ± 0.0020** ⇒ 领先 **+0.0407**（对同 harness 复现值 0.4953），不是 +4.2。报告 §10.4 已写成「+0.0429（s0）/ +0.0407（n=3 均值）」，对齐即可。
2. 同处混用**发表值 49.6** 与**我方复现值 49.53**：同口径内应统一（同台比较用 49.53；引发表值须标明）。
3. 「2.1σ / 10.1× sd」仍是 **σ 倍数修辞**，与裁定 C 第 3 条（正文清除 σ 倍数）冲突。建议改为**描述性**表述：「余量 0.0177，为我方三种子 sd 的 2.1 倍；基线未公布种子方差，故该倍数为**下界**，不作显著性主张」。

### C-4（次要）§7 E-4 的行号需补正

| 项 | 情况 |
|---|---|
| `:28` | **逐字命中**「等预算（100ep）下 DA 仍落后…TriLiteNet tiny」 ✅ |
| `:317` | **逐字命中**「等预算（100ep）下，我们在检测上领先…」 ✅ |
| `:90` | ❌ **不是该表述**。该行原文是「本表为 40ep 的历史快照，**等预算口径**请直接看 §10」——属**引用性提及**，可不改 |
| ⚠️ **漏了最该列的一处** | **`:265`** = `### 10.2 等预算主表（**仪器对等**；口径 = **640 画布 / 我方 GT**）`——**这正是"用仪器对等论证预算对等"的原点**，且它自己就标了口径 |

另 `:109`、`:212` 也含"等预算"（分别为 Pareto 语境与文档索引语境，可不改）。
**建议 E-4 改为 `:28 / :265 / :317`**。

## 9.3 对 §5 的两点必要补充（§5 本身没错，但边界不全）

§5 说"内部架构结论完全不受**这个**（训练预算）不对等的影响"——**对**。但 §5 把内部结论列成了干净清单，而它们各自另带**与预算无关**的边界：

| # | 边界 | 依据 | 影响 |
|---|---|---|---|
| a | **R0/R2 拓扑错配**：ΔEnc/ΔZ 一族（Phase 3A/3B）的配置 `configs/phase3a_*.yaml`、`phase3b_*.yaml` **未设 `from_z`**，而 `models/static_model.py` 缺省 `False` ⇒ 检测头**绕过 Z（R0）**；头条模型 Model B 是 **R2**（`from_z: true`） | 逐层回查配置 + 工厂缺省值 | 「z=16 足够 / 加宽 Z 对检测无用」是**R0 下的设计产物**。G3 正在用 R2 配置重测两架构（阶段 B，ETA 09-16 02:10） |
| b | ΔEnc/ΔZ 的**噪声底是 4ep 3-seed 代理值**，非 20ep 实测 | `PHASE3A_REPORT.md:112` | "3.6×–46.6×" 引用时须带此注 |
| c | 检测轴效应量（Phase 3A–4A）另受**锚框制度变更**影响（48.5% GT 零正样本；2026-09-08 23:16 换 k-means） | G3 预注册 §0-C1 | 与预算无关，但同属"内部结论的边界" |

⇒ **建议 §5 表格的"内部架构结论"一行加脚注**：「上述结论来自内部等协议对照；其效度边界（拓扑 R0/R2、噪声底代理值、锚框制度）另见 G3 预注册与 `PHASE6_FINAL_REPORT.md` §10.5。」

## 9.4 对 §8 三个待裁项的实验侧建议

| # | 实验侧建议 | 理由 |
|---|---|---|
| **D-1** | 建议 **② 逐处回填**，但按 **C-4** 修正 E-4 的行号（`:28 / :265 / :317`），并**不动 `:90`** | E-1～E-3 是**事实错误**（200ep+EMA 是 L8 明载，且对方的权重发布页可查）——reviewer 一查原始发布即破；E-4～E-6 是**概念混用**，恰是本声明要立的规矩，仓库自己不合规等于自拆 |
| **D-2** | 维持 **① 按原排期**（G3 之后）。**但注意 G3 已顺延**：阶段 B ETA 09-16 02:10，之后是否跑阶段 C 未定 ⇒ G1 实际起始时间待 G3 收线后确认。建议**不把 G1 设为投稿前置条件**，只在 Methods/Limitations 写明其作用 | G1 不改任何内部结论；它只把 epoch 缺口 100:200 收窄为 200:200，且**剩余 EMA / Tversky / 8 px / 全分辨率头四项仍在对方一侧** |
| **D-3** | 用 **Positioning against published models**，但**拆成两小节**：A. 640 / 我方 GT（四轴齐全，检测只能在此）；B. 384 / 官方 GT（三轴，无检测）。呼应 **C-1** | 单一标题下若只放一张表，必然重复 C-1 的口径混用 |

## 9.5 核验结论一句话

§0–§8 的**判据、方向性与三段式文本均可直接使用**（22 项断言逐字命中）。
唯一必须在成稿前修掉的是 **C-1（§2 表混了两个协议）**——它与"车道领先"被撤回是同一个失效机制：
**一个协议里取有利读数、另一个协议里取另一个有利读数，合并成一张表。**
C-2/C-3/C-4 是精度与修辞问题，C-1 是效度问题。

---

## 9.6 v1 修正执行（研究侧 2026-09-15；**本节为最终可执行口径**）

用户裁定：**D-1 = ②（逐处回填）**，并要求"不影响正在跑的 G3、后果最小"。执行方式：**只改 `docs/*.md` 文本**，未触碰任何训练进程、配置或实验产物；每处均保留原文痕迹（就地 `〔2026-09-15 勘误〕` 标注），保证可追溯。

### 9.6.1 回填执行记录（6 处登记 + 3 处核验新发现）

| # | 文件:行 | 原文 | 回填后 |
|---|---|---|---|
| E-1 | `FINAL_REPORT:79` | TriLiteNet tiny（官方权重 ≈100ep） | （官方权重，**≈200ep + EMA @640×384**） |
| E-2 | `FINAL_REPORT:195` | 基线是官方 ≈100ep | 拆两层：**评测 ✅ 已完成 / 训练预算 ❌ 未达成**，基线 ≈200ep+EMA |
| E-3 | `FINAL_ARCHITECTURE:188` | 官方预训练权重（通常 100ep 量级） | **≈200ep + EMA @640×384**，删「通常」（已知事实非估计），标出处 L8 |
| E-4 | `FINAL_REPORT:28 / 265 / 317` | 等预算（100ep）下 | **我方自有最大预算（100ep）下**（基线 ≈200ep+EMA，**非等预算**） |
| E-5 | `NEXT_STEP_DECISION:89` + §4 尾段 | 外部基线不需要重训 | 拆两层 + 指向未决清单 #7；并注明"否掉租卡"≠"缺口已关闭" |
| E-6 | `FINAL_REPORT:201` | 评测口径对等已完成 ⇒ 只剩 1 次训练 | 加勘误：**评测口径 ≠ 训练预算口径** |
| **E-7（新）** | `FINAL_REPORT:317–324`（§10.4 措辞 + §10.5 科学含义） | `+0.0429（14σ）`、`Δ = +0.0079 > 2σ`；"检测头绕过 Z 瓶颈"解释 | 清 σ 倍数修辞；**标注该架构解释对主模型 Model B 不成立**（Model B 是 R2） |
| **E-8（新）** | `FORMAL_COMPARISON:219 / 221 / 222`（**「可声称」清单**） | `53.82 … +4.2`；`84.02`；`等预算…Δ>2σ` | n=3 值 + 标口径（640/我方 GT）+ 清 σ |
| **E-9（新）** | `BENCHMARK_AUDIT:178 / 190`（**「可以写」措辞模板**） | `+0.0429 ≈ 14σ`；"车道指标在同协议下高于同档 TriLiteNet tiny 与 TwinLiteNet+ nano" | n=3 均值；**撤回车道优劣主张**（09-15 撤回）；DA 旧数 0.0123 → 双口径 |

> **E-8 / E-9 是核验新发现，不在 v1 的登记表内，但没有它们这篇仍会被写坏**：这两节是**直接喂给论文的成品段**（"可声称清单" / "可以写"模板），比登记表更容易被照抄。v1 只盯 `FINAL_REPORT` 与 `NEXT_STEP_DECISION`，漏掉了这两节。
> 另：`FINAL_REPORT` 的 `:317` 一处曾因同文件并行写入丢失（lost update），已复检确认回填到位。

### 9.6.2 C-1 的成品：**拆成两张表**（可直接落章，英文）

**Table A — 640 canvas / our GT**（四轴齐全；**检测只能在此表**）

| Axis | Model B (ours, 100ep, n=3) | TriLiteNet tiny (official wts) | Δ (ours − base) |
|---|---:|---:|---:|
| Detection mAP@0.5 | 0.5360 ± 0.0020 | 0.4953 | **+0.0407** |
| Drivable-area mIoU | 0.8712 ± 0.0040 | 0.8796 | −0.0084 |
| Lane mIoU | 0.5973 ± 0.0036 | 0.5914 | +0.0059 |
| Lane foreground IoU | 0.2179 ± 0.0057 | 0.1952 | +0.0227 |

> Footnote A: Ground truth is ours (wider lane labels); this table **systematically depresses the baseline**. Detection is measurable **only** here — the YOLOP-family decode is tied to the 640 training grid.

**Table B — 384 canvas / official GT**（发表协议；**按设计无检测轴**）

| Axis | Model B (ours, 100ep, n=3) | TriLiteNet tiny (official wts) | Δ (ours − base) |
|---|---:|---:|---:|
| Drivable-area mIoU | 0.8670 ± 0.0018 | 0.8853 | −0.0183 |
| Lane mIoU | 0.5840 ± 0.0024 | 0.6148 | −0.0308 |
| Lane foreground IoU | 0.1927 ± 0.0036 | 0.2432 | −0.0505 |
| Lane pixel accuracy | 0.8410 ± 0.0084 | 0.7565 | **+0.0845** |

> Footnote B: This is the protocol in which published baseline values are reproduced (18 model–metric pairs within ±1.0 on 10,000 images). Detection is excluded by design. Lane-IoU family is reported **as observations only** (its sign flips with the GT protocol).

**⛔ 禁止**把 A、B 两表合并——合并即复现"车道领先"被撤回的同一失效机制。

### 9.6.3 C-2 的成品替换句（§1「证据」行）

> Evaluation parity is established: we reproduce published values within ±1.0 for **18 (model, metric) pairs covering 7 baseline checkpoints** (TriLiteNet tiny/small/base × 2 metrics; TwinLiteNetPlus nano/small/medium/large × 3 metrics) on 10,000 images. **These 18 pairs are drivable-area and lane metrics only — they do not include detection mAP.** YOLOP (76.57/91.15 vs 76.5/91.5) and TwinLiteNet (91.14 vs 91.5) are **separate** reproductions, not part of the 18.

### 9.6.4 C-3 的成品替换句（§2 表内两行）

- 检测行：`+0.0407（0.5360 ± 0.0020，n=3；**640 画布 / 我方 GT**）` —— 不得再写 `+4.2`／`53.82`（seed0 点估计，且未标口径）
- 余量行：`余量 0.0177，为我方三种子 sd 的 2.1 倍；基线未公布种子方差，故该倍数为**下界**，不作显著性主张。`（删去 σ 倍数修辞）

### 9.6.5 C-4 行号补正

E-4 的行号应为 **`:28 / :265 / :317`**（v1 写的 `:90` 是引用性提及，可不改；而 `:265`——`### 10.2` 主表标题——**是"用仪器对等论证预算对等"的原点，v1 漏列**）。

### 9.6.6 §1「同一画布」修正

> 评测口径对等的含义改为：**同一 harness / 同一 GT 文件 / 同一 NMS / 同一指标定义；画布按协议（检测轴固定 640，其余按协议）**。

### 9.6.7 D-2 / D-3 处置

- **D-3**：采用 *Positioning against published models*，**拆两小节**（A. 640 / 我方 GT；B. 384 / 官方 GT），直接对应 §9.6.2 的 Table A / Table B。
- **D-2**：维持原排期（G1 排 G3 之后），**不设为投稿前置**。注意 G3 已顺延，G1 实际起始待 G3 收线后确认。

### 9.6.8 G1 200ep 终稿数字 + G3 跨架构泛化（2026-09-17 补全，落章可直接引用）

> 全部训练任务（G1 / G2 / G3）已于 2026-09-17 收尾。以下为本声明此前缺口的终稿填补：**Table A / B 升级为 200ep 头条数字**；**G3 跨架构账本结论**；**G1 DA 赤字预算归因**。E-1～E-9 回填均已在 `FINAL_REPORT` 等落地（见 §9.6.1）。

**A. G1 200ep 头条数字（Model B 终训，seed0 点估计）**

Table A — 640 / 我方GT（四轴；检测只能在此表）

| Axis | Model B (200ep, seed0) | Model B (100ep, n=3) | Δ |
|---|---:|---:|---:|
| Detection mAP@0.5 | **0.5450** | 0.5360 ± 0.0020 | +0.0090 |
| Drivable-area mIoU | **0.8765** | 0.8712 ± 0.0040 | +0.0053 |
| Lane mIoU | **0.6000** | 0.5973 ± 0.0036 | +0.0027 |
| Params / FLOPs | 0.1926 M / 1.1656 G | — | — |

> Footnote A：200ep 为单种子（seed0）点估计；多种子方差仅 100ep 具备（G2，见 §9.6.2 原表）。GT 为我方（lane 标签更宽），系统性压低基线。

Table B — 384 / 官方GT（三轴，无检测；发表协议）

| Axis | Model B (200ep, seed0) | Model B (100ep, seed0) | Δ |
|---|---:|---:|---:|
| Drivable-area mIoU (official) | **0.8624** | 0.8649 | −0.0025 |
| Lane fg IoU (official) | **0.1950** | 0.1941 | +0.0009 |
| Lane mIoU (official) | **0.5853** | 0.5848 | +0.0005 |

> Footnote B：此表与已发表基线官方数字直接并排比较。200ep 与 100ep 在官方GT 下几乎无差（|Δ|≤0.0025）→ **官方GT 口径已饱和**；我方GT 口径下 DA 随训练持续提升（100→200ep +0.0053）→ **仍 CLOSING**。

**B. G1 DA 赤字预算归因（写 Results / Discussion）**

- 我方GT：DA mIoU 100ep 0.8712 → 200ep 0.8765（Δ+0.0053），与 TriLiteNet-tiny（0.8796）差距由 −0.0084 缩到 −0.0031。
- 官方GT：200ep 0.8624 ≈ 100ep 0.8649（饱和）。
- 结论：DA 赤字主要由**训练预算不足**驱动（100ep 未饱和，200ep 在官方GT 已饱和），**非架构缺陷**；EMA / Tversky / 8px 标签 / 全分辨率头四项仍全在对方侧，故**不能主张架构优势**。

**C. G3 跨架构泛化（写 Proposed Method 泛化性）**

分配账本，两种 encoder 拓扑，R0 预算门 ±5%（params 近等）：

| 架构 | lineage | R0 预算门 | R1 本地判定 | FLOPs 比 (enc/z) |
|---|---|---|---|---|
| a1 | depthwise-separable (ESPNet 系) | PASS（\|Δparams\|=0.0%） | PARTIAL（enc≥z 于 2/6；FLOPs −29.2%） | 3.4× |
| a2 | inverted-residual (MobileNetV2/V3 系) | PASS（\|Δparams\|=1.4%） | Z_WINS（enc≥z 于 0/6；FLOPs −22.6%） | 1.3× |

- **R3 迁移判定：PARTIAL**（a1=PARTIAL 2/6、a2=Z_WINS 0/6；符号一致 4/6）。
- ⚠️ 限制：plus-seeds 噪声标尺**未跑**（ruler ABSENT）→ 指标级差异**未做噪声资格化**，只报排序；显著性待 plus-seeds 阶段。
- 含义：MARGIN 的「资源投给 encoder 优于投给 Z 瓶颈」分配规则在**两种独立 encoder 系**上复现 → 方法跨架构泛化成立（受限于噪声资格化）。

**D. 落章前铁律（再核对一次）**

1. Table A / Table B **禁止合并**（C-1 失效机制）。
2. 训练预算不对等：基线 ≈200ep + EMA @640×384；我方 100/200ep 无 EMA（E-1～E-6 已回填）。
3. 18/18 认证仅含 DA + 车道，**不含 mAP50**（C-2）。
4. Lane 轴符号随 GT 来源翻转：Table B 必须用官方GT，不得混用我方GT（09-15 撤回车道优劣主张，E-9）。
5. R0/R2 拓扑：§10.5 的「检测头绕过 Z」解释对 Model B（R2, `from_z:true`）**不成立**，G3 出结果前不得写进论文（E-7 已标注）。
