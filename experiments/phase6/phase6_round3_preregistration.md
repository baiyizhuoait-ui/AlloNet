# Phase 6 — Round 3 Pre-registration (cross-architecture validation)

**Frozen:** 2026-09-10 23:2x CST, BEFORE any Round 3 training. **Repo HEAD:** `9fb29e7` (+ uncommitted Round 3 additions).
**Scope:** 1-seed screening of two hypotheses on a SECOND architecture. **Not** a re-run of Phase 2–5.
**Rule of this document:** everything below is fixed now. Any later change is a `§12.x`-style logged amendment, never a silent edit.

---

## 1. Question

Round 1–2 established a bottleneck *ordering* (detection is supervision-limited; lane is space-limited) **inside one architecture family (R2)**.
Round 3 asks the only question that raises or kills that claim:

> Is the ordering a property of the **task**, or an artefact of **R2**?

Novelty discipline (binding): "it is a different model" is **not** the claim. The claim under test is
*measurement-driven resource allocation*, i.e. that these bottlenecks are **measurable and transferable**.

## 2. Hypotheses (falsifiable form)

| # | Statement | Second architecture | Primary read-out |
|---|---|---|---|
| **H1** | Detection's binding constraint includes **supervision / assignment**, not only feature capacity | **YOLOP** (7 940 846 params) | mAP50 dose–response over an anchor-quality ladder |
| **H2** | Under extreme compression, lane accuracy responds more to **spatial fidelity** than to **channel width** | **TwinLiteNetPlus** (0.033–1.944 M) | lane mIoU: spatial gain vs channel gain |
| **H3** | Different tasks are limited by different resource dimensions ⇒ asymmetric allocation beats uniform | — | **H3 status is defined as H1 ∩ H2.** It is NOT separately testable here |

**H3 is the weakest of the three** and was already tested and *not* supported in Round 2 (EXP-08a NOT SUPPORTED, EXP-08b UNRESOLVED). Round 3 may therefore only *confirm* H3 via a conjunction; it can never "rescue" it. If H1 and H2 split, H3 is reported as **REJECTED (conjunction fails)** regardless of either arm's individual strength.

**Two testbeds, declared up front.** No single off-the-shelf architecture offers both a detection head and a lane head *without already having a high-resolution lane path*. Measured, not assumed (§6). So H1→YOLOP and H2→TwinLiteNetPlus. The cross-architecture evidence for H1 and H2 therefore does **not** share a testbed. This is a declared limitation, not a hidden one.

## 3. Instrument discipline (the ruler is held fixed)

- The intervention is **the anchor set**, nothing else. Architecture, params, FLOPs, batch, epochs, optimizer, schedule, dataset, augmentation, initialisation and seed are identical across arms.
- **The assignment rule is fixed at the project rule** `0.5 < box/anchor < 2.0` (`losses/yolo_loss.py`:106) for **all** arms, including on YOLOP.
  *Why:* YOLOP's native matcher uses `ANCHOR_THRESHOLD = 4.0` **plus** 5-neighbour offset expansion (`lib/core/postprocess.py`). Two different rulers cannot be pooled, and the R2 effect whose transferability we are testing was measured under the project rule. Holding the ruler fixed and varying only the anchors is the single-variable design.
  *Registered confound:* under YOLOP's **native** rule the shipped anchors leave only **0.6 %** zero-positive GT, under the project rule **49.4 %** (§4). The native-rule numbers are reported alongside every claim and annotated `native-rule, do not pool`.
- Anchor sets used for training are read from `experiments/phase6/phase6_round3_anchor_summary.json` — the STEP-1 artefact. The analysed set and the trained set are the **same object**. `shipped` is re-read live from the built model.
- The anchor set travels with the checkpoint (`anchors.json`) and the eval side applies it via `YOLOP_ANCHORS_JSON`. Without this, eval would silently decode with the shipped anchors and every arm would score identically.

## 4. STEP-1 result already on the table (zero-training, 0 GPU) — the factual basis for the design

Measured on 3 000 `tri_train` images / **32 421 GT boxes** (side p10 7.3, p50 19.2, p90 80.3 px; **aspect h/w median 0.80**).

| arm | anchors (3 levels × 3) | mean best-IoU | zero-pos *(project)* | zero-pos *(native)* | matches/GT |
|---|---|---|---|---|---|
| **A0** shipped (YOLOP's own) | 3×9,5×11,4×20 / 7×18,6×39,12×31 / 19×50,38×81,68×157 | 0.4161 | **49.41 %** | 0.57 % | 0.67 |
| **A1** aspect-flip of A0 | 9×3,11×5,20×4 / 18×7,39×6,31×12 / 50×19,81×38,157×68 | 0.5419 | 16.66 % | 0.33 % | 1.37 |
| **A2** k-means refit | 9×8,18×15,32×24 / 49×38,80×52,65×102 / 124×82,166×136,237×214 | **0.6805** | **3.74 %** | 0.04 % | 2.01 |
| ctx R2 old default | — | 0.4069 | 48.51 % | 0.42 % | 0.69 |
| ctx R2 k-means | — | 0.6805 | 3.86 % | 0.05 % | 2.01 |

Three facts this fixes:

1. **The "original = corrected" worry was wrong.** YOLOP's shipped anchors are *not* autoanchor-optimal for this GT distribution: they are **all tall** (h/w 2.13–6.50) against a near-square GT (median 0.80), and they leave **49.4 %** of GT with no positive assignment under the project rule — statistically indistinguishable from R2's *known-broken* pre-2026-09-08 default (48.5 %). The contrast exists after all.
2. **The hole is a property of the ruler, not only the anchors.** The same shipped set goes 49.4 % → 0.6 % merely by moving `anchor_t` 2.0 → 4.0. Any claim about "supervision holes" must name the rule.
3. **The k-means repair is architecture-independent.** The refit on YOLOP's 3-level/3-anchor layout reproduces R2's k-means set **value-for-value**. Anchors are a function of (dataset, rule, k); only the *consequences* can be architecture-specific.

### 4.1 EXP-9A arms (3 points, pre-registered dose ladder)

| arm | anchors | mean best-IoU | role |
|---|---|---|---|
| `A0` | shipped | 0.4161 | **degraded end** (incumbent, to be beaten or exonerated) |
| `A1` | aspect-flip | 0.5419 | **intermediate dose** |
| `A2` | k-means refit | 0.6805 | **repaired end** |

A 3-point ladder replaces the originally planned "degrade / repair" pair: the measured facts show `A0` already **is** the degraded end, so a synthetic degradation arm would be redundant, and a 3-point dose–response is strictly more informative than a 2-point contrast (it can distinguish "no effect" from "non-monotone effect", which a 2-arm test cannot).

### 4.2 EXP-9B arms (3 points, FLOPs-parity controlled)

| arm | lane branch input | role |
|---|---|---|
| `B0` baseline | stock TwinLiteNetPlus (1/8 learned + 1/2 & 1/4 **raw RGB**) | reference |
| `B1` spatial | **+** one small **learned 1/4** lateral (1×1 projection of the encoder's 1/4 activation, `c_lat` channels) concatenated into the lane branch at the same insertion point | spatial intervention |
| `B2` channel | **no** new lateral; lane-branch width increased by Δc at the existing resolution | channel control |

`Δc` is solved by binary search so that FLOPs(`B2`) = FLOPs(`B1`) within **±5 %** (the project's budget-equivalence tolerance). If the search cannot hit ±5 %, the arm pair is **not** run and the reason is logged — a non-parity pair would be uninterpretable.

### 4.3 Anti-drift clause

No arm may be added, dropped or re-parameterised after seeing a result. In particular: no "add one more seed because it is close", no 4-epoch substitute for a 20-epoch arm, no swapping the primary metric.

## 5. Budget calibration (measured, not guessed)

The readiness probe showed YOLOP at 7.94 M params could be 45–90 min/epoch — an estimate, explicitly flagged as such. Therefore:

1. A **micro-probe** (≈60 steps, batch 8) is run first and must report ms/step + peak GPU memory.
2. The epoch count / subset size for the real arms is then frozen **from the measured rate**, targeting ≈60–90 min per arm at IDENTICAL settings for all three arms.
3. The frozen numbers are written into the chain script and never differ between arms.
4. A `budget calibration` line is recorded in `phase6_round3_statistics.csv` so the choice is auditable.

*This step is anchor-set-agnostic: it is executed with whichever arm runs first, and it cannot favour an arm because no arm has produced a metric yet.*

## 6. Noise scale and gates

R2's noise rulers (4ep 1× = 0.0146; 20ep σ = 0.0048) belong to **R2** and are **not** assumed to transfer. Round 3 is 1-seed screening, so the second architecture's seed noise is **unmeasured**.

Pre-registered handling:
- The screening gate uses **2σ_R2 = 0.0096 mAP50** (and 0.0096 mIoU) as a **conservative magnitude floor only**, labelled `floor = 2σ_R2, architecture-noise unmeasured`.
- A screening verdict of SUPPORTED additionally requires **strict monotonicity** across the 3-point ladder (Spearman ρ = +1 on the pre-registered order A0 < A1 < A2 for detection; B0 < B1 and B0 < B2 for lane).
- Anything that passes the floor but fails monotonicity → **WEAK SUPPORT**.
- Confirmatory 2–3 seed runs are launched **only** if screening returns SUPPORTED. Never launched to convert a WEAK into a SUPPORTED.

| tier | condition |
|---|---|
| **SUPPORTED** | sign correct **and** monotone (ρ=+1) **and** span ≥ 0.0096 |
| **WEAK SUPPORT** | sign correct, but monotone fails **or** span < 0.0096 |
| **REJECTED** | sign wrong (flat counts as wrong at the floor: a ladder spanning < 0.0096 with ρ ≠ +1 is reported as REJECTED for the *mechanism*, with the raw numbers shown) |

## 7. EXP-9B statistics (pre-registered formulas)

```
Spatial gain     = lane_mIoU(B1) - lane_mIoU(B0)
Channel gain     = lane_mIoU(B2) - lane_mIoU(B0)
Spatial advantage = Spatial gain - Channel gain      <- the quantity under test
```
Secondary (reported, not primary): `lane_fg_iou`, `lane_pixel_acc`. **FPS is not a criterion** (project convention).
`da_mIoU` is reported as a **guard**: an intervention that also moves DA means the manipulation was not lane-local, and the row is annotated `da-moved`.

## 8. Verdict mapping (Case A–D) and final recommendation

| case | condition | consequence |
|---|---|---|
| **A** | H1 SUPPORTED **and** H2 SUPPORTED | strong evidence for task-conditioned bottleneck dimensions → Round 4 new-model construction may start |
| **B** | only H2 | lane spatial bottleneck appears transferable; detection assignment may be architecture-dependent → Round 4 around lane + allocation only |
| **C** | only H1 | supervision bottleneck appears transferable → **redefine** the architecture hypothesis |
| **D** | neither | Round 1–2 were probably R2-specific → **do not pile on new modules**; choose a third architecture, or reposition the contribution as an architecture-specific empirical study, or run the mechanism-review fallback |

`Final architecture recommendation ∈ {PROCEED, REVISE, STOP}` is a **derived** field: it must equal the action implied by the case above. It may not be written independently of the case.

**Hard rule:** a single-architecture observation is **never** upgraded to a general law. Every Round-3 sentence that generalises must cite two architectures with the ruler (§3) held fixed.

## 9. Deliverables

`experiments/phase6/phase6_round3_architecture_audit.md` · `phase6_round3_detection.csv` · `phase6_round3_lane.csv` · `phase6_round3_statistics.csv` · `phase6_round3_decision.md` · updated `phase6_novelty_matrix.csv`.

## 10. Execution order (linear; no parallel or unordered expansion)

```
STEP-0  architecture audit (0 GPU)                       [DONE 2026-09-10]
STEP-1  zero-training assignment + lane baselines (0 GPU) [DONE 2026-09-10]
STEP-2  micro-probe -> freeze budget -> 3x EXP-9A arms, then 3x EXP-9B arms
STEP-3  tier verdicts -> Case A-D -> decision.md + novelty matrix
```
STEP-2 does **not** auto-proceed to 2/3-seed confirmation; that is a separate, user-approved gate.

## 11. Amendments

*(none yet — any future entry states date, what changed, why, and what was already observed at the time)*

### 11.1 — 2026-09-10 23:4x CST: STEP-2 budget frozen; H1 read-out is split into an exact instrument claim and a power-limited accuracy claim

**Observed at the time of this amendment: no Round 3 arm had run.** STEP-0/STEP-1 were complete (zero-training, 0 GPU); the only GPU work so far was the 60-step micro-probe, which produces no arm metric. Nothing below is informed by any arm result.

**(a) Frozen budget, from the measured micro-probe (pre-registered §5 slot filled in).**

| family | batch | images | epochs | steps | measured rate | projected wall | peak mem |
|---|---|---|---|---|---|---|---|
| YOLOP (EXP-9A) | 8 | 4 000 of 69 863 | 2 | 1 000 | 4 887 ms/step | **≈81 min/arm** | 8 139 / 8 151 MiB |
| TLP-small (EXP-9B) | 16 | full 69 863 | 3 | 13 098 | ≈360 ms/step | **≈79 min/arm** | ≈4 340 MiB |

Both land inside the pre-registered 60–90 min window, and the two families are matched to within ~3 % of each other in wall clock so neither architecture receives a time advantage.

*Rejected configuration, recorded so it is not retried:* YOLOP batch 16 measured **55 073 ms/step at peak 16 017 MiB** on an 8 151 MiB card — an oversubscribed run that thrashes into shared memory (11× slower than batch 8). TLP batch 24 was also rejected on throughput grounds (23.9 ms/img vs 20.5 ms/img at batch 16), not memory.

**(b) Declared deviation: YOLOP cannot use the full split.** R2's screening budget was **full `tri_train` (69 863) at batch 16, ≈26 min per 4-epoch arm**, because R2 is 0.2 M params. At batch 8 the same split costs **11.9 h/epoch** for YOLOP. YOLOP therefore trains on a **4 000-image subset (5.7 % of `tri_train`)**. Consequence, declared: Round 3's YOLOP arms have far fewer optimiser steps than R2's screening arms, so *cross-architecture comparisons of trained accuracy are budget-confounded* and are reported with that label. The anchor ladder itself remains FLOPs-neutral and single-variable.

**(c) Registered design choice: EXP-9B lr = 1e-4** (vs 1e-3 for EXP-9A). The arms start from the **released** TwinLiteNetPlus checkpoint, whose lane head is already at mIoU 0.6018 and whose whole family spans only +0.031 mIoU over 58× parameters. A 1e-3 LR over a 2–3 epoch budget would move the pretrained extractor far enough that all three arms converge toward "equally damaged", destroying the contrast. 1e-4 lets the added zero-init paths learn while the released extractor stays near its optimum. Fixed before any arm ran; not tuned after seeing a result.

**(d) Substantive correction — the H1 read-out must be split, because the affordable budget is probably underpowered for trained mAP50.**

R2's *own* numbers constrain what a ~1 000-step ladder can show. In R2's 4-epoch screening (17 464 steps, full data), the assignment main effect on mAP50 was **−0.0044**, i.e. the *wrong sign and below* R2's own 1× noise unit of 0.0146; the positive anchor effect (+0.14 mAP50) appeared only at the **20-epoch** confirmatory budget (≈87 320 steps). Round 3's YOLOP arms are ~1 000 steps — roughly **1/87** of that. The instrument is the same and the contrast magnitude is the same (≈48–49 % → ≈4 % zero-positive), so the honest expectation is a flat trained ladder.

Consequently H1 is decomposed, and this split is fixed now:

| component | evidence | status |
|---|---|---|
| **H1-instrument** — the shipped anchor set fails to assign positive targets to a large majority of real GT under the project rule | **STEP-1**, exact, zero-training, 0 GPU: zero-positive rate, mean best IoU, matches/GT | **primary**, and unaffected by budget |
| **H1-accuracy** — that degeneracy costs detection accuracy at matched budget | EXP-9A trained ladder (mAP50) | **secondary**, power-limited |

**Pre-committed interpretation rule (prevents post-hoc spin in either direction):**

- If the trained ladder is flat — span < 0.0096 **and** ρ ≠ +1 — the verdict string is **`UNDERPOWERED-AT-AFFORDABLE-BUDGET`**. It may **not** be reported as REJECTED (absence of evidence at 1/87 of the resolving budget), and it may **not** be reported as SUPPORTED. H1-accuracy is recorded as **UNRESOLVED on this architecture**.
- `SUPPORTED` for H1-accuracy still requires the §6 gate in full: correct sign **and** monotone ρ = +1 **and** span ≥ 0.0096.
- The §8 Case A–D mapping is evaluated with H1-accuracy set to UNRESOLVED, and every Case sentence that depends on H1 must carry the `power-limited` annotation.

**(e) Added mechanistic read-out (free, already logged): the detection-loss trajectory.** Under a degenerate anchor set fewer GT become positive targets, so a YOLOP arm trained on `A0` should show a *systematically lower* `det` loss at matched steps while generalising no better. That divergence between "lower training loss" and "no accuracy gain" is the signature of a supervision bottleneck rather than a capacity bottleneck, and it is measurable at any budget. `det_loss_first` / `det_loss_last` are recorded per arm in `phase6_round3_trained.csv` for this purpose.

**(f) Eval protocol:** all arms of both families are evaluated on the **full** `tri_val` (10 000 images) with the project's existing evaluator; the YOLOP arms pass their anchor set through `YOLOP_ANCHORS_JSON` so the analysed set, the trained set and the decoded set are one object. Each arm's row goes to `phase6_round3_trained.csv` (trained arms) — deliberately a different file from `phase6_round3_detection.csv` (zero-training assignment diagnostics), so the two evidence classes can never be confused.

### 11.2 — 2026-09-11 09:5x CST: STEP-2 首次运行因主机意外断电作废；补臂内 checkpoint/续训（设计不变）

**发生了什么（事后取证，非推测）**

STEP-2 于 2026-09-10 23:37:35 启动，`A0_shipped` 训练至 `step 220/500`（epoch 1）后进程消失。取证结论：

| 证据 | 结论 |
|---|---|
| System Event **41** (Kernel-Power)「系统已在未先正常关机的情况下重新启动」 | 非正常关机 |
| System Event **6008**「上一次系统的关闭是意外的」+ Kernel-Boot **Id=20**「上一次关机的成功状态为 false」 | 非正常关机 |
| `LastBootUpTime = 2026-09-11 09:48:11` | 主机整夜处于断电/关机状态，非休眠 |
| **无 minidump、无 BugCheck、近 7 天无 Event 4101 (TDR)、无 nvlddmkm 错误** | 不是蓝屏，**不是 GPU 驱动崩溃**，OS 未记录任何故障 |
| 训练日志末条 `23:55:23`，文件 mtime `23:56`，之后戛然而止 | 断电时刻 ≈23:56，与日志截断完全吻合 |
| 旁证：2026-09-09 00:41 有一次 `0x00000116 (VIDEO_TDR_FAILURE)` 导致的非正常重启 | 该主机 3 天内非正常关机 2 次 |

**结论：这是主机层面的突发断电/硬关机，与本项目代码、与 OOM、与 GPU 资源耗尽均无关。** 未注册任何与本实验设计有关的失败。

**损失**：`A0_shipped` 19 分钟进度（220/1000 步）全部丢失——因为臂内不落盘，只在 epoch 结束存 checkpoint。其余 5 臂从未启动。**零有效结果**。仓库、STEP-0/1 全部产物、权重、数据集经 `git fsck` + 逐项核对确认**完好无损**。

**修复（仅鲁棒性，不改设计、不改预算、不改判据）**

1. **臂内滚动 checkpoint**：两个训练器新增 `--ckpt-every N`（YOLOP/链中 100 步、TLP/链中 200 步），写 `checkpoint.pt` + `progress.json{epoch,step,done}`。**原子写入**——先写 `*.tmp` 再 `os.replace()`，因此断电恰落在写盘瞬间也不会留下半个损坏的 checkpoint。
2. **断点续训** `--resume`：从 `progress.json` 恢复 `(epoch, step)`，重启后只补未完成部分。**最坏损失从"整臂 81 min"降到 ≤ `ckpt-every` 步（YOLOP ≈8 min、TLP ≈1.2 min）。**
3. **每轮确定性 shuffle**：loader 改为按 `(seed, epoch)` 单独播种（`seed*1000+ep`），使任一 epoch 的批顺序仅由轮号即可复现。**理由**：否则续训会静默重放一个与中断前**不同**的顺序，属静默协议漂移。三臂共用同一方案，比较有效性不受影响。（已注册：首次运行的数据顺序因此与本次不同，但首次运行未产出任何可用结果。）
4. **已完成臂的早退守卫**：若 `progress.json.done=true` 且 `metrics.json` 存在，直接退出，**避免用空 loss 覆盖已写好的 metrics.json**（该 bug 已在续训自测中被发现并修掉）。

**验证（实测，非声明）**：YOLOP 臂训练至 `step 10/30` 后被 `timeout` 强杀 → `progress.json={"epoch":1,"step":10}` → 续训日志 `RESUME ... epoch=1 step=10`，从 `ep1 step 15` 继续，`1–10` 步未重复，两轮跑完 `done:true`。TLP 臂同样通过（并验证了"已完成则全部跳过"路径）。`bash -n` 与 `py_compile` 均通过。

**未变更**：冻结预算（YOLOP 4000×2ep@bs8；TLP 全量×3ep@bs16）、lr（1e-3 / 1e-4）、判据（§6/§7）、H1 拆分与 `UNDERPOWERED-AT-AFFORDABLE-BUDGET` 规则（§11.1(d)）、Case A–D 映射（§8）**全部原样**。链子 `--resume` 幂等：已完成臂跳过，中断臂接着跑。

### 11.3 — 2026-09-11 10:2x CST：更正 §5 预算冻结——batch 8 测到的不是"慢"，是**显存悬崖失速**；YOLOP batch 8 → 6

**触发**：STEP-2 重启前复核时发现 `batch 6`（480 ms/step）与 `batch 8`（4824 ms/step）相差 10×。批量只差 25%、时间差 10 倍，违反任何缩放律，故做受控复测并采集 GPU 遥测。

**受控复测**（同一脚本、同一数据、同一 seed，交替采样；实测值，非声明）

| batch | ms/step | 峰值显存 | 占卡比 | SM 时钟 | 功耗 | util | 每图 |
|---|---|---|---|---|---|---|---|
| 6 | **306** | 6085 MiB | **74.7 %** | 2925 / 3090 MHz | **83–85 W** | 90–95 % | 51 ms |
| 8 | **5532** | 8139 MiB | **99.85 %** | **2970 / 3090 MHz（顶格）** | **28–33 W** | 100 % | 692 ms |

**判据（关键）**：batch 8 下 SM 时钟已拉到**满频**却仅耗 **30 W**，util 却报 100%。**时钟顶格 + 功耗极低 = GPU 在空等，不是在算。** batch 8 比 batch 6 只多做 33% 的活，却慢 **18×**，单位时间做功反而更少 —— 这是**阈值效应（踩到显存天花板）**，不是平滑的容量限制。

机制：Windows/WDDM 下超额申请**不抛 OOM**，而由驱动把显存**静默换页到主机内存**（走 PCIe），于是每个 step 都在等内存事务。可用显存实际小于标称 8151 MiB（显示驱动有预留），batch 8 的 8139 MiB 已越线。

**推论 —— 本步的限制不是"显存不够"，而是"跨过了那条线"。退回线内是零成本的**

| 配置 | 单臂墙钟 | 3 臂 YOLOP |
|---|---|---|
| batch 8（旧冻结） | 1000 步 × 5.53 s ≈ **92 min** | ≈ 4.6 h |
| batch 6（新冻结） | 1333 步 × 0.306 s ≈ **6.8 min** | ≈ **20 min** |

**修订（唯一改动：YOLOP `batch_size 8 → 6`）**

- YOLOP：图像 4000、epochs 2、seed 0、lr 1e-3、wd 5e-4、grad-clip 10 **全部不变**。
- TLP：**不动**（峰值 ≈4340 MiB = 53 %，远在悬崖之下；batch 16 已是该家族最优）。
- **注册的副作用（必须披露）**：batch 变小 ⇒ 每 epoch 步数 500 → 667、总优化步数 1000 → 1333（+33 %）；梯度噪声与 BN 统计随之改变。
  - **不损害比较有效性**：六臂在各自家族内配置完全一致，臂间为单变量对照。
  - **与 R2 的横向对照**：R2 筛选用 batch 16，故 Round 3 的绝对数值与 R2 **不构成数值等价**；R2 仅作定性参照（§11.1(d) 的检力讨论与 batch 无关，不受影响）。
- **本次修订的性质**：§5 冻结源自一次**病态测量**（把失速读数当成了正常速率）。以可事前验证的物理机制更正测量误差，属修订的正当用途；**不是**在看到训练结果后调参（此刻六臂仍无一产出结果）。
- **判据全部不变**：§6 噪声门、§7 公式、§8 Case A–D、§11.1 的 H1 拆分与 `UNDERPOWERED-AT-AFFORDABLE-BUDGET` 规则**原样**。

**防复发（工程，本次最重要的系统性收获）**：两个训练器新增**显存悬崖守卫**（⚠️ **本句有误，见 §11.4 更正**：该次编辑实际只对 YOLOP 生效，TLP 的检查块漏写、`vram_warned` 标志成了死变量）—— 每个 step 比较 `max_memory_reserved / total_memory`：**>0.96 立即 `RuntimeError` 中止**该臂（附明确指引），**>0.90 打印一次告警**。今后任何"显存贴着天花板"的配置都会**大声失败**，而不是静默慢 18 倍。

**对"要不要租卡"的直接影响（记录在此以免事后重构）**：修好 batch 后，YOLOP 三臂从 ≈4.6 h 降到 ≈20 min；剩余墙钟由 **TLP 主导（≈79 min/臂 × 3 ≈ 4 h）**，而 TLP 峰值仅 53 %、**不受显存限制**。因此就当前冻结预算而言，Round 3 **无需租卡即可在本机跑完**。

---

## 11.4 —— EXP-9A 仪器失效、作废与重跑（2026-09-11，训练前三项均已修复）

### (a) 发现过程

首轮 EXP-9A 三臂 10:29–10:54 训完即出数字。**触发排查的不是数字本身，而是一条结构不变量**：三臂
`lane_mIoU`(0.6038) 与 `da_mIoU`(0.9115) **逐位相同**。lane/DA 头是联合训练的，若真加载了各自权重不可能四位小数全同。由此定位 D1。

**D1 掩盖了 D2**：只修 D1，会得到一批"有区分度、看起来像结果"的数字（0.0677/0.0971/0.1049），
而它们产自一个输入分布错配的仪器。两者必须一起修。

### (b) 缺陷清单（三项，已修 + 实测验证）

| # | 缺陷 | 诊断证据 | 修复与验证 |
|---|---|---|---|
| **D1** | `evaluate_baseline.build_yolop` 无权重入口，恒读 `weights/YOLOP_End-to-end.pth`；链子只传锚框 | 三臂 lane/DA 逐位相同 | 加 `YOLOP_WEIGHTS` + 三重守卫；实测 `delta=1.07e+03 / 542 tensors`、缺权重硬报错、缺文件硬报错 |
| **D2** | 训练器喂 `[0,1]` 原图，评估器按 ImageNet 归一化（同一数据集、两种分布） | 修 D1 后 200 图 mAP50 0.0677 vs 发布 0.7712 | 抽出唯一函数 `normalize_batch()` 供两侧共用；1 epoch 标定 → **0.5151**（原 0.0677） |
| **D3** | TLP 训练器**缺**显存悬崖守卫（§11.3 声称"两个训练器"有误：该次编辑只对 YOLOP 生效，TLP 的 `vram_warned` 成了死变量） | 代码审查 | 补入检查块；实测 batch 32 → **ABORT 120.5 %** |

### (c) 证据分级（作废 vs 保留）

**作废、不得引用**（已归档 `round3_invalid_evidence/`，含 README 说明）：

| 来源 | 数字 | 原因 |
|---|---|---|
| 首轮全量评估 | mAP50 A0=0.7657 / A1=0.0000 / A2=0.0005 | D1：评的是发布模型 |
| 修 D1 后 200 图 | mAP50 A0=0.0677 / A1=0.0971 / A2=0.1049 | D2：输入分布错配 |

> ⚠️ 第二行的排序 A2>A1>A0 **恰与 H1 预测同向**。但该仪器已被证明**没有测量目标变量**，
> 且存在纯机械解释（kmeans 锚框数值更大，在错配下受害更小）。**记录在案，不作为证据。**

**不受影响、仍然有效**：

- STEP-0 架构审计、STEP-1 零成本分配诊断；**H1-instrument = SUPPORTED**（span 0.4567 = 47.6× 下限，与任何训练无关）
- TLP 四档 lane 基线（nano/small/medium/large = 0.5861/0.6018/0.6089/0.6170）
- **EXP-9B 全套**：其评估显式传 `--round3-weights` 且有 strict mismatch 校验；训练/评估两侧同为 `norm="unit"`；
  step-0 恒等 0.000e+00。**TLP 训练器唯一改动是"补上守卫"，不含任何数学变更**，故 B0 已完成的
  进度（epoch2/step3600）**可合法续训**。

### (d) 重跑配置冻结（EXP-9A）

| 项 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| lr | 1e-3（借用 R2 **从零训练**配方） | **1e-4**（预训练微调标准，AdamW） | 判据**事前钉死**：只看"是否保住发布权重"，**不看三臂排序**。实测 2ep/4000 图：1e-3→0.5104，1e-4→**0.5656** |
| batch | 6 | 6 | §11.3 不变 |
| 预算 | 4000 图 × 2 ep @bs6 = 1333 步 | **全量 69 863 图 × 1 ep @bs6 = 11 644 步 ≈45 min/臂** | §5 原规则"按实测速率冻结到 60–90 min/臂"。bs6 实测 234 ms/step，原 1000 步是在**病态速率**（4887 ms）下定的 |
| seed / wd / 调度 | 0 / 5e-4 / CosineAnnealing / grad-clip 10 | **不变** | — |
| 判据（§6/§7/§8/§11.1） | — | **一律不变** | — |

**必须披露**：本阶段已**第二次**因"病态速率"修订冻结值（§11.3 与本节），两次根因同为显存悬崖。
修正后步数为原预算的 **11.6×**，且首次使用全量训练数据 —— 这是对 §11.1(d) 记录的"可负担预算检力不足"
的**源头缓解**，而非事后调参。

**待用户确认**：预算方案（全量×1ep 推荐 / 全量×2ep / 4000图×16ep）。确认前不启动重跑。

### (e) 新增门禁（4 道，全部带自检）

1. **权重加载证明**：eval 日志必须含 `WEIGHTS OVERRIDE` 且 `trained-vs-released delta > 0`；逐位相同则拒绝出分
2. **输入一致性门禁** `scripts/round3_input_parity_check.py`：L0 数据集确定性 / L1 训练与评估张量**逐元素相同** /
   L2 非平凡（两种归一化必须不同）/ L3 **结构**（两个训练器不得含私有归一化常量，必须走共享函数）。
   链子启动即跑，fail-closed。自检：植入私有常量 → 正确 FAIL
3. **显存悬崖守卫**：两个训练器**均有**（>0.96 中止 / >0.90 告警）
4. **结构不变量**：三臂 `lane_mIoU`/`da_mIoU` 不得逐位相同（D1 的探测器）

### (f) 不变声明

H1/H2/H3 定义、H1 的三项拆分、`UNDERPOWERED-AT-AFFORDABLE-BUDGET` 判停写法、Case A–D 映射、
噪声门与判据公式、以及"禁止用不显著冒充证伪"—— **全部原样，未作任何放宽。**

## 11.5 Amendment — STEP-2 completion, D5 defect, lane-monotonicity ambiguity

*Appended 2026-09-11 after the STEP-2 chain ran to completion. Nothing above is modified.*

### (a) What changed, and what was already observed when it changed

The chain launched 11:53:02 under the user-selected option **O1 (EXP-9A: full `tri_train`
69 863 imgs x 1 ep @ bs6, lr 1e-4; EXP-9B: full x 3 ep @ bs16)** and finished 16:49:10
(4 h 56 m), all six arms evaluated. At the time this entry was written the verdicts had
**already been computed** by `phase6_round3_verdict.py` and are recorded in
`phase6_round3_decision.md`. No threshold, tier rule, order or formula in sections 6-8 was
changed before, during or after the run -- including after seeing the numbers.

**Change:** two engineering defects (b) and one pre-registration ambiguity (c) are recorded
here. Both verdicts are unaffected by either (argued below).

### (b) D5 — CSV column shift (same failure class as a silent NA)

The first STEP-2 result write joined fields with `,` without quoting. The intervention label
`learned 1/4 tap into lane head (zero-init, +204 params)` contains a comma, so every
subsequent cell in the B1 row shifted by one: `lane_mIoU` read `0.8568` (the `da_mIoU`
value) instead of `0.6004`. B2 was shifted identically. The chain's own printed table and any
`column -s, -t` view rendered it *plausibly* -- the corruption was invisible to the eye.

*Caught by:* the verdict script reading the file, not by inspection.

*Fix (structural, not a patch):* the appender now writes via `csv.writer` (auto-quoting) and
then **reads the file back with `csv.DictReader`**, asserting the appended row's `mAP50`,
`lane_mIoU` and `intervention` equal what was just written; any mismatch aborts the append.
All six arms re-appended, each printing `[readback OK]`. The corrupted file is preserved as
`round3_invalid_evidence/phase6_round3_trained.COLSHIFT.csv`.

*Verification:* every cell of the rebuilt CSV was cross-checked against its source
`*_eval/metrics.json` (6/6 exact), plus structural width checks on all four result tables.

### (c) Lane-monotonicity ambiguity (section 6 prose vs section 6 code) — recorded, not silently resolved

Section 6 prose defines lane monotonicity as "`B0 < B1` and `B0 < B2`". The rule as encoded
in `verdict_exp9b` tests only `B0 < B1`. On this data the two readings disagree
(`B0 = 0.5988`, `B2 = 0.5985`, so `B0 < B2` fails).

**The H2 verdict is robust to the disagreement.** The tier function returns WEAK SUPPORT
whenever the sign is correct and the ladder is not simultaneously (monotone *and* >= floor);
under either reading the ladder fails that conjunction (`span = 0.0019 < 0.0096`), so both
readings yield WEAK SUPPORT. Recorded for auditability and to be disambiguated before any
Round 4 pre-registration is frozen.

### (d) Verdicts (details and raw numbers in `phase6_round3_decision.md`)

| block | tier | basis |
|---|---|---|
| H1.instrument | SUPPORTED | zero-training, span 0.4567 = 47.6x floor, strict monotone |
| H1.accuracy | SUPPORTED | mAP50 0.5407 / 0.6846 / 0.7400, rho = +1, span = 20.8x floor |
| H2.lane | WEAK SUPPORT | spatial advantage +0.0019 = 0.20x floor, ladder not monotone |

**Case C -> `REVISE (redefine architecture hypothesis)`**, derived from section 8, not
written independently.

### (e) What has NOT happened

- No 2/3-seed confirmation was launched. Per section 6, confirmatory seeds follow a
  SUPPORTED screening; H2 screened WEAK, and a WEAK is never converted by re-running.
- No Round 4 work has begun. Case C is a finding, not an authorisation.
