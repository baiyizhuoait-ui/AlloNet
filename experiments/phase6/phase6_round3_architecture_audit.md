# Phase 6 — Round 3 Architecture Audit

**Generated:** 2026-09-10 23:4x CST · **Repo HEAD:** `9fb29e7` (+ uncommitted Round 3 additions) · **GPU:** RTX 5060 Laptop 8 GB
**Method:** every number below was **measured** (model construction + forward hooks + parameter counting + the real FLOPs counter), not transcribed from a README. Anything inferred is marked *inferred*.
**Status:** read-only audit. No training was run to produce this file.

---

## 0. Executive summary

| question | answer |
|---|---|
| Is a second architecture available at all? | **Yes**, two usable ones (YOLOP for H1, TwinLiteNetPlus for H2) |
| Is YOLOP a valid independent testbed for H1? | **Yes** — and, measured, its detection head carries the *same pathology class* as R2's known-broken baseline |
| Is YOLOP a valid testbed for H2? | **No** — its lane branch already consumes a 1/2-resolution feature and already outputs at 1/1. No headroom for a "higher resolution" intervention |
| Is TwinLiteNetPlus a valid testbed for H2? | **Yes, with a correction** — the lane head does see 1/2 and 1/4 *resolution*, but only as **unlearned average-pooled RGB**; the reachable headroom is a **learned** high-resolution tap |
| Does one architecture cover everything? | **No.** Round 3 therefore uses **two** second architectures. This is a declared limitation (pre-registration §2) |
| Correction issued | the readiness study's claim "YOLOP's own anchors are autoanchor outputs ⇒ no contrast" is **falsified** — see §5 |

**Scale reference.** R2 (the incumbent family) is **0.2014 M / 1.0796 G**. YOLOP is **7.9408 M / 30.8208 G** (39.4× params, 28.5× FLOPs); TwinLiteNetPlus spans **0.0334–1.9439 M** with the primary preset `small` at **0.1216 M / 4.667 G**.

---

## 1. Candidate survey (what was available)

| architecture | det | DA | lane | params (measured) | code location | weights | in-repo train entry |
|---|---|---|---|---|---|---|---|
| **YOLOP** | ✅ | ✅ | ✅ | **7 940 846** | **outside the repo** (`../YOLOP`) | `weights/YOLOP_End-to-end.pth` (95.7 MB) | **none** → adapter written (`scripts/phase6_round3_yolop_train.py`) |
| HybridNets | ✅ | ✅ | ✅ | not probed | outside the repo | `weights/hybridnets.pth` | none |
| TwinLiteNet | ❌ | ✅ | ✅ | — | in repo `baselines/` | ✅ | ✅ |
| **TwinLiteNetPlus** | ❌ | ✅ | ✅ | nano 33 379 / **small 121 552** / medium 478 876 / large 1 943 911 | in repo `baselines/` | ✅ 4 presets | ✅ |
| TriLiteNet | ❌ | ✅ | ✅ | tiny 151 413 / small 592 338 / base 2 350 206 | in repo `baselines/` | ✅ | ✅ |

`evaluation/evaluate_baseline.py` itself already registers TwinLiteNet / TwinLiteNetPlus / TriLiteNet as **two-task** (`with_det=(name != "TwinLiteNet" and not name.startswith("TwinLiteNetPlus"))`), so the three-task candidates are YOLOP and HybridNets only. HybridNets was not probed because YOLOP already satisfies H1 and the H2 slot is filled by a family with a much finer compression gradient.

**Selection (user-approved, option 1.A): two testbeds** — H1→YOLOP, H2→TwinLiteNetPlus(family).

---

## 2. YOLOP — measured structure

**Provenance:** `hustvl/YOLOP` @ `8d8f68df318c71f01d6f813c024df646c7d1978f` (2023-10-20), external working tree **clean**.
`evaluation/evaluate_baseline.py::build_yolop()` resolves the code via `sys.path.insert(0, ROOT/../YOLOP)`.

| item | measured value |
|---|---|
| params / FLOPs | **7 940 846 (7.9408 M)** / **30.8208 G** (640×640) |
| backbone | `Focus` → 4× `BottleneckCSP`, strides **2 / 4 / 8 / 16 / 32** |
| **detection feature levels** | blocks **17 / 20 / 23** (`BottleneckCSP`), feature maps **80×80 (s8) / 40×40 (s16) / 20×20 (s32)** |
| detection head | `Detect` @ block **24** (16 182 params), `nl=3`, `na=3` → 9 anchors, `stride=[8,16,32]` |
| shipped anchors (px on the 640 canvas) | L1 `3×9, 5×11, 4×20` · L2 `7×18, 6×39, 12×31` · L3 `19×50, 38×81, 68×157` — **all tall, h/w 2.13–6.50** |
| **DA feature source** | block **33** (`Conv`, 148 params) → **s1 = 640×640**; path s8→s4→s2→s1 |
| **lane feature source** | block **42** (`Conv`, 148 params) → **s1 = 640×640**; path s8→s4→**s2 (320×320, blocks 39/40)**→s1 |
| detection bypass | **present** — det reads FPN s8/16/32 directly and completely bypasses the seg/lane up-sampling tower |
| lane high-resolution path | **already present and already reaches s1** |
| output shapes | det `25200×6` (decoded); DA `1×2×640×640`; lane `1×2×640×640` |
| parameter distribution | backbone/neck ≈ **7.92 M of 7.94 M (99.7 %)**; all three task heads together ≈ **16.5 k (0.21 %)** |

### 2.1 Why YOLOP is a valid independent testbed for H1

1. **Not a re-skin of R2.** R2 = a 0.2014 M light encoder + one Z bottleneck + task projections. YOLOP = CSPDarknet trunk + FPN/PAN + SPP + three heads. 39.4× the parameters, 28.5× the FLOPs, and a completely different information route.
2. **Anchor-based detection**, i.e. the *same family* as R2's R0 head, so the intervention (the anchor set) is meaningful in both.
3. **Same asymmetric signature, different implementation.** det reads deep FPN levels while DA/lane climb a separate up-sampling tower — structurally analogous to R2's "detection head bypasses Z", but built from unrelated code.
4. **High-capacity control.** Round 2's closing result (EXP-09) was that at high capacity the supervision lever **vanishes** (SUBSTITUTION, Δ_anchor = −0.0007). YOLOP is a genuinely high-capacity architecture, so the direction of the test is **predictable in advance**: finding a live supervision lever here would *refute* "capacity absorbs supervision"; finding none *supports* its transferability. A pre-registered, two-sided prediction.
5. **Evaluation side already wired** — `build_yolop` constructs it, `evaluate_baseline.py` scores all three tasks, the checkpoint is in-repo. Only the *training* path was missing.

### 2.2 Why YOLOP is the WRONG testbed for H2

Measured: the lane branch reads **s2 (320×320)** and already emits at **s1 (640×640)**. The specification's intervention is "add one very small *higher-resolution* feature path". On YOLOP there is no higher resolution to add (the input side is the only place left, which is meaningless). Forcing it would mean *removing* resolution and restoring it — a different experiment, on an external repository's code. **H2 is therefore not run on YOLOP.**

---

## 3. TwinLiteNetPlus — measured structure

In-repo (`baselines/TwinLiteNetPlus/`), ESPNet-style encoder; 4 presets from `model/config.py`.

| preset | params | FLOPs (G) | encoder out (learned) | `inp2` tap | `inp1` tap |
|---|---|---|---|---|---|
| nano | 33 379 | 1.588 (bare) / 1.886 (eval path) | `(16, 80, 80)` = **1/8** | `(3, 160, 160)` = 1/4 | `(3, 320, 320)` = 1/2 |
| **small** (primary) | **121 552** | **4.213 / 4.667** | `(32, 80, 80)` = **1/8** | `(3, 160, 160)` = 1/4 | `(3, 320, 320)` = 1/2 |
| medium | 478 876 | — | `(64, 80, 80)` = 1/8 | 1/4 | 1/2 |
| large | 1 943 911 | — | `(128, 80, 80)` = 1/8 | 1/4 | 1/2 |

Parameter distribution (preset `small`): encoder **105 022**, `caam` 4 354, `conv_caam` 4 656, lane branch (`up_1_ll` 1 952 + `up_2_ll` 1 696 + `out_ll` 112) = **3 760**, DA branch the same 3 760.

### 3.1 The critical measurement (and the correction it forces)

`Encoder.forward` returns `(out_encoder, inp1, inp2)` where
`inp1 = AvgDownsampler(1)(input)` and `inp2 = AvgDownsampler(2)(input)`
(`model/model.py`, `Encoder.forward`). Those are **average-pooled raw RGB with 3 channels** — *unlearned*. They are concatenated into the up-conv blocks via `torch.cat([x, ori_img], dim=1)`.

So the honest statement is:

> The TwinLiteNetPlus lane head **does** receive 1/2 and 1/4 *resolution*, but only as unlearned RGB averages. **No learned feature above 1/8 reaches it.** The reachable spatial headroom is therefore **not more resolution — it is a learned high-resolution tap.**

This is a genuine correction to the readiness study, which said only "lane and DA share the 1/8 encoder output, no high-resolution bypass". The shared *learned* source is 1/8, yes — but the head is not resolution-starved in the naive sense, and a design that simply "adds a 1/4 lateral" would have been built on a mis-statement. `models/round3/tlp_variants.py` is built on the measured version.

### 3.2 Why TwinLiteNetPlus is a valid independent testbed for H2

1. **Structured like R2's control pair, not like R2.** "low-resolution shared learned representation → task head, with a choice between injecting high-resolution information vs adding width" is exactly the R2 `l14f1` (1/4 lateral) vs `lch72` (channel-only) contrast.
2. **A fine compression gradient is built in**: 0.0334 / 0.1216 / 0.4789 / 1.9439 M — a **58×** range straddling R2's 0.2014 M operating point. The primary preset is **`small`** (0.1216 M), chosen *pre hoc* because it is the closest to R2's operating scale: this makes the transfer test an **interpolation**, not an extrapolation.
3. **Everything is in-repo** — code, weights, training entry. No external dependency, so it passes the hygiene gate without an exception.
4. **Deterministic construction of the arms** (see §6): both interventions are verified **bit-identical to baseline at step 0**, and their FLOPs are matched to **3.48 %** (small) / **0.95 %** (nano), inside the project's ±5 % budget-equivalence tolerance.

### 3.3 Zero-training compression gradient (2000-image `tri_val` subset, released weights, pure eval)

| preset | params | FLOPs (G) | **lane_mIoU** | lane_fg_iou | da_mIoU | da_fg_iou |
|---|---|---|---|---|---|---|
| nano | 33 379 | 1.886 | 0.5861 | 0.1845 | 0.8615 | 0.7736 |
| **small** | 121 552 | 4.667 | **0.6018** | 0.2159 | 0.8994 | 0.8339 |
| medium | 478 876 | 15.425 | 0.6089 | 0.2299 | 0.9189 | 0.8663 |
| large | 1 943 911 | 58.569 | 0.6170 | 0.2464 | 0.9277 | 0.8806 |

**Reading:** across a **58×** parameter increase, lane_mIoU moves only **+0.031** (0.5861→0.6170) while da_mIoU moves **+0.066**. Lane is close to saturated with respect to *capacity* in this family.

This is the third independent echo of the same phenomenon: R2's EXP-02 measured a spatial-vs-channel lane difference of only **0.0007** (< 2σ), and R2's Phase 3A found DA/lane saturating at `E-large`. It is therefore *pre-registered expectation*, not a surprise, that EXP-9B will land in the **WEAK SUPPORT** band. That is exactly why the pre-registration fixes a WEAK tier and forbids "adding seeds until it exists".

*Comparability caveat (declared):* these numbers come from a 2000-image subset, whereas R2's lane_mIoU values were computed over the full 10 000-image `tri_val`. The **gradient** above is internally consistent (same subset, same protocol); the **absolute** cross-architecture comparison is not like-for-like and is not used for any claim.

---

## 4. Assignment-rule differences (a registered confound, measured)

| | matcher | threshold | neighbour expansion |
|---|---|---|---|
| **trac / R2 (the ruler Round 3 holds fixed)** | `losses/yolo_loss.py:build_targets` | `0.5 < box/anchor < 2.0` | **none** |
| YOLOP native | `lib/core/postprocess.py:build_targets` | `TRAIN.ANCHOR_THRESHOLD = 4.0` (`lib/config/default.py`:100) | **5-neighbour offsets**, `g = 0.5` |

Two different rulers cannot be pooled. Round 3 therefore trains **all** YOLOP arms with the **project rule** (pre-registration §3), so the only thing that changes across arms is the anchor set. The native-rule readings are carried alongside as a measured confound:

| anchor set | zero-positive *(project rule)* | zero-positive *(native rule t=4.0)* |
|---|---|---|
| YOLOP shipped | **49.41 %** | 0.57 % |
| aspect-flip | 16.66 % | 0.33 % |
| k-means refit | **3.74 %** | 0.04 % |

**A "supervision hole" is a property of the ruler as much as of the anchors** — the same shipped set goes 49.4 % → 0.6 % by moving `t` from 2.0 to 4.0. Every Round 3 statement about holes names its rule.

---

## 5. Correction issued (readiness study → this audit)

The readiness study (`PHASE6_ROUND3_READINESS_STUDY.md` §4) argued that EXP-9A's contrast had **degenerated to zero** on YOLOP, on the grounds that YOLOP's shipped anchors *are* autoanchor (`k-means + genetic evolution`) outputs for BDD100K.

**That was an inference, and the measurement falsifies it.** `NEED_AUTOANCHOR = False` plus the presence of `lib/utils/autoanchor.py` shows only that the tools exist and that the hard-coded set is used as-is — it does not show the hard-coded set is data-optimal. Measured on 32 421 GT boxes:

* YOLOP's shipped anchors leave **49.41 %** of GT with zero positive assignment under the project rule — statistically indistinguishable from R2's **known-broken** pre-2026-09-08 default (**48.51 %**).
* They are **all tall** (h/w 2.13–6.50) against a GT whose aspect h/w median is **0.80**.
* Their mean best-anchor IoU is **0.4161** versus **0.6805** for a k-means refit.

So the contrast does **not** degenerate; it is in fact *stronger* than assumed. Consequences:
* the planned synthetic "degradation" arm is **dropped as redundant** — the shipped set already *is* the degraded end;
* EXP-9A becomes a 3-point **dose ladder** (0.4161 / 0.5419 / 0.6805 mean best-IoU), which can distinguish "no effect" from "non-monotone effect" — something a 2-arm contrast cannot do.

**A second, unplanned finding:** the k-means refit reproduces R2's k-means anchor set **value-for-value** under a different architecture's level layout, because anchors are a function of (dataset, rule, k) and nothing else. The *repair* is architecture-independent; only its *consequences* can be architecture-specific. That is precisely the structure H1 needs to be testable across architectures.

---

## 6. Engineering controls put in place

| risk (all three are instances of this repo's recurring silent-failure class) | control |
|---|---|
| **Second-architecture code lives outside the repo** (`sys.path.insert(ROOT/../YOLOP)`); a rename there would break or, worse, silently swap the model | commit `8d8f68d` frozen in this document **and** registered in `scripts/hygiene_exceptions.txt` (option 3.A) |
| **Anchors are hard-coded inside the external repo** (4 sites in `lib/models/YOLOP.py`) | never edited upstream; overridden **from the trac side** on both the buffers (`det.anchors`, `det.anchor_grid`) after construction |
| **Eval would silently decode with the shipped anchors** and score every arm identically | the anchor set travels with the checkpoint (`anchors.json`) and eval applies it via `YOLOP_ANCHORS_JSON`; a missing/invalid path is **fatal**, not ignored |
| **Analysed anchor set ≠ trained anchor set** | the trainer reads the STEP-1 artefact `phase6_round3_anchor_summary.json`; `shipped` is re-read live from the built model. `flip`/`kmeans` come from the file. No copied literals |
| **No in-repo training path for YOLOP** | `scripts/phase6_round3_yolop_train.py` — thin adapter, R2 protocol, anchor-set switchable |
| **Arm pairs that are not FLOPs-matched** would be uninterpretable | `scripts/phase6_round3_lane_probe.py` measures real FLOPs and refuses (`NOT-RUN-able`) if the gap exceeds ±5 % |
| **Added lanes that are not exact no-ops at step 0** would confound init with intervention | `selfcheck()` asserts `max|Δ| = 0` vs baseline for both arms (it fired on its author's first, wrong version — the gate works) |
| **Nested, unregistered `.git` under `baselines/*/`** | recorded here as a known hygiene gap; not touched in this round |

## 7. What this audit does NOT establish

* No claim that any Round 3 hypothesis is true. H1/H2/H3 remain **PENDING** until STEP-2 trains.
* HybridNets was not probed; if a third testbed is ever needed, it must get the same hook-based audit before any use.
* The absolute cross-architecture lane_mIoU comparison (§3.3 caveat) is not like-for-like.
* YOLOP's FLOPs figure is measured at 640×640 through the *evaluation* path; the training path's FLOPs differ slightly (different seg head invocation), which does not affect the audit's conclusions.
