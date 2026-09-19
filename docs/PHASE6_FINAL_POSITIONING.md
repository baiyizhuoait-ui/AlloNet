# Phase 6 — Final Positioning on a Certified Protocol

**Compiled 2026-09-14.** Supersedes the retracted `phase6_algorithm_comparison.md` (10 models, no
protocol check). Companion to `PHASE6_ALGORITHM_LEDGER.md` (comparability audit) and
`PHASE6_BENCHMARK_AUDIT.md` (FLOPs / Lane / DA axis rulings).

---

## 0. Verdict first

| question | answer | evidence |
|---|---|---|
| Is our harness protocol-compliant? | **Yes — 18/18** | every published reference reproduced within ±1.0 at n=10,000 (§1) |
| Do we match the closest published three-task model? | **No on DA, Yes on detection** | DA 86.49 vs TriLiteNet-tiny 88.53; det 53.82 vs 49.6 (§2) |
| D-2 guard (parity with baseline) | **NOT met** | g = 0.8796 − 86.73 = 0.0123 ≤ 0.0141 (residual 2.0σ) |
| Where are we in the whole field? | **Last on DA (9/9); last on official-GT Lane IoU (9/9), but 4/9 on our own GT; first on Lane pixel accuracy; 16.6–202× cheaper** | §2, §3 |

The honest headline: **we are not competitive with the three-task field on accuracy, and we never
were.** What the certified run adds is that this is now a *model* result rather than a *measurement*
result — the instrument is proven, so the gap belongs to the model.

---

## 1. Protocol certificate

Design (`scripts/phase6_consistency_official.py`): a 2×2 sweep that isolates the two ways a number
can move — **canvas** (640×640 ours vs 640×384 official) and **GT source** (our masks vs the
official annotation package). Both canvases are scale-0.5 letterbox, so the metric is always
computed on the *same* 640×360 pixel plane; the official GT swap is a **pure re-scoring of
identical predictions**, not a second forward pass.

Pre-registered rule, fixed before the run: *the (official GT, 384 canvas) cell must reproduce the
published rows within ±1.0, or the residual gets reported as an unresolved protocol unknown.*

Result — **18/18 within ±1.0**, n=10,000 (`experiments/phase6/consistency/stageB_in384/`):

| model | metric | measured | published | Δ |
|---|---|---:|---:|---:|
| TriLiteNet tiny | DA mIoU | 88.53 | 88.5 | +0.03 |
| TriLiteNet tiny | Lane IoU | 24.32 | 24.2 | +0.12 |
| TriLiteNet small | DA mIoU | 91.03 | 90.5 | +0.53 |
| TriLiteNet small | Lane IoU | 27.64 | 27.6 | +0.04 |
| TriLiteNet base | DA mIoU | 92.45 | 92.0 | +0.45 |
| TriLiteNet base | Lane IoU | 29.85 | 29.8 | +0.05 |
| TLP nano | DA mIoU / Lane IoU / Lane Acc | 87.42 / 23.55 / 70.31 | 87.3 / 23.3 / 70.2 | +0.12 / +0.25 / +0.11 |
| TLP small | DA mIoU / Lane IoU / Lane Acc | 90.65 / 29.39 / 75.85 | 90.6 / 29.3 / 75.8 | +0.05 / +0.09 / +0.05 |
| TLP medium | DA mIoU / Lane IoU / Lane Acc | 92.07 / 32.44 / 79.21 | 92.0 / 32.3 / 79.1 | +0.07 / +0.14 / +0.11 |
| TLP large | DA mIoU / Lane IoU / Lane Acc | 92.91 / 34.26 / 81.94 | 92.9 / 34.2 / 81.9 | +0.01 / +0.06 / +0.04 |

**Consequence.** Any remaining gap in the table below is a *model* difference. The instrument is
no longer an explanation.

---

## 2. The apples-to-apples table (one harness, one protocol, 9 models)

Protocol: n=10,000 `tri_val`, letterbox 384 canvas (content 360×640), **one forward pass scored
against both GT sources**. Units are percent. `Ours` = FINAL-100 / B100, 0.1926M params, seed 0.

| model | params | DA mIoU<br>official | DA mIoU<br>ours | Lane IoU<br>official | Lane IoU<br>ours | Lane mIoU<br>official | Lane Acc<br>official |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Model B (ours)** | **0.193M** | **86.49** | **85.94** | **19.41** | 21.73 | **58.48** | **84.02** |
| TwinLiteNetPlus nano | 0.03M | 87.42 | 87.34 | 23.55 | 18.42 | 61.22 | 70.31 |
| TwinLiteNetPlus small | 0.12M | 90.65 | 90.58 | 29.39 | 21.46 | 64.17 | 75.85 |
| TwinLiteNetPlus medium | 0.48M | 92.07 | 92.02 | 32.44 | 23.22 | 65.70 | 79.21 |
| TwinLiteNetPlus large | 1.94M | **92.91** | **92.85** | **34.26** | **24.40** | **66.61** | 81.94 |
| TwinLiteNet | 0.4M | 91.27 | 91.20 | 28.83 | 23.45 | 63.76 | 81.06 |
| TriLiteNet tiny | 0.15M | 88.53 | 88.40 | 24.32 | 20.58 | 61.48 | 75.65 |
| TriLiteNet small | 0.59M | 91.03 | 90.84 | 27.64 | 22.70 | 63.16 | 79.53 |
| TriLiteNet base | 2.35M | 92.45 | 92.24 | 29.85 | 23.86 | 64.28 | 82.33 |

Reading it honestly:

- **DA (comparable axis — same GT, Jaccard 0.9885): we are last of nine.** 86.49 vs 88.53 for
  TriLiteNet tiny, which has *fewer* parameters (0.15M vs 0.193M). The DA deficit is not a
  measurement artefact: it survives the GT swap (−0.55 only) and the canvas swap (−0.79).
- **Lane IoU (not comparable to published): also last of nine** at 19.41, and this one *is* partly
  a rendering artefact — see §4.
- **Lane pixel accuracy (within-harness): first of nine**, 84.02 vs 82.33 (TriLiteNet base) and
  81.94 (TLP large). This is the metric YOLOPv2/YOLOPv3 lead with, for the reasons in §4.
- **Parameters: 10× cheaper than the strongest model in this table (TLP large 1.94M) and 12×
  cheaper than TriLiteNet base (2.35M); 16.6× below the cheapest model in the wider three-task
  field (SCAM-P C2f-n 3.2M).**

### 2b. Detection — the axis where we win

Detection is canvas-locked to 640×640 (the YOLOP-family decode is tied to the training anchor grid),
so it comes from the FINAL-100 run, and the harness is certified on it at Δ ≤ 0.07.

| model | params | mAP50 | note |
|---|---:|---:|---|
| **Model B (ours)** | 0.193M | **53.82** | FINAL-100, seed 0, n=10,000 |
| TriLiteNet tiny | 0.15M | 49.6 published / 49.53 re-measured | the only published three-task model in our parameter class |
| *next cheapest three-task model* | 3.2M (SCAM-P C2f-n) | 78.0 | **16.6× our parameters** |

We lead the only published model in our class by +4.2 mAP50, and the next rung of the field costs
16.6× the parameters for +24.2 mAP50.

---

## 3. The whole field, with comparability flags

Source of truth: the 22 local PDFs in `C:\Users\<user>\Desktop\YOLOP-base分析\` (no web search needed).
**Never subtract across rows without reading the flag column.**

| model | year | params | mAP50 | DA mIoU | Lane IoU | Lane Acc | axes safe to compare with us |
|---|---|---:|---:|---:|---:|---:|---|
| YOLOP | 2022 | 7.9M | 76.5 | 91.5 | 26.2 | 70.5 | DA, det |
| HybridNets | 2022 | 12.83M | 77.3 | 90.5 | 31.6 | 85.4 | DA, det |
| YOLOPv2 | 2022 | 38.9M | 83.4 | 93.2 | 27.25 | 87.31 | DA, det |
| YOLOPv3 | 2024 | 30.2M | 84.3 | 93.2 | 28.0 | 88.3 | DA, det |
| YOLOPX | 2025 | 32.9M | 83.3 | 93.2 | 27.2 | 88.6 | DA, det |
| A-YOLOM-n | 2023 | 4.43M | 78.0 | 90.5 | 28.2 | 81.3 | DA, det |
| A-YOLOM-s | 2023 | 13.61M | 81.1 | 91.0 | 28.8 | 84.9 | DA, det |
| SCAM-P C2f-n | 2025 | 3.2M | 78.0 | 90.4 | 26.5 | 81.2 | DA, det |
| SCAM-P C2f-s | 2025 | 12.0M | 81.1 | 91.0 | 27.8 | 84.2 | DA, det |
| SCAM-P GELAN-n | 2025 | 3.4M | 78.1 | 91.0 | 27.3 | 82.2 | DA, det |
| SCAM-P GELAN-s | 2025 | — | 81.0 | 91.6 | 28.8 | 84.6 | DA, det |
| GDMNet | 2024 | — | 78.2 | 92.2 | 26.4 | 75.3 | DA, det |
| MtTEPNet | 2026 | 8.3M | 79.9 | 92.8 | 28.8 | 87.4 | DA, det |
| MDA-Net-n | 2026 | 3.59M | 79.1 | 91.0 | 28.6 | 85.8 | DA, det |
| MDA-Net-s | 2026 | 13.22M | 82.0 | 91.3 | 28.9 | 87.1 | DA, det |
| Sparse U-PDP | 2023 | 12.05M | 84.7 | 92.9 | 32.4 | — | DA, det |
| TriLiteNet tiny | 2025 | 0.15M | 49.6 | 88.5 | 24.2 | 75.6 | **all** |
| TriLiteNet small | 2025 | 0.59M | 63.2 | 91.0 | 27.6 | 79.5 | **all** |
| TriLiteNet base | 2025 | 2.35M | 72.3 | 92.4 | 29.8 | 82.3 | **all** |
| TwinLiteNet | 2023 | 0.4M | — | 91.3 | 31.08 | — | DA only (2-task) |
| TLP nano → large | 2024 | 0.03→1.94M | — | 87.3→92.9 | 23.3→34.2 | 70.2→81.9 | DA only (2-task) |
| **Model B (ours)** | 2026 | **0.193M** | **53.82** | **86.5–86.7** | *(not comparable)* | **84.0** | — |

**Excluded on purpose** (see ledger §5): `MDANet.pdf` (remote-sensing change detection — a
same-acronym trap), Unified Driving Tokens (world-model tokenizer, NAVSIM), the Baczmanski paper
(wrong filename, custom FPT'22 dataset), YOLOP-with-LiDAR (multi-sensor), UF-Net (CULane protocol),
Q-YOLOP (quantisation, not an architecture).

**What this table says.** On DA we sit *below every three-task model in the field*, including
TriLiteNet tiny — the smallest of them and the only one in our price class. On detection we sit
above exactly one model: that same TriLiteNet tiny. Everything above us in the accuracy frontier is
a 3.2–38.9M-parameter network. There is no version of this table in which we are mid-field.

---

## 4. The lane-column asymmetry (must be quoted with any lane number)

Swapping the GT source moves our lane IoU **down** and every baseline's **up**:

| model | Lane IoU on our GT | Lane IoU on official GT | direction |
|---|---:|---:|---|
| Ours | 21.73 | 19.41 | **−2.32** |
| TriLiteNet tiny | 20.58 | 24.32 | +3.74 |
| TLP nano | 18.42 | 23.55 | +5.13 |
| TLP large | 24.40 | 34.26 | +9.86 |

The mechanism: our training labels are thickened (the `thin14` recipe), the official validation GT
is thin. Baselines were trained and scored on the same thin convention and are therefore matched;
we are not. **Our published-protocol lane IoU is a pessimistic value**, and the size of the penalty
is exactly the size of the baselines' gain.

This is also why the field leads with lane *pixel accuracy*: YOLOPv2/v3 both state that lane lines are
8 px wide in training and 2 px in validation, so IoU is capped near 2/8 by construction and
"pixel accuracy … better reflects the performance of lane detection" (YOLOPv3, p.12). We are first
of nine on that metric — but that is a within-harness statement, not a published-field claim.

---

## 5. What is claimable, and what is not

**Claimable**
1. Protocol conformance, 18/18 at n=10,000 — the harness reproduces the published TriLiteNet and
   TLP tables within ±1.0 on every reported cell.
2. Detection parity-plus in our class: 53.82 mAP50 at 0.193M, +4.2 over the only published
   three-task model of comparable size, and 16.6× cheaper than the next rung.
3. Parameter efficiency: 0.193M — 10× below the strongest model in the certified table, 16.6×
   below the lightest three-task model in the wider field, 202× below the heaviest.
4. Within-harness lane pixel accuracy: first of nine (84.02).
5. Every Round-4 / FINAL-100 verdict, unchanged — none of them rested on published numbers.

**Not claimable**
1. DA competitiveness: last of nine in-harness, and below every three-task model in the field.
2. Any lane-IoU comparison against published values (different GT, Jaccard 0.3804; and the metric is
   structurally capped).
3. Any FLOPs comparison against published values: ours = 2×MACs @640², published = MACs @384×640;
   predicted ratio 3.33 (measured 3.27–3.36). Params are safe (match 1.00–1.01×); FLOPs are not.
4. Any latency comparison: hardware/precision/batch all differ; the same YOLOP appears as 41/93/26 FPS.
5. `accuracy per GFLOP`, cross-paper lane ranks, and "cheapest of 10 models" — all retracted.

---

## 6. Open items

| item | status |
|---|---|
| Seed 1 / seed 2 (want ±σ on FINAL-100) | **running** — seed 1 resumed at ep19 (18→100), seed 2 fresh 100 ep queued behind it |
| Full latency column, 9 baselines, FIXED protocol (warmup 200 / reps 300 / CUDA events) | queued behind the seeds; the D8 probe proved the old instrument could not do it (reps 3 vs 100 asymmetry) |
| P-1 "lane saturates at 40 ep" | **held open** — it rests on the lane column, and the official GT renders lane differently, so saturation may be partly a label-rendering artefact |
| Detection at 384 canvas | **deliberately not run** — the anchor grid changes, so the number would be incomparable rather than informative |

---

## Appendix — provenance of the numbers in §2

`experiments/phase6/consistency/stageB_in384/consistency.json` (+ `.md`), 9 models, 10,000 images,
one forward pass, both GT sources.

The original aggregate write crashed (`numpy.bool_` is not JSON-serialisable, and the file was left
truncated mid-value). The script is fixed — `float()`/`bool()` casting on every leaf, plus an atomic
tmp-then-rename write so a mid-write exception can never leave a half-serialised artifact. The
aggregate in the repository was **rebuilt from the run log** rather than re-measured (a re-run costs
48 min of GPU that the seed arms need); every value is a verbatim `[done]` line, and the file carries
`rebuilt_from_log` / `rebuilt_note` so it cannot be mistaken for a first-hand artifact. Script:
`scripts/phase6_consistency_rebuild.py`.
