# Phase 6 — Cross-Algorithm Ledger & Comparability Audit

**Compiled 2026-09-14.** Source of truth: the 22 local PDFs in
`C:\Users\<user>\Desktop\YOLOP-base分析\` (text dump in `.workbuddy/pdftext/`), the vendored
upstream repos under `baselines/`, and our own harness. **No web search was needed** — every
published number below is traceable to a local file.

---

## 0. Why this document exists

The previous table (`phase6_algorithm_comparison.md`) ranked 10 models on one harness and called
it "positioning". That was wrong for a reason that had nothing to do with arithmetic: it compared
our numbers with **published** numbers without ever checking whether the two sides measured the
same thing. This document does that check first, then reports only what survives.

---

## 1. The ground-truth audit (decisive; zero training, CPU only)

Script: `scripts/phase6_official_gt_audit.py`. Inputs: our `data/bdd100k/{lanes,segments}/masks`
vs the official package `data/bdd100k/official_eval/` (staged by
`scripts/phase6_extract_official_gt.sh` from the Google-Drive zips the upstream READMEs point to).
400 images of `tri_val`, native resolution, identical image IDs (10,000/10,000 overlap).

| axis | our file | official package | foreground | **Jaccard(ours, pkg)** |
|---|---|---|---|---|
| **DA** | values {0,1,2} | values {0,127,191,223,239} (soft edges) | 16.528% vs 16.680% | **0.9885** |
| **Lane** | values {0, 2…48} (18 values) | values {0,255} | 0.710% vs 0.650% | **0.3804** |

Decisive cross-check: applying the official threshold `>1` to **our** lane file selects
**100.000%** of pixels (our background is 255) — the two files cannot be the same encoding.

**Verdict.**
- **DA is the same annotation** (Jaccard 0.99). Our DA numbers are directly comparable to published DA.
- **Lane is a different annotation** (Jaccard 0.38). Two ~2-px-wide lane sets that disagree
  pixel-wise at 1-px scale produce exactly this: same density (0.71% vs 0.65%), low overlap.
  Lane numbers are **not** comparable, and no scale factor can repair them.

### 1b. Why "Lane IoU" is a structurally capped metric anyway

Both YOLOPv2 and YOLOPv3 state the convention explicitly (verified from the PDFs):

> "we set the width of the lane lines to **8 pixels in the training set** and **2 pixels in the
> validation set**. This … will result in predicted lane lines being **significantly wider than
> the ground truth**. As a result, the evaluation metric IoU commonly remains low. Conversely,
> pixel accuracy … better reflects the performance of lane detection. Therefore, we emphasize
> pixel accuracy in this work." — `YOLOPV3.pdf` p.12

So published Lane IoU is bounded near 2/8 of the achievable overlap **by construction**. This is
why the whole field sits in 23–34 and why two of the three papers in this family lead with pixel
accuracy instead. Any lane ranking built on Lane IoU is ranking a saturated metric.

---

## 2. Axis-by-axis comparability verdict

| axis | verdict | evidence |
|---|---|---|
| **Detection (mAP50)** | **COMPARABLE** | Our harness reproduces TriLiteNet tiny/small/base at 49.53 / 63.26 / 72.35 vs published 49.6 / 63.2 / 72.3 → Δ ≤ 0.07 |
| **DA (mIoU)** | **COMPARABLE** | Same GT file (Jaccard 0.9885); our harness gives tiny 87.96 vs published 88.5 → Δ 0.5 |
| **Lane (IoU / fgIoU)** | **NOT COMPARABLE** | Different GT rendering (Jaccard 0.3804) *and* a different metric definition on our side; published convention caps the value |
| **Params** | **COMPARABLE** | Matches to 1.00–1.01× on all 7 models where both sides report it |
| **FLOPs** | **NOT COMPARABLE** | Ours = 2 × MACs (thop) @ 640×640; published = MACs @ 384×640. Predicted ratio 2.0 × 1.6667 = **3.333**; measured 3.27–3.36 (mean 3.32) on the 7 param-matched models. Never place side by side |
| **FPS / latency** | **NOT COMPARABLE** | Hardware, batch, precision all differ (TITAN XP / A5000 / RTX 4090 / Orin / V100). Same YOLOP appears as 41 / 93 / 26 FPS |

---

## 3. Group A — full three-task, camera-only, BDD100K val

| model | params | FLOPs (as published) | input | mAP50 | Recall | DA mIoU | Lane Acc | Lane IoU | speed | source (local) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOP (2022) | 7.9M | 18.6B / 9.38G † | 640×384 | 76.5 | 89.2 | 91.5 | 70.5 | 26.2 | 41 FPS (TITAN XP) | YOLOPv2 T2–T4 |
| HybridNets (2022) | 12.83M | 15.6B | 640×384 | 77.3 | 92.8 | 90.5 | 85.4 | **31.6** | 37 ms (V100) | YOLOPv2 T2–T4 |
| YOLOPv2 (2022) | 38.9M | — | 640 | 83.4 | 91.1 | 93.2 | 87.31 | 27.25 | 91 FPS | YOLOPv2 T1–T4 |
| YOLOPv3 (2024) | 30.2M | — | 640 | **84.3** | 96.9 | 93.2 | 88.3 | 28.0 | — | YOLOPv3 T4–T5 |
| YOLOPX (2025) | — (GH: 32.9M) | — | 640×640 | — (GH: 83.3) | — | — (GH: 93.2) | 88.6 | 27.2 | 47 FPS (GH) | YOLOPX T2 |
| A-YOLOM-n (2023) | 4.43M | — | 640×640 | 78.0 | 85.3 | 90.5 | 81.3 | 28.2 | 39.9 FPS (bs1) | ref. tables |
| A-YOLOM-s (2023) | 13.61M | — | 640×640 | 81.1 | 86.9 | 91.0 | 84.9 | 28.8 | 39.7 FPS (bs1) | ref. tables |
| SCAM-P C2f-n (2025) | 3.2M | — | 640×640 | 78.0 | 85.4 | 90.4 | 81.2 | 26.5 | 107.1 FPS (Orin FP16) | SCAM-P |
| SCAM-P C2f-s (2025) | 12.0M | — | 640×640 | 81.1 | 87.1 | 91.0 | 84.2 | 27.8 | 77.8 FPS (Orin FP16) | SCAM-P |
| SCAM-P GELAN-n (2025) | 3.4M | — | 640×640 | 78.1 | 85.8 | 91.0 | 82.2 | 27.3 | 56.4 FPS (Orin) | SCAM-P |
| SCAM-P GELAN-s (2025) | — | — | 640×640 | 81.0 | 86.8 | 91.6 | 84.6 | 28.8 | **230.5 FPS (4090)** | SCAM-P |
| GDMNet (2024) | — | — | — | 78.2 | 89.7 | **92.2** | 75.3 | 26.4 | 56.5 FPS | GDMNet T1 ✔verified |
| MtTEPNet (2026) | 8.3M | 13.8G | 640×640 | 79.9 | 89.8 | **92.8** | 87.4 | 28.8 | 38 FPS | MtTEPNet |
| MDA-Net-n (2026) | 3.59M | — | 640×640 | 79.1 | 86.8 | 91.0 | 85.8 | 28.6 | 80.26 FPS (bs1) | MDA-Net ✔verified |
| MDA-Net-s (2026) | 13.22M | — | 640×640 | **82.0** | 88.3 | 91.3 | 87.1 | 28.9 | 79.13 FPS (bs1) | MDA-Net ✔verified |
| Sparse U-PDP (2023) | 12.05M | 15.1G | — | **84.7** | — | **92.9** | — | **32.4** | 29 FPS | Sparse U-PDP ✔verified |
| TriLiteNet tiny (2025) | 0.15M | 0.55G | 640×384 | 49.6 | 76.5 | 88.5 | 75.6 | 24.2 | 185 FPS (4090) | triLiteNet |
| TriLiteNet small (2025) | 0.59M | 1.99G | 640×384 | 63.2 | 81.6 | 91.0 | 79.5 | 27.6 | 151 FPS (4090) | triLiteNet |
| TriLiteNet base (2025) | 2.35M | 7.72G | 640×384 | 72.3 | 85.6 | 92.4 | 82.3 | 29.8 | 105 FPS (4090) | triLiteNet |
| **Model B (ours, 100 ep, s0)** | **0.193M** | **1.166G (true FLOPs @640×640)** | 640×640 | **53.82** | — | **86.73** | — | *(not comparable)* | 2.396 ms | our harness |

† YOLOP's FLOPs is published as 18.6B (HybridNets) *and* 9.38G (TriLiteNet) *and* 17.32G
(Baczmanski et al.) — three values for one model. This alone disqualifies the FLOPs axis.

**What is defensible from this table:**
1. **Detection** (comparable axis): ours 53.82 beats TriLiteNet tiny 49.6 at 0.193M vs 0.15M params,
   while being *cheaper in our common harness* (1.166G vs 1.80G, −35%). Next step up is
   A-YOLOM-n 78.0 at **23× our parameters**.
2. **DA** (comparable axis): ours 86.73 sits **below** TriLiteNet tiny 88.5/published, 87.96/our
   harness → the deficit is real, and it is *within the same model family's own tiny config*.
3. The accuracy frontier is dense above us: five models at mAP50 81–85 (Sparse U-PDP 84.7,
   YOLOPv3 84.3, YOLOPv2 83.4, YOLOPX 83.3, MDA-Net-s 82.0) — all at 13–39M params.
4. Nobody in this family exceeds Lane IoU 32.4 and most sit 26–29. The lane axis is where the
   whole field is stuck — which is a *finding about the field*, not a weakness of a small model.

---

## 4. Group B — two-task (DA + lane, no detection head)

Structurally ineligible for a three-task comparison; listed as segmentation-efficiency reference.

| model | params | FLOPs | input | DA mIoU | Lane Acc | Lane IoU | speed |
|---|---|---|---|---|---|---|---|
| TwinLiteNet (2023) | 0.4M | — | 640×360 | 91.3 | — | 31.08 | 415 FPS (A5000) / 60 (Xavier NX) |
| TwinLiteNet+ Nano | 0.03M | 0.57G | 640×384 | 87.3 | 70.2 | 23.3 | — |
| TwinLiteNet+ Small | 0.12M | 1.40G | 640×384 | 90.6 | 75.8 | 29.3 | — |
| TwinLiteNet+ Medium | 0.48M | 4.63G | 640×384 | 92.0 | 79.1 | 32.3 | — |
| TwinLiteNet+ Large | 1.94M | 17.58G | 640×384 | 92.9 | 81.9 | **34.2** | 109 FPS (bs1) |

These four share one source (`baselines/TwinLiteNetPlus/README.md` official table + paper), so
their internal ordering *is* readable — and it shows DA/lane accuracy climbing monotonically with
capacity, i.e. **this family is not saturated** in the way the accuracy table's lane column is.

---

## 5. Group C — excluded, with reasons (this is where a careless table goes wrong)

| item | what it actually is | why excluded | source |
|---|---|---|---|
| `MDANet.pdf` | **Remote-sensing change detection** (Siamese + DFM + ARM + CSFM, BTCDD / LEVIR-CD) | Different task domain entirely. Same acronym as the driving model below — a real trap | `MDANet.txt` T2–T3 |
| MDA-Net / MDANet (TITS 2026) | The driving model (YOLO11 + GADC + SADF + TriD-Sample) | **Included** in Group A — do not confuse with the row above | `MDA-Net.txt` |
| `Unified Driving Tokens.pdf` | Discrete visual tokenizer for driving world models / planning on **NAVSIM** | Not a panoptic perception model; 20M readout head + 1B transformer | `Unified_Driving_Tokens.txt` p.1 |
| `A panoramic driving perception fusion algorithm...pdf` | Actually **"Detection-segmentation CNN for autonomous vehicle perception"** (Baczmanski et al., AGH Kraków, MMAR 2023) — *filename does not match the paper* | Evaluated on a **custom FPT'22 dataset**, not BDD100K. Useful only for its FLOPs cross-check (YOLOP 17.32G @640×640, HybridNets 14.53G @640×384, MultiTask V3 25.44G @512×320) | `A_panoramic…txt` |
| `YOLOP with LiDAR Fusion.pdf` | "A panoramic driving perception fusion algorithm based on multi-task learning" (PLOS ONE, Wu et al.) — LiDAR + camera **fusion** | Multi-sensor; outside a camera-only comparison | `YOLOP_with_LiDAR_Fusion.txt` p.1 |
| UF-Net | Three-task, but lane evaluated on **CULane**; params 26.3M / 48.9M; mIoU 79.8/81.6, mAP50 79.2/80.1 | Different dataset + metric protocol ⇒ not placeable in this table | `UF-Net.txt` T5–T6 |
| Q-YOLOP | Quantization-aware training of YOLOP | Deployment technique, not a competing architecture | `~/Downloads/Q-YOLOP.pdf` |
| YOLOP-MG (Sensors 2024) | Genuine three-task BDD100K model: mAP50 81.4, lane IoU 28.9, DA mIoU 92.6, 61 FPS | **Not in the table above only because params/FLOPs are not reported in the local text** — add if needed | `YOLOP-MG.txt` p.1, T4 |

---

## 6. Retracted / kept

**Retracted** (all were built on cross-paper comparison without a protocol check):
- lane cross-paper ranks; the `accuracy per GFLOP` headline (numerator invalid *and* denominator
  now known to be a different unit; the ratio is also a fragile metric by construction);
- "cheapest of 10 evaluated models" as a field-level claim.

**Kept** (these are within-table statements and survive untouched):
- all parameter comparisons; all *within-harness* comparisons;
- the detection axis — the only axis empirically validated against published values (Δ ≤ 0.07);
- the DA axis, now positively confirmed by the 0.9885 Jaccard;
- the Pareto dominance relations computed inside one harness;
- every FINAL-100 / Round-4 verdict (they never referenced published numbers).

**Held open, not withdrawn:** P-1 ("lane saturates at 40 ep") rests on the lane column; the
official GT renders lane differently, so "saturation" may be partly an artefact of *our* label
rendering. Re-check after the official-GT re-run.

---

## 7. What the official package now unlocks

The zips are staged and audited (`data/bdd100k/official_eval/`, val split only, 10,000 ids each,
matching our split list 10000/10000). That makes three things possible for the first time:

1. **Conformance test** — run each baseline at *its own* native resolution against the official
   GT; TriLiteNet tiny must move from our 19.5 to ≈24.2. If it does, our lane pipeline is proven
   conformant and our model's lane number becomes publishable.
2. **Controlled comparison** — evaluate every model at a common 640×640 against the official GT so
   the whole table is apples-to-apples.
3. **Full latency column** — 9 baselines measured under the corrected protocol (warmup 200 /
   reps 300 / CUDA events), which the D8 probe showed the old instrument could not do.

All three are inference-only and must run with the GPU to itself. **They are queued behind the
seed-1/seed-2 training and were not started during it.**
