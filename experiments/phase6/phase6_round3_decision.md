# Phase 6 Round 3 — decision report (cross-architecture validation)

**Date:** 2026-09-11 · **Chain:** `21a61fe` · **Budget:** EXP-9A 69 863 imgs × 1 ep @ bs6, lr 1e-4; EXP-9B 69 863 imgs × 3 ep @ bs16
**Chain wall clock:** 11:53:02 → 16:49:10 (4 h 56 m). Ran to completion, no interruption.

## 1. Verdict (mechanical, computed by `scripts/phase6_round3_verdict.py`)

| block | verdict | raw span | ratio to floor |
|---|---|---|---|
| **H1.instrument** (zero-training, exact) | **SUPPORTED** | zero-pos 0.4941 → 0.1666 → 0.0374 | **47.6×** |
| **H1.accuracy** (trained ladder) | **SUPPORTED** | mAP50 0.5407 → 0.6846 → 0.7400, ρ=+1 | **20.8×** |
| **H2.lane** (spatial vs channel, FLOPs-parity) | **WEAK SUPPORT** | lanemIoU advantage +0.0019 | **0.20×** |

```
CASE C -> REVISE (redefine architecture hypothesis)
```

`Final architecture recommendation = REVISE`. This is **derived** from the case map (prereg §8), not written independently; Case C is the only case with H1 alone.

## 2. Raw results

**EXP-9A — YOLOP, project assignment rule held fixed, anchor set is the only variable**

| arm | anchor set | mAP50 | mAP50-95 | n_pred | det_loss_last | wall |
|---|---|---|---|---|---|---|
| A0_shipped | YOLOP own (live) | 0.5407 | 0.2369 | 625 107 | 0.1145 | 42.5 min |
| A1_flip | aspect-flip of A0 | 0.6846 | 0.3598 | 982 623 | 0.1145 | 42.0 min |
| A2_kmeans | IoU-k-means refit (n=3000) | 0.7400 | 0.3961 | 1 187 603 | 0.0829 | 42.6 min |

Detection AP50 by size also rises monotonically (small 0.4654/0.5552/0.6003; medium 0.4482/0.8162/0.8358; large 0.2095/0.4571/0.8232).

**EXP-9B — TwinLiteNetPlus-small, DA branch untouched, ±5% FLOPs parity**

| arm | intervention (+params) | lane_mIoU | lane_fg_iou | da_mIoU (guard) |
|---|---|---|---|---|
| B0_baseline | stock lane head | 0.5988 | 0.2201 | 0.8563 |
| B1_spatial | learned 1/4 tap (+204) | 0.6004 | 0.2230 | 0.8568 |
| B2_channel | 1×1 widen at 1/4 (+263) | 0.5985 | 0.2196 | 0.8567 |

## 3. What each verdict means, and what it does NOT claim

### H1 — SUPPORTED, and now architecture-general

H1 says: *the anchor/assignment configuration, not model capacity, limits achievable detection accuracy* — because a bad anchor set leaves most real GT without a positive sample, so the detector is trained to suppress them.

Round 3 makes this a **two-architecture** observation with the ruler (§3) held fixed at the project rule `0.5 < box/anchor < 2.0`:

- **R2** (trac): k-means anchors vs the old default → **+0.14 mAP50** at 20 ep, 2 seeds.
- **YOLOP** (this round): the same intervention direction, a 3-point ladder → **+0.1993 mAP50** at 11 643 steps, strictly monotone in both the zero-training instrument (`zero_pos_rate`, `mean_best_iou`) and the trained metric.

This is the sentence Round 3 is allowed to generalise, and it satisfies the hard rule: two architectures, one ruler, plus an instrument that is *exact* (zero-training) on the second architecture.

**Not claimed:** that the effect size transfers (0.14 at 20 ep vs 0.20 at ~11.6 k steps are different budgets and different models — the *direction* and the *mechanism* transfer, the magnitude does not). Not claimed for a third architecture.

### H2 — WEAK SUPPORT: the sign is right, the size is not resolvable

Primary quantity (prereg §7): `spatial advantage = (B1−B0) − (B2−B0) = +0.0016 − (−0.0003) = +0.0019`.
Sign is correct (spatial > channel) but the magnitude is **0.20× the conservative floor** (0.0096), and the ladder is **not monotone** under the §6 order (`B0 < B2` fails: 0.5988 → 0.5985).

Two independent facts, both measured, both consistent with lane being near its ceiling in this family:

1. B0 (0.5988) is **below the released pretrained** TLP-small lane_mIoU measured at STEP-1 (0.6018) — three epochs on the project loss did not improve lane at all, so all three arms sit on the same plateau and the intervention has almost nothing to bite on.
2. STEP-1's four-preset sweep: 58× parameters (nano → large) buys only **+0.031 lane_mIoU**. Lane is capacity-saturated in this family; adding +200–260 parameters at the 1/4 stage cannot be expected to move it.

**This is not evidence of absence.** Per the pre-registered honesty contract, a sub-floor effect is not a refutation of the mechanism — it is an inability to resolve it at this budget with this backbone. The correct label is WEAK SUPPORT, and Round 4 must not treat H2 as disproved *or* as established.

### Case C → REVISE

Case C: H1 alone. Prereg §8 consequence: *the supervision bottleneck appears transferable → redefine the architecture hypothesis.*

Read concretely: the transferable, reproducible bottleneck found so far is the **supervision/assignment** dimension — an architecture-general property of how targets are bound to predictions. The **lane channel-vs-spatial allocation** dimension is *not* established on the second architecture, and the second architecture gave an independent reason why (lane saturation). Therefore Round 4 must **not** be built around "lane spatial > channel" alone, and the H-32/H-33 asymmetric-architecture candidate must be re-derived rather than carried forward as if H2 had passed.

## 4. Pre-registration discrepancy found while adjudicating (recorded, not silently resolved)

Prereg §6 prose defines lane monotonicity as "`B0 < B1` **and** `B0 < B2`", but the pre-registered code (`verdict_exp9b`) tests only `B0 < B1`. These disagree on this data (`B0 < B2` fails).

The verdict is **robust to the discrepancy**: the tier function returns WEAK SUPPORT whenever the sign is correct and the ladder is not (monotone *and* ≥ floor), so both readings give WEAK SUPPORT. Recorded here so it is auditable, and flagged for disambiguation before any Round 4 prereg is frozen.

## 5. Instrument defects found while Round 3 was running

Full forensics in `round3_invalid_evidence/README.md` (D1–D3, from the aborted first run) plus:

- **D5 — CSV column shift.** The first STEP-2 write joined fields with `,` and never quoted them; the comma inside `learned 1/4 tap into lane head (zero-init, +204 params)` shifted every later cell by one, so `lane_mIoU` silently read `da_mIoU`'s value (0.8568 instead of 0.6004). A table off by one column is the same failure class as a silent NA — it still looks like a number. The chain log's printed table (and any `column -s, -t` view) displayed it "aligned but wrong".
  - **Fix, structural:** the appender now writes through `csv.writer` (auto-quoting) and then **reads the file back with `csv.DictReader`**, asserting the appended row's `mAP50`, `lane_mIoU` and `intervention` equal what was just written; a mismatch aborts the append. Six arms re-appended, all `[readback OK]`. The shifted file is archived as `round3_invalid_evidence/phase6_round3_trained.COLSHIFT.csv`.
  - **Caught by:** the verdict script reading the CSV, not by eye — the rendered table looked fine.

## 6. Consequences for Round 4 (not started; requires separate approval)

Per the standing protocol, Round 4 does not auto-start. What Case C implies for its design:

1. **Re-anchor the architecture hypothesis on the supervision/assignment dimension** — that is the axis with two-architecture support and an exact (zero-training) instrument.
2. **Do not promote** "task-conditional channel-vs-spatial allocation" to a paper claim on this evidence; its novelty-matrix status is downgraded accordingly.
3. If the lane/allocation question is to be pursued, it needs a **backbone with lane headroom** (the TLP-small lane head is saturated) or an explicit power analysis — not a bigger budget on the same saturated head.
4. Before any Round 4 prereg is frozen: disambiguate the §6 lane-monotonicity prose/code gap.

## 7. Provenance

- Verdict logic encoded in `scripts/phase6_round3_verdict.py` **before** any trained metric existed (selftest 6/6; refuses a Case when an arm is missing).
- Results: `phase6_round3_trained.csv` (6 rows, readback-verified), `phase6_round3_statistics.csv` (31 rows).
- Zero-training instrument + lane baselines: `phase6_round3_detection.csv`, `phase6_round3_lane.csv`.
- Budget re-freeze and all amendments: `phase6_round3_preregistration.md` §11.1–§11.4.
