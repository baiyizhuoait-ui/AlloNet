# Phase 6 / FINAL-100 — Pre-registration of the paper's main run

**Status: FROZEN before any 100-epoch metric exists.**
Written 2026-09-13 18:30, ~5 min after launch, while the run was at epoch 1.
Nothing in §5–§6 may be edited after the first `metrics.json` appears.

---

## §1 Run identity

| field | value |
|---|---|
| tag / cell | `B100` |
| config | `configs/phase6_r4_R2_thin14_z16.yaml` — **sha1 `f3b8b9a95`…** (byte-identical to the 20ep and 40ep arms; only `--epochs` differs) |
| trainer | `training/train.py` sha1 `88ee5640f`… |
| runner | `scripts/phase6_final100_run.sh` sha1 `3ed1f2e3b`… (frozen copy of `phase6_round4_run.sh`; `diff` = 4 path substitutions + log name only) |
| git | `02096b38c` |
| epochs / seed / batch | **100** / **0** / 16 |
| LR schedule | `CosineAnnealingLR(T_max = epochs × len(loader) = 100 × 4366 = 436,600 steps)`, lr 1e-3 → 0 |
| started | 2026-09-13 **18:25:13** |
| expected finish | 2026-09-14 **≈06:00–07:00** (measured 133–135 ms/step at ep1 ⇒ 9.7–9.8 min/epoch; 20ep arms ran 96 ms/step ⇒ 7.0 min/epoch) |
| outdir / eval | `experiments/phase6/final/B100` → `..._eval/metrics.json` |

**Why a separate directory.** Round 4 is closed, committed (`d8ba8bf`) and archived. Its four tables back the frozen UNRESOLVED verdict. Appending a new arm into those CSVs would put a row into tables that are already final. This run answers a *different* question ("same budget as the published baselines") and gets its own dir and its own tables.

**Why 100 epochs at all.** Every external baseline in `experiments/BASELINE_RESULTS.md` was evaluated from **official released weights** (full training schedule). Our own evidence is 20 ep × 3 seeds and 40 ep × 1 seed. Detection, DA and lane were all measured through *our* evaluation script on the same `tri_val` 10k, 640×640 letterbox protocol, and that script reproduces the published numbers (YOLOP 76.57 vs 76.5 official; TwinLiteNet 91.14 vs 91.5). **So instrument parity is already established; the training budget is the only unmatched factor.** This run closes that one factor, and nothing else.

---

## §2 Frozen reference table (copied from the tables, not retyped)

**Ours, Model B** — 0.1926 M params / **1.1656 GFLOPs** @640²

| cell | epochs | seed | mAP50 | mAP50-95 | DA mIoU | lane mIoU | lane_fg |
|---|---|---:|---:|---:|---:|---:|---:|
| R4-R2thin14 | 20 | 0 | 0.4954 | 0.2146 | 0.8528 | 0.5946 | 0.2121 |
| R4-R2thin14 | 20 | 1 | 0.4967 | 0.2168 | 0.8587 | 0.5936 | 0.2105 |
| R4-R2thin14 | 20 | 2 | 0.5012 | 0.2186 | 0.8570 | 0.5994 | 0.2199 |
| **20ep mean (N=3)** | 20 | — | **0.4978** | 0.2167 | **0.8562** | **0.5959** | **0.2142** |
| R5-R2thin40 | 40 | 0 | 0.5198 | — | 0.8594 | 0.5992 | 0.2203 |
| R6-R2thinL4 (lr 1e-4) | 20 | 0 | 0.3648 | — | 0.8301 | — | 0.1902 |

**External baselines** — official released weights, our `tri_val` 10k protocol

| model | params | FLOPs@640 | mAP50 | DA mIoU | lane mIoU | lane_fg |
|---|---:|---:|---:|---:|---:|---:|
| TriLiteNet tiny | 0.151 M | 1.8 G | **0.4953** | **0.8796** | 0.5914 | 0.1952 |
| TwinLiteNetPlus nano | 0.033 M | 1.9 G | — (no det head) | 0.8634 | 0.5866 | 0.1859 |

**Gap of ours (40 ep, seed 0) vs TriLiteNet tiny:** mAP50 **+0.0245**, DA **−0.0202**, lane **+0.0078**, lane_fg **+0.0251**.

---

## §3 Noise model — and an explicit correction

**Correction to an earlier statement of mine.** I previously described the three Model-B seeds as "40 ep". They are **20 ep** (`R4-R2thin14`, seeds 0/1/2). At 40 ep we have **N = 1** (seed 0 only). The paired 20→40 comparison is therefore seed-matched and clean (0.4954 → 0.5198, +0.0244), but the 40 ep *level* rests on a single seed.

Seed sd is available only at 20 ep (N=3):

| axis | sd @20ep | 2σ |
|---|---:|---:|
| mAP50 | 0.0030 | 0.0061 |
| DA mIoU | 0.0030 | 0.0061 |
| lane mIoU | 0.0031 | 0.0062 |
| lane_fg | 0.0050 | 0.0101 |

**Stated assumption (not silent):** this sd is used as the noise estimate for the 100 ep run. Seed noise is not expected to shrink with more epochs, so if anything this under-states the uncertainty. It is the only noise estimate the project has at this budget and re-estimating it from the 100 ep run itself would be circular (§8 forbids re-estimating σ from the arm being judged).

---

## §4 Cost model of the epoch lever (frozen)

Measured, seed-matched, 20 → 40 ep (F9, z = Δ/σ with σ from §3):

| axis | 20ep (s0) | 40ep (s0) | Δ | z |
|---|---:|---:|---:|---:|
| mAP50 | 0.4954 | 0.5198 | **+0.0244** | +5.08 |
| DA mIoU | 0.8528 | 0.8594 | **+0.0066** | +6.00 |
| lane mIoU | 0.5946 | 0.5992 | +0.0046 | +2.42 |
| lane_fg | 0.2121 | 0.2203 | +0.0082 | +2.65 |

Two facts that must travel together with any use of this table:
1. **DA is the most epoch-sensitive axis** (z = +6.00, highest of the four). The epoch lever is pointed exactly at the one axis where we trail.
2. **The remaining deficit was measured with a 100-eponent handicap.** DA is 0.0202 behind an opponent at ≈100 ep while we are at 40 ep. That number is therefore an *upper* bound on the true architectural deficit; it is not evidence that the deficit is structural.

Diminishing returns are expected (cosine annealing puts the largest LR steps early). Naive 3× extrapolation of the 20→40 increment would give DA +0.020 — i.e. **parity**. Decay-corrected estimates land at +0.006…+0.012 → DA ≈ 0.865…0.871. **The two extrapolations disagree by exactly the amount that decides the question, which is why the decision rules below are written as a boundary test rather than a point prediction.**

---

## §5 Predictions (frozen)

- **P-1 (monotone in budget).** All four axes beat their 40 ep value: mAP50 > 0.5198, DA > 0.8594, lane > 0.5992, lane_fg > 0.2203.
- **P-2 (the deficit survives).** DA_100 **< 0.8796** — at equal budget we still do **not** match TriLiteNet tiny on drivable-area segmentation.
- **P-3 (the detection lead becomes significant).** mAP50_100 ≥ 0.4953 + 2σ = **0.5014**, i.e. the lead over TriLiteNet tiny exceeds 2σ for the first time. (At 40 ep it is +0.0245 on N=1 with no seed evidence; at 20 ep the 3-seed mean is only +0.0025 = 0.4σ.)
- **P-4 (the lane lead widens).** lane_mIoU_100 > 0.5914 and lane_fg_100 > 0.1952, both already true at 40 ep, with lane_fg margin growing.

---

## §6 Decision rules (frozen)

Let `g = 0.8796 − DA_100` be the residual DA deficit, and let `2σ = 0.0061`.

- **D-1.** If `g > 0.0141` (equivalently `DA_100 < 0.8655`): the deficit survives 100 epochs at > 2σ. **Report it as a real, budget-independent deficit and stop after 1 seed.** A single seed is sufficient for a negative claim of this magnitude (the effect is ≳ 2.3σ wide).
- **D-2.** If `g ≤ 0.0141` (`DA_100 ≥ 0.8655`): the residual sits inside the seed-noise band. **One seed cannot separate "still behind" from "parity".** Seeds 1 and 2 must be added before *any* parity language is used. Until then the honest phrasing is "within seed noise at equal budget".
- **D-3.** If `DA_100 ≥ 0.8796` (P-2 falsified): parity on DA is directly demonstrated and the paper's main table changes character — this becomes a claim that must be replicated across seeds (D-2 rules apply with the sign reversed).

**Paper-facing rule that follows from D-1/D-2:** a claim of the form "we match the published baseline on DA" requires the 3-seed interval. A claim of the form "we do not match it on DA, but we lead on detection and lane at 35% fewer FLOPs" is supported by 1 seed and is the claim that holds under the larger part of the prediction space.

---

## §7 Scope: what this run cannot change

| item | why it is out of reach |
|---|---|
| **F4** (h8 dead rung) | A deterministic architecture failure — two independent backbones produced bit-identical 0.4959 with fg = 0.0. The training budget is not a variable in it. |
| **F8** (U⁻ cannot be rebuilt inside U's budget) | Settled by the zero-training cost model (uniform + 1/4 h16 = 1.7248 G > U's 1.6650 G). Not a budget question. |
| **F5 deviation** (lateral kept at h16 without an h16 control) | Requires a different arm, not more epochs. |
| **The Round-4 verdict** | **UNRESOLVED stands.** This is a support measurement for the "mechanism / analysis" paper, not a route to Level 3. |
| **Absolute SOTA** | Out of range and not the claim. We beat TriLiteNet tiny on params? No — 1.27× more params, 35% fewer FLOPs. The honest headline is FLOPs + detection + lane, with DA conceded. |

---

## §8 Falsification and stop discipline

- Any of P-1…P-4 may fail; a failure is a result and will be reported as one.
- **No re-estimation of σ from this arm.** σ stays as measured at 20 ep (§3) — otherwise the boundary in §6 would be tuned by the thing it judges.
- **No mid-run metric peeking to change the plan.** The eval runs once, after epoch 100, through `evaluate_baseline.py --baseline OursStatic` — the same entry point used for all 21 Round-4 arms.
- **Stop condition:** this run, then D-1/D-2, then stop. Seeds 1–2 are *conditional* on D-2, and the B′ arm (lean + 1/4 h16, no lateral, 1.1394 G) is a separate optional item that cannot change F4 or F8 either.

---

## §9 Outcome (appended 2026-09-14 06:25; §5–§6 above are UNTOUCHED)

**Run completed:** 2026-09-13 18:25:13 → 2026-09-14 06:25:06, **719 min**, peak per-process `self` **2328 MiB / 8151 MiB = 28%**. Eval 276 s. No ABORT, no STOP_CHAIN, no VRAM cliff.

| axis | 20ep (s0) | 40ep (s0) | **100ep (s0)** | TriLiteNet tiny | 100ep gap | 2σ |
|---|---:|---:|---:|---:|---:|---:|
| mAP50 | 0.4954 | 0.5198 | **0.5382** | 0.4953 | **+0.0429** | ±0.0060 |
| DA mIoU | 0.8528 | 0.8594 | **0.8673** | 0.8796 | **−0.0123** | ±0.0060 |
| Lane mIoU | 0.5946 | 0.5992 | **0.5978** | 0.5914 | +0.0064 | ±0.0062 |
| Lane fgIoU | 0.2121 | 0.2203 | **0.2187** | 0.1952 | +0.0235 | ±0.0100 |

(mAP50-95 = 0.2451; params 0.1926 M / 1.1656 G, byte-identical config to the 20ep/40ep arms.)

### Prediction verdicts

- **P-1 — FAILED, partially (2 / 4 axes).** mAP50 ✓ (0.5382 > 0.5198), DA ✓ (0.8673 > 0.8594), but **lane_mIoU ✗ (0.5978 < 0.5992, −0.45σ)** and **lane_fg ✗ (0.2187 < 0.2203, −0.32σ)**. Lane is **saturated at 40 ep**; the epoch lever is dead on that axis. This is the most informative negative result of the run.
- **P-2 — HOLDS.** DA_100 = 0.8673 < 0.8796. The deficit survives at equal budget.
- **P-3 — HOLDS, with room.** 0.5382 ≫ 0.5014. The detection lead is **+0.0429 ≈ 14σ**, up from +0.0245 on N=1; it crosses 2σ for the first time.
- **P-4 — PARTIALLY FALSIFIED.** Both lane axes still lead, but the margins **shrank** (+0.0078→+0.0064, +0.0251→+0.0235), so the "margin growing" clause fails.

### Decision-rule outcome

`g = 0.8796 − 0.8673 = ` **`0.0123`** ≤ `0.0141` → **D-2 fires.**

The residual sits at **2.0σ**, i.e. inside the guard band and short of the 2.3σ negative threshold of D-1. Consequences, exactly as written:

1. **No parity language is permitted** until seeds 1–2 are added. The honest phrasing is "behind by 0.0123 (≈2σ, N=1)".
2. The **1-seed claim is the one the paper may use**, per the paper-facing rule at the end of §6: *"we lead on detection (+0.0429, 14σ) and lane (+0.0064 / +0.0235) at 35% fewer FLOPs, and we trail on DA by 0.0123 — i.e. roughly half of the original 0.0202 deficit was a training-budget artefact."*
3. Seeds 1–2 remain **conditional** (≈12 h each, ≈24 h total on the local machine) and are now a PI resource decision, not a scientific unknown.

### §8 compliance check

- No σ was re-estimated from this arm — σ remains the 20 ep, N=3 value from §3, as required.
- No mid-run metric peeking: the single eval ran once, after epoch 100, through the same `evaluate_baseline.py` entry point as all 21 Round-4 arms.
- §7 scope re-confirmed: **F4, F8, the F5 h16 deviation and the Round-4 UNRESOLVED verdict are all unchanged.** This run is a support measurement, not a route to Level 3.

### One residual defect carried forward

**D8 is still open.** This eval measured `p50 = 2.355 ms`, while Round 4 measured 2.663 / 7.670 ms for the same architecture — the ≥2.9× latency dispersion persists. The paper's latency column therefore still requires a dedicated zero-training re-measurement (same card, no concurrent compute) and must not cite training-time readings.
