#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / Round 4 closure -- registry sync (ZERO GPU).

Updates the three living registry tables to the post-Round-4 state:
  experiments/phase6/phase6_experiment_registry.csv      (EXP-10..13 -> DONE)
  experiments/phase6/phase6_architecture_hypotheses.csv  (H-36 scope, H-37 outcome)
  experiments/phase6/phase6_novelty_matrix.csv           (H-36/H-37/H-32/H-35 wording)

Discipline (lesson of D5 / 13.1): a hand-written CSV row once lost its column
alignment because an unquoted comma split a field. Therefore every table is
rewritten through csv.DictWriter (which quotes automatically) and re-read with a
per-row column-count assertion. Any mismatch aborts WITHOUT writing.

Every table is backed up to <name>.prer4final.bak first.
"""
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "experiments", "phase6")


def load(name):
    p = os.path.join(E, name)
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    hdr = list(rows[0].keys())
    for i, r in enumerate(rows):
        assert len(r) == len(hdr), f"{name}: row {i} has {len(r)} cells, header {len(hdr)}"
    return p, hdr, rows


def save(p, hdr, rows, name):
    shutil.copy2(p, p + ".prer4final.bak")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    # readback: column count + row count + no dropped key
    with open(p, newline="") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == len(rows), f"{name}: readback row count {len(back)} != {len(rows)}"
    for i, r in enumerate(back):
        assert len(r) == len(hdr), f"{name}: readback row {i} misaligned"
        assert all(v is not None for v in r.values()), f"{name}: readback row {i} has None"
    print(f"[{name}] rows={len(back)} cols={len(hdr)}  readback OK (backup .prer4final.bak)")


# ---------------------------------------------------------------- 1) registry
p, hdr, rows = load("phase6_experiment_registry.csv")
LOG = open(os.path.join(E, "phase6_round4_preregistration.md"), encoding="utf-8").read()
assert "13.13" in LOG, "prereg 13.13 missing - did the closure record land?"

REG = {
    "EXP-10": dict(
        status="DONE (Round 4 stage A, 3 arms x 20ep seed0; chain closed 2026-09-13 09:40:39)",
        key_result=(
            "ladder MEASURED, and the pre-registered prediction P2 is REFUTED. lane_mIoU: "
            "1/8 h32 = 0.5827 (3 seeds) -> 1/4 h32 = 0.5951 (+0.0124) -> 1/4 h16 = 0.5946/0.5936/0.5994 "
            "(3 seeds, mean 0.5959, +0.0132 >= 2sigma) at only +8.0% FLOPs -> 1/4 h8 = 0.4959 with "
            "lane_fg_iou = 0.0000 (DEAD). So resolution is the operative variable and width is inert "
            "(h16 reproduces h32) -- but the repair has a POSITIVE floor (+5.54% FLOPs for h16 without "
            "lateral), NOT the predicted 'zero or negative'. The h8 collapse is deterministic, not noise: "
            "two independent trunks (lean R3min and uniform U-minus) output the SAME 0.4959 / fg 0.0, and "
            "0.9917 pixel-acc with (0.9917+0)/2 = 0.49585 is the all-background signature; eval loaded 257 "
            "keys with missing=0/unexpected=0. F5 = PASS (up-only 0.5923 >= 0.5913) but only at h32."),
        evidence_tier="CONFIRMED (ladder measured; P2 refuted; h8 floor triple-checked)"),
    "EXP-11": dict(
        status="COMPLETE (3-seed means; F3 recomputed after the D7 relabel)",
        key_result=(
            "implementation isolation PROVEN not asserted (parsed-YAML model-tree delta = {detection.anchors} "
            "for both capacity levels; params identical to the digit; FLOPs identical to 4 decimals). "
            "Measured effect at lean capacity: F3 = 0.4950 - 0.3505 = +0.1445 mAP50 (2sigma = 0.0096, ~15sigma). "
            "The Round-4 report's earlier value 0.0462 was computed against the D7-contaminated group mean "
            "(two arms labelled 'old' had actually trained with the code-default k-means anchors); D7 is now "
            "closed EMPIRICALLY: the genuine old-anchor arms landed at 0.3452 / 0.3521, both inside 2sigma of "
            "the historical seed0 0.3543, while the mislabelled pair sits 15sigma away. The report now carries "
            "a fail-loud guard that refuses to emit any verdict if a base_* cell reads like k-means."),
        evidence_tier="CONFIRMED (F3 PASS, 15 sigma, 3 seeds, isolation proven)"),
    "EXP-12": dict(
        status="DONE (4 competing models, all seeds landed; Round 4 closed 2026-09-13)",
        key_result=(
            "Pareto front, no domination - and the 'free asymmetric allocation' candidate is refuted WITHOUT "
            "training. U uniform 0.3339M/1.6650G mAP50 0.5382; S naive full 1/4 0.2019M/1.6391G 0.4981; "
            "B thin 1/4 h16 0.1926M/1.1656G 0.4978 (3 seeds each); U-minus uniform+1/4 h8 0.3224M/1.6158G "
            "DEAD on the h8 floor. B == S on all three axes within the frozen noise scale (max |d| = 0.0003 "
            "mAP50 = 0.03 sigma) at -28.9% FLOPs => F6 PASS. U still wins detection (+0.0404 = +8.1%) but costs "
            "x1.73 params / x1.43 FLOPs, so there is NO domination either way; B leads on per-FLOP utility by "
            "+32% (+45% on lane). F8 FAIL is structural: rebuilding U-minus at h16 costs 1.7248G (no lateral) "
            "or 1.7641G (with lateral), both > U 1.6650G, so no uniform option can host a free lane repair "
            "(scripts/phase6_round4b_cost_probe.py, zero GPU)."),
        evidence_tier="COMPLETE / PARTIAL claim: Pareto trade-off CONFIRMED, domination NOT SUPPORTED"),
    "EXP-13": dict(
        status="COMPLETE (2x2 x 3 seeds, D7 fixed - first clean version)",
        key_result=(
            "clean 2x2 at 3 seeds: Baseline (lean+old) 0.3505 +- 0.0047; +Detection (lean+km) 0.4950 +- 0.0038 "
            "(+0.1445 = 15 sigma, zero FLOPs); +Lane (lean+full 1/4+old) 0.3588 +- 0.0031 (+0.0083, ns, but "
            "lane_mIoU +0.0141); +Both 0.4981 +- 0.0060 (+0.1475); +Full model B (thin 1/4 h16+km) 0.4978 "
            "+- 0.0030 at -28.9% FLOPs. Two readings: (1) anchor provenance is the dominant detection lever "
            "and the lane repair does not perturb detection (P4 CONFIRMED); (2) the two levers are ORTHOGONAL, "
            "not substitutes - interaction on the lane axis = -0.0052 (< 2sigma) ADDITIVE, and on lane_mIoU "
            "-0.0017 (< 2sigma) ADDITIVE. H-36's substitution law therefore applies specifically to the "
            "supervision x encoder-capacity pair, not to every other lever."),
        evidence_tier="COMPLETE (4 cells x 3 seeds)"),
}
for r in rows:
    if r["exp_id"] in REG:
        r.update(REG[r["exp_id"]])
save(p, hdr, rows, "phase6_experiment_registry.csv")

# ------------------------------------------------------- 2) architecture hypotheses
p, hdr, rows = load("phase6_architecture_hypotheses.csv")
HYP = {
    "H-36": dict(
        status=("ESTABLISHED (screening tier) -- and its SCOPE was NARROWED by Round 4's clean 2x2. "
                "The substitution is specific to the supervision x ENCODER-CAPACITY pair: on the "
                "lane-resolution axis the same anchor lever is ADDITIVE (interaction -0.0052 < 2sigma, "
                "EXP-13 3-seed). Do not generalise it to 'supervision substitutes for any other lever'."),
        next_action=("If pursued: extend C3/C4 to 3 seeds to lift H-36 itself off the screening tier. "
                     "The Round 4 lane axis is NOT an instance of this law and must not be cited as one."),
    ),
    "H-37": dict(
        status=("TESTED -- Round 4 closed 2026-09-13. Adjudicated mechanically by F1..F8 (thresholds frozen "
                "before any Round 4 metric existed): 6 PASS / 2 FAIL, verdict UNRESOLVED (bounded strictly "
                "between Level 2 and Level 3; Level 3 blocked ONLY by F4). SUPPORTED in direction: the "
                "low-price dimensions do pay (supervision 0 FLOPs -> +0.1407 mAP50; lane resolution "
                "+5.54% FLOPs -> +0.0132 lane_mIoU). NOT supported in strength: asymmetric allocation does "
                "NOT dominate uniform expansion (U wins detection +0.0404 at x1.73 params), so only a Pareto "
                "trade-off may be claimed. The 'free asymmetric' candidate is refuted by the cost model."),
        next_action=("STOP. Write up as a mechanism/analysis paper (no SOTA claim, no 'better architecture' "
                     "claim). Disclose three deviations: F4's literal failure, F5 having no h16 control, and "
                     "the gap in the prereg's level ladder. The only optional arm left is lean+1/4 h16 "
                     "without lateral (B' = 1.1394G, ~2.3 GPU-h) - it cannot make F4 pass and cannot reach "
                     "Level 3."),
    ),
}
for r in rows:
    if r["id"] in HYP:
        r.update(HYP[r["id"]])
save(p, hdr, rows, "phase6_architecture_hypotheses.csv")

# ---------------------------------------------------------------- 3) novelty matrix
p, hdr, rows = load("phase6_novelty_matrix.csv")
NEW = {
    "capacity-supervision coupling (H-34)": dict(
        status=("SUPERSEDED by H-36. Round 4 adds a SCOPE LIMIT to the successor: the substitution is "
                "supervision x encoder-capacity only. On the lane-resolution axis the anchor lever is "
                "additive (interaction -0.0052 < 2sigma).")),
    "task-conditional channel-vs-spatial allocation (EXP-01 H-35)": dict(
        status=("DOWNGRADED -- and Round 4 turns the diagnosis into a measured boundary. On the lane axis "
                "WIDTH is inert (1/4 h16 reproduces 1/4 h32: 0.5959 vs 0.5951, within 2sigma, at -28.9% "
                "FLOPs; h8 is a hard floor with a deterministic collapse) while RESOLUTION is the operative "
                "variable. So 'channel-vs-spatial' does not survive as a task-conditional claim; what "
                "survives is the much narrower 'resolution over width' cost rule.")),
    "bottleneck-profile-derived asymmetric architecture (H-32/H-33)": dict(
        status=("DEAD as a module-first claim -- re-expressed as H-37. Round 4 measured the naive "
                "spatial-heavy reading against its own cost-stripped variant: S (1.6391G) vs B (1.1656G) "
                "differ by at most 0.0003 mAP50 / 0.0008 lane_mIoU / 0.0012 DA, i.e. S is equivalent to B "
                "within the frozen noise scale while costing +40.6% FLOPs. F6 PASS, F8 FAIL.")),
    "supervision-capacity substitution at extreme compression (H-36 / EXP-09C)": dict(
        status=("ESTABLISHED (Round 4 did NOT extend it - it BOUNDED it). EXP-13 completed the second 2x2 "
                "at 3 seeds (D7 fixed) and shows the substitution lives on the encoder-capacity axis only: "
                "interaction on the lane axis = -0.0052 (< 2sigma) ADDITIVE. New empirical content = the "
                "scope restriction, not a stronger law."),
        new_empirical_law=("partial -- one architecture for the law itself; Round 4 supplies the scope "
                           "boundary (substitution is lever-specific) at 3 seeds")),
    "marginal-price resource allocation at extreme multi-task compression (H-37 / Round 4)": dict(
        status=("TESTED, Round 4 closed 2026-09-13 -> UNRESOLVED (6 PASS / 2 FAIL), bounded strictly "
                "between Level 2 and Level 3. What may be claimed: (a) the measured marginal-price table, "
                "(b) B == S at -28.9% FLOPs, (c) the lane repair's POSITIVE floor (+5.54% FLOPs) plus the "
                "h8 deterministic floor. What may NOT be claimed: domination of uniform expansion "
                "(U wins detection +0.0404 at x1.73 params) or of any published method."),
        expected_gain=("-28.9% FLOPs vs the naive spatial-heavy model at lane/det parity: MEASURED, F6 PASS. "
                       "Detection win: MEASURED FALSE (U wins). Domination of published baselines: NOT "
                       "ACHIEVED - TriLiteNet tiny DA 0.8796 vs ours 0.8629; TLP nano matches us at 1/10 "
                       "params."),
        risk=("the lane axis has only ONE architecture of support (Round 3 H2 WEAK); detection on ours does "
              "not beat published tiny models; F4 failed so the level ladder is not satisfied")),
}
for r in rows:
    key = r["candidate"]
    if key in NEW:
        r.update(NEW[key])
save(p, hdr, rows, "phase6_novelty_matrix.csv")

print("\n[registry-sync] all three tables updated and read back")
sys.exit(0)
