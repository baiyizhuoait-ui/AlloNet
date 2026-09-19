#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 4 -- repair the H-34 row defect and apply the novelty-matrix updates.

DEFECT FOUND (same class as Round-3 D5): phase6_architecture_hypotheses.csv row
H-34 carries 13 fields instead of 11, because its `status` value contains three
unquoted commas. Fields 9/10/11 are the fragments of `status`; field 12 is the
real `next_action`. Silent, and it makes every later column of that row wrong for
any consumer that trusts column position.

Repair = rejoin 9..11 into `status` and move 12 to `next_action`. Then the row is
rewritten through csv.writer so the file is properly quoted from now on.
"""
import csv
import os

D = "~/ai_study/trac/experiments/phase6"
os.chdir(D)
HYP = "phase6_architecture_hypotheses.csv"
NOV = "phase6_novelty_matrix.csv"

# ------------------------------------------------------------- repair H-34
rows = list(csv.reader(open(HYP, newline="", encoding="utf-8-sig")))
hdr, body = rows[0], rows[1:]
assert len(hdr) == 11
bad = [i for i, r in enumerate(body) if len(r) != 11]
print("[hypotheses] rows with wrong column count:", [body[i][0] for i in bad])
STATUS = ("REJECTED (screening - EXP-07 4ep 2x2: interaction -0.0164 = 1.12x noise, below the 2x gate, "
          "and NEGATIVE in 13/13 metric faces i.e. direction contradicts H-M). "
          "20ep EXP-09C then CONTRADICTED it outright (interaction -0.1435): the axes are SUBSTITUTES, "
          "not complementary. Superseded by H-36")
NEXT = ("do NOT launch the gated 20ep for H-34. Superseded by H-36 (substitution). "
        "Decisive follow-up = A-uniform + k-means anchors at 20ep; gaining a +0.14 mAP50 effect from "
        "capacity is no longer expected (EXP-07 predicted, EXP-09C confirmed)")
for i in bad:
    r = body[i]
    assert r[0] == "H-34", f"unexpected malformed row {r[0]}"
    fixed = r[:9] + [STATUS, NEXT]
    assert len(fixed) == 11
    body[i] = fixed
    print("[hypotheses] H-34 repaired: 13 -> 11 fields")
with open(HYP, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([hdr] + body)
back = list(csv.reader(open(HYP, newline="", encoding="utf-8-sig")))
assert all(len(r) == 11 for r in back), "repair failed"
assert [r for r in back if r[0] == "H-34"][0][9].startswith("REJECTED")
assert [r for r in back if r[0] in ("H-36", "H-37")]
print(f"[hypotheses] readback OK ({len(back)-1} hypotheses, all 11 cols)")

# --------------------------------------------------- novelty matrix updates
rows = list(csv.reader(open(NOV, newline="", encoding="utf-8-sig")))
hdr, body = rows[0], rows[1:]
UPD = {
    "bottleneck-profile-derived asymmetric architecture":
        ("DEAD as a module-first claim -- re-expressed as H-37 (marginal-price allocation). The naive "
         "spatial-heavy reading is Pareto-dominated by its own cost-stripped variant (S 1.6391G vs "
         "B 1.1656G, same lane within 2 sigma if F1/F6 pass)"),
    "task-conditional channel-vs-spatial allocation":
        ("DOWNGRADED -- H2 WEAK on the second architecture (saturated head), and the R2-family lane head "
         "is near its plateau. Any revival needs a lane-headroom backbone plus an explicit power analysis"),
    "capacity-supervision coupling (H-34)":
        ("SUPERSEDED -- H-34 (capacity amplifies the value of supervision) was REJECTED at 4ep and then "
         "CONTRADICTED at 20ep by EXP-09C: interaction -0.1435, i.e. the axes are SUBSTITUTES. See the "
         "new H-36 row"),
    "cross-architecture supervision/assignment bottleneck (Round 3 H1)":
        ("ESTABLISHED (2 architectures, ruler held fixed) -- this is the axis the Round 4 allocation rule "
         "is anchored on. Its weakness is that the bottleneck may be a training-config choice rather than "
         "an architecture property; Round 4 answers that by pricing the config choice against capacity"),
}
for r in body:
    for k, v in UPD.items():
        if r[0].startswith(k):
            r[9] = v
            print("[novelty] updated status:", k[:52])

NEW = [
    ["supervision-capacity substitution at extreme compression (H-36 / EXP-09C)",
     "MED -- assignment is heavily studied; the capacity interaction is not",
     "PARTIAL (better assignment helps small objects -- YOLO26 STAL and the DALA/RLA family)",
     "YES -- both levers measured in the same factorial at fixed params/FLOPs",
     "YES -- supervision and capacity compete for the same headroom",
     "partial -- one architecture, 20ep, 2 of 4 cells n=1",
     "zero inference cost (training config only)",
     "high-capacity d_anchor = -0.0007 vs low-capacity +0.1428",
     "screening tier needs seed extension before any external claim",
     "ESTABLISHED (screening) -- gated seed extension; NOT a Round 4 mandatory arm"],
    ["marginal-price resource allocation at extreme multi-task compression (H-37 / Round 4)",
     "MED-HIGH -- MDANet (IEEE TITS 2026) and MT-TPPNet (Computers 2025) are adjacent and share the vocabulary",
     "partial (per-dimension allocation and asymmetric projections are known)",
     "YES -- allocation derived from measured marginal prices under budget parity",
     "YES -- an allocation RULE, not a module",
     "not established",
     "budget-parity by construction; zero new model code",
     "predicted -28.9% FLOPs vs the naive spatial-heavy model at lane/det parity (test F6)",
     "the lane axis has only ONE architecture of support (Round 3 H2 WEAK); the detection axis has two",
     "ACTIVE -- Round 4 primary; adjudicated mechanically by F1..F8"],
]
have = {r[0] for r in body}
for r in NEW:
    if r[0] not in have:
        body.append(r)
        print("[novelty] +", r[0][:60])
with open(NOV, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows([hdr] + body)
back = list(csv.reader(open(NOV, newline="", encoding="utf-8-sig")))
assert len(back) == len(body) + 1
assert all(len(r) == 10 for r in back), "novelty column count drifted"
assert back[-1][0] == NEW[-1][0]
print(f"[novelty] readback OK ({len(back)-1} rows)")
print("\nREPAIR + UPDATE VERIFIED")
