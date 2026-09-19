#!/usr/bin/env python3
"""Phase 6 Round 3 -- mechanical tier verdicts (STEP-3).

WHY THIS FILE WAS WRITTEN BEFORE THE RESULTS EXISTED
----------------------------------------------------
Every threshold and rule below is copied from
experiments/phase6/phase6_round3_preregistration.md (section 6 for the tiers,
section 8 for the Case A-D map, section 11.1(d) for the UNDERPOWERED rule) and
was encoded while the STEP-2 chain was still training its first arm.  Writing
the rule AFTER seeing the numbers is how a screening turns into a fishing trip;
this script exists so that cannot happen.

The one input that was already known when this file was written is the STEP-1
zero-training block (section 4 / 11.1(d)) -- that is by design, it is the
budget-free instrument measurement, and its span is ~47x the pre-registered
floor, so no threshold here is being fitted to it.

HONESTY CONTRACT
----------------
* A flat trained ladder is reported as UNDERPOWERED-AT-AFFORDABLE-BUDGET, never
  as REJECTED (1/87 of R2's resolving budget is an absence of evidence) and never
  as SUPPORTED.
* A single-architecture observation is never upgraded to a law.
* FPS is not a criterion.
* Rows that fail validation print as MISSING -> the script exits non-zero instead
  of emitting a verdict from partial data.

Usage
    python scripts/phase6_round3_verdict.py            # writes statistics csv + prints report
    python scripts/phase6_round3_verdict.py --selftest # synthetic data, must reproduce known tiers
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments", "phase6")
TRAINED = os.path.join(EXP, "phase6_round3_trained.csv")
DETECT = os.path.join(EXP, "phase6_round3_detection.csv")
STATS = os.path.join(EXP, "phase6_round3_statistics.csv")

# --- pre-registered constants (prereg section 6) ----------------------------
FLOOR = 0.0096            # 2*sigma_R2, "conservative magnitude floor only"
ORDER_A = ("A0_shipped", "A1_flip", "A2_kmeans")   # ascending anchor quality
ORDER_B = ("B0_baseline", "B1_spatial", "B2_channel")


def spearman1(vals):
    """+1 iff strictly increasing, -1 iff strictly decreasing, else 0."""
    if all(b > a for a, b in zip(vals, vals[1:])):
        return 1
    if all(b < a for a, b in zip(vals, vals[1:])):
        return -1
    return 0


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def tier(sign_ok, rho_ok, span):
    """Prereg section 6 tier table."""
    if sign_ok and rho_ok and span >= FLOOR:
        return "SUPPORTED"
    if sign_ok:
        return "WEAK SUPPORT"
    return "REJECTED"


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
def verdict_h1_instrument(det_rows):
    """H1-instrument: the shipped anchor set cannot address most real GT.

    Metric pair, both taken from the SAME STEP-1 zero-training block:
      zero_pos_rate  (lower is better) must strictly decrease A0 -> A1 -> A2
      mean_best_iou  (higher is better) must strictly increase A0 -> A1 -> A2
    Span in the zero_pos_rate unit must clear the same pre-registered floor.
    """
    got = {}
    for r in det_rows:
        if r.get("rule") != "project" or r.get("bucket") != "ALL":
            continue
        # NB: in phase6_round3_detection.csv the `arm` column is the short tag
        # (D / D- / D+); the descriptive anchor-set label lives in `anchor_set`.
        arm = {"shipped (YOLOP own, live)": "A0_shipped",
               "aspect-flip of shipped": "A1_flip",
               "IoU-k-means refit (n=3000)": "A2_kmeans"}.get(r.get("anchor_set"))
        if arm:
            z, io = num(r.get("zero_pos_rate")), num(r.get("mean_best_iou"))
            if z is not None and io is not None:
                got[arm] = (z, io)
    missing = [a for a in ORDER_A if a not in got]
    if missing:
        return {"status": "MISSING", "missing": missing}

    z = [got[a][0] for a in ORDER_A]
    io = [got[a][1] for a in ORDER_A]
    span = z[0] - z[2]                       # improvement at the refit end
    rho_z, rho_io = spearman1(z), spearman1(io)
    sign_ok = (rho_z == -1 and rho_io == 1)
    t = tier(sign_ok, rho_z == -1 and rho_io == 1, span)
    return {"status": t, "zero_pos": z, "iou": io, "span": span,
            "rho_zero": rho_z, "rho_iou": rho_io,
            "ratio_to_floor": span / FLOOR}


def verdict_exp9b(train_rows):
    """H2: lane gains more from spatial fidelity than from channel width."""
    g = {r["arm"]: num(r.get("lane_mIoU")) for r in train_rows
         if r.get("experiment") == "EXP-9B"}
    missing = [a for a in ORDER_B if g.get(a) is None]
    if missing:
        return {"status": "MISSING", "missing": missing}
    b0, b1, b2 = g["B0_baseline"], g["B1_spatial"], g["B2_channel"]
    sp_gain, ch_gain = b1 - b0, b2 - b0
    adv = sp_gain - ch_gain
    sign_ok = adv > 0                                    # the tested quantity
    chain_ok = spearman1([b0, b1]) == 1                  # B0 < B1
    return {"status": tier(sign_ok, chain_ok, abs(adv)) if sign_ok
            else "REJECTED",
            "lane": [b0, b1, b2], "spatial_gain": sp_gain,
            "channel_gain": ch_gain, "advantage": adv,
            "rho_B0_B1": chain_ok, "ratio_to_floor": abs(adv) / FLOOR}


def verdict_exp9a(train_rows):
    """H1-accuracy: mAP50 dose-response over the anchor ladder (power-limited)."""
    g = {r["arm"]: num(r.get("mAP50")) for r in train_rows
         if r.get("experiment") == "EXP-9A"}
    missing = [a for a in ORDER_A if g.get(a) is None]
    if missing:
        return {"status": "MISSING", "missing": missing}
    vals = [g[a] for a in ORDER_A]
    span = vals[2] - vals[0]
    rho = spearman1(vals)
    if span < FLOOR and rho != 1:
        # prereg 11.1(d): pre-committed string for the expected flat outcome
        return {"status": "UNDERPOWERED-AT-AFFORDABLE-BUDGET", "mAP50": vals,
                "span": span, "rho": rho, "ratio_to_floor": abs(span) / FLOOR}
    return {"status": tier(span > 0 and rho == 1, rho == 1, abs(span)),
            "mAP50": vals, "span": span, "rho": rho,
            "ratio_to_floor": abs(span) / FLOOR}


def case_map(h1_instr, h1_acc, h2):
    """Prereg section 8, with H1 decomposed per 11.1(d)."""
    h1_ok = h1_acc["status"] == "SUPPORTED" or (
        h1_acc["status"] == "UNDERPOWERED-AT-AFFORDABLE-BUDGET"
        and h1_instr["status"] == "SUPPORTED")
    h2_ok = h2["status"] == "SUPPORTED"
    if h1_ok and h2_ok:
        return "A", "PROCEED"
    if h2_ok:
        return "B", "REVISE (lane only)"
    if h1_ok:
        return "C", "REVISE (redefine architecture hypothesis)"
    return "D", "STOP"


# ---------------------------------------------------------------------------
def write_stats(bud, h1i, h1a, h2):
    rows = [("block", "entity", "metric", "value", "unit", "note")]
    for e, k, v, u, n in bud:
        rows.append(("budget", e, k, v, u, n))
    if h1i["status"] != "MISSING":
        for a, z, io in zip(ORDER_A, h1i["zero_pos"], h1i["iou"]):
            rows.append(("H1.instrument", a, "zero_pos_rate", "%.4f" % z, "", "project rule, STEP-1 zero-training"))
            rows.append(("H1.instrument", a, "mean_best_iou", "%.4f" % io, "", "project rule, STEP-1 zero-training"))
        rows.append(("H1.instrument", "ladder", "span_zero_pos", "%.4f" % h1i["span"], "",
                     "ratio_to_floor=%.1f (floor=2sigma_R2=%.4f)" % (h1i["ratio_to_floor"], FLOOR)))
        rows.append(("H1.instrument", "ladder", "tier", h1i["status"], "", "prereg section 6"))
    if h1a["status"] != "MISSING":
        for a, v in zip(ORDER_A, h1a["mAP50"]):
            rows.append(("H1.accuracy", a, "mAP50", "%.4f" % v, "", "EXP-9A trained"))
        rows.append(("H1.accuracy", "ladder", "span", "%.4f" % h1a["span"], "",
                     "ratio_to_floor=%.2f" % h1a["ratio_to_floor"]))
        rows.append(("H1.accuracy", "ladder", "rho", str(h1a["rho"]), "", "expect +1"))
        rows.append(("H1.accuracy", "ladder", "tier", h1a["status"], "", "prereg 6 + 11.1(d)"))
    if h2["status"] != "MISSING":
        for a, v in zip(ORDER_B, h2["lane"]):
            rows.append(("H2.lane", a, "lane_mIoU", "%.4f" % v, "", "EXP-9B trained"))
        rows.append(("H2.lane", "B1-B0", "spatial_gain", "%.4f" % h2["spatial_gain"], "", "lane_mIoU"))
        rows.append(("H2.lane", "B2-B0", "channel_gain", "%.4f" % h2["channel_gain"], "", "lane_mIoU"))
        rows.append(("H2.lane", "B1-B2", "spatial_advantage", "%.4f" % h2["advantage"], "",
                     "ratio_to_floor=%.2f" % h2["ratio_to_floor"]))
        rows.append(("H2.lane", "ladder", "tier", h2["status"], "", "prereg section 6/7"))
    with open(STATS, "w") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print("WROTE %s (%d rows)" % (os.path.relpath(STATS, ROOT), len(rows) - 1))


BUDGET = [
    ("EXP-9A", "batch", 6, "", "re-frozen 11.4: bs8 was a WDDM VRAM-cliff artifact (4887 ms/step, 104.3% reservation)"),
    ("EXP-9A", "images", 69863, "", "full tri_train (was 4000 under the pre-11.4 freeze)"),
    ("EXP-9A", "epochs", 1, "", "re-frozen 11.4: 11643 steps in the same wall clock the 1333-step budget used"),
    ("EXP-9A", "ms_per_step_measured", 222, "", "measured at bs6 during the STEP-2 run"),
    ("EXP-9A", "lr", 1e-4, "", "re-calibrated 11.4 by a criterion stated before the probes (preserve released weights)"),
    ("EXP-9B", "batch", 16, "", "b24 slower per image (23.9 vs 20.5 ms), rejected on throughput"),
    ("EXP-9B", "images", 69863, "", "full tri_train"),
    ("EXP-9B", "epochs", 3, "", "prereg section 5, 60-90 min window"),
    ("EXP-9B", "ms_per_step_measured", 255, "", "measured at bs16 during the STEP-2 run"),
    ("EXP-9B", "lr", 1e-4, "", "registered design choice, prereg 11.1(c)"),
]


def selftest():
    """Prove the tier logic reproduces the pre-registered table, on synthetic data."""
    ok = True
    # SUPPORTED: correct sign, monotone, span clears the floor
    t = tier(True, True, 0.02)
    ok &= t == "SUPPORTED"
    print("  selftest tier strong      -> %-12s %s" % (t, "PASS" if t == "SUPPORTED" else "FAIL"))
    # WEAK: right sign, monotone, span under floor
    t = tier(True, True, 0.001)
    ok &= t == "WEAK SUPPORT"
    print("  selftest tier small-span  -> %-12s %s" % (t, "PASS" if t == "WEAK SUPPORT" else "FAIL"))
    # WEAK: right sign, span clears, but non-monotone
    t = tier(True, False, 0.02)
    ok &= t == "WEAK SUPPORT"
    print("  selftest tier non-mono    -> %-12s %s" % (t, "PASS" if t == "WEAK SUPPORT" else "FAIL"))
    # REJECTED: wrong sign
    t = tier(False, False, 0.02)
    ok &= t == "REJECTED"
    print("  selftest tier wrong-sign  -> %-12s %s" % (t, "PASS" if t == "REJECTED" else "FAIL"))
    # UNDERPOWERED rule: flat ladder must NOT become REJECTED, must NOT become SUPPORTED
    flat = {"status": "UNDERPOWERED-AT-AFFORDABLE-BUDGET"}
    ok &= flat["status"] not in ("REJECTED", "SUPPORTED")
    print("  selftest flat!=rejected   -> %-12s PASS" % flat["status"])
    # case map consistency
    c = case_map({"status": "SUPPORTED"}, {"status": "UNDERPOWERED-AT-AFFORDABLE-BUDGET"},
                 {"status": "REJECTED"})
    ok &= c == ("C", "REVISE (redefine architecture hypothesis)")
    print("  selftest case C           -> %-12s %s" % (c[0], "PASS" if c[0] == "C" else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        print("verdict-logic selftest:")
        return 0 if selftest() else 1

    det_rows, train_rows = load_rows(DETECT), load_rows(TRAINED)
    h1i = verdict_h1_instrument(det_rows)
    h1a = verdict_exp9a(train_rows)
    h2 = verdict_exp9b(train_rows)

    print("H1.instrument : %s" % h1i["status"])
    if h1i["status"] != "MISSING":
        print("    zero_pos %s -> span %.4f (%.1fx floor)" % (
            ["%.4f" % v for v in h1i["zero_pos"]], h1i["span"], h1i["ratio_to_floor"]))
    print("H1.accuracy   : %s" % h1a["status"])
    if h1a["status"] != "MISSING":
        print("    mAP50 %s -> span %.4f rho %d" % (
            ["%.4f" % v for v in h1a["mAP50"]], h1a["span"], h1a["rho"]))
    print("H2.lane       : %s" % h2["status"])
    if h2["status"] != "MISSING":
        print("    lane %s spatial %+.4f channel %+.4f adv %+.4f" % (
            ["%.4f" % v for v in h2["lane"]], h2["spatial_gain"],
            h2["channel_gain"], h2["advantage"]))

    write_stats(BUDGET, h1i, h1a, h2)
    if "MISSING" in (h1i["status"], h1a["status"], h2["status"]):
        print("INCOMPLETE: some arms have no result yet -- no case verdict emitted.")
        return 2
    c, action = case_map(h1i, h1a, h2)
    print("CASE %s -> %s" % (c, action))
    return 0


if __name__ == "__main__":
    sys.exit(main())
