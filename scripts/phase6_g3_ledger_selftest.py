#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-test for scripts/phase6_g3_ledger.py -- run it BEFORE the data exists.

The judging code must be exercised while the outcome is still unknown, otherwise
a bug in the judge is indistinguishable from a real result.  Each case below
writes a synthetic g3_results.csv into a temp dir, runs the ledger on it, and
asserts the frozen verdict (R0/R1/R3).  No real data, no GPU, nothing written
into experiments/.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "scripts", "phase6_g3_ledger.py")
PY = sys.executable

HDR = ["variant", "cell", "z", "encoder", "epochs", "params_M", "flops_G",
       "fps", "mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg",
       "peak_gpu_mem_mib", "final_train_loss", "train_wall_min", "seed",
       "source", "git_commit"]

GEOM = {  # tier -> (params_M, flops_G) mimicking the solved G3 geometry
    "a1_zspend": (0.3019, 2.0889), "a1_encspend": (0.3020, 1.4790),
    "a2_zspend": (0.2881, 2.0865), "a2_encspend": (0.2921, 1.6154),
}


def row(cell, z, enc, params, flops, seed, metrics, source="trained"):
    m = dict(zip(["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU",
                  "lane_fg"], metrics))
    return ["g3", cell, z, enc, 20, params, flops, 100.0,
            m["mAP50"], m["mAP50_95"], m["da_mIoU"], m["da_fg"],
            m["lane_mIoU"], m["lane_fg"], "2000", "0.2", "150", seed,
            source, "deadbeef"]


# metrics order: mAP50, mAP50_95, da_mIoU, da_fg, lane_mIoU, lane_fg
def archname(arch):
    return "dws" if arch == "a1" else "ir"


def build(arch, base_m, zspend_m, encspend_m, params_override=None):
    pz, fz = GEOM[f"{arch}_zspend"]
    pe, fe = GEOM[f"{arch}_encspend"]
    if params_override:
        pe = params_override
    rows = []
    # 3 seeds of the base cell (so the ruler exists)
    for s in (0, 1, 2):
        rows.append(row(f"{arch}_base", 16, archname(arch), 0.2014, 1.0796, s,
                        base_m))
    rows.append(row(f"{arch}_zspend", 128, archname(arch), pz, fz, 0, zspend_m))
    rows.append(row(f"{arch}_encspend", 16, archname(arch), pe, fe, 0,
                    encspend_m))
    return rows


def run(rows, expect):
    d = tempfile.mkdtemp(prefix="g3selftest_")
    p = os.path.join(d, "g3_results.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)
    out = subprocess.run([PY, LEDGER, "--csv", p, "--outdir", d],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:])
        print(out.stderr[-3000:])
        raise SystemExit("ledger crashed")
    with open(os.path.join(d, "g3_ledger.json")) as fh:
        j = json.load(fh)
    got = j.get("transfer_verdict")
    status = "OK " if got == expect else "FAIL"
    print(f"  {status} expected={expect:18s} got={got:18s} "
          f"({j.get('transfer_detail','')[:70]})")
    return got == expect


def main():
    print("=" * 84)
    print("G3 LEDGER SELF-TEST -- frozen decision rules, synthetic data")
    print("=" * 84)
    ok = True

    # Case 1: both architectures dominate, signs agree -> TRANSFERRED
    b = [0.45, 0.15, 0.850, 0.770, 0.590, 0.200]
    z1 = [0.451, 0.150, 0.849, 0.769, 0.591, 0.201]     # Z spend: flat
    e1 = [0.500, 0.170, 0.862, 0.780, 0.600, 0.210]     # enc spend: better
    ok &= run(build("a1", b, z1, e1) + build("a2", b, z1, e1), "TRANSFERRED")

    # Case 2: arch-2 has no encoder advantage (Z_WINS) -> NOT_TRANSFERRED
    z2 = [0.451, 0.150, 0.849, 0.769, 0.591, 0.201]
    e2 = [0.440, 0.145, 0.845, 0.765, 0.588, 0.198]
    ok &= run(build("a1", b, z1, e1) + build("a2", b, z2, e2),
              "NOT_TRANSFERRED")

    # Case 3: budget gate fails on arch-2 (encoder tier 12% bigger) -> INCONCLUSIVE
    ok &= run(build("a1", b, z1, e1) +
              build("a2", b, z1, e1, params_override=0.2881 * 1.12),
              "INCONCLUSIVE")

    # Case 4: missing the enc-spend cell on arch-2 -> INCONCLUSIVE
    rows = build("a1", b, z1, e1) + build("a2", b, z1, e1)
    rows = [r for r in rows if not (r[1].startswith("a2") and "encspend" in r[1])]
    ok &= run(rows, "INCONCLUSIVE")

    # Case 5: a failed-no-metrics row must never be scored
    rows = build("a1", b, z1, e1) + build("a2", b, z1, e1)
    rows.append(row("a2_zspend", 128, "ir", 0.2881, 2.0865, 0, [0.9] * 6,
                    source="failed-no-metrics"))
    ok &= run(rows, "TRANSFERRED")

    print()
    if ok:
        print("ALL 5 CASES PASS -- the judge behaves as pre-registered")
    else:
        raise SystemExit("SELF-TEST FAILED")


if __name__ == "__main__":
    main()
