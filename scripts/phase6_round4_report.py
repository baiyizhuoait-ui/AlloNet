#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 Round 4 -- mechanical report + Level adjudicator.

WRITTEN BEFORE ANY ROUND-4 METRIC EXISTED. The F1..F8 thresholds below are copied
from experiments/phase6/phase6_round4_preregistration.md section 7 and must not be
edited after the numbers are read. Any post-hoc change goes to the prereg's
append-only section 13 and is reflected here with an explicit comment.

Fail-closed by default: if a cell required by a test is missing, that test reports
PENDING and the Level block is NOT emitted (no "partial conclusions").
    python scripts/phase6_round4_report.py                 # strict
    python scripts/phase6_round4_report.py --allow-partial  # tables with NA
    python scripts/phase6_round4_report.py --selftest       # threshold self-check
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "experiments")

# ---------------------------------------------------------------- frozen consts
SIG = {"mAP50": 0.0048, "lane_mIoU": 0.0019, "lane_fg_iou": 0.0031, "da_mIoU": 0.00115}
REF = {  # measured before Round 4; see prereg section 7
    "lean_km_lane_mean": 0.5827,      # lean + k-means, 20ep, 3 seeds
    "full14_km_lane_mean": 0.5951,    # lean + full 1/4 + k-means, 20ep, 3 seeds
    "full14_km_map_mean": 0.4981,     # same cells, mAP50
    "lean_old_map_s0": 0.3543,        # lean + old anchors, 20ep, seed0
}
TH = {
    "F1": REF["full14_km_lane_mean"] - 2 * SIG["lane_mIoU"],   # 0.5913
    "F2": REF["lean_km_lane_mean"] + 2 * SIG["lane_mIoU"],     # 0.5865
    "F3": 2 * SIG["mAP50"],                                    # 0.0096
    "F6_lane": REF["full14_km_lane_mean"] - 2 * SIG["lane_mIoU"],
    "F6_map": REF["full14_km_map_mean"] - 2 * SIG["mAP50"],    # 0.4885
    "F6_da": 2 * SIG["da_mIoU"],                               # 0.0023
    "F7": 2 * SIG["mAP50"],
}

# ------------------------------------------------------------------- cell table
# name -> training outdir (the eval dir is "<outdir>_eval"), reused or new.
R4 = "experiments/phase6/round4/"
CELLS = {
    "base_s0": "experiments/phase4a/exp4A_r2_z16_e20",
    # ---- D7 FIX (2026-09-13, prereg 13.4 + 13.13) ------------------------------
    # r4_R4A0s1 / r4_R4A0s2 were trained with the CODE-DEFAULT anchors, and commit
    # c8aca35 (2026-09-08) had switched that default to IoU-k-means; configs/
    # phase4a_r2_z16.yaml has no anchors block. So those two arms are lean+k-means,
    # NOT the old-anchor baseline. The true old-anchor arms were trained in Round 4B
    # as r4_R4A0o1 / r4_R4A0o2 (configs/phase6_r4_lean_old.yaml pins the historical
    # anchors explicitly). F3's previous value 0.0462 was computed against the
    # contaminated group mean; the D7 guard below makes this un-repeatable.
    "base_s1": R4 + "r4_R4A0o1",
    "base_s2": R4 + "r4_R4A0o2",
    "det_s0": "experiments/phase6c/exp9_r2z16_km",
    "det_s1": "experiments/phase4a/exp4B_danc_z16_e20_s1",
    "det_s2": "experiments/phase4a/exp4B_danc_z16_e20_s2",
    "lane_s0": "experiments/phase4a/exp4B_l14f1_z16_e20",
    "lane_s1": "experiments/phase4a/exp4B_l14f1_z16_e20_s1",
    "lane_s2": "experiments/phase4a/exp4B_l14f1_z16_e20_s2",
    "both_s0": "experiments/phase6/exp6_combo20",
    "both_s1": "experiments/phase6/exp6_combo20_s1",
    "both_s2": "experiments/phase6/exp6_combo20_s2",
    "R2thin_s0": R4 + "r4_R4R2thin",
    "R2thin_s1": R4 + "r4_R4R2s1",   # Round 4B (prereg 13.5): Model B -> 3 seeds
    "R2thin_s2": R4 + "r4_R4R2s2",
    "R3min_s0": R4 + "r4_R4R3min",
    "R1up_s0": R4 + "r4_R4R1up",
    "U_s0": "experiments/phase6c/exp9_aunif_km",
    "U_s1": R4 + "r4_R4Us1",
    "U_s2": R4 + "r4_R4Us2",
    "Umin_s0": R4 + "r4_R4Umin",
}

CELL_META = {  # name: (model, construction, params, flops_G)  [params/FLOPs measured]
    "base_s0": ("baseline", "lean + old anchors", 201366, 1.0796),
    "base_s1": ("baseline", "lean + old anchors", 201366, 1.0796),
    "base_s2": ("baseline", "lean + old anchors", 201366, 1.0796),
    "det_s0": ("+Detection", "lean + k-means", 201366, 1.0796),
    "det_s1": ("+Detection", "lean + k-means", 201366, 1.0796),
    "det_s2": ("+Detection", "lean + k-means", 201366, 1.0796),
    "lane_s0": ("+Lane", "lean + full 1/4 + old", 201878, 1.6391),
    "lane_s1": ("+Lane", "lean + full 1/4 + old", 201878, 1.6391),
    "lane_s2": ("+Lane", "lean + full 1/4 + old", 201878, 1.6391),
    "both_s0": ("+Both", "lean + full 1/4 + k-means", 201878, 1.6391),
    "both_s1": ("+Both", "lean + full 1/4 + k-means", 201878, 1.6391),
    "both_s2": ("+Both", "lean + full 1/4 + k-means", 201878, 1.6391),
    "R2thin_s0": ("B", "lean + thin 1/4 h16 + k-means", 192566, 1.1656),
    "R2thin_s1": ("B", "lean + thin 1/4 h16 + k-means", 192566, 1.1656),
    "R2thin_s2": ("B", "lean + thin 1/4 h16 + k-means", 192566, 1.1656),
    "R3min_s0": ("R3min", "lean + min 1/4 h8 + k-means", 189638, 1.0174),
    "R1up_s0": ("R1up", "lean + 1/4 h32, no lateral + k-means", 201366, 1.6129),
    "U_s0": ("U", "uniform + 1/8 + k-means", 333862, 1.6650),
    "U_s1": ("U", "uniform + 1/8 + k-means", 333862, 1.6650),
    "U_s2": ("U", "uniform + 1/8 + k-means", 333862, 1.6650),
    "Umin_s0": ("Uminus", "uniform + 1/4 h8 + k-means", 322390, 1.6158),
}

KEYS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg_iou", "lane_mIoU", "lane_fg_iou",
        "det_AP50_small", "det_AP50_medium", "det_AP50_large",
        "det_AP5095_small", "det_AP5095_medium", "det_AP5095_large",
        "det_recall50_small", "det_recall50_medium", "det_recall50_large",
        "p50_latency_ms", "p95_latency_ms", "eval_fps", "gpu_memory_mib",
        "parameters", "flops"]


def load(name):
    mp = os.path.join(ROOT, CELLS[name] + "_eval", "metrics.json")
    if not os.path.exists(mp):
        return None
    d = json.load(open(mp))
    d["_metric_path"] = os.path.relpath(mp, ROOT)
    return d


def mean_of(cells, key, names=None):
    vals = [cells[n][key] for n in (names or cells) if n in cells and cells[n]
            and key in cells[n]]
    return (st.fmean(vals), st.stdev(vals) if len(vals) > 1 else float("nan"),
            len(vals)) if vals else (float("nan"), float("nan"), 0)


def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        assert abs(TH["F1"] - 0.5913) < 5e-5, TH["F1"]
        assert abs(TH["F2"] - 0.5865) < 5e-5, TH["F2"]
        assert abs(TH["F3"] - 0.0096) < 5e-5, TH["F3"]
        assert abs(TH["F6_map"] - 0.4885) < 5e-5, TH["F6_map"]
        assert abs(TH["F6_da"] - 0.0023) < 5e-5, TH["F6_da"]
        print("[selftest] OK -- F1/F2/F3/F6 thresholds match prereg section 7")
        return 0

    cells = {k: load(k) for k in CELLS}
    missing = [k for k in CELLS if cells[k] is None]

    # ---- D7 fail-loud guard (prereg 13.4 / 13.13) -----------------------------
    # The old-anchor baseline mAP50 is ~0.345-0.354. If a "base_*" cell reads like
    # k-means (>0.45), the cell map is wrong AGAIN - refuse to emit any verdict
    # rather than silently inflating F3. That mistake already cost us one round.
    for _n in ("base_s0", "base_s1", "base_s2"):
        _c = cells.get(_n)
        if _c and float(_c.get("mAP50", 0.0)) > 0.45:
            print(f"[D7 GUARD] {_n} mAP50={float(_c['mAP50']):.4f} > 0.45 -> this cell is "
                  f"NOT the old-anchor baseline (prereg 13.4). Refusing to emit verdicts.")
            return 2
    print("[D7 GUARD] ok - base_s0/s1/s2 all read like the old-anchor baseline")
    print(f"[report] cells present {len(CELLS)-len(missing)}/{len(CELLS)}")
    if missing:
        print("[report] MISSING:", ", ".join(missing))

    def M(keys, key):
        return mean_of(cells, key, keys)

    # ---------------------------------------------------------------- tables
    rows = []
    for n, c in cells.items():
        if not c:
            continue
        model, con, pp, fl = CELL_META[n]
        seed = n.rsplit("_s", 1)[-1]
        rows.append(dict(model=model, cell=n, construction=con, seed=seed,
                         params=pp, flops_G=fl,
                         **{k: c.get(k) for k in KEYS},
                         metric_path=c["_metric_path"]))
    with open(os.path.join(E, "phase6", "phase6_final_models.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)

    # ablation: per component, 3-seed mean/sd
    abl = []
    comps = [("Baseline", ["base_s0", "base_s1", "base_s2"]),
             ("+Detection component", ["det_s0", "det_s1", "det_s2"]),
             ("+Lane component", ["lane_s0", "lane_s1", "lane_s2"]),
             ("+Both", ["both_s0", "both_s1", "both_s2"]),
             ("+Full model (B, thin 1/4)", ["R2thin_s0", "R2thin_s1", "R2thin_s2"])]
    bmap = M(comps[0][1], "mAP50")
    for name, keys in comps:
        have = [k for k in keys if cells.get(k)]
        m, s, k = M(keys, "mAP50")
        lm, _, _ = M(keys, "lane_mIoU")
        dm, _, _ = M(keys, "da_mIoU")
        con = CELL_META[keys[0]][1]
        abl.append(dict(arm=name, construction=con, n_seeds=k,
                        params=CELL_META[keys[0]][2], flops_G=CELL_META[keys[0]][3],
                        mAP50_mean=fmt(m), mAP50_sd=fmt(s),
                        d_mAP50_vs_baseline=fmt(m - bmap[0]),
                        lane_mIoU_mean=fmt(lm), da_mIoU_mean=fmt(dm),
                        seeds=",".join(x.rsplit("_s", 1)[-1] for x in have)))
    with open(os.path.join(E, "phase6", "phase6_ablation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(abl[0].keys()))
        w.writeheader()
        w.writerows(abl)

    # efficiency: per-task utility + Pareto (maximise mAP50 & lane, minimise FLOPs)
    # Explicit model->cells map (seed means) rather than a prefix filter, so that
    # Model S (the lean + full 1/4 + k-means cells that DOUBLE as the EXP-13
    # "+Both" ablation arm) is present in the competition table.
    EFFSET = [("U", ["U_s0", "U_s1", "U_s2"]),
              ("S", ["both_s0", "both_s1", "both_s2"]),
              ("B", ["R2thin_s0", "R2thin_s1", "R2thin_s2"]),
              ("Uminus", ["Umin_s0"]),
              ("R3min", ["R3min_s0"]),
              ("R1uponly", ["R1up_s0"]),
              ("lean+km (reference)", ["det_s0", "det_s1", "det_s2"])]
    eff = []
    for model, keys in EFFSET:
        have = [k for k in keys if cells.get(k)]
        if not have:
            continue
        pp, fl = CELL_META[have[0]][2], CELL_META[have[0]][3]
        mp, _, n = mean_of(cells, "mAP50", have)
        lmn, _, _ = mean_of(cells, "lane_mIoU", have)
        dmn, _, _ = mean_of(cells, "da_mIoU", have)
        eff.append(dict(model=model, construction=CELL_META[have[0]][1], n_seeds=n,
                        params=pp, flops_G=fl, mAP50=round(mp, 4), lane_mIoU=round(lmn, 4),
                        da_mIoU=round(dmn, 4),
                        mAP50_per_GFLOPs=fmt(mp / fl), lane_mIoU_per_GFLOPs=fmt(lmn / fl),
                        mAP50_per_Mparams=fmt(mp / (pp / 1e6)),
                        lane_mIoU_per_Mparams=fmt(lmn / (pp / 1e6))))

    def dominated(x, others):
        for y in others:
            if y is x:
                continue
            if (y["flops_G"] <= x["flops_G"]
                    and y["mAP50"] >= x["mAP50"] and y["lane_mIoU"] >= x["lane_mIoU"]
                    and (y["flops_G"] < x["flops_G"] or y["mAP50"] > x["mAP50"]
                         or y["lane_mIoU"] > x["lane_mIoU"])):
                return True
        return False

    for x in eff:
        x["pareto_non_dominated"] = not dominated(x, eff)
    if eff:
        with open(os.path.join(E, "phase6", "phase6_efficiency.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(eff[0].keys()))
            w.writeheader()
            w.writerows(eff)

    # ---------------------------------------------------------------- F tests
    res, pending = [], []

    def need(keys, label):
        miss = [k for k in keys if not cells.get(k)]
        if miss:
            pending.append((label, miss))
            return False
        return True

    def add(tid, qty, obs, ref, thr, ok, basis):
        res.append(dict(test_id=tid, quantity=qty, observed=fmt(obs), reference=fmt(ref),
                        threshold=fmt(thr), verdict=("PASS" if ok is True else
                                                     "FAIL" if ok is False else "PENDING"),
                        basis=basis))

    if need(["R2thin_s0"], "F1"):
        o = cells["R2thin_s0"]["lane_mIoU"]
        add("F1", "lane_mIoU(R2 thin 1/4)", o, REF["full14_km_lane_mean"], TH["F1"],
            o >= TH["F1"], "thin keeps full's lane gain?")
    else:
        add("F1", "lane_mIoU(R2 thin 1/4)", None, REF["full14_km_lane_mean"], TH["F1"], None, "cell missing")

    if need(["R2thin_s0"], "F2"):
        o = cells["R2thin_s0"]["lane_mIoU"]
        add("F2", "lane_mIoU(R2 thin 1/4)", o, REF["lean_km_lane_mean"], TH["F2"],
            o >= TH["F2"], "repair real vs 1/8?")
    else:
        add("F2", "lane_mIoU(R2 thin 1/4)", None, REF["lean_km_lane_mean"], TH["F2"], None, "cell missing")

    if need(["base_s0", "base_s1", "base_s2", "det_s0", "det_s1", "det_s2"], "F3"):
        d = M(["det_s0", "det_s1", "det_s2"], "mAP50")[0] - M(["base_s0", "base_s1", "base_s2"], "mAP50")[0]
        add("F3", "d mAP50 (k-means - old), lean", d, 0.0, TH["F3"], d >= TH["F3"],
            "supervision integration magnitude, 3 seeds")
    else:
        add("F3", "d mAP50 (k-means - old), lean", None, 0.0, TH["F3"], None, "cells missing")

    if need(["R3min_s0"], "F4"):
        o = cells["R3min_s0"]["lane_mIoU"]
        add("F4", "lane_mIoU(R3 min 1/4)", o, REF["lean_km_lane_mean"], TH["F2"],
            o >= TH["F2"], "minimum-cost rung also works?")
    else:
        add("F4", "lane_mIoU(R3 min 1/4)", None, REF["lean_km_lane_mean"], TH["F2"], None, "cell missing")

    if need(["R1up_s0"], "F5"):
        o = cells["R1up_s0"]["lane_mIoU"]
        add("F5", "lane_mIoU(R1 up-only 1/4)", o, REF["full14_km_lane_mean"], TH["F1"],
            o >= TH["F1"], "PASS => the s1 lateral is NOT necessary and must be removed")
    else:
        add("F5", "lane_mIoU(R1 up-only 1/4)", None, REF["full14_km_lane_mean"], TH["F1"], None, "cell missing")

    if need(["R2thin_s0"], "F6"):
        c = cells["R2thin_s0"]
        ok = (c["lane_mIoU"] >= TH["F6_lane"] and c["mAP50"] >= TH["F6_map"]
              and cells["both_s0"] and abs(c["da_mIoU"] - cells["both_s0"]["da_mIoU"]) <= TH["F6_da"])
        add("F6", "B vs S: lane>=thr & mAP>=thr & |dDA|<=thr", c["lane_mIoU"],
            REF["full14_km_lane_mean"], TH["F6_lane"], ok,
            f"mAP50={fmt(c['mAP50'])}>={fmt(TH['F6_map'])}; |dDA| vs S checked when both_s0 present")
    else:
        add("F6", "B vs S equivalence at -28.9% FLOPs", None, None, TH["F6_lane"], None, "cells missing")

    if need(["R2thin_s0", "det_s0"], "F7"):
        d = abs(cells["R2thin_s0"]["mAP50"] - cells["det_s0"]["mAP50"])
        add("F7", "|mAP50(B) - mAP50(lean+km)|", d, 0.0, TH["F7"], d <= TH["F7"],
            "lane branch must not perturb detection")
    else:
        add("F7", "|mAP50(B) - mAP50(lean+km)|", None, 0.0, TH["F7"], None, "cells missing")

    if need(["Umin_s0", "U_s0"], "F8"):
        um, u = cells["Umin_s0"], cells["U_s0"]
        ok = (CELL_META["Umin_s0"][3] <= CELL_META["U_s0"][3]
              and um["lane_mIoU"] >= u["lane_mIoU"] + 2 * SIG["lane_mIoU"]
              and um["mAP50"] >= u["mAP50"] - TH["F7"])
        add("F8", "U- dominates U (flops<=, lane+, mAP-<=2sig)", um["lane_mIoU"], u["lane_mIoU"],
            u["lane_mIoU"] + 2 * SIG["lane_mIoU"], ok,
            f"flops {CELL_META['Umin_s0'][3]} <= {CELL_META['U_s0'][3]}; mAP "
            f"{fmt(um['mAP50'])} >= {fmt(u['mAP50'] - TH['F7'])}")
    else:
        add("F8", "U- dominates U", None, None, None, None, "cells missing")

    # ---- supplementary 3-seed rows (NOT part of the frozen ladder) ------------
    # Model B's seeds 1/2 were purchased in Round 4B with the stated purpose of
    # meeting the Level-3 3-seed requirement (prereg 13.5). F1/F2/F6/F7 are frozen
    # as seed-0 tests, so they are left untouched above; these 'b*' rows show the
    # same quantities on the 3-seed mean so a reader can see the verdict does not
    # hinge on the single seed. The noise scale is NOT re-estimated.
    _b3 = ["R2thin_s0", "R2thin_s1", "R2thin_s2"]
    if all(cells.get(k) for k in _b3 + ["both_s0", "both_s1", "both_s2", "det_s0"]):
        _s3 = ["both_s0", "both_s1", "both_s2"]
        _d3 = ["det_s0", "det_s1", "det_s2"]
        bl = M(_b3, "lane_mIoU")[0]; bm = M(_b3, "mAP50")[0]; bd = M(_b3, "da_mIoU")[0]
        sl = M(_s3, "lane_mIoU")[0]; sd = M(_s3, "da_mIoU")[0]; dm = M(_d3, "mAP50")[0]
        add("F1b*", "lane_mIoU(B) 3-seed mean", bl, REF["full14_km_lane_mean"], TH["F1"],
            bl >= TH["F1"], "supplementary: B n=3 (seeds 0,1,2)")
        add("F2b*", "lane_mIoU(B) 3-seed mean", bl, REF["lean_km_lane_mean"], TH["F2"],
            bl >= TH["F2"], "supplementary: B n=3 (seeds 0,1,2)")
        add("F6b*", "B vs S: lane & mAP & |dDA|, 3-seed means", bl, sl, TH["F6_lane"],
            (bl >= TH["F6_lane"] and bm >= TH["F6_map"] and abs(bd - sd) <= TH["F6_da"]),
            f"mAP50={fmt(bm)}>={fmt(TH['F6_map'])}; |dDA|={fmt(abs(bd - sd))}<={fmt(TH['F6_da'])}")
        add("F7b*", "|mAP50(B) - mAP50(lean+km)|, 3-seed means", abs(bm - dm), 0.0, TH["F7"],
            abs(bm - dm) <= TH["F7"], "supplementary: B n=3 vs lean+km n=3")

    with open(os.path.join(E, "phase6", "phase6_final_statistics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    print("\n=== F tests ===")
    for r in res:
        print(f"  {r['test_id']:3s} {r['verdict']:7s} obs={r['observed']:>9s} thr={r['threshold']:>9s}  {r['basis']}")

    # ---------------------------------------------------------------- level
    def v(t):
        return {r["test_id"]: r["verdict"] for r in res}[t]

    allpass = all(v(t) == "PASS" for t in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"])
    if pending:
        print("\n[level] PENDING -- missing cells:",
              "; ".join(f"{l}:{','.join(m)}" for l, m in pending))
        print("[level] Level block withheld (fail-closed).")
    else:
        f1, f2, f3, f4 = v("F1"), v("F2"), v("F3"), v("F4")
        f6, f8 = v("F6"), v("F8")
        if f1 == "FAIL" and f2 == "FAIL":
            lvl = "Level 1 (only modules; no principle)" if f3 == "PASS" else "Level 0/1 (inconclusive)"
        elif f3 == "PASS" and (f1 == "FAIL" or f2 == "FAIL"):
            lvl = "Level 2 (task-specific bottleneck found; not a model claim)"
        elif f1 == "PASS" and f2 == "PASS" and f4 == "PASS":
            lvl = "Level 4 (principle + cross-architecture + equal-budget efficiency)" \
                if (f6 == "PASS" or f8 == "PASS") else \
                "Level 3 (task-conditioned allocation supported; equal-budget efficiency not yet shown)"
        else:
            lvl = "UNRESOLVED"
        print(f"\n[level] {lvl}")
        print(f"[stop] {'STOP searching new modules; write the final docs.' if lvl.startswith('Level 3') or lvl.startswith('Level 4') else 'do not escalate without new evidence.'}")

    print(f"\n[report] wrote phase6_final_models.csv ({len(rows)} rows), "
          f"phase6_ablation.csv ({len(abl)}), phase6_efficiency.csv ({len(eff)}), "
          f"phase6_final_statistics.csv ({len(res)})")
    return 0 if (not missing or a.allow_partial) else 1


if __name__ == "__main__":
    sys.exit(main())
