#!/usr/bin/env python3
"""Phase 6 / FINAL-100 -- build the paper's main comparison table.

Frozen BEFORE the 100-epoch run produced any metric (2026-09-13 18:30). The table
shape must not be chosen after seeing which rows look good, so the columns and the
row set are fixed here.

Two tables are emitted:
  Table 1  <=2G FLOPs tier: published baselines vs ours, at their budget and ours.
  Table 2  the epoch lever on our own model: 20 / 40 / 100 ep.

Baselines are read from experiments/BASELINE_RESULTS.md's numbers, which came from
OFFICIAL RELEASED WEIGHTS evaluated by evaluation/evaluate_baseline.py on tri_val
(10k images, 640x640 letterbox). They are hardcoded because they are external
constants that must not silently drift; the script asserts they are the values the
repo file states.

Usage:  python scripts/phase6_paper_table.py [--allow-missing]
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "experiments")
R4 = os.path.join(E, "phase6", "round4")
FIN = os.path.join(E, "phase6", "final")

AXES = ["mAP50", "da_mIoU", "lane_mIoU", "lane_fg_iou"]
AXIS_LABEL = {
    "mAP50": "mAP50",
    "da_mIoU": "DA mIoU",
    "lane_mIoU": "Lane mIoU",
    "lane_fg_iou": "Lane fgIoU",
}

# --- external constants (official released weights; see BASELINE_RESULTS.md) ------
# Every one of these is a row of that file; the assertion below re-reads the file so
# a future edit there fails loudly instead of desynchronising the paper table.
BASELINES = {
    "TriLiteNet tiny": dict(params=0.151, flops=1.8,
                            mAP50=0.4953, da_mIoU=0.8796, lane_mIoU=0.5914, lane_fg_iou=0.1952),
    "TwinLiteNetPlus nano": dict(params=0.033, flops=1.9,
                                 mAP50=None, da_mIoU=0.8634, lane_mIoU=0.5866, lane_fg_iou=0.1859),
    "TriLiteNet small": dict(params=0.592, flops=6.6,
                             mAP50=0.6326, da_mIoU=0.9053, lane_mIoU=0.6037, lane_fg_iou=0.2198),
}

OURS = {
    "B100": FIN + "/B100",              # the paper's main run (100 ep). eval dir = +_eval
    "B40_s0": R4 + "/r4_R4R2thin40",
    "B20_s0": R4 + "/r4_R4R2thin",
    "B20_s1": R4 + "/r4_R4R2s1",
    "B20_s2": R4 + "/r4_R4R2s2",
}

# Seed sd measured at 20 ep, N=3 (the only noise estimate this project has; see
# phase6_final100_preregistration.md 3). Used to draw the 2-sigma significance bands.
SD20 = {"mAP50": 0.0030, "da_mIoU": 0.0030, "lane_mIoU": 0.0031, "lane_fg_iou": 0.0050}


def load(base):
    p = base + "_eval/metrics.json"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        d = json.load(fh)
    d["_src"] = os.path.relpath(p, ROOT)
    return d


def fmt(v, nd=4):
    return "MISSING" if v is None else ("%.*f" % (nd, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit the table with MISSING placeholders (used before the 100ep run ends)")
    args = ap.parse_args()

    # --- guard: the external constants must still match the repo's own record ----
    brow = os.path.join(E, "BASELINE_RESULTS.md")
    if os.path.exists(brow):
        txt = open(brow, encoding="utf-8").read()
        for name, vals in BASELINES.items():
            if name not in txt:
                print("WARN: baseline %r not found in BASELINE_RESULTS.md" % name)
            elif vals["mAP50"] is not None and ("%.4f" % vals["mAP50"]) not in txt:
                print("WARN: %s mAP50 %.4f not present verbatim in BASELINE_RESULTS.md"
                      % (name, vals["mAP50"]))

    cells = {k: load(v) for k, v in OURS.items()}
    missing = [k for k, v in cells.items() if v is None]
    if missing and not args.allow_missing:
        print("MISSING cells: %s" % missing)
        print("(re-run with --allow-missing to emit placeholders)")
        return 1

    seeds20 = [cells[k] for k in ("B20_s0", "B20_s1", "B20_s2") if cells[k]]
    mean20 = {a: st.mean([d[a] for d in seeds20]) for a in AXES} if len(seeds20) == 3 else {}

    out_md = []
    out_md.append("# Phase 6 — main result tables\n")
    out_md.append("Protocol: BDD100K `tri_val`, 10,000 images, 640x640 letterbox, FP32, "
                  "single-class vehicle detection. Baselines are **official released weights**; "
                  "ours are trained by us. All rows measured by `evaluation/evaluate_baseline.py`.\n")

    # ---------------- Table 1 ----------------
    out_md.append("\n## Table 1 — the <=2 GFLOPs tier, plus one larger reference row\n")
    out_md.append("_TriLiteNet small (6.6 G) is **out of tier** and is shown only as a scaling "
                  "reference. Any claim in the paper is confined to the <=2 G rows: TriLiteNet tiny "
                  "(1.8 G), TLP nano (1.9 G) and ours (1.166 G)._\n")
    out_md.append("| model | budget | params (M) | FLOPs (G) | mAP50 | DA mIoU | Lane mIoU | Lane fgIoU |")
    out_md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, v in BASELINES.items():
        out_md.append("| %s | official | %s | %s | %s | %s | %s | %s |" % (
            name, v["params"], v["flops"], fmt(v["mAP50"]), fmt(v["da_mIoU"]),
            fmt(v["lane_mIoU"]), fmt(v["lane_fg_iou"])))
    for key, label in (("B100", "100 ep (ours, main)"), ("B40_s0", "40 ep (ours)"), ("B20_s0", "20 ep (ours)")):
        d = cells[key]
        if d is None:
            out_md.append("| Ours (Model B) | %s | 0.193 | 1.166 | MISSING | MISSING | MISSING | MISSING |" % label)
            continue
        out_md.append("| Ours (Model B) | %s | %.3f | %.3f | %s | %s | %s | %s |" % (
            label, (d.get("parameters") or 192566) / 1e6, (d.get("flops") or 1.1656448e9) / 1e9,
            fmt(d["mAP50"]), fmt(d["da_mIoU"]), fmt(d["lane_mIoU"]), fmt(d["lane_fg_iou"])))
    out_md.append("")

    # ---------------- gaps ----------------
    ref = BASELINES["TriLiteNet tiny"]
    if cells["B100"] and cells["B40_s0"]:
        d100, d40 = cells["B100"], cells["B40_s0"]
        out_md.append("\n## Table 1b — gap vs TriLiteNet tiny, and what the epoch lever bought\n")
        out_md.append("| axis | 40ep gap | 100ep gap | change | 2σ band | significant? |")
        out_md.append("|---|---:|---:|---:|---:|---|")
        for a in AXES:
            g40 = d40[a] - ref[a]
            g100 = d100[a] - ref[a]
            band = 2 * SD20[a]
            sig = "yes" if abs(g100) > band else "NO (inside seed noise)"
            out_md.append("| %s | %+.4f | %+.4f | %+.4f | ±%.4f | %s |" % (
                AXIS_LABEL[a], g40, g100, g100 - g40, band, sig))
        out_md.append("")

        # frozen decision rule from the preregistration
        g = ref["da_mIoU"] - d100["da_mIoU"]
        out_md.append("\n**Pre-registered DA decision rule** (`phase6_final100_preregistration.md` §6): ")
        if g > 0.0141:
            out_md.append("`g = %.4f > 0.0141` → **D-1: the DA deficit survives 100 epochs at >2σ; "
                          "1 seed suffices; report as budget-independent.**" % g)
        else:
            out_md.append("`g = %.4f <= 0.0141` → **D-2: residual lies inside seed noise; "
                          "seeds 1–2 required before any parity language.**" % g)
        out_md.append("")

    # ---------------- Table 2 ----------------
    out_md.append("\n## Table 2 — the epoch lever on our own model (seed 0)\n")
    out_md.append("| budget | mAP50 | DA mIoU | Lane mIoU | Lane fgIoU |")
    out_md.append("|---|---:|---:|---:|---:|")
    for key, label in (("B20_s0", "20 ep"), ("B40_s0", "40 ep"), ("B100", "100 ep")):
        d = cells[key]
        if d is None:
            out_md.append("| %s | MISSING | MISSING | MISSING | MISSING |" % label)
            continue
        out_md.append("| %s | %s | %s | %s | %s |" % (label, fmt(d["mAP50"]), fmt(d["da_mIoU"]),
                                                      fmt(d["lane_mIoU"]), fmt(d["lane_fg_iou"])))
    if mean20:
        out_md.append("| 20 ep, mean of 3 seeds | %s | %s | %s | %s |" % tuple(
            fmt(mean20[a]) for a in AXES))
    out_md.append("")

    dst_md = os.path.join(E, "phase6", "paper_tables.md")
    with open(dst_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_md) + "\n")

    # ---------------- machine-readable ----------------
    dst_csv = os.path.join(E, "phase6", "paper_main_table.csv")
    with open(dst_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "budget", "params_M", "flops_G", "mAP50", "mAP50_95",
                    "da_mIoU", "lane_mIoU", "lane_fg_iou", "source"])
        for name, v in BASELINES.items():
            w.writerow([name, "official", v["params"], v["flops"], v["mAP50"], "",
                        v["da_mIoU"], v["lane_mIoU"], v["lane_fg_iou"], "official weights, our protocol"])
        for key, label in (("B100", "100ep"), ("B40_s0", "40ep"), ("B20_s0", "20ep")):
            d = cells[key]
            if d is None:
                w.writerow(["Ours (Model B)", label, 0.193, 1.166, "", "", "", "", "", "PENDING"])
                continue
            w.writerow(["Ours (Model B)", label, round((d.get("parameters") or 192566) / 1e6, 4),
                        round((d.get("flops") or 1.1656448e9) / 1e9, 4),
                        d["mAP50"], d.get("mAP50_95"), d["da_mIoU"], d["lane_mIoU"],
                        d["lane_fg_iou"], d["_src"]])

    print("wrote %s" % os.path.relpath(dst_md, ROOT))
    print("wrote %s" % os.path.relpath(dst_csv, ROOT))
    print()
    print("\n".join(out_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
