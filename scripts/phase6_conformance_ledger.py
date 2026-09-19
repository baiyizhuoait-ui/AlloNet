#!/usr/bin/env python3
"""Phase 6 -- benchmark conformance ledger.

Answers one question per row: does OUR number for a model match the number its
authors published? A column where the answers are 'no, by an unknown constant' is
not comparable across papers, no matter how internally consistent it is.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments", "phase6")

# our harness (BASELINE_RESULTS.md 2026-08-29 + final_metrics.csv 2026-09-14)
MINE = {
    "TriLiteNet tiny":  dict(params=0.151, flops=1.80, map50=0.4953, da=0.8796, lane_fg=0.1952, lineacc=0.6790),
    "TriLiteNet small": dict(params=0.592, flops=6.60, map50=0.6326, da=0.9053, lane_fg=0.2198, lineacc=0.7064),
    "TriLiteNet base":  dict(params=2.350, flops=25.4, map50=0.7235, da=0.9203, lane_fg=0.2371, lineacc=0.7301),
    "TLP nano":         dict(params=0.033, flops=1.90, map50=None,   da=0.8634, lane_fg=0.1859, lineacc=0.6710),
    "TLP small":        dict(params=0.122, flops=4.70, map50=None,   da=0.8986, lane_fg=0.2165, lineacc=0.7062),
    "TLP medium":       dict(params=0.479, flops=15.4, map50=None,   da=0.9191, lane_fg=0.2297, lineacc=0.7175),
    "TLP large":        dict(params=1.944, flops=58.6, map50=None,   da=0.9279, lane_fg=0.2450, lineacc=0.7448),
    "TwinLiteNet":      dict(params=0.440, flops=14.1, map50=None,   da=0.9114, lane_fg=0.2281, lineacc=0.7203),
    "YOLOP":            dict(params=7.940, flops=31.3, map50=0.7657, da=0.9115, lane_fg=0.2247, lineacc=0.7896),
}

# official. src "repo" = upstream README vendored in baselines/ (verifiable in-tree);
#              src "cited" = quoted in review, NOT verified in this repo.
OFFICIAL = {
    "TriLiteNet tiny":  dict(params=0.15, flops=0.55, map50=49.6, da=None, lane_fg=24.2, lineacc=76.5, src="cited"),
    "TriLiteNet small": dict(params=0.59, flops=1.99, map50=63.2, da=None, lane_fg=27.6, lineacc=81.6, src="cited"),
    "TriLiteNet base":  dict(params=2.35, flops=7.72, map50=72.3, da=None, lane_fg=29.8, lineacc=85.6, src="cited"),
    "TLP nano":         dict(params=0.03, flops=0.57, map50=None, da=87.3, lane_fg=23.3, lineacc=70.2, src="repo"),
    "TLP small":        dict(params=0.12, flops=1.40, map50=None, da=90.6, lane_fg=29.3, lineacc=75.8, src="repo"),
    "TLP medium":       dict(params=0.48, flops=4.63, map50=None, da=92.0, lane_fg=32.3, lineacc=79.1, src="repo"),
    "TLP large":        dict(params=1.94, flops=17.58, map50=None, da=92.9, lane_fg=34.2, lineacc=81.9, src="repo"),
    "TwinLiteNet":      dict(params=0.44, flops=3.90, map50=None, da=91.3, lane_fg=31.1, lineacc=77.8, src="repo"),
    "YOLOP":            dict(params=5.53, flops=8.11, map50=76.5, da=91.6, lane_fg=26.5, lineacc=None, src="repo"),
}

L = []
A = L.append
A("### Conformance ledger: our harness vs the number the authors published\n")
A("| model | src | params mine/off | FLOPs mine/off (ratio) | mAP50 mine/off | DA mine/off | LaneIoU mine/off (ratio) | LaneAcc mine/off |")
A("|---|---|---|---|---|---|---|---|")
for k in MINE:
    m, o = MINE[k], OFFICIAL[k]
    f = m["flops"] / o["flops"]
    lr = m["lane_fg"] / (o["lane_fg"] / 100.0) if o["lane_fg"] else None
    A("| %s | %s | %.3f / %.2f | %.2f / %.2f (**%.2fx**) | %s / %s | %s / %s | %.4f / %.2f (**%.2f**) | %s / %s |" % (
        k, o["src"], m["params"], o["params"], m["flops"], o["flops"], f,
        "-" if m["map50"] is None else "%.4f" % m["map50"],
        "-" if o["map50"] is None else "%.1f" % o["map50"],
        "%.4f" % m["da"], "-" if o["da"] is None else "%.1f" % o["da"],
        m["lane_fg"], o["lane_fg"], lr,
        "%.4f" % m["lineacc"], "-" if o["lineacc"] is None else "%.1f" % o["lineacc"]))

fs = [MINE[k]["flops"] / OFFICIAL[k]["flops"] for k in MINE]
ls = [MINE[k]["lane_fg"] / (OFFICIAL[k]["lane_fg"] / 100.0) for k in MINE]
A("\n**FLOPs ratio:** min %.2f, max %.2f, mean %.2f. Expected from the protocol"
  " difference alone: 2.0 (MACs -> FLOPs) x 1.6667 (384x640 -> 640x640) = **3.333**.\n" % (
      min(fs), max(fs), sum(fs) / len(fs)))
A("**LaneIoU ratio:** min %.2f, max %.2f, mean %.2f. Spread %.2f - no single constant"
  " explains it, so this column cannot be repaired by a scale factor.\n" % (
      min(ls), max(ls), sum(ls) / len(ls), max(ls) - min(ls)))

A("\n### Published reference table already vendored in-tree (`baselines/TwinLiteNetPlus/README.md`)\n")
A("| Model | DA mIoU (%) | Lane Acc (%) | Lane IoU (%) | FLOPs | #Params |")
A("|---|---|---|---|---|---|")
for r in [
    ("DeepLabV3+", 90.9, "--", 29.8, "30.7G", "15.4M"),
    ("SegForme", 92.3, "--", 31.7, "12.1G", "7.2M"),
    ("R-CNNP", 90.2, "--", 24.0, "--", "--"),
    ("YOLOP", 91.6, "--", 26.5, "8.11G", "5.53M"),
    ("IALaneNet (ResNet-18)", 90.54, "--", 30.39, "89.83G", "17.05M"),
    ("IALaneNet (ResNet-34)", 90.61, "--", 30.46, "139.46G", "27.16M"),
    ("IALaneNet (ConvNeXt-tiny)", 91.29, "--", 31.48, "96.52G", "18.35M"),
    ("IALaneNet (ConvNeXt-small)", 91.72, "--", 32.53, "200.07G", "39.97M"),
    ("YOLOv8 (multi)", 84.2, 81.7, 24.3, "--", "--"),
    ("Sparse U-PDP", 91.5, "--", 31.2, "--", "--"),
    ("TwinLiteNet", 91.3, 77.8, 31.1, "3.9G", "0.44M"),
    ("TwinLiteNet+ Nano", 87.3, 70.2, 23.3, "0.57G", "0.03M"),
    ("TwinLiteNet+ Small", 90.6, 75.8, 29.3, "1.40G", "0.12M"),
    ("TwinLiteNet+ Medium", 92.0, 79.1, 32.3, "4.63G", "0.48M"),
    ("TwinLiteNet+ Large", 92.9, 81.9, 34.2, "17.58G", "1.94M"),
]:
    A("| %s | %s | %s | %s | %s | %s |" % r)

os.makedirs(OUT, exist_ok=True)
p = os.path.join(OUT, "phase6_conformance_ledger.md")
open(p, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("\nwrote", p)
