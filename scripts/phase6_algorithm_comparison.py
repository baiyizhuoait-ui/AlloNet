#!/usr/bin/env python3
"""Phase 6 -- cross-algorithm positioning table.

Sources (all measured by evaluation/evaluate_baseline.py on the same protocol):
  * external rows : experiments/BASELINE_RESULTS.md   (2026-08-29, full tri_val 10k)
  * ours          : experiments/phase6/final_metrics.csv   (FINAL-100, seed 0)

Latency:
  * "fixed_ms"    : D8 probe FIXED protocol (warmup 200 / reps 300 / CUDA events), 2026-09-14
  * "old_ms"      : as-shipped instrument -- KNOWN BIASED (warmup 10, reps 3 ours / 100 baselines).
                    Reproduced as 4.3x too high on ours in the D8 null replicate. Kept only to
                    document the retraction; do NOT use for any claim.
"""
import csv
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments", "phase6")

# name, family, params_M, flops_G, mAP50, da_mIoU, da_fg_iou, lane_mIoU, lane_fg_iou
ROWS = [
    ("YOLOP",                  "external", 7.940, 31.3, 0.7657, 0.9115, 0.8556, 0.6038, 0.2247),
    ("TwinLiteNet",            "external", 0.440, 14.1, None,   0.9114, 0.8551, 0.6077, 0.2281),
    ("TwinLiteNetPlus nano",   "external", 0.033, 1.9,  None,   0.8634, 0.7776, 0.5866, 0.1859),
    ("TwinLiteNetPlus small",  "external", 0.122, 4.7,  None,   0.8986, 0.8336, 0.6019, 0.2165),
    ("TwinLiteNetPlus medium", "external", 0.479, 15.4, None,   0.9191, 0.8672, 0.6087, 0.2297),
    ("TwinLiteNetPlus large",  "external", 1.944, 58.6, None,   0.9279, 0.8815, 0.6161, 0.2450),
    ("TriLiteNet tiny",        "external", 0.151, 1.8,  0.4953, 0.8796, 0.8045, 0.5914, 0.1952),
    ("TriLiteNet small",       "external", 0.592, 6.6,  0.6326, 0.9053, 0.8458, 0.6037, 0.2198),
    ("TriLiteNet base",        "external", 2.350, 25.4, 0.7235, 0.9203, 0.8697, 0.6123, 0.2371),
    ("Ours (Model B)",         "ours",     0.1926, 1.1656, 0.5382, 0.8673, 0.7888, 0.5978, 0.2187),
]

FIXED_MS = {"Ours (Model B)": 2.3964, "TriLiteNet tiny": 3.5096,
            "TriLiteNet small": 5.3836, "TwinLiteNetPlus nano": 5.0314}
OLD_MS = {"YOLOP": 14.6, "TwinLiteNet": 9.0, "TwinLiteNetPlus nano": 5.2,
          "TwinLiteNetPlus small": 6.1, "TwinLiteNetPlus medium": 9.3,
          "TwinLiteNetPlus large": 15.0, "TriLiteNet tiny": 4.4,
          "TriLiteNet small": 6.8, "TriLiteNet base": 9.4}

KEY = {"mAP50": 4, "da_mIoU": 5, "da_fg_iou": 6, "lane_mIoU": 7, "lane_fg_iou": 8}


def rank(metric):
    """rank 1 = best, only over rows that have the metric"""
    vals = [(r[0], r[KEY[metric]]) for r in ROWS if r[KEY[metric]] is not None]
    vals.sort(key=lambda t: -t[1])
    return {n: i + 1 for i, (n, _) in enumerate(vals)}, len(vals)


def pareto(metric, cost=3):
    """non-dominated set: cheaper AND >= metric everywhere. cost=flops index"""
    out = {}
    for r in ROWS:
        v = r[KEY[metric]]
        if v is None:
            continue
        dominated_by = [o[0] for o in ROWS
                        if o[KEY[metric]] is not None and o[0] != r[0]
                        and o[cost] <= r[cost] and o[KEY[metric]] >= v
                        and (o[cost] < r[cost] or o[KEY[metric]] > v)]
        out[r[0]] = dominated_by
    return out


ranks = {m: rank(m) for m in KEY}
dom_lane = pareto("lane_mIoU")
dom_da = pareto("da_mIoU")

os.makedirs(OUT, exist_ok=True)
csv_path = os.path.join(OUT, "phase6_algorithm_comparison.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "family", "params_M", "flops_G", "mAP50", "da_mIoU", "da_fg_iou",
                "lane_mIoU", "lane_fg_iou", "lane_mIoU_per_GFLOP", "da_mIoU_per_GFLOP",
                "fixed_p50_ms", "old_biased_p50_ms",
                "rank_mAP50", "rank_da_mIoU", "rank_lane_mIoU", "rank_lane_fg_iou"])
    for r in ROWS:
        n = r[0]
        w.writerow([n, r[1], r[2], r[3],
                    "" if r[4] is None else r[4], r[5], r[6], r[7], r[8],
                    round(r[7] / r[3], 4), round(r[5] / r[3], 4),
                    FIXED_MS.get(n, ""), OLD_MS.get(n, ""),
                    ranks["mAP50"][0].get(n, ""), ranks["da_mIoU"][0].get(n, ""),
                    ranks["lane_mIoU"][0].get(n, ""), ranks["lane_fg_iou"][0].get(n, "")])
print("wrote", csv_path)

lines = []
A = lines.append
A("# Phase 6 - cross-algorithm positioning (10 models, one harness)\n")
A("> **READ THE PROTOCOL COLUMN BEFORE QUOTING ANY NUMBER HERE.**\n"
  "> This table is internally comparable and externally un-comparable. Audit: "
  "`docs/PHASE6_BENCHMARK_AUDIT.md`.\n"
  "> - **FLOPs** = 2 x MACs (thop) at **640x640**. Published FLOPs are MACs at **384x640**. "
  "Ratio 3.27-3.36 on the 7 models whose params also match. **Never put our FLOPs next to a "
  "published FLOPs.**\n"
  "> - **Lane columns**: our value / published Lane IoU = 0.71-0.85 (spread 0.14, not a "
  "constant) => **not cross-paper comparable, and not repairable by a scale factor.**\n"
  "> - **DA column**: within ~1 point of published => roughly comparable.\n"
  "> - **Detection column**: reproduces published mAP50 to within 0.1 => comparable.\n"
  "> - **TwinLiteNet / TwinLiteNet+ have no detection head** (2-task, not 3-task).\n"
  "> - `accuracy per GFLOP` is a **secondary** indicator only - a tiny model can win it "
  "trivially. Prefer the Pareto frontier and budget-constrained accuracy.\n")
A("Protocol: BDD100K `tri_val`, 10,000 images, 640x640, FP32, single-class vehicle detection.\n"
  "External rows = official released weights; ours = trained by us (100 ep, seed 0).\n"
  "Every row measured by `evaluation/evaluate_baseline.py`.\n")
A("\n## Full table (sorted by cost)\n")
A("| model | params (M) | FLOPs (G) | mAP50 | DA mIoU | DA fgIoU | Lane mIoU | Lane fgIoU |")
A("|---|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(ROWS, key=lambda x: x[3]):
    A("| %s%s | %.3f | %.3f | %s | %.4f | %.4f | %.4f | %.4f |" % (
        r[0], " **(ours)**" if r[1] == "ours" else "", r[2], r[3],
        "-" if r[4] is None else "%.4f" % r[4], r[5], r[6], r[7], r[8]))

A("\n## Rank of ours among the 10 models\n")
A("| axis | ours | rank | #models | best |")
A("|---|---:|---:|---:|---:|")
for m, label in (("mAP50", "mAP50"), ("da_mIoU", "DA mIoU"),
                 ("lane_mIoU", "Lane mIoU"), ("lane_fg_iou", "Lane fgIoU")):
    rk, tot = ranks[m]
    if "Ours (Model B)" in rk:
        best = max(r[KEY[m]] for r in ROWS if r[KEY[m]] is not None)
        A("| %s | %.4f | **%d / %d** | %d | %.4f |" % (
            label, [r[KEY[m]] for r in ROWS if r[0] == "Ours (Model B)"][0],
            rk["Ours (Model B)"], tot, tot, best))
A("\nOur FLOPs (1.166 G) is the **lowest of all 10** - rank 1 / 10 on cost.\n")

A("\n## Accuracy per GFLOP (the efficiency axis)\n")
A("| model | FLOPs (G) | Lane mIoU / GFLOP | DA mIoU / GFLOP |")
A("|---|---:|---:|---:|")
for r in sorted(ROWS, key=lambda x: -(x[7] / x[3])):
    A("| %s%s | %.3f | **%.4f** | %.4f |" % (
        r[0], " **(ours)**" if r[1] == "ours" else "", r[3], r[7] / r[3], r[5] / r[3]))

A("\n## Pareto check\n")
A("On `(cost, Lane mIoU)`, ours is dominated by **nobody** and strictly dominates:")
for n, by in dom_lane.items():
    if by:
        A("- %s is strictly dominated by %s" % (n, ", ".join(by)))
A("\nOn `(cost, DA mIoU)`, ours is dominated by **nobody**, and dominates:")
for n, by in dom_da.items():
    if by:
        A("- %s is strictly dominated by %s" % (n, ", ".join(by)))

A("\n## Latency (batch 1, FP32) - and the retracted column\n")
A("| model | FIXED p50 (ms) | old as-shipped p50 (ms) | note |")
A("|---|---:|---:|---|")
for r in sorted(ROWS, key=lambda x: FIXED_MS.get(x[0], 99)):
    n = r[0]
    A("| %s%s | %s | %s | %s |" % (
        n, " **(ours)**" if r[1] == "ours" else "",
        "**%.3f**" % FIXED_MS[n] if n in FIXED_MS else "not measured",
        "%.1f" % OLD_MS[n] if n in OLD_MS else "-",
        "retracted: biased instrument" if n in OLD_MS else ""))
A("\nThe as-shipped instrument (warmup 10 / reps 3 for ours, reps 100 for baselines) returned "
  "10.2152 ms for a model whose true p50 is 2.3964 ms - a 4.3x inflation - and scattered five "
  "identical checkpoints over 1.73-4.51x. Only the FIXED column is usable; a full sweep of the "
  "remaining 5 baselines still needs to run.\n")

md_path = os.path.join(OUT, "phase6_algorithm_comparison.md")
open(md_path, "w").write("\n".join(lines) + "\n")
print("wrote", md_path)
print("\n".join(lines[:60]))
