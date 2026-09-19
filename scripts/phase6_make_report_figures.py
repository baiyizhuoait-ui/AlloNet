#!/usr/bin/env python3
"""Phase 6 -- build the full figure set for the technical report.

Every number here is read from a committed artifact table; nothing is invented.
Sources are named in each figure's docstring and in the report next to the image.

Output: experiments/phase6/figures/*.png  (200 dpi, English labels -- the report
prose is Chinese, but the plots stay ASCII so they never depend on a CJK font
being installed in the matplotlib environment).

Run:  gpu_env/bin/python scripts/phase6_make_report_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "experiments", "phase6", "figures")
os.makedirs(FIG, exist_ok=True)

OURS = "#c0392b"       # our model -- red, stands out in every plot
BASE = "#2c3e50"       # published baselines
ACC1 = "#2980b9"       # accent 1
ACC2 = "#16a085"       # accent 2
ACC3 = "#e67e22"       # accent 3
GREY = "#95a5a6"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.autolayout": False,
})

_saved = []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _saved.append(name)
    print(f"  [fig] {name}")


# --------------------------------------------------------------------------
# 0. research roadmap
# --------------------------------------------------------------------------
def fig01_roadmap():
    phases = [
        ("P0-1", "Scaffold &\nbaseline repro", "harness matches\npublished numbers"),
        ("P2A-C", "Where is the\nbottleneck?", "per-task profile\nof every resource"),
        ("P3A", "Encoder\ncapacity sweep", "E-small/base/large\nmonotone 6/6"),
        ("P3B", "Encoder x Z\ninteraction", "detection most\nsensitive axis"),
        ("P3C", "Fixed-budget\nallocation", "encoder-heavy >\nZ-heavy"),
        ("P4-5", "Architecture\n& lanes", "R0 vs R2;\n1/8 lane ceiling"),
        ("P6", "Allocation rule\n& field position", "marginal-price rule;\ncertified field table"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_axis_off()
    n = len(phases)
    w, gap = 1.52, 0.42
    for i, (tag, title, out) in enumerate(phases):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.42), w, 0.86, boxstyle="round,pad=0.05",
                                    linewidth=1.3, edgecolor=OURS if i == n - 1 else ACC1,
                                    facecolor="#fdf2f0" if i == n - 1 else "#eef4fa"))
        ax.text(x + w / 2, 1.15, tag, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=OURS if i == n - 1 else ACC1)
        ax.text(x + w / 2, 0.85, title, ha="center", va="center", fontsize=8.6)
        ax.text(x + w / 2, 0.22, out, ha="center", va="top", fontsize=7.6, color="#444",
                style="italic")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, 0.85), (x + w + gap - 0.03, 0.85),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.2, color=GREY))
    ax.set_xlim(-0.3, n * (w + gap))
    ax.set_ylim(-0.25, 1.62)
    ax.set_title("Phase 0 -> 6: what each stage answered", fontsize=11.5, fontweight="bold",
                 pad=8)
    save(fig, "fig01_roadmap.png")


# --------------------------------------------------------------------------
# 1. budget curve
# --------------------------------------------------------------------------
BUDGET = {
    20: dict(mAP=0.4978, mAP_sd=0.0030, DA=0.8562, DA_sd=0.0030, lane=0.5959, lane_sd=0.0031, n=3),
    40: dict(mAP=0.5198, mAP_sd=None, DA=0.8594, DA_sd=None, lane=0.5992, lane_sd=None, n=1),
    100: dict(mAP=0.5382, mAP_sd=None, DA=0.8673, DA_sd=None, lane=0.5978, lane_sd=None, n=1),
}
TINY_DA = 0.8796
FROZEN_2SIG = 0.0023
EMPIRICAL_2SD = 0.0061


def fig02_budget_curve():
    eps = sorted(BUDGET)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    specs = [("mAP", "detection mAP50", ACC1), ("DA", "drivable-area mIoU", ACC2),
             ("lane", "lane mIoU", ACC3)]
    for ax, (key, label, col) in zip(axes, specs):
        m = [BUDGET[e][key] for e in eps]
        sd = [BUDGET[e][f"{key}_sd"] for e in eps]
        ax.errorbar(eps, m, yerr=[s if s else 0 for s in sd], marker="o", ms=6.5,
                    lw=2, color=col, capsize=4, capthick=1.2, zorder=3)
        for e, v, s in zip(eps, m, sd):
            ax.annotate(f"{v:.4f}" + ("" if s else "  (n=1)"), (e, v),
                        textcoords="offset points", xytext=(0, 12), ha="center",
                        va="bottom", fontsize=7.4)
        ax.set_xticks(eps)
        ax.set_xlim(11, 122)
        ax.set_xlabel("training budget (epochs)")
        ax.set_ylabel(label)
        ax.set_title(label)
        if key == "DA":
            ax.set_ylim(0.848, 0.888)
            ax.axhline(TINY_DA, ls="--", lw=1.4, color=BASE)
            ax.annotate(f"TriLiteNet tiny, same harness: {TINY_DA}", (13, TINY_DA),
                        textcoords="offset points", xytext=(2, -13), ha="left",
                        va="top", fontsize=7.4, color=BASE)
    axes[0].set_ylim(0.488, 0.549)
    axes[2].set_ylim(0.588, 0.603)
    fig.suptitle("Fig. 1  The only variable is the epoch budget: same config, same 0.1926 M params, "
                 "same 1.1656 GFLOPs", fontsize=11.5, fontweight="bold", y=1.06)
    save(fig, "fig02_budget_curve.png")


def fig03_gap_closure():
    eps = sorted(BUDGET)
    gap = [BUDGET[e]["DA"] - TINY_DA for e in eps]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhspan(-FROZEN_2SIG, 0, color=ACC2, alpha=0.13,
               label=f"frozen 2 sigma = {FROZEN_2SIG} (D-2 guard band)")
    ax.axhspan(-EMPIRICAL_2SD, 0, color=ACC3, alpha=0.10,
               label=f"family-measured 2 sd = {EMPIRICAL_2SD}")
    ax.plot(eps, gap, marker="o", ms=8, lw=2.2, color=OURS, zorder=4)
    for e, g in zip(eps, gap):
        ax.annotate(f"{g:+.4f}", (e, g), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=OURS, fontweight="bold")
    for a, b in zip(eps, eps[1:]):
        d = BUDGET[b]["DA"] - BUDGET[a]["DA"]
        mid = (a + b) / 2
        stronger = "beyond 2sd" if abs(d) >= EMPIRICAL_2SD else "within noise (2sd)"
        ax.annotate("", (b, (BUDGET[b]["DA"] - TINY_DA)), (a, (BUDGET[a]["DA"] - TINY_DA)),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1, ls=":"))
        ax.text(mid, (BUDGET[a]["DA"] + BUDGET[b]["DA"]) / 2 - TINY_DA - 0.0016,
                f" +{d:.4f}\n {stronger}", ha="center", fontsize=7.3, color="#555")
    ax.set_xticks(eps)
    ax.set_xlabel("training budget (epochs)")
    ax.set_ylabel("DA mIoU gap vs TriLiteNet tiny")
    ax.set_title("Fig. 2  The deficit shrinks, but only the 40->100 step\n"
                 "survives the family's OWN seed sd", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7.4)
    save(fig, "fig03_gap_closure.png")


# --------------------------------------------------------------------------
# 2. capacity allocation
# --------------------------------------------------------------------------
# phase3A (z16) + phase3B (z32/z128) -- identical everything but encoder/z
CAP = {
    "mAP50": {
        "E-small": [0.2687, 0.2810, 0.2639],
        "E-base":  [0.3204, 0.3222, 0.3161],
        "E-large": [0.3585, 0.3494, 0.3436],
    },
    "da_mIoU": {
        "E-small": [0.8423, 0.8456, 0.8490],
        "E-base":  [0.8564, 0.8527, 0.8522],
        "E-large": [0.8589, 0.8546, 0.8575],
    },
    "lane_mIoU": {
        "E-small": [0.5818, 0.5841, 0.5855],
        "E-base":  [0.5867, 0.5875, 0.5877],
        "E-large": [0.5878, 0.5893, 0.5925],
    },
}
CAP_PARAMS = {
    "E-small": [0.1013, 0.1135, 0.1867],
    "E-base":  [0.1889, 0.2027, 0.2858],
    "E-large": [0.2915, 0.3068, 0.3984],
}
ZLAB = ["z=16", "z=32", "z=128"]


def fig04_capacity_heatmap():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for ax, (k, d) in zip(axes, CAP.items()):
        encs = list(d)
        M = np.array([d[e] for e in encs])
        im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(3), ZLAB)
        ax.set_yticks(range(3), encs)
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                norm = (v - M.min()) / (M.max() - M.min() + 1e-9)
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8.2,
                        color="white" if norm > 0.6 else "#111")
        ax.set_title(k)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Fig. 3  Encoder capacity x bottleneck width (Z), 20 ep, identical protocol\n"
                 "detection tracks ENCODER capacity; widening Z alone buys almost nothing",
                 fontsize=11.2, fontweight="bold", y=1.14)
    save(fig, "fig04_capacity_heatmap.png")


def fig05_marginal_price():
    """Marginal price of each resource dimension, measured on our own arm family."""
    levers = [
        # label, d_metric(primary), d_flops, metric name, zero_cost flag
        ("encoder capacity\nE-small -> E-large\n(detection)", 0.0898, 0.7061, "mAP50"),
        ("encoder capacity\nE-small -> E-large\n(lane)", 0.0060, 0.7061, "lane mIoU"),
        ("Z width\nz=16 -> z=128\n(detection)", -0.0043, 0.9634, "mAP50"),
        ("Z width\nz=16 -> z=128\n(lane)", 0.0010, 0.9634, "lane mIoU"),
        ("lane resolution\n1/8 -> 1/4 (detection)", 0.0000, 0.5595, "mAP50"),
        ("lane resolution\n1/8 -> 1/4 (lane)", 0.0124, 0.5595, "lane mIoU"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    labels = [l[0] for l in levers]
    vals = [l[1] / l[2] for l in levers]
    cols = [OURS if v > 0.02 else (ACC1 if v > 0 else GREY) for v in vals]
    y = np.arange(len(levers))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yy, v in zip(y, vals):
        if v < 0:
            ax.text(0.0015, yy, f"{v:+.4f}", va="center", ha="left", fontsize=8, color=GREY)
        else:
            ax.text(v + 0.0022, yy, f"{v:+.4f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y, labels, fontsize=7.8)
    ax.set_xlabel("metric delta per +1 GFLOPs spent")
    ax.axvline(0, color="#333", lw=1)
    # the two zero-inference-cost levers cannot be drawn as a ratio -- annotate instead
    ax.text(0.118, y[2] - 0.05,
            "not on this scale (zero inference cost):\n"
            "  anchor supervision  ->  mAP50 +0.1445 at 0 GFLOPs\n"
            "  lane width h32->h16 ->  lane parity at -0.4736 GFLOPs",
            fontsize=7.6, color="#444", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#ccc"))
    ax.set_title("Fig. 4  Marginal price of each resource dimension\n"
                 "encoder capacity dominates detection; 1/4 lane resolution is the "
                 "cheapest productive lane lever", fontweight="bold")
    ax.set_xlim(-0.006, 0.150)
    save(fig, "fig05_marginal_price.png")


def fig06_pareto_ours():
    """Within-harness Pareto over our own arms (params vs the two axes)."""
    arms = [
        # label, params(M), flops(G), mAP50, lane_mIoU, family
        ("R3min (min 1/4 h8)", 0.1896, 1.0174, 0.4912, 0.4959, "dead"),
        ("B (thin 1/4 h16)", 0.1926, 1.1656, 0.4978, 0.5959, "lean"),
        ("R1up (1/4 h32)", 0.2014, 1.6129, 0.4942, 0.5923, "lean"),
        ("lean+km (ref)", 0.2014, 1.0796, 0.4950, 0.5827, "lean"),
        ("S (full 1/4 h32)", 0.2019, 1.6391, 0.4981, 0.5951, "lean"),
        ("Uminus (1/4 h8)", 0.3224, 1.6158, 0.5326, 0.4959, "uniform"),
        ("U (uniform)", 0.3339, 1.6650, 0.5382, 0.5851, "uniform"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    fam_col = {"lean": ACC1, "uniform": ACC3, "dead": GREY}
    for ax, (idx, name) in zip(axes, [(3, "mAP50"), (4, "lane mIoU")]):
        for lab, pm, fl, mp, ln, fam in arms:
            v = mp if name == "mAP50" else ln
            dead = fam == "dead"
            ax.scatter(pm, v, s=90, color=OURS if lab.startswith("B (") else fam_col[fam],
                       marker="X" if dead else "o", zorder=4,
                       edgecolor="white", linewidth=1.1)
            ax.annotate(lab, (pm, v), textcoords="offset points", xytext=(6, 5),
                        fontsize=7.3, color="#333")
        ax.set_xlabel("parameters (M)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs parameters")
        ax.set_xlim(0.17, 0.36)
    axes[1].axhline(0.5913, ls="--", lw=1.2, color=ACC2)
    axes[1].annotate("lane target (full-1/4 level, 0.5913)", (0.176, 0.5913),
                     textcoords="offset points", xytext=(0, 5), fontsize=7.2, color=ACC2)
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in fam_col.items()]
    axes[0].legend(handles=h, fontsize=7.6, loc="lower right",
                   title="allocation family", title_fontsize=7.6)
    fig.suptitle("Fig. 5  Pareto view inside one harness: B reaches the lane target at "
                 "-28.9% FLOPs vs S, while uniform expansion buys detection only",
                 fontsize=11, fontweight="bold", y=1.04)
    save(fig, "fig06_pareto_ours.png")


def fig07_BvsSvsU():
    rows = [
        ("mAP50", 0.5382, 0.4981, 0.4978, 0.0404, "U wins +0.0404\nat 1.73x params"),
        ("DA mIoU", 0.8629, 0.8550, 0.8562, 0.0067, "U wins +0.0067"),
        ("lane mIoU", 0.5872, 0.5951, 0.5959, -0.0087, "B wins +0.0087\nover U"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 4.3))
    x = np.arange(len(rows))
    w = 0.26
    U = [r[1] for r in rows]
    S = [r[2] for r in rows]
    B = [r[3] for r in rows]
    ax.bar(x - w, [v - 0.84 for v in U], w, bottom=0.84, color=ACC3, label="U uniform 0.334 M / 1.665 G")
    ax.bar(x, [v - 0.84 for v in S], w, bottom=0.84, color=ACC1, label="S naive 1/4 0.202 M / 1.639 G")
    ax.bar(x + w, [v - 0.84 for v in B], w, bottom=0.84, color=OURS, label="B thin 1/4 h16 0.193 M / 1.166 G")
    for xi, (name, u, s, b, d, note) in zip(x, rows):
        for off, v in ((-w, u), (0, s), (w, b)):
            ax.annotate(f"{v:.4f}", (xi + off, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.2)
        ax.text(xi, 0.845, note, ha="center", fontsize=7.2, color=OURS, va="bottom")
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_ylabel("score (axis starts at 0.84)")
    ax.set_ylim(0.84, 0.965)
    ax.legend(fontsize=7.8, loc="upper center", ncol=3)
    ax.set_title("Fig. 6  Three allocation shapes at ~equal budget (3 seeds each)\n"
                 "B == S on every axis within noise, for 28.9% fewer FLOPs",
                 fontweight="bold")
    save(fig, "fig07_BvsSvsU.png")


# --------------------------------------------------------------------------
# 3. ablations
# --------------------------------------------------------------------------
def fig08_ablation_2x2():
    cells = {
        ("old anchors", "no lane repair"): (0.3505, 0.5839),
        ("k-means anchors", "no lane repair"): (0.4950, 0.5827),
        ("old anchors", "full 1/4"): (0.3588, 0.5980),
        ("k-means anchors", "full 1/4"): (0.4981, 0.5951),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    for ax, (metric, idx) in zip(axes, [("mAP50", 0), ("lane mIoU", 1)]):
        for lane_arm, style in [("no lane repair", dict(ls="--", marker="s")),
                                ("full 1/4", dict(ls="-", marker="o"))]:
            xs = [0, 1]
            ys = [cells[("old anchors", lane_arm)][idx],
                  cells[("k-means anchors", lane_arm)][idx]]
            ax.plot(xs, ys, color=ACC1 if lane_arm == "no lane repair" else OURS,
                    lw=2, ms=8, label=lane_arm, **style)
            for xx, yy in zip(xs, ys):
                ax.annotate(f"{yy:.4f}", (xx, yy), textcoords="offset points",
                            xytext=(0, 9 if lane_arm == "full 1/4" else -14),
                            ha="center", fontsize=7.4)
        ax.set_xticks([0, 1], ["old anchors", "k-means anchors"])
        ax.set_ylabel(metric)
        ax.set_title(f"interaction on {metric}: "
                     + ("-0.0052 mAP50 (additive)" if idx == 0 else "-0.0017 (additive)"))
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 7  2x2 ablation, 3 seeds per cell: the two levers are ORTHOGONAL\n"
                 "anchors move detection by +0.1445 and leave lane alone; "
                 "the lane repair moves lane and leaves detection alone",
                 fontsize=10.8, fontweight="bold", y=1.09)
    save(fig, "fig08_ablation_2x2.png")


def fig09_lane_ladder():
    rungs = [
        ("1/8 h32\n(baseline)", 1.0796, 0.5827, 0.0000, False),
        ("1/4 h32\n(full)", 1.6391, 0.5951, 1.0000, False),
        ("1/4 h16\n(thin = B)", 1.1656, 0.5959, 0.9884, False),
        ("1/4 h8\n(min)", 1.0174, 0.4959, 0.0000, True),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    x = np.arange(len(rungs))
    cols = [GREY if d else (OURS if "thin" in r[0] else ACC1) for r in rungs for d in [r[4]]]
    ax.bar(x, [r[2] for r in rungs], 0.55, color=cols)
    for xi, r in zip(x, rungs):
        ax.annotate(f"{r[2]:.4f}", (xi, r[2]), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8.6, fontweight="bold")
        ax.annotate(f"{r[1]:.4f} G", (xi, 0.487), ha="center", fontsize=7.4, color="#444")
    ax.axhline(0.5913, ls="--", lw=1.2, color=ACC2)
    ax.annotate("PASS bar 0.5913", (3.42, 0.5913), fontsize=7.4, color=ACC2, va="bottom",
                ha="right")
    ax.set_xticks(x, [r[0] for r in rungs])
    ax.set_ylabel("lane mIoU")
    ax.set_ylim(0.485, 0.605)
    ax.set_title("Fig. 8  The lane-resolution ladder: resolution is the operative variable,\n"
                 "width is inert (h16 == h32 at -28.9% FLOPs), and h8 collapses "
                 "deterministically", fontweight="bold")
    ax.annotate("lane_fg_iou = 0.0000\n(all-background signature)", (3, 0.4959),
                textcoords="offset points", xytext=(-6, 22), fontsize=7.4, color=GREY,
                ha="right")
    save(fig, "fig09_lane_ladder.png")


def fig10_anchor_two_arch():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))

    ax = axes[0]
    sets = ["shipped", "aspect-flip", "k-means refit"]
    best = [0.4161, 0.5419, 0.6805]
    zero = [0.4941, 0.1666, 0.0374]
    xs = np.arange(3)
    ax.bar(xs, best, 0.5, color=[GREY, ACC1, OURS])
    ax.set_ylabel("mean best IoU (box-anchor match)", color=ACC1)
    ax.set_xticks(xs, sets)
    ax.set_ylim(0, 0.86)
    ax2 = ax.twinx()
    ax2.plot(xs, zero, marker="D", ms=7, lw=2, color=ACC3, label="zero-positive rate")
    ax2.set_ylabel("zero-positive anchor rate", color=ACC3)
    ax2.grid(False)
    ax2.set_ylim(0, 0.56)
    for xi, v in zip(xs, zero):
        ax2.annotate(f"{v * 100:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 7),
                     fontsize=7.4, color=ACC3, ha="center")
    for xi, v in zip(xs, best):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 4),
                    fontsize=7.6, ha="center")
    ax.set_title("zero-training diagnostic: the anchor set itself\n"
                 "~50% of anchors receive no positive at all under the shipped set")

    ax = axes[1]
    labels = ["YOLOP\n(2nd architecture)", "R2\n(home architecture)"]
    deltas = [0.1993, 0.1445]
    floors = ["mAP50 floor = 0.0073\n= 27x the delta", "2 sigma = 0.0096\n= 15x the delta"]
    xs = np.arange(2)
    ax.bar(xs, deltas, 0.42, color=[ACC1, OURS])
    for xi, (d, f) in enumerate(zip(deltas, floors)):
        ax.annotate(f"+{d:.4f}", (xi, d), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9, fontweight="bold")
        ax.annotate(f, (xi, d / 2), ha="center", va="center", fontsize=7.2, color="white")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("mAP50 gain from the supervision fix alone")
    ax.set_title("trained effect on BOTH architectures\n"
                 "direction and mechanism transfer; magnitude is not claimed to")
    fig.suptitle("Fig. 9  Anchor assignment is a zero-cost, cross-architecture detection lever",
                 fontsize=11, fontweight="bold", y=1.04)
    save(fig, "fig10_anchor_two_arch.png")


def fig11_encoder_z_interaction():
    data = [
        ("E-base\nz=32", [-0.0105, -0.0070, -0.0015], [0.0073, 0.0142, 0.0021]),
        ("E-base\nz=128", [0.0005, -0.0109, -0.0027], [0.0073, 0.0142, 0.0021]),
        ("E-large\nz=32", [-0.0214, -0.0076, -0.0008], [0.0073, 0.0142, 0.0021]),
        ("E-large\nz=128", [-0.0101, -0.0081, 0.0010], [0.0073, 0.0142, 0.0021]),
    ]
    tasks = ["detection", "DA", "lane"]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    x = np.arange(len(data))
    w = 0.24
    for i, (t, col) in enumerate(zip(tasks, [ACC1, ACC2, ACC3])):
        vals = [d[1][i] for d in data]
        ax.bar(x + (i - 1) * w, vals, w, color=col, label=t)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.annotate(f"{v:+.4f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3 if v > 0 else -9), ha="center", fontsize=6.8)
    ax.axhline(0, color="#333", lw=1)
    for i, (t, thr) in enumerate(zip(tasks, [0.0073, 0.0142, 0.0021])):
        ax.plot([-0.5, 3.5], [thr, thr], ls=":", lw=1.3, color=[ACC1, ACC2, ACC3][i], alpha=0.8)
        ax.annotate(f"{t} noise floor {thr}", (3.5, thr), fontsize=6.9,
                    color=[ACC1, ACC2, ACC3][i], va="bottom", ha="right")
    ax.set_xticks(x, [d[0] for d in data])
    ax.set_ylabel("interaction term (delta dCap, k-means minus old)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(-0.030, 0.020)
    ax.set_title("Fig. 10  Encoder x Z interaction: only the DETECTION faces clear their own\n"
                 "noise floor -- supervision and capacity compete for detection headroom, "
                 "not for DA or lane", fontweight="bold")
    save(fig, "fig11_encoder_z_interaction.png")


# --------------------------------------------------------------------------
# 4. field position
# --------------------------------------------------------------------------
STAGEB = {
    "TriLiteNet tiny": dict(da_o=0.8853, da_u=0.8840, lane_o=0.2432, lane_u=0.2058, pub_lane=0.242),
    "TriLiteNet small": dict(da_o=0.9103, da_u=0.9084, lane_o=0.2764, lane_u=0.2270, pub_lane=0.276),
    "TriLiteNet base": dict(da_o=0.9245, da_u=0.9224, lane_o=0.2985, lane_u=0.2386, pub_lane=0.298),
    "TLP nano": dict(da_o=0.8742, da_u=0.8734, lane_o=0.2355, lane_u=0.1842, pub_lane=0.233),
    "TLP small": dict(da_o=0.9065, da_u=0.9058, lane_o=0.2939, lane_u=0.2146, pub_lane=0.293),
    "TLP medium": dict(da_o=0.9207, da_u=0.9202, lane_o=0.3244, lane_u=0.2322, pub_lane=0.323),
    "TLP large": dict(da_o=0.9291, da_u=0.9285, lane_o=0.3426, lane_u=0.2440, pub_lane=0.342),
    "TwinLiteNet": dict(da_o=0.9127, da_u=0.9120, lane_o=0.2883, lane_u=0.2345, pub_lane=0.311),
    "Ours Model B": dict(da_o=0.8649, da_u=0.8594, lane_o=0.1941, lane_u=0.2173, pub_lane=None),
}


def fig12_dual_gt():
    names = list(STAGEB)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.4))
    ax = axes[0]
    w = 0.38
    o = [STAGEB[n]["da_o"] for n in names]
    u = [STAGEB[n]["da_u"] for n in names]
    ax.bar(x - w / 2, o, w, color=ACC1, label="official GT")
    ax.bar(x + w / 2, u, w, color=ACC2, label="our GT")
    for xi, (a, b) in enumerate(zip(o, u)):
        ax.annotate(f"{a * 100:.2f}", (xi - w / 2, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
        ax.annotate(f"{b * 100:.2f}", (xi + w / 2, b), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
    ax.set_xticks(x, names, rotation=38, ha="right", fontsize=7.6)
    ax.set_ylim(0.84, 0.945)
    ax.set_ylabel("DA mIoU")
    ax.legend(fontsize=8)
    ax.set_title("DA: JS = 0.9885 -- same annotation, COMPARABLE")
    ax.patches[16].set_edgecolor(OURS)
    ax.patches[16].set_linewidth(2)
    ax.patches[17].set_edgecolor(OURS)
    ax.patches[17].set_linewidth(2)

    ax = axes[1]
    o = [STAGEB[n]["lane_o"] for n in names]
    u = [STAGEB[n]["lane_u"] for n in names]
    ax.bar(x - w / 2, o, w, color=ACC1, label="official GT")
    ax.bar(x + w / 2, u, w, color=ACC2, label="our GT")
    for xi, (a, b) in enumerate(zip(o, u)):
        ax.annotate(f"{a * 100:.2f}", (xi - w / 2, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
        ax.annotate(f"{b * 100:.2f}", (xi + w / 2, b), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=6.6)
    ax.set_xticks(x, names, rotation=38, ha="right", fontsize=7.6)
    ax.set_ylim(0.16, 0.38)
    ax.set_ylabel("lane foreground IoU")
    ax.legend(fontsize=8)
    ax.set_title("lane: JS = 0.3804 -- different annotation, NOT comparable")
    n_ours = names.index("Ours Model B")
    ax.annotate("", (n_ours + w / 2, STAGEB["Ours Model B"]["lane_u"]),
                (n_ours - w / 2, STAGEB["Ours Model B"]["lane_o"]),
                arrowprops=dict(arrowstyle="-|>", color=OURS, lw=2.0))
    for i in range(len(names) - 1):
        ax.annotate("", (i + w / 2, STAGEB[names[i]]["lane_u"]),
                    (i - w / 2, STAGEB[names[i]]["lane_o"]),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1))
    ax.text(0.02, 0.97, "switching to the official GT moves every\n"
                        "baseline UP and our own row DOWN\n"
                        "-> our lane number is PESSIMISTIC",
            transform=ax.transAxes, fontsize=7.2, color=OURS, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fdf2f0", ec=OURS, lw=0.8))
    fig.suptitle("Fig. 11  9 models, one harness, 10,000 images, 384 canvas, two ground truths\n"
                 "18/18 published references reproduced within +-1.0",
                 fontsize=11, fontweight="bold", y=1.05)
    save(fig, "fig12_dual_gt.png")


FIELD = [
    # name, params_M, DA, mAP50, laneIoU, group
    ("YOLOP", 7.9, 91.5, 76.5, 26.2),
    ("HybridNets", 12.83, 90.5, 77.3, 31.6),
    ("YOLOPv2", 38.9, 93.2, 83.4, 27.25),
    ("YOLOPv3", 30.2, 93.2, 84.3, 28.0),
    ("YOLOPX", 32.9, 93.2, 83.3, 27.2),
    ("A-YOLOM-n", 4.43, 90.5, 78.0, 28.2),
    ("A-YOLOM-s", 13.61, 91.0, 81.1, 28.8),
    ("SCAM-P C2f-n", 3.2, 90.4, 78.0, 26.5),
    ("SCAM-P C2f-s", 12.0, 91.0, 81.1, 27.8),
    ("SCAM-P GELAN-n", 3.4, 91.0, 78.1, 27.3),
    ("MtTEPNet", 8.3, 92.8, 79.9, 28.8),
    ("MDA-Net-n", 3.59, 91.0, 79.1, 28.6),
    ("MDA-Net-s", 13.22, 91.3, 82.0, 28.9),
    ("Sparse U-PDP", 12.05, 92.9, 84.7, 32.4),
    ("TriLiteNet tiny", 0.15, 88.5, 49.6, 24.2),
    ("TriLiteNet small", 0.59, 91.0, 63.2, 27.6),
    ("TriLiteNet base", 2.35, 92.4, 72.3, 29.8),
    ("TwinLiteNet+ nano", 0.03, 87.3, None, 23.3),
    ("TwinLiteNet+ small", 0.12, 90.6, None, 29.3),
    ("TwinLiteNet+ medium", 0.48, 92.0, None, 32.3),
    ("TwinLiteNet+ large", 1.94, 92.9, None, 34.2),
    ("Model B (ours)", 0.193, 86.49, 53.82, None),
]


def fig13_field_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    ax = axes[0]
    for name, pm, da, mp, lane in FIELD:
        ours = name.startswith("Model B")
        ax.scatter(pm, da, s=140 if ours else 62, color=OURS if ours else BASE,
                   marker="*" if ours else "o", zorder=5 if ours else 3,
                   edgecolor="white", linewidth=1)
    for name, pm, da, mp, lane in FIELD:
        if name in ("Model B (ours)", "TriLiteNet tiny", "TwinLiteNet+ nano", "TwinLiteNet+ large"):
            ax.annotate(name, (pm, da), textcoords="offset points", xytext=(7, -3),
                        fontsize=7.6, color=OURS if name.startswith("Model B") else "#333")
    ax.annotate("Sparse U-PDP", (12.05, 92.9), textcoords="offset points", xytext=(-6, 7),
                fontsize=7.6, color="#333", ha="right")
    ax.axhline(86.49, ls=":", lw=1.1, color=OURS, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("parameters (M, log scale)")
    ax.set_ylabel("DA mIoU (%)")
    ax.set_ylim(84.5, 95)
    ax.set_xlim(0.022, 70)
    ax.set_title("DA mIoU vs parameters -- 22 published models + ours\n"
                 "the two cheapest competitors still sit +0.81 and +2.01 points ABOVE us")

    ax = axes[1]
    pts = [(mp, lane) for name, pm, da, mp, lane in FIELD if mp is not None and lane is not None]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=58, color=BASE,
               edgecolor="white", linewidth=1, label="published")
    ax.scatter([53.82], [19.41], s=150, color=OURS, marker="*", zorder=5,
               edgecolor="white", linewidth=1.2, label="Model B, scored on official GT")
    ax.scatter([53.82], [21.87], s=110, color=OURS, marker="^", zorder=5,
               edgecolor="white", linewidth=1.2, label="Model B, scored on our GT")
    ax.annotate("", (53.82, 19.41), (53.82, 21.87),
                arrowprops=dict(arrowstyle="-", color=OURS, lw=1.2, ls=":"))
    ax.set_xlim(45, 88)
    ax.set_ylim(17.5, 36)
    ax.set_xlabel("mAP50 (%)")
    ax.set_ylabel("lane IoU (%)")
    ax.axhspan(23, 34.2, color=ACC3, alpha=0.09)
    ax.annotate("the entire published field is confined\nto lane IoU 23-34.2: the metric is\n"
                "capped by construction\n(train strokes 8 px, val 2 px)",
                (0.30, 0.30), xycoords="axes fraction", fontsize=7.4, color="#444",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6ec", ec="#e0c9a6"))
    ax.legend(fontsize=7.6, loc="lower right")
    ax.set_title("detection vs lane IoU, whole field\n"
                 "nobody escapes the lane ceiling")
    fig.suptitle("Fig. 12  Where we actually sit in the field (all numbers from the 22 local PDFs)\n"
                 "DA is the validated comparable axis; our DA point is placed at the "
                 "official-GT value", fontsize=10.8, fontweight="bold", y=1.07)
    save(fig, "fig13_field_scatter.png")


def fig14_conformance():
    checks = [
        ("TriLiteNet tiny / DA", 88.53, 88.5), ("TriLiteNet tiny / laneIoU", 24.32, 24.2),
        ("TriLiteNet small / DA", 91.03, 90.5), ("TriLiteNet small / laneIoU", 27.64, 27.6),
        ("TriLiteNet base / DA", 92.45, 92.0), ("TriLiteNet base / laneIoU", 29.85, 29.8),
        ("TLP nano / DA", 87.42, 87.3), ("TLP nano / laneIoU", 23.55, 23.3),
        ("TLP nano / laneAcc", 70.31, 70.2), ("TLP small / DA", 90.65, 90.6),
        ("TLP small / laneIoU", 29.39, 29.3), ("TLP small / laneAcc", 75.85, 75.8),
        ("TLP medium / DA", 92.07, 92.0), ("TLP medium / laneIoU", 32.44, 32.3),
        ("TLP medium / laneAcc", 79.21, 79.1), ("TLP large / DA", 92.91, 92.9),
        ("TLP large / laneIoU", 34.26, 34.2), ("TLP large / laneAcc", 81.94, 81.9),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    meas = [c[1] for c in checks]
    pub = [c[2] for c in checks]
    ax.scatter(pub, meas, s=64, color=ACC1, edgecolor="white", linewidth=1, zorder=4)
    lim = [22, 95]
    ax.plot(lim, lim, color="#333", lw=1, ls="-")
    ax.fill_between(lim, [l - 1 for l in lim], [l + 1 for l in lim], color=ACC2, alpha=0.16,
                    label="+-1.0 tolerance band")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("value published by the authors")
    ax.set_ylabel("value measured by our harness")
    ax.set_title("Fig. 13  Protocol certificate: 18/18 published references\n"
                 "reproduced within +-1.0 (n = 10,000 images)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.04, 0.94, "all N=18 checks pass\nmax |delta| = 0.53", transform=ax.transAxes,
            fontsize=8.4, va="top", color=ACC2, fontweight="bold")
    save(fig, "fig14_conformance.png")


# --------------------------------------------------------------------------
# 5. measurement credibility
# --------------------------------------------------------------------------
def fig15_gt_audit():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))

    ax = axes[0]
    ax.bar(["DA", "lane"], [0.9885, 0.3804], 0.45, color=[ACC2, OURS])
    for x, v in zip([0, 1], [0.9885, 0.3804]):
        ax.annotate(f"{v:.4f}", (x, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.90, ls="--", lw=1.2, color="#333")
    ax.annotate("comparability bar 0.90", (1.42, 0.90), ha="right", fontsize=7.4)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Jaccard(our masks, official package)")
    ax.set_title("pixel-level agreement of the two label sets\n(400 images, native resolution)")

    ax = axes[1]
    ax.bar([0, 1], [16.528, 16.680], 0.35, color=ACC1, label="DA foreground %")
    ax.bar([0.42, 1.42], [0.710, 0.650], 0.35, color=ACC3, label="lane foreground %")
    ax.set_xticks([0.21, 1.21], ["ours", "official"])
    ax.set_ylabel("foreground fraction (%)")
    ax.set_title("foreground density matches on both axes\n"
                 "-- the lane disagreement is a 1-px rendering shift,\n"
                 "not a different region")
    ax.legend(fontsize=7.8)

    ax = axes[2]
    ax.axis("off")
    ax.text(0.5, 0.94, "the decisive cross-check", ha="center", fontsize=10,
            fontweight="bold", transform=ax.transAxes)
    txt = (
        "Apply the official rule (>1) to OUR lane file:\n"
        "    100.000% of pixels are selected\n\n"
        "Our lane background is 255. The official\n"
        "background is 0. The two files cannot be\n"
        "the same encoding.\n\n"
        "DA: official rule selects 4.8% of our file ->\n"
        "the encodings agree.\n\n"
        "CONSEQUENCE FOR THE PAPER\n"
        "  DA axis  -> directly comparable\n"
        "  lane axis -> NOT comparable; no scale\n"
        "               factor can repair it"
    )
    ax.text(0.02, 0.80, txt, fontsize=7.9, va="top", family="monospace",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f7f9fb", ec="#b9c6d3"))
    fig.suptitle("Fig. 14  Ground-truth audit: which axis may legally be compared to a paper",
                 fontsize=11, fontweight="bold", y=1.05)
    save(fig, "fig15_gt_audit.png")


def fig16_d8_instrument():
    ours_ship = [4.51, 3.59, 1.73, 3.85, 2.21]
    ours_fix = [1.04, 1.04, 1.12, 1.06, 1.15]
    labels = ["B100 s0", "B40 s0", "B20 s0", "B20 s1", "B20 s2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))

    ax = axes[0]
    x = np.arange(5)
    w = 0.38
    ax.bar(x - w / 2, ours_ship, w, color=OURS, label="as shipped (warmup 10 / reps 3)")
    ax.bar(x + w / 2, ours_fix, w, color=ACC2, label="FIXED (warmup 200 / reps 300 / CUDA events)")
    for xi, (a, b) in enumerate(zip(ours_ship, ours_fix)):
        ax.annotate(f"{a:.2f}x", (xi - w / 2, a), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.2)
        ax.annotate(f"{b:.2f}x", (xi + w / 2, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.2)
    ax.axhline(1.0, ls="--", lw=1.2, color="#333")
    ax.set_xticks(x, labels)
    ax.set_ylabel("spread across a 10-session null replicate (x)")
    ax.set_title("five checkpoints, IDENTICAL architecture (192,566 params / 1.1656 GFLOPs)\n"
                 "a good instrument puts them on 1.00x")
    ax.legend(fontsize=7.8)

    ax = axes[1]
    models = ["Ours B100", "Ours B40", "Ours B20 s0", "Ours B20 s1", "Ours B20 s2",
              "TL tiny", "TL small", "TLP nano"]
    ship_o = [10.2152, 2.5938, 2.4145, 2.67, 2.3071, 4.2529, 5.6009, 7.1488]
    ship_b = [2.3005, 2.9733, 2.4978, 2.7691, 2.4481, 4.3067, 10.0174, 3.9734]
    fixed = [2.3964, 2.3829, 2.3843, 2.8122, 2.3238, 3.5096, 5.3836, 5.0314]
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, ship_o, w, color=OURS, label="as shipped, OUR rows (reps=3)")
    ax.bar(x, ship_b, w, color=BASE, label="as shipped, BASELINE rows (reps=100)")
    ax.bar(x + w, fixed, w, color=ACC2, label="FIXED protocol, symmetric")
    ax.set_xticks(x, models, rotation=34, ha="right", fontsize=7.2)
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("the published table's two blocks were never measured\n"
                 "by the same instrument -- the column is not a comparison")
    ax.legend(fontsize=7.4)
    fig.suptitle("Fig. 15  D8: locating the latency defect instead of reporting around it",
                 fontsize=11, fontweight="bold", y=1.03)
    save(fig, "fig16_d8_instrument.png")


def fig17_efficiency():
    rows = [(n, pm, mp) for n, pm, da, mp, lane in FIELD if mp is not None]
    rows.sort(key=lambda t: t[2] / t[1], reverse=True)
    names = [r[0] for r in rows]
    eff = [r[2] / r[1] for r in rows]      # mAP50 per M params
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    cols = [OURS if n.startswith("Model B") else BASE for n in names]
    x = np.arange(len(names))
    ax.bar(x, eff, 0.6, color=cols)
    for xi, v in zip(x, eff):
        ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.6)
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=7.2)
    ax.set_ylabel("mAP50 per million parameters")
    top = names[0]
    ax.annotate(f"{top} stays ahead of us here\n(330.7 vs 278.9): the honest\n"
                f"statement is SECOND, at 2.6x\nthe next competitor",
                (0.30, 0.72), xycoords="axes fraction", fontsize=7.6, color="#444",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fdf6ec", ec="#e0c9a6"))
    ax.set_title("Fig. 16  Detection accuracy per parameter\n"
                 "our strongest efficiency axis -- but it is 2nd place, not 1st",
                 fontweight="bold")
    save(fig, "fig17_efficiency.png")


def main():
    print(f"writing figures to {FIG}")
    fig01_roadmap()
    fig02_budget_curve()
    fig03_gap_closure()
    fig04_capacity_heatmap()
    fig05_marginal_price()
    fig06_pareto_ours()
    fig07_BvsSvsU()
    fig08_ablation_2x2()
    fig09_lane_ladder()
    fig10_anchor_two_arch()
    fig11_encoder_z_interaction()
    fig12_dual_gt()
    fig13_field_scatter()
    fig14_conformance()
    fig15_gt_audit()
    fig16_d8_instrument()
    fig17_efficiency()
    print(f"\n{len(_saved)} figures written")
    for n in _saved:
        p = os.path.join(FIG, n)
        print(f"  {os.path.getsize(p) / 1024:8.1f} KB  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
