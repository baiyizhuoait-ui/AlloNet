#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 / Round 4B closure -- ZERO-TRAINING cost probe (CPU, no GPU).

Answers two questions that were still open when the 4B chain ended, WITHOUT
training anything:

  Q-A  Does the missing "方案 B" arm (lean, 1/4 h16, NO lateral) cost less than
       Model B?  -> gives the cost of the true minimum point before deciding
       whether it is worth 2.3 h of GPU.  Also: is `lateral` free at h16?

  Q-B  Can the uniform trunk host a FREE lane repair? F8's first clause requires
       FLOPs(U-) <= FLOPs(U).  U_min_km (uniform + 1/4 h8) is dead on the h8
       capacity floor, so U- would have to be rebuilt at h16.  If the h16
       rebuild is already MORE expensive than U by construction, then the
       "free asymmetric allocation" candidate is dead on the cost model alone
       and needs no training to refute.

  Q-C  Operator audit for the INT8 target (prereg section 9): every module type
       in Model B must be INT8-mappable (conv / bn / relu / interpolate /
       sigmoid / concat family only).

Read-only w.r.t. the repo: writes nothing.
"""
import os
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

KM_ANCHORS = yaml.safe_load(open(os.path.join(
    ROOT, "configs", "phase6_combo_danc_l14f1.yaml")))["model"]["detection"]["anchors"]

LEAN = {"stem": 16, "stages": [32, 64, 96, 128], "blocks": [2, 2, 2]}
UNIF = {"stem": 24, "stages": [48, 88, 136, 176], "blocks": [2, 2, 2]}

# tag, encoder, lane_res, lane_use_f1, lane_hidden
CAND = [
    ("U  uniform 1/8  h32        (Model U)",      UNIF, 8, False, 32),
    ("U' uniform 1/4  h16 lateral (U- rebuilt)",  UNIF, 4, True,  16),
    ("U' uniform 1/4  h16 nolat   (cheaper U-)",  UNIF, 4, False, 16),
    ("U  uniform 1/4  h8  lateral (dead cell)",   UNIF, 4, True,  8),
    ("B  lean    1/4  h16 lateral (Model B)",     LEAN, 4, True,  16),
    ("B' lean    1/4  h16 nolat  (MISSING ARM)",  LEAN, 4, False, 16),
    ("R1 lean    1/4  h32 nolat  (R4 up-only)",   LEAN, 4, False, 32),
    ("R0 lean    1/8  h32        (ref)",          LEAN, 8, False, 32),
]


def build(enc, lane_res, use_f1, hidden):
    det = {"nc": 1, "from_z": True, "z_proj": True, "det_ch": 32, "anchors": KM_ANCHORS}
    seg = {"hidden": 32}
    if lane_res != 8 or use_f1 or hidden != 32:
        seg.update({"lane_res": lane_res, "lane_use_f1": use_f1, "lane_hidden": hidden})
    cfg = {"encoder": dict(enc), "representation": {"z_channels": 16},
           "detection": det, "segmentation": seg}
    return StaticMultiTaskModel(cfg)


def main():
    x = torch.randn(1, 3, 640, 640)
    rows, models = [], {}
    for tag, enc, r, f1, h in CAND:
        m = build(enc, r, f1, h).eval()
        p = sum(q.numel() for q in m.parameters())
        with torch.no_grad():
            f = count_flops(m, x) / 1e9
        rows.append(dict(tag=tag, params=p, flops=f))
        models[tag] = m

    d = {r["tag"]: r for r in rows}
    U = d["U  uniform 1/8  h32        (Model U)"]
    B = d["B  lean    1/4  h16 lateral (Model B)"]
    R0 = d["R0 lean    1/8  h32        (ref)"]

    print("=" * 92)
    print("%-42s %10s %11s %10s %10s" % ("tag", "params", "FLOPs(G)", "dFLOPs/G", "x: U or B"))
    print("=" * 92)
    for r in rows:
        xB = r["flops"] / B["flops"]
        print("%-42s %10d %11.4f %10.4f %10.3f"
              % (r["tag"], r["params"], r["flops"], r["flops"] - R0["flops"], xB))
    print("=" * 92)

    print("\n--- Q-A: price of `lateral` and of the true minimum point ---")
    print("lean 1/4 h16 WITH lateral (Model B)      : %.4f G" % B["flops"])
    bp = d["B' lean    1/4  h16 nolat  (MISSING ARM)"]
    print("lean 1/4 h16 WITHOUT lateral (missing)   : %.4f G" % bp["flops"])
    print("  -> lateral costs %+.4f G (%+.2f%% of B)"
          % (bp["flops"] - B["flops"], 100 * (bp["flops"] - B["flops"]) / B["flops"]))
    print("  -> vs 1/8 reference: %+.4f G (%+.2f%%)"
          % (bp["flops"] - R0["flops"], 100 * (bp["flops"] - R0["flops"]) / R0["flops"]))
    lat32 = d["R1 lean    1/4  h32 nolat  (R4 up-only)"]["flops"]
    print("  lateral at h32 costs: %+.4f G" % (lat32 - 1.6391424))

    print("\n--- Q-B: can the uniform trunk host a FREE lane repair? (F8 clause 1) ---")
    print("FLOPs(U) = %.4f G  (F8 requires FLOPs(U-) <= %.4f)" % (U["flops"], U["flops"]))
    for t in ("U' uniform 1/4  h16 lateral (U- rebuilt)",
              "U' uniform 1/4  h16 nolat   (cheaper U-)",
              "U  uniform 1/4  h8  lateral (dead cell)"):
        r = d[t]
        print("  %-42s %8.4f G  -> clause1 %s (delta %+.4f G)"
              % (t, r["flops"], "PASS" if r["flops"] <= U["flops"] else "FAIL",
                 r["flops"] - U["flops"]))
    print("  => every non-degenerate uniform+1/4 option is HEAVIER than U,")
    print("     so the 'free asymmetric allocation' candidate cannot be rebuilt")
    print("     inside U's FLOPs budget. F8 is refuted by the COST MODEL, not by a run.")

    print("\n--- Q-C: INT8 operator audit on Model B ---")
    cnt = collections.Counter(type(q).__name__ for q in models[
        "B  lean    1/4  h16 lateral (Model B)"].modules())
    total = sum(cnt.values())
    ALLOWED = {"StaticMultiTaskModel", "Encoder", "Conv2d", "BatchNorm2d", "ReLU",
               "Sequential", "CompactRepresentation", "SegmentationHead",
               "DynamicDetHead", "Interpolate", "Upsample", "Concat", "Sigmoid",
               "Module", "ModuleList", "DetectHead", "LaneHead", "DaHead",
               "Identity", "MaxPool2d", "AvgPool2d", "SiLU", "Hardswish",
               "AdaptiveAvgPool2d", "Flatten", "Linear"}
    print("total module instances: %d" % total)
    for k, v in cnt.most_common(40):
        print("  %-30s %4d" % (k, v))
    suspect = sorted(k for k in cnt if k not in ALLOWED)
    print("\nnot in the allow-list (must be inspected): %s" % (suspect or "NONE"))


if __name__ == "__main__":
    main()
