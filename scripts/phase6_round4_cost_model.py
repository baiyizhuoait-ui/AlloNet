#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 Round 4 -- EXP-10 / EXP-12 cost model (ZERO-TRAINING, CPU).

Purpose: place every Round 4 candidate on the (params, FLOPs) plane BEFORE any
training run, so that (a) the lane-repair cost ladder is pre-registered with
measured numbers instead of guesses, and (b) EXP-12's equal-budget claim is a
measurement, not an intention.

No new model code is introduced: every tier is an existing configuration knob
(`lane_res`, `lane_use_f1`, `lane_hidden`, encoder width). This is deliberate --
Architecture Design Rule 1 (each compute unit must answer a bottleneck evidence)
forbids adding modules to hit a cost target.

Usage:  python scripts/phase6_round4_cost_model.py
Output: experiments/phase6/phase6_round4_cost.csv  (+ stdout table)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

# --- the project assignment-rule ruler, held FIXED across all Round 4 cells ---
# k-means anchors are read from the frozen Round-3 config, never re-derived here,
# so every Round 4 cell inherits the identical anchor set + identical matching rule.
KM_ANCHORS = yaml.safe_load(open(os.path.join(
    ROOT, "configs", "phase6_combo_danc_l14f1.yaml")))["model"]["detection"]["anchors"]

LEAN_ENC = {"stem": 16, "stages": [32, 64, 96, 128], "blocks": [2, 2, 2]}
UNIF_ENC = {"stem": 24, "stages": [48, 88, 136, 176], "blocks": [2, 2, 2]}

# tag, encoder, segmentation overrides, supervision tags
CANDIDATES = [
    # ---- EXP-10 lane spatial-repair cost ladder (lean backbone, km anchors) ----
    ("R0_no_repair_1_8", LEAN_ENC, {"hidden": 32},                       "km"),
    ("R1_full_1_4_h32",  LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 32},                  "km"),
    ("R2_thin_1_4_h16",  LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 16},                  "km"),
    ("R3_min_1_4_h8",    LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 8},                   "km"),
    ("R4_uponly_1_4_h32", LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": False,
                                     "lane_hidden": 32},                 "km"),
    # ---- EXP-12 competing models -------------------------------------------
    ("B_lean_km",        LEAN_ENC, {"hidden": 32},                       "km"),
    ("B_thin_km",        LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 16},                  "km"),
    ("S_full_km",        LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 32},                  "km"),
    ("S_full_old",       LEAN_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 32},                  "old"),
    ("U_floor_km",       UNIF_ENC, {"hidden": 32},                       "km"),
    # ---- does the uniform model get the spatial repair for ~free? -----------
    ("U_thin_km",        UNIF_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 16},                  "km"),
    ("U_min_km",         UNIF_ENC, {"hidden": 32, "lane_res": 4, "lane_use_f1": True,
                                    "lane_hidden": 8},                   "km"),
]


def build(enc, seg, sup):
    det = {"nc": 1, "from_z": True, "z_proj": True, "det_ch": 32}
    if sup == "km":
        det["anchors"] = KM_ANCHORS
    cfg = {"encoder": dict(enc), "representation": {"z_channels": 16},
           "detection": det, "segmentation": dict(seg)}
    return StaticMultiTaskModel(cfg)


def main():
    x = torch.randn(1, 3, 640, 640)
    rows = []
    base = None
    for tag, enc, seg, sup in CANDIDATES:
        m = build(enc, seg, sup).eval()
        p = sum(q.numel() for q in m.parameters())
        with torch.no_grad():
            f = count_flops(m, x)
        rec = {"tag": tag, "encoder": "lean" if enc is LEAN_ENC else "uniform",
               "lane_res": int(seg.get("lane_res", 8)),
               "lane_use_f1": bool(seg.get("lane_use_f1", False)),
               "lane_hidden": int(seg.get("lane_hidden", seg.get("hidden", 32))),
               "supervision": sup, "params": p, "flops_G": f / 1e9}
        rows.append(rec)
        print(f"{tag:22s} params={p:8d} ({p/1e6:.4f}M)  FLOPs={f/1e9:7.4f} G")
        if tag == "R0_no_repair_1_8":
            base = rec

    # deltas vs the lean/1-8 rung, on both axes
    for r in rows:
        r["d_params"] = r["params"] - base["params"]
        r["d_flops_G"] = round(r["flops_G"] - base["flops_G"], 4)
        r["flops_x"] = round(r["flops_G"] / base["flops_G"], 4)

    out = os.path.join(ROOT, "experiments", "phase6", "phase6_round4_cost.csv")
    cols = ["tag", "encoder", "supervision", "lane_res", "lane_use_f1",
            "lane_hidden", "params", "params_M", "flops_G", "d_params",
            "d_flops_G", "flops_x"]
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            r["params_M"] = round(r["params"] / 1e6, 4)
            w.writerow([r[c] for c in cols])
    print(f"\n[cost] wrote {out}")

    # readback self-check (same discipline as the Round-3 CSV appender)
    with open(out) as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == len(rows), "readback row count mismatch"
    for a, b in zip(rows, back):
        assert a["tag"] == b["tag"]
        assert abs(float(b["params"]) - a["params"]) < 1e-6
        assert abs(float(b["flops_G"]) - a["flops_G"]) < 1e-6
    print("[cost] readback OK")


if __name__ == "__main__":
    main()
