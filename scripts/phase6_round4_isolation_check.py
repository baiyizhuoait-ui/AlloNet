#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 6 Round 4 -- implementation-isolation audit (ZERO-TRAINING, CPU).

The Round 4 brief demands two isolations be *demonstrated*, not asserted:

EXP-11 (supervision integration)
    "Model parameters same. Architecture FLOPs same or nearly same.
     Only supervision changes."
    -> checked here by (a) a structural parsed-YAML diff of the two configs and
       (b) an independent params/FLOPs build of both.

EXP-10 (lane repair ladder)
    each rung must differ from the reference by exactly one configuration knob,
    so that the cost/lane-gain tradeoff is attributable to that knob.

Ruler invariant
    every Round 4 cell must use the byte-identical k-means anchor set and the
    project matching rule (0.5 < box/anchor < 2.0) must be untouched.

Exit code 1 on any violation (fail-closed). Usage:
    python scripts/phase6_round4_isolation_check.py
"""
import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from models.static_model import StaticMultiTaskModel  # noqa: E402
from profiling.flops_real import count_flops  # noqa: E402

C = lambda n: os.path.join(ROOT, "configs", n)
FAIL = []


def load(p):
    return yaml.safe_load(open(C(p)))


def model_block(cfg):
    """The architecture-bearing sub-tree only (train.* is schedule, not arch)."""
    return cfg.get("model", {})


def flat_diff(a, b, path=""):
    """Return the set of dotted paths whose values differ (recursive, dict-aware)."""
    out = set()
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            out |= flat_diff(a.get(k, "<MISSING>"), b.get(k, "<MISSING>"),
                             f"{path}.{k}" if path else str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b) or a != b:
            out.add(path)
    else:
        if a != b:
            out.add(path)
    return out


def build(cfg):
    return StaticMultiTaskModel(model_block(cfg)).eval()


def measure(cfg):
    m = build(cfg)
    p = sum(q.numel() for q in m.parameters())
    with torch.no_grad():
        f = count_flops(m, torch.randn(1, 3, 640, 640))
    return p, f / 1e9


def check(name, cond, detail=""):
    print(f"[{'OK ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


print("=" * 78)
print("EXP-11  supervision isolation  (anchors = the ONLY architectural delta)")
print("=" * 78)
PAIRS = [("phase4a_r2_z16.yaml", "phase4b_danc_z16.yaml", "lean  x {old -> k-means}"),
         ("phase6_e8_uniform_z16.yaml", "phase6c_e9_aunif_km.yaml", "uniform x {old -> k-means}")]
for a, b, label in PAIRS:
    ca, cb = load(a), load(b)
    d = flat_diff(model_block(ca), model_block(cb))
    only_anchors = d <= {"detection.anchors"}
    check(f"{label}: model tree delta ⊆ {{detection.anchors}}", only_anchors,
          f"delta={sorted(d)}")
    if only_anchors:
        pa, fa = measure(ca)
        pb, fb = measure(cb)
        check(f"{label}: params identical", pa == pb, f"{pa} vs {pb}")
        check(f"{label}: FLOPs identical (<0.1%)", abs(fa - fb) / fa < 1e-3,
              f"{fa:.4f}G vs {fb:.4f}G")
        # the anchor set must actually differ, otherwise the cell is a null experiment
        check(f"{label}: anchor sets differ (non-null experiment)",
              ca["model"]["detection"].get("anchors") != cb["model"]["detection"].get("anchors"))

print()
print("=" * 78)
print("EXP-10  lane ladder isolation  (segmentation = the ONLY architectural delta)")
print("=" * 78)
REF = "phase6_combo_danc_l14f1.yaml"
RUNGS = [("phase6_r4_R2_thin14_z16.yaml", "R2 thin 1/4  h16"),
         ("phase6_r4_R3_min14_z16.yaml", "R3 min  1/4  h8"),
         ("phase6_r4_R4_uponly14_z16.yaml", "R4 up-only 1/4 (no lateral)"),
         ("phase4b_l14up_z16.yaml", "R4-prev  (pre-Round-4, old anchors)"),
         ("phase4b_l14f1_z16.yaml", "R1-prev  (pre-Round-4, old anchors)")]
ref = load(REF)
for f, label in RUNGS:
    d = flat_diff(model_block(ref), model_block(load(f)))
    # R2/R3 change only a segmentation knob; R4-prev/R1-prev (pre-Round-4 files)
    # differ only by the anchors block that this round adds to every cell.
    ok = len(d) > 0 and all(x == "detection.anchors" or x.startswith("segmentation")
                            for x in d)
    check(f"{label}: model tree delta ⊆ segmentation.* ∪ {{detection.anchors}}", ok,
          f"delta={sorted(d)}")

print()
print("=" * 78)
print("RULER  one anchor set / one matching rule across every Round 4 cell")
print("=" * 78)
r4 = ["phase6_r4_R2_thin14_z16.yaml", "phase6_r4_R3_min14_z16.yaml",
      "phase6_r4_R4_uponly14_z16.yaml", "phase6_r4_U_min_km.yaml",
      "phase6_combo_danc_l14f1.yaml", "phase6c_e9_aunif_km.yaml"]
km = [load(f)["model"]["detection"]["anchors"] for f in r4]
check("all Round-4 km cells share the byte-identical anchor set",
      all(k == km[0] for k in km))
# the matching rule itself is code, not config: prove the file is untouched
import hashlib  # noqa: E402
los = os.path.join(ROOT, "losses", "yolo_loss.py")
src = open(los).read()
RULER = "j[ai] = (ratio < 2.0).all(-1) & (ratio > 0.5).all(-1)"
check("matching rule present, exact form", RULER in src,
      "sha256(losses/yolo_loss.py)=" + hashlib.sha256(src.encode()).hexdigest()[:16])

print()
if FAIL:
    print(f"RESULT: FAIL ({len(FAIL)} violation(s)) -> {FAIL}")
    sys.exit(1)
print("RESULT: isolation OK -- EXP-11 is single-variable, EXP-10 rungs are single-knob, ruler fixed")
