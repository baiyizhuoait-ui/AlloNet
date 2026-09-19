#!/usr/bin/env python3
"""G3 forensic: read the anchor buffer out of every historical checkpoint.

Why: the k-means anchor set became the code default on 2026-09-08 23:16
(commit c8aca35).  Every cell whose config omitted an `anchors:` key silently
inherited whatever the default was ON THE DAY IT RAN.  The arch-1 allocation
ledger's decisive "like-for-like" section compares a phase2d cell with a
phase3a cell, i.e. possibly two different supervision regimes.  Run dates alone
are not proof, so read the anchors the model actually carries.

Classification of a checkpoint's anchor set:
  KM   == DEFAULT_ANCHORS_3S (k-means, the corrected set)
  OLD  == the pre-2026-09-08 aspect-flipped set
  ?    == neither (config wrote a custom set)
"""
import os
import sys

ROOT = "~/ai_study/trac"
sys.path.insert(0, ROOT)

import torch  # noqa: E402

KM = [[[9, 8], [18, 15], [32, 24]],
      [[49, 38], [80, 52], [65, 102]],
      [[124, 82], [166, 136], [237, 214]]]
OLD = [[[4, 12], [7, 19], [11, 28]],
       [[17, 40], [25, 58], [38, 89]],
       [[62, 136], [88, 206], [124, 412]]]


def classify(a):
    a = a.tolist() if torch.is_tensor(a) else a
    def as_int(x):
        return [[[int(round(v)) for v in p] for p in s] for s in x]
    if as_int(a) == KM:
        return "KM  (k-means, corrected)"
    if as_int(a) == OLD:
        return "OLD (aspect-flipped)"
    return "?   (custom)"


def anchors_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck)
    hits = {k: v for k, v in sd.items()
            if "anchor" in k.lower() and torch.is_tensor(v)}
    if not hits:
        return None, sorted(k for k in sd if "det" in k.lower())[:4]
    return hits, None


CELLS = [
    # (label, path) -- arch-1 ledger inputs first
    ("arch1 A) E-base z16  (phase2d, ledger anchor)",
     "experiments/phase2d/expD_z16_e20/checkpoint.pt"),
    ("arch1 A) E-base z128 (phase2d, ledger Z-spend)",
     "experiments/phase2d/expD_z128_e20/checkpoint.pt"),
    ("arch1 B) E-small z16 (phase3a)",
     "experiments/phase3a/exp3A_esmall_z16/checkpoint.pt"),
    ("arch1 B) E-base z16  (phase3a, reused from 2D)",
     "experiments/phase3a/exp3A_ebase_z16/checkpoint.pt"),
    ("arch1 B) E-large z16 (phase3a, ledger enc-spend)",
     "experiments/phase3a/exp3A_elarge_z16/checkpoint.pt"),
    # phase4a
    ("phase4a  r2 z16", "experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt"),
    ("phase4a  r2 z32", "experiments/phase4a/exp4A_r2_z32_e20/checkpoint.pt"),
    ("phase4a  r2 z128", "experiments/phase4a/exp4A_r2_z128_e20/checkpoint.pt"),
    # round 4 / phase 6 (these configs write anchors explicitly)
    ("r4 R2thin14 seed0 (FINAL-100 seed0 ancestor)",
     "experiments/phase6/final/B100/checkpoint.pt"),
    ("r4 R2thin14 seed1", "experiments/phase6/final/B100_s1/checkpoint.pt"),
    ("r4 R2thin14 seed2", "experiments/phase6/final/B100_s2/checkpoint.pt"),
]

print("=" * 82)
print("ANCHOR FORENSICS -- which supervision regime did each cell actually run?")
print("=" * 82)
rows = []
for label, rel in CELLS:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"{label:56s} MISSING")
        rows.append((label, "MISSING", ""))
        continue
    hits, fallback = anchors_of(p)
    if not hits:
        print(f"{label:56s} no anchor tensor; det keys={fallback}")
        rows.append((label, "NO_ANCHOR_BUFFER", ""))
        continue
    for k, v in hits.items():
        cls = classify(v)
        print(f"{label:56s} {k:28s} {cls}")
        rows.append((label, k, cls))

print()
km = sum(1 for r in rows if r[2].startswith("KM"))
old = sum(1 for r in rows if r[2].startswith("OLD"))
print(f"summary: KM={km}  OLD={old}  other={len(rows)-km-old}")
