#!/usr/bin/env python3
"""Round 4B config isolation proof (no training, no GPU).

Same discipline as scripts/phase6_round4_isolation_check.py: every new arm must be
provably a single-knob edit of its base config, compared as PARSED DICTS (so YAML
reformatting cannot hide or fake a difference).

Also re-asserts the D7 fact this round depends on: the code default anchor set is
currently the k-means set, i.e. a config WITHOUT an anchors block does NOT mean
"historical anchors".
"""
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIST_ANCHORS = [[[4, 12], [7, 19], [11, 28]],
                [[17, 40], [25, 58], [38, 89]],
                [[62, 136], [88, 206], [124, 412]]]
KM_ANCHORS = [[[9, 8], [18, 15], [32, 24]],
              [[49, 38], [80, 52], [65, 102]],
              [[124, 82], [166, 136], [237, 214]]]

failures = []


def flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
    else:
        out[p] = d
    return out


def load(rel):
    with open(os.path.join(ROOT, rel)) as fh:
        return yaml.safe_load(fh)


def check(label, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {label}")
    if detail:
        print(f"        {detail}")
    if not ok:
        failures.append(label)


CASES = [
    ("configs/phase6_r4_lean_old.yaml", "configs/phase4a_r2_z16.yaml",
     {"model.detection.anchors", "train.anchors"},
     "EXP-13 C1 (lean + HISTORICAL anchors) differs from phase4a_r2_z16 by anchors only"),
    ("configs/phase6_r4_R2thin_lr1e4_z16.yaml", "configs/phase6_r4_R2_thin14_z16.yaml",
     {"train.lr"},
     "EXP-10b (lr probe) differs from R2_thin by train.lr only"),
]

print("== Round 4B isolation ==")
for new, base, expect, label in CASES:
    a, b = flat(load(new)), flat(load(base))
    delta = {k for k in (set(a) | set(b)) if a.get(k) != b.get(k)}
    check(label, delta == expect,
          f"delta={sorted(delta)} expected={sorted(expect)}")

# --- D7 anchor provenance ---------------------------------------------------
lean = load("configs/phase6_r4_lean_old.yaml")
check("lean_old pins the HISTORICAL aspect-flipped anchors in-file",
      lean["model"]["detection"]["anchors"] == HIST_ANCHORS
      and lean["train"]["anchors"] == HIST_ANCHORS,
      "both model.detection.anchors and train.anchors match git c8aca35^ values")

base_cfg = load("configs/phase4a_r2_z16.yaml")
check("the base config really has no anchors block (this is how D7 happened)",
      "anchors" not in (base_cfg["model"].get("detection") or {})
      and "anchors" not in (base_cfg.get("train") or {}),
      f"model.detection keys={sorted(base_cfg['model']['detection'])}")

src = open(os.path.join(ROOT, "models", "representation", "det_from_z.py")).read()
check("code DEFAULT_ANCHORS_3S is the k-means set, NOT the historical one",
      "9, 8" in src.replace(" ", " ") or "[9, 8]" in src,
      "so 'no anchors block' means k-means; supervision must live in the config")

# the two Round-4B anchors must be mutually exclusive with the code default, else
# the 'old' arm would silently still be km again
check("historical set != k-means set",
      HIST_ANCHORS != KM_ANCHORS, "pinning is meaningful")

print()
if failures:
    print(f"ISOLATION: {len(failures)} FAILURE(S)")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ISOLATION: all checks passed")
