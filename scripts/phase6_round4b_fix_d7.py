#!/usr/bin/env python3
"""Defect D7 provenance repair (0 GPU): relabel the two Round-4 baseline arms that
were LABELLED `old anchors` but provably trained k-means.

PRIMARY EVIDENCE (config level)
  * configs/phase4a_r2_z16.yaml has no `anchors` block.
  * models/representation/det_from_z.py DEFAULT_ANCHORS_3S is the IoU-k-means set as
    of commit c8aca35 (2026-09-08); the historical aspect-flipped set was demoted.
  * scripts/phase6_round4_run.sh takes `supervision` as a CSV label only -- there is
    no code path by which it selects anchors.
  => an arm tagged `old` against that config trains k-means.

CORROBORATION (metric level)
  the affected rows report mAP50 well above the historical old-anchor cell (0.3543)
  and inside the lean+km band. This script FAILS LOUDLY if that is not true, i.e. if
  the primary conclusion above were wrong.

ACTION
  cell  R4-A0lean_old -> R4-A0lean_km   (that is what these runs actually are:
  the lean+k-means cell's missing seeds 1 and 2 -- so the defect converts into a
  seed completion for the OTHER cell, at zero cost)
  supervision `old` -> `km`
  Metrics are NOT touched. Readback-verified, row-count-verified.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [("experiments/phase6/phase6_round4_results.csv", "cell"),
         ("experiments/phase6/phase6_round4_metrics.csv", "cell")]
OLD, NEW = "R4-A0lean_old", "R4-A0lean_km"
KM_BAND = (0.47, 0.52)      # lean+km @20ep measured 0.4950
HIST_OLD = 0.3543           # historical old-anchor cell, seed 0

changed_total = 0
for rel, keycol in FILES:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"[skip] {rel} not present")
        continue
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys()) if rows else []
    if keycol not in fields:
        print(f"[skip] {rel}: no '{keycol}' column")
        continue

    hit = [r for r in rows if r.get(keycol) == OLD]
    if not hit:
        print(f"[ok  ] {rel}: no rows labelled {OLD} (already repaired)")
        continue

    # fail loud if the metric does NOT corroborate the config-level conclusion
    for r in hit:
        try:
            v = float(r.get("mAP50") or "nan")
        except ValueError:
            v = float("nan")
        ok = (KM_BAND[0] <= v <= KM_BAND[1])
        print(f"[{'ok  ' if ok else 'FAIL'}] {rel} seed={r.get('seed')} mAP50={v} "
              f"km-band={KM_BAND} (historical old = {HIST_OLD})")
        if not ok:
            print("       REFUSING to relabel: the metric does not match the k-means "
                  "signature, so the D7 conclusion is unsafe.")
            sys.exit(1)
        if float(r.get("seed") or -1) not in (1, 2):
            print(f"       REFUSING: unexpected seed {r.get('seed')}")
            sys.exit(1)

    n_before = len(rows)
    for r in rows:
        if r.get(keycol) == OLD:
            r[keycol] = NEW
            if "supervision" in fields and r.get("supervision") == "old":
                r["supervision"] = "km"

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # readback self-verification (same discipline as the Round-3 D5 fix)
    with open(path, newline="") as fh:
        back = list(csv.DictReader(fh))
    assert len(back) == n_before, f"row count changed {n_before} -> {len(back)}"
    assert not [r for r in back if r.get(keycol) == OLD], "relabel did not take"
    moved = [r for r in back if r.get(keycol) == NEW]
    assert len(moved) == len(hit), "wrong number of rows relabelled"
    for r in moved:
        assert KM_BAND[0] <= float(r["mAP50"]) <= KM_BAND[1], "metric drifted"
    print(f"[done] {rel}: {len(hit)} row(s) relabelled {OLD} -> {NEW} "
          f"(readback OK, total rows {len(back)})")
    changed_total += len(hit)

print(f"\nD7 REPAIR: {changed_total} row(s) relabelled. Metrics unchanged.")
print("NEW arms pinning the historical anchors will be appended as "
      "R4-A0lean_old seed 1,2 by the Round 4B chain.")
