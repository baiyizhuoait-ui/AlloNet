#!/usr/bin/env python3
"""G1 -- budget curve for the DA deficit (zero-GPU, read-only).

Question this answers
---------------------
Round 4's preregistration (phase6_round4_preregistration.md, 13.5) asserted, in writing,
that our DA gap to TriLiteNet-tiny "does not change with the epoch count" -- i.e. that it
is structural rather than a training-budget artefact.  FINAL-100 then measured the gap
closing from -0.0202 (40 ep) to -0.0123 (100 ep).  One of the two is wrong, and the
direction matters: if the gap closes with budget, a large part of the DA deficit is
under-training, not architecture.

What this script does
---------------------
Collects every arm that is the *same model* (matched by a parameter/FLOP fingerprint,
not by cell name -- the cells are called R4-R2thin14 / R5-R2thin40 / B100 for the same
network) and reports the budget curve with per-point seed dispersion and the significance
of each step, against the preregistration-frozen 2-sigma floor.

Design rules honoured
---------------------
* Arms are matched by (variant, z, encoder, params_M, flops_G) fingerprint and the
  fingerprint must be byte-identical across budget points, otherwise the script refuses
  to draw the curve (fail-closed: a curve across different models is not a budget curve).
* Hyper-parameter negatives are EXCLUDED by an explicit allow-list, not by a heuristic:
  R6-R2thinL4 is the same network at lr=1e-4 and is the F10 negative control.  Mixing it
  in would silently turn a learning-rate effect into a budget effect.
* Missing seeds are reported as missing.  The script never interpolates and never
  substitutes a mean for an absent point.
* The noise floor is the FROZEN one (Round 3/4 preregistration), not re-estimated from
  these data -- re-estimating it after seeing the numbers is the failure mode the
  preregistration forbids.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TriLiteNet-tiny published DA mIoU on the published protocol (the reference level).
TINY_DA = 0.8796

# Frozen noise floor: 2*sigma(da_mIoU), from round4 preregistration 7 (0.0011 -> 2s=0.0023).
TWO_SIGMA_DA = 0.0023

# The same network appears under several cell names.  These are the ones that differ from
# each other ONLY by `--epochs` (verified below by the parameter fingerprint).
BUDGET_CELLS = {"R4-R2thin14", "R5-R2thin40", "B100"}

# Registered exclusions -- same network, different single knob, must not enter a budget curve.
EXCLUDED_CELLS = {
    "R6-R2thinL4": "lr 1e-4 (F10 learning-rate negative control) -- not a budget point",
}

FINGERPRINT_FIELDS = ("variant", "z", "encoder", "params_M", "flops_G")


def _f(row, key, default=None):
    v = row.get(key)
    if v in (None, "", "NA", "nan"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r.get("cell")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join("experiments", "phase6"))
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    rows = []
    for name in ("phase6_round4_results.csv", "final_results.csv"):
        src = os.path.join(outdir, name)
        for r in load(src):
            r["_src"] = name
            rows.append(r)

    # ---- partition -------------------------------------------------------
    kept, excluded, other = [], [], []
    for r in rows:
        cell = r["cell"]
        if cell in BUDGET_CELLS:
            kept.append(r)
        elif cell in EXCLUDED_CELLS:
            excluded.append(r)
        else:
            other.append(r)

    print("=" * 78)
    print("G1 budget curve -- same model, varying only --epochs")
    print("=" * 78)
    print(f"rows scanned      : {len(rows)}  ({len(kept)} kept / {len(excluded)} registered-excluded / {len(other)} unrelated)")
    for r in excluded:
        print(f"  excluded  {r['cell']:<14} epochs={r.get('epochs')} :: {EXCLUDED_CELLS[r['cell']]}")
    if not kept:
        print("\nFAIL: no budget arms found. Nothing to report.")
        return 2

    # ---- fingerprint identity check (fail-closed) ------------------------
    fps = {tuple(r.get(k) for k in FINGERPRINT_FIELDS) for r in kept}
    if len(fps) != 1:
        print("\nFAIL (fail-closed): the kept arms do NOT share one fingerprint, so they are not")
        print("the same model and this is not a budget curve. Fingerprints found:")
        for fp in sorted(fps):
            print("   ", dict(zip(FINGERPRINT_FIELDS, fp)))
        return 2
    fp = next(iter(fps))
    print("\nfingerprint (identical across all arms -- this is what licenses the comparison):")
    print("   " + "  ".join(f"{k}={v}" for k, v in zip(FINGERPRINT_FIELDS, fp)))

    # ---- group by epoch --------------------------------------------------
    by_ep: dict[int, list] = {}
    for r in kept:
        ep = int(float(r["epochs"]))
        by_ep.setdefault(ep, []).append(r)

    print("\n" + "-" * 78)
    print(f"{'epochs':>6} {'n':>3} {'mAP50 mean+-sd':>22} {'DA mean+-sd':>22} {'DA gap vs tiny':>16}")
    print("-" * 78)
    stats = {}
    for ep in sorted(by_ep):
        rs = by_ep[ep]
        da = [_f(r, "da_mIoU") for r in rs]
        mp = [_f(r, "mAP50") for r in rs]
        da = [v for v in da if v is not None]
        mp = [v for v in mp if v is not None]
        seeds = sorted(int(float(r["seed"])) for r in rs)
        da_m = statistics.mean(da)
        mp_m = statistics.mean(mp)
        da_sd = statistics.stdev(da) if len(da) > 1 else None
        mp_sd = statistics.stdev(mp) if len(mp) > 1 else None
        stats[ep] = dict(n=len(rs), seeds=seeds, da=da_m, da_sd=da_sd,
                         mp=mp_m, mp_sd=mp_sd, gap=da_m - TINY_DA,
                         sources=sorted({r["_src"] for r in rs}))
        sd_s = f"{da_sd:.4f}" if da_sd is not None else "n/a"
        msd_s = f"{mp_sd:.4f}" if mp_sd is not None else "n/a"
        print(f"{ep:>6} {len(rs):>3} {mp_m:>10.4f} +- {msd_s:<9} {da_m:>10.4f} +- {sd_s:<9} {da_m - TINY_DA:>+16.4f}")

    eps = sorted(stats)
    print("\nseeds per budget point (a point with n=1 has no error bar -- say so in the paper):")
    for ep in eps:
        s = stats[ep]
        flag = "" if s["n"] > 1 else "   <-- single seed, NO error bar"
        print(f"   {ep:>3} ep : seeds {s['seeds']}{flag}")

    # ---- step significance ----------------------------------------------
    # Two rulers.  The frozen one is the preregistration's, and it is the one that gates.
    # The empirical one is what this family ACTUALLY shows seed-to-seed; it is reported for
    # sensitivity only and is explicitly NOT used to re-gate (preregistration forbids
    # re-estimating the noise scale after seeing the numbers).
    multi = [ep for ep in eps if stats[ep]["n"] > 1 and stats[ep]["da_sd"] is not None]
    emp2 = 2 * statistics.mean([stats[ep]["da_sd"] for ep in multi]) if multi else None

    print("\n" + "-" * 78)
    print("step test -- two rulers (the frozen one gates; the empirical one is sensitivity only)")
    print(f"   frozen     2*sigma = {TWO_SIGMA_DA:.4f}   (Round 3/4 preregistration)")
    if emp2 is not None:
        biggest = max(multi, key=lambda e: stats[e]["da_sd"])
        print(f"   empirical  2*sd    = {emp2:.4f}    (within-arm seed sd {stats[biggest]['da_sd']:.4f}"
              f" at {biggest}ep, n={stats[biggest]['n']})")
        if stats[biggest]["da_sd"] > 1.5 * (TWO_SIGMA_DA / 2):
            print(f"   ** the frozen sigma is {stats[biggest]['da_sd'] / (TWO_SIGMA_DA / 2):.1f}x SMALLER than this family's"
                  " own seed sd **")
    else:
        print("   empirical  2*sd    = n/a (no budget point has >1 seed yet)")
    print("-" * 78)
    verdicts = []
    for a, b in zip(eps, eps[1:]):
        d = stats[b]["gap"] - stats[a]["gap"]
        closing = d > 0
        real = abs(d) >= TWO_SIGMA_DA
        real_emp = (emp2 is not None) and (abs(d) >= emp2)
        verdicts.append((a, b, d, closing, real, real_emp))
        def tag(ok):
            return ("CLOSING" if closing else "WIDENING") if ok else "within noise"
        emp_s = tag(real_emp) if emp2 is not None else "n/a"
        print(f"   {a:>3} -> {b:>3} ep : d(gap) = {d:>+8.4f}   frozen: {tag(real):<10}  empirical: {emp_s}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    frozen_closing = [v for v in verdicts if v[4] and v[3]]
    if frozen_closing and len(frozen_closing) == len(verdicts):
        print("VERDICT (frozen ruler): STILL CLOSING -- every measured step closes the gap")
        print("         beyond the frozen floor, so the preregistration's 'does not change with")
        print("         epoch count' is contradicted by its own successor data.")
    elif frozen_closing:
        print("VERDICT (frozen ruler): PARTIALLY CLOSING -- report per-step, do not generalise.")
    else:
        print("VERDICT (frozen ruler): FLAT / WITHIN NOISE -- the deficit is consistent with")
        print("         being structural.")
    if emp2 is not None:
        emp_closing = [v for v in verdicts if v[5] and v[3]]
        if len(emp_closing) < len(frozen_closing):
            print(f"STRONGER CAVEAT: under this family's OWN seed sd ({emp2 / 2:.4f}), only "
                  f"{len(emp_closing)}/{len(verdicts)} step(s) survive.")
            print("         The frozen ruler is 1.5x+ tighter than the family's actual dispersion,")
            print("         so the frozen verdict is the optimistic one. Report both; claim the weaker.")
    print("=" * 78)

    need = [ep for ep in eps if stats[ep]["n"] == 1]
    if need:
        print("\nCAVEAT (must travel with these numbers): single-seed budget point(s): "
              + ", ".join(f"{e}ep" for e in need))
        print("        seed1/2 for 100ep come from the running seeds chain; re-run this script")
        print("        once they land -- the 100ep point will then carry an error bar.")
    top = max(eps)
    print(f"\ndata completeness: {top}ep has {stats[top]['n']} seed(s). "
          f"Sources: " + ", ".join(sorted({s for ep in eps for s in stats[ep]['sources']})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
