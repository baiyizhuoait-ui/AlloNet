#!/usr/bin/env python3
"""G2 -- seed dispersion at 100 ep, and what it does to the frozen noise ruler.

Zero GPU, read-only.  Closes the G2 gap in docs/PHASE6_PUBLICATION_STRATEGY.md.

What G2 was for
---------------
Every budget point on the G1 curve was single-seed except 20 ep.  Without an
error bar at 100 ep the "gap closes with budget" argument is a comparison of
one point against a band, which a reviewer can dismiss.  With seed1/seed2 on
B100 the 100 ep end now has n=3.

The question this script must answer HONESTLY is not "are the numbers stable"
-- it is *whether the noise ruler the project froze before seeing these data is
still valid at 100 ep*.  Round 3/4 froze per-metric sigmas:

    SIG = {mAP50: 0.0048, lane_mIoU: 0.0019, lane_fg_iou: 0.0031,
           da_mIoU: 0.00115}

derived from 20 ep arms.  A ruler estimated at 20 ep is not automatically the
ruler at 100 ep: longer training gives more room to diverge, so sd can grow.
If it grew, every existing dominance claim that rests on the old floor is
weaker than it was stated to be, and this script must say so rather than
quietly re-scale and keep the claim.

Design rules honoured
---------------------
* The measured sd is reported *next to* the frozen sigma, never substituted
  for it.  Re-estimating a preregistered floor after seeing the data is the
  failure mode the preregistration forbids; what is allowed is reporting that
  the floor is wrong and showing by how much.
* Arms are matched to the budget family by the (variant, z, encoder, params_M,
  flops_G) fingerprint, not by cell name -- B100 and R4-R2thin14 are the same
  network under different names.  A fingerprint mismatch aborts.
* Missing seeds are reported as missing.  No interpolation, no mean-substitution.
* Two-sample contrasts use the pooled sd with n-1 weighting and require the
  |delta| to beat 2 * pooled sd before it is called separable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- frozen consts
# From scripts/phase6_round4_report.py (Round 3/4 preregistration section 7).
FROZEN_SIG = {
    "mAP50": 0.0048,
    "lane_mIoU": 0.0019,
    "lane_fg_iou": 0.0031,
    "da_mIoU": 0.00115,
}
# CSV column -> frozen-ruler key (the ruler predates the current column names).
RULER_KEY = {
    "mAP50": "mAP50",
    "da_mIoU": "da_mIoU",
    "lane_mIoU": "lane_mIoU",
    "lane_fg": "lane_fg_iou",
}

METRICS = ["mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"]
HIGHER_IS_BETTER = {"mAP50", "mAP50_95", "da_mIoU", "da_fg", "lane_mIoU", "lane_fg"}

FINGERPRINT_FIELDS = ("variant", "z", "encoder", "params_M", "flops_G")

# The same network under three names, differing only by --epochs.  Verified by
# fingerprint below.  R6-R2thinL4 shares the fingerprint but is an lr control,
# so it is excluded by name rather than by heuristic.
BUDGET_CELLS = {"R4-R2thin14", "R5-R2thin40", "B100"}
EXCLUDED_CELLS = {"R6-R2thinL4": "lr 1e-4 (F10 learning-rate negative control)"}


def _f(row, key):
    v = row.get(key)
    if v in (None, "", "NA", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load(outdir, name):
    path = os.path.join(outdir, name)
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r.get("cell")]


def summarise(rows, metric):
    """n, mean, sample sd (n-1), min, max, seeds -- with seeds reported."""
    vals, seeds = [], []
    for r in rows:
        v = _f(r, metric)
        if v is not None:
            vals.append(v)
            seeds.append(r.get("seed"))
    if not vals:
        return dict(n=0, mean=None, sd=None, mn=None, mx=None, seeds=[])
    return dict(
        n=len(vals),
        mean=round(st.mean(vals), 6),
        sd=round(st.stdev(vals), 6) if len(vals) > 1 else None,
        mn=round(min(vals), 6),
        mx=round(max(vals), 6),
        seeds=seeds,
    )


def pooled_sd(a, b):
    """Pooled sample sd for two independent groups with n-1 weighting."""
    if a["sd"] is None or b["sd"] is None:
        return None
    na, nb = a["n"], b["n"]
    if na < 2 or nb < 2:
        return None
    num = (na - 1) * a["sd"] ** 2 + (nb - 1) * b["sd"] ** 2
    return (num / (na + nb - 2)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join("experiments", "phase6"))
    ap.add_argument("--out-sub", default="g2")
    args = ap.parse_args()

    base = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    dest = os.path.join(base, args.out_sub)
    os.makedirs(dest, exist_ok=True)

    rows = []
    for name in ("phase6_round4_results.csv", "final_results.csv"):
        for r in load(base, name):
            r["_src"] = name
            rows.append(r)

    # ---- select the budget family ----------------------------------------
    fam, excl = [], []
    for r in rows:
        if r["cell"] in EXCLUDED_CELLS:
            excl.append(r)
        elif r["cell"] in BUDGET_CELLS:
            fam.append(r)

    print("=" * 82)
    print("G2 -- seed dispersion at 100 ep, against the frozen Round 3/4 ruler")
    print("=" * 82)
    print(f"rows scanned: {len(rows)}   budget family: {len(fam)}   registered-excluded: {len(excl)}")
    for r in excl:
        print(f"  excluded {r['cell']:<14} :: {EXCLUDED_CELLS[r['cell']]}")

    if not fam:
        print("\nFAIL: no budget-family rows found. Nothing to report.")
        return 2

    # fingerprint must be one and the same across the whole family
    fps = {tuple(r.get(k) for k in FINGERPRINT_FIELDS) for r in fam}
    if len(fps) != 1:
        print("\nFAIL (fail-closed): the family does not share one fingerprint, so this is")
        print("not a budget curve and the dispersion below would mix models:")
        for fp in sorted(fps):
            print("   ", dict(zip(FINGERPRINT_FIELDS, fp)))
        return 2
    fp = next(iter(fps))
    print(f"fingerprint OK: {dict(zip(FINGERPRINT_FIELDS, fp))}")

    # ---- split by epoch budget ------------------------------------------
    by_ep = {}
    for r in fam:
        ep = r.get("epochs")
        if ep is None:
            continue
        by_ep.setdefault(int(float(ep)), []).append(r)
    budgets = sorted(by_ep)

    print("\n" + "-" * 82)
    print("[1] DISPERSION AT EACH BUDGET POINT  (higher is better on every metric)")
    print("-" * 82)
    header = f"{'metric':<11}" + "".join(f"{str(e) + 'ep':>26}" for e in budgets)
    print(header)
    table = {}
    for m in METRICS:
        cells = []
        for e in budgets:
            s = summarise(by_ep[e], m)
            table.setdefault(m, {})[e] = s
            if s["n"] == 0:
                cells.append(f"{'-':>26}")
            elif s["sd"] is None:
                cells.append(f"{s['mean']:.4f} (n=1)            ")
            else:
                cells.append(f"{s['mean']:.4f}+-{s['sd']:.4f} (n={s['n']})".rjust(26))
        print(f"{m:<11}" + "".join(cells))

    # ---- the actual G2 verdict: measured sd vs frozen ruler -------------
    print("\n" + "-" * 82)
    print("[2] G2 VERDICT -- is the frozen ruler still valid at the largest budget?")
    print("-" * 82)
    biggest = budgets[-1]
    print(f"    reference budget = {biggest} ep  (n={len(by_ep[biggest])} seeds)")
    print(f"\n    {'metric':<11}{'measured sd':>13}{'frozen sigma':>14}{'ratio':>9}   verdict")
    verdicts = {}
    for m, key in RULER_KEY.items():
        s = table[m][biggest]
        fr = FROZEN_SIG[key]
        if s["n"] < 2 or s["sd"] is None:
            verdicts[m] = dict(measured_sd=None, frozen=fr, ratio=None, verdict="NO_DATA")
            print(f"    {m:<11}{'n<2':>13}{fr:>14.5f}{'-':>9}   NO_DATA")
            continue
        ratio = s["sd"] / fr
        if ratio <= 0.85:
            v = "frozen is CONSERVATIVE (safe)"
        elif ratio < 1.15:
            v = "CONSISTENT with frozen"
        else:
            v = f"frozen is {ratio:.1f}x TOO TIGHT -- claims need re-checking"
        verdicts[m] = dict(measured_sd=s["sd"], frozen=fr,
                           ratio=round(ratio, 3), verdict=v)
        print(f"    {m:<11}{s['sd']:>13.5f}{fr:>14.5f}{ratio:>9.2f}   {v}")

    # ---- what it costs the budget argument ------------------------------
    print("\n" + "-" * 82)
    print("[3] BUDGET CONTRAST -- smallest vs largest budget, on the MEASURED ruler")
    print("-" * 82)
    print(f"    {'metric':<11}{'delta':>10}{'2*pooled_sd':>14}{'x floor':>10}   separable?")
    contrasts = {}
    small = budgets[0]
    for m in METRICS:
        a, b = table[m][small], table[m][biggest]
        if a["mean"] is None or b["mean"] is None:
            continue
        d = b["mean"] - a["mean"]
        ps = pooled_sd(a, b)
        if ps is None or ps == 0:
            contrasts[m] = dict(delta=round(d, 6), pooled_sd=ps,
                                floor=None, separable=None)
            print(f"    {m:<11}{d:>+10.4f}{'n/a':>14}{'-':>10}   NO_ERROR_BAR")
            continue
        floor = 2 * ps
        ok = abs(d) > floor
        contrasts[m] = dict(delta=round(d, 6), pooled_sd=round(ps, 6),
                            floor=round(floor, 6), separable=bool(ok),
                            x_floor=round(abs(d) / floor, 2))
        print(f"    {m:<11}{d:>+10.4f}{floor:>14.4f}{abs(d) / floor:>9.2f}x   "
              f"{'YES' if ok else 'no (inside noise)'}")

    # ---- note the fps column is not a model property --------------------
    fps_stats = table.get("fps", {}).get(biggest)
    if fps_stats is None:
        fs = summarise(by_ep[biggest], "fps")
        if fs["n"] > 1 and fs["sd"] is not None:
            print("\n" + "-" * 82)
            print("[4] CAVEAT -- the `fps` column is a host measurement, not a model property")
            print("-" * 82)
            print(f"    100 ep, {fs['n']} seeds, IDENTICAL weights shape: "
                  f"{fs['mn']:.1f} .. {fs['mx']:.1f} fps")
            print(f"    spread {fs['mx'] - fs['mn']:.1f} fps = "
                  f"{(fs['mx'] - fs['mn']) / fs['mean'] * 100:.0f}% of the mean. "
                  "This is machine load, not the seed.")
            print("    Never quote fps as a model difference without pinning the host.")

    out = dict(
        fingerprint=dict(zip(FINGERPRINT_FIELDS, fp)),
        frozen_sigma=FROZEN_SIG,
        budgets=budgets,
        dispersion={m: {str(e): table[m][e] for e in budgets} for m in METRICS},
        g2_verdict=verdicts,
        budget_contrast={"small": small, "big": biggest, "metrics": contrasts},
        _note=("Measured sd is reported beside the frozen sigma, never in place of "
               "it. Frozen sigmas come from 20 ep arms; a change in them at 100 ep "
               "is a finding about the ruler, not a licence to re-freeze it."),
    )
    p = os.path.join(dest, "g2_seed_variance.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    c = os.path.join(dest, "g2_seed_dispersion.csv")
    with open(c, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "epochs", "n", "mean", "sd", "min", "max", "seeds"])
        for m in METRICS:
            for e in budgets:
                s = table[m][e]
                w.writerow([m, e, s["n"], s["mean"], s["sd"], s["mn"], s["mx"],
                            "|".join(str(x) for x in s["seeds"])])

    print(f"\nwrote {os.path.relpath(p, ROOT)}")
    print(f"wrote {os.path.relpath(c, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
